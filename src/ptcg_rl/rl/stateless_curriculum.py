"""Single-writer durable curriculum for the clean stateless policy lineage."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.stateless_curriculum_codec import (
    decode_curriculum_payload,
    encode_curriculum_payload,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONFIG_DOMAIN = b"ptcg-rl/stateless-curriculum-config/v1\x00"
_POLICY_DOMAIN = b"ptcg-rl/stateless-durable-policy/v1\x00"
_MEMBER_DOMAIN = b"ptcg-rl/stateless-pfsp-member/v1\x00"
_ASSIGNMENT_DOMAIN = b"ptcg-rl/stateless-curriculum-assignment/v1\x00"
_STATE_DOMAIN = b"ptcg-rl/stateless-curriculum-state/v1\x00"
_EXTERNAL_STATE_FORMAT = "simple_stateless_curriculum_state_pointer_v2"
_LOGGER = logging.getLogger(__name__)
_LOGGED_FLOOR_RELAXATIONS: set[tuple[int, float, float]] = set()
_FLOOR_RELAXATION_LOG_LOCK = threading.Lock()
_STATE_ENCODING_CACHE_LOCK = threading.Lock()
_STATE_ENCODING_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[StatelessCurriculumState], str | None, bytes | None],
] = {}

CurriculumLane = Literal["mirror", "pfsp", "scripted"]
_CURRICULUM_LANES: tuple[CurriculumLane, ...] = (
    "mirror",
    "pfsp",
    "scripted",
)
MemberSource = Literal["historical_anchor", "fixed_stateless_anchor", "past_self"]
MemberStatus = Literal["active", "retiring"]
TerminalStatus = Literal[
    "engine_terminal",
    "infrastructure_error",
    "step_limit",
    "window_cutoff",
    "cancelled",
    "stale",
    "duplicate",
]


class OpponentLaneMix(BaseModel):
    """Fixed top-level target mass for the three opponent lanes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mirror: float = 0.50
    pfsp: float = 0.35
    scripted: float = 0.15

    @field_validator("mirror", "pfsp", "scripted")
    @classmethod
    def valid_probability(cls, value: float) -> float:
        """Require finite non-negative lane masses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("opponent lane masses must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def sums_to_one(self) -> OpponentLaneMix:
        """Keep top-level mass fixed instead of silently redistributing it."""
        if not math.isclose(
            self.mirror + self.pfsp + self.scripted,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("opponent lane masses must sum to one")
        return self

    def as_mapping(self) -> dict[CurriculumLane, float]:
        """Return lane masses in stable scheduling order."""
        return {
            "mirror": self.mirror,
            "pfsp": self.pfsp,
            "scripted": self.scripted,
        }


class MatchupPfspConfig(BaseModel):
    """Online exact-matchup PFSP controls maintained only in Hydra profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    power: float = Field(default=2.0, gt=0.0)
    epsilon: float = Field(default=0.05, gt=0.0)
    ema_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_evidence: int = Field(default=16, ge=1)
    probability_floor: float = Field(default=0.001, ge=0.0, lt=1.0)
    probability_cap: float = Field(default=0.75, gt=0.0, le=1.0)
    active_artifacts_per_window: int | None = Field(default=None, ge=1)


class PastSelfRetentionConfig(BaseModel):
    """Bounded past-self age coverage, eviction immunity, and reentry controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    age_bucket_ratio: float = Field(default=2.0, gt=1.0)
    difficulty_immunity_enabled: bool = True
    reentry_interval_admissions: int | None = Field(default=None, ge=1)
    reentry_minimum_age_bucket: int = Field(default=2, ge=1)

    @field_validator("age_bucket_ratio")
    @classmethod
    def finite_age_bucket_ratio(cls, value: float) -> float:
        """Require a finite geometric age multiplier."""
        if not math.isfinite(value):
            raise ValueError("past-self age bucket ratio must be finite")
        return value


class StatelessCurriculumConfig(BaseModel):
    """Resolved dynamic-pool identity and mechanical capacity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_mix: OpponentLaneMix = Field(default_factory=OpponentLaneMix)
    pfsp: MatchupPfspConfig = Field(default_factory=MatchupPfspConfig)
    past_self_retention: PastSelfRetentionConfig = Field(
        default_factory=PastSelfRetentionConfig
    )
    replaceable_snapshot_capacity: int = Field(default=4, ge=1)
    past_self_admission_interval_versions: int = Field(default=1, ge=1)
    retain_all_past_self: bool = False
    assignment_seed: int = 0
    pinned_manifest_fingerprint: str
    scripted_manifest_fingerprint: str
    scripted_sampling_fingerprint: str | None = None
    training_roster_fingerprint: str

    @field_validator(
        "pinned_manifest_fingerprint",
        "scripted_manifest_fingerprint",
        "scripted_sampling_fingerprint",
        "training_roster_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str | None) -> str | None:
        """Require resolved immutable roster/manifests."""
        return None if value is None else _require_fingerprint(value)

    @property
    def fingerprint(self) -> str:
        """Return the resume-critical curriculum configuration identity."""
        payload = self.model_dump(mode="json")
        if self.scripted_sampling_fingerprint is None:
            payload.pop("scripted_sampling_fingerprint")
        if not self.retain_all_past_self:
            payload.pop("retain_all_past_self")
        pfsp = payload["pfsp"]
        if isinstance(pfsp, dict) and pfsp.get("active_artifacts_per_window") is None:
            pfsp.pop("active_artifacts_per_window")
        return _fingerprint(_CONFIG_DOMAIN, payload)


class DurablePolicyArtifact(BaseModel):
    """Verified immutable policy checkpoint eligible for past-self admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=0)
    policy_path: Path
    policy_size_bytes: int = Field(ge=1)
    policy_sha256: str
    policy_model_fingerprint: str
    model_config_fingerprint: str
    exact_registry_fingerprint: str
    active_exact_deck_digests: tuple[str, ...]
    input_contract_fingerprint: str
    training_roster_fingerprint: str

    @field_validator(
        "policy_sha256",
        "policy_model_fingerprint",
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "input_contract_fingerprint",
        "training_roster_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require complete pair and policy input identities."""
        return _require_fingerprint(value)

    @field_validator("active_exact_deck_digests")
    @classmethod
    def valid_exact_routes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require distinct sorted exact routes in every admitted pair."""
        normalized = tuple(_require_fingerprint(value) for value in values)
        if not normalized or tuple(sorted(set(normalized))) != normalized:
            raise ValueError("active exact deck digests must be sorted and unique")
        return normalized

    @property
    def fingerprint(self) -> str:
        """Return the path-independent policy content identity."""
        payload = self.model_dump(
            mode="json",
            exclude={"policy_path"},
        )
        return _fingerprint(_POLICY_DOMAIN, payload)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_pair_payload(cls, value: Any) -> Any:
        """Read old nested pair records as their policy-only projection."""
        if not isinstance(value, Mapping) or "learner_state_path" not in value:
            return value
        fields = {
            "version",
            "policy_path",
            "policy_size_bytes",
            "policy_sha256",
            "policy_model_fingerprint",
            "model_config_fingerprint",
            "exact_registry_fingerprint",
            "active_exact_deck_digests",
            "input_contract_fingerprint",
            "training_roster_fingerprint",
        }
        return {name: item for name, item in value.items() if name in fields}

    @classmethod
    def from_manifest(cls, path: Path) -> Self:
        """Project an exact-pair manifest to its immutable policy artifact."""
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint pair manifest must be a mapping")
        if payload.get("format") != "exact_policy_learner_pair_v1":
            raise ValueError("checkpoint pair format is unsupported")
        policy = _required_mapping(payload, "policy")
        metadata = _required_mapping(payload, "metadata")
        identity = _required_mapping(metadata, "simple_stateless_identity")
        policy_path = Path(str(policy["path"])).resolve()
        artifact = cls(
            version=int(payload["version"]),
            policy_path=policy_path,
            policy_size_bytes=int(policy["size_bytes"]),
            policy_sha256=str(policy["sha256"]),
            policy_model_fingerprint=str(policy["model_fingerprint"]),
            model_config_fingerprint=str(identity["model_config_fingerprint"]),
            exact_registry_fingerprint=str(identity["exact_registry_fingerprint"]),
            active_exact_deck_digests=tuple(
                str(value) for value in identity["active_exact_deck_digests"]
            ),
            input_contract_fingerprint=str(identity["input_contract_fingerprint"]),
            training_roster_fingerprint=str(identity["training_roster_fingerprint"]),
        )
        artifact.verify()
        return artifact

    def verify(self) -> None:
        """Recheck the immutable policy bytes."""
        _verify_file(
            self.policy_path,
            size_bytes=self.policy_size_bytes,
            sha256=self.policy_sha256,
            label="policy checkpoint",
        )


@dataclass(frozen=True, slots=True)
class VerifiedPolicyPublication:
    """Same-process capability for policy bytes verified while publishing."""

    artifact: DurablePolicyArtifact
    device: int
    inode: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def capture(cls, artifact: DurablePolicyArtifact) -> Self:
        """Capture the immutable inode identity returned by a durable writer."""
        stat = artifact.policy_path.stat()
        if stat.st_size != artifact.policy_size_bytes:
            raise ValueError("published policy size changed before capture")
        return cls(
            artifact=artifact,
            device=stat.st_dev,
            inode=stat.st_ino,
            modified_ns=stat.st_mtime_ns,
            changed_ns=stat.st_ctime_ns,
        )

    def verify_unchanged(self) -> None:
        """Reject replacement or in-place mutation without rereading the file."""
        stat = self.artifact.policy_path.stat()
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        expected = (
            self.device,
            self.inode,
            self.artifact.policy_size_bytes,
            self.modified_ns,
            self.changed_ns,
        )
        if identity != expected:
            raise ValueError("verified policy publication changed after capture")


# Compatibility for callers that imported the old, overly broad name. New
# persisted curriculum state contains only the policy fields above.
DurablePolicyPairArtifact = DurablePolicyArtifact


class PfspMember(BaseModel):
    """One immutable `(pilot artifact, exact opponent deck)` sampling unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member_id: str
    snapshot_id: str
    source: MemberSource
    pilot_artifact_fingerprint: str
    bundle_fingerprint: str
    exact_deck_digest: str
    policy_path: Path
    policy_size_bytes: int = Field(ge=1)
    policy_sha256: str
    input_contract_fingerprint: str
    exact_registry_fingerprint: str
    base_weight: float = Field(default=1.0, gt=0.0)
    sampling_floor: float = Field(default=0.0, ge=0.0, lt=1.0)
    pinned: bool = False
    status: MemberStatus = "active"
    admission_generation: int = Field(default=0, ge=0)
    leases: int = Field(default=0, ge=0)
    pair: DurablePolicyArtifact | None = None

    @field_validator("member_id", "snapshot_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        """Reject ambiguous member/snapshot aliases."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("PFSP member identifiers must be non-empty")
        return normalized

    @field_validator(
        "pilot_artifact_fingerprint",
        "bundle_fingerprint",
        "exact_deck_digest",
        "policy_sha256",
        "input_contract_fingerprint",
        "exact_registry_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require content identity for every sampled opponent bundle."""
        return _require_fingerprint(value)

    @model_validator(mode="after")
    def coherent_source(self) -> PfspMember:
        """Keep pinned anchors and replaceable past-self semantics disjoint."""
        if self.source in {"historical_anchor", "fixed_stateless_anchor"}:
            if not self.pinned or self.pair is not None:
                raise ValueError("protected anchors must be pinned without a pair")
        elif self.pinned or self.pair is None:
            raise ValueError(
                "past-self members need a durable policy and are replaceable"
            )
        if self.pair is not None:
            if self.policy_path.resolve() != self.pair.policy_path.resolve():
                raise ValueError("past-self member path differs from policy artifact")
            if self.policy_size_bytes != self.pair.policy_size_bytes:
                raise ValueError("past-self member size differs from policy artifact")
            if self.policy_sha256 != self.pair.policy_sha256:
                raise ValueError("past-self member policy artifact differs from member")
            if self.pilot_artifact_fingerprint != self.pair.policy_model_fingerprint:
                raise ValueError("past-self member pilot differs from policy artifact")
            if self.exact_deck_digest not in self.pair.active_exact_deck_digests:
                raise ValueError("past-self member route is absent from its policy")
            if self.input_contract_fingerprint != self.pair.input_contract_fingerprint:
                raise ValueError("past-self member input contract differs from policy")
            if self.exact_registry_fingerprint != self.pair.exact_registry_fingerprint:
                raise ValueError(
                    "past-self member registry differs from policy artifact"
                )
        return self

    def verify(self) -> None:
        """Verify the immutable member artifact before load or exact resume."""
        if self.pair is not None:
            # The model validator already requires the member and pair to name
            # the same path, size, and digest. Hash those bytes only once.
            self.pair.verify()
            return
        _verify_file(
            self.policy_path,
            size_bytes=self.policy_size_bytes,
            sha256=self.policy_sha256,
            label=f"PFSP member {self.member_id}",
        )


def verify_member_artifacts(members: Sequence[PfspMember]) -> None:
    """Hash each distinct policy once without retaining the audit read cache."""
    verified: set[tuple[Path, int, str]] = set()
    for member in members:
        identity = (
            member.policy_path.resolve(),
            member.policy_size_bytes,
            member.policy_sha256,
        )
        if identity in verified:
            continue
        member.verify()
        _discard_audit_page_cache(member.policy_path)
        verified.add(identity)


class ScriptedCurriculumBundle(BaseModel):
    """Small immutable scripted-lane sampling entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_id: str
    artifact_fingerprint: str
    exact_deck_digest: str
    base_weight: float = Field(default=1.0, gt=0.0)

    @field_validator("opponent_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        """Reject anonymous scripted opponents."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("scripted opponent ID must be non-empty")
        return normalized

    @field_validator("artifact_fingerprint", "exact_deck_digest")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable script and deck identities."""
        return _require_fingerprint(value)


class OnlineStatistic(BaseModel):
    """Valid terminal count and candidate-score EMA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int = Field(default=0, ge=0)
    score_ema: float = Field(default=0.5, ge=0.0, le=1.0)
    last_assignment_cursor: int = Field(default=0, ge=0)


class CurriculumAssignment(BaseModel):
    """Immutable assignment identity persisted until terminal/cancellation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    cursor: int = Field(ge=0)
    generation: int = Field(ge=0)
    lane: CurriculumLane
    candidate_deck_digest: str
    candidate_seat: Literal[0, 1]
    opponent_id: str
    opponent_artifact_fingerprint: str
    opponent_pilot_fingerprint: str
    opponent_deck_digest: str
    member_id: str = ""

    @field_validator(
        "candidate_deck_digest",
        "opponent_artifact_fingerprint",
        "opponent_pilot_fingerprint",
        "opponent_deck_digest",
    )
    @classmethod
    def valid_optional_fingerprint(cls, value: str) -> str:
        """Allow empty mirror identities, otherwise require SHA-256."""
        return "" if not value else _require_fingerprint(value)


class CurriculumEvent(BaseModel):
    """Durable admission/retirement audit event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    generation: int = Field(ge=0)
    kind: Literal[
        "admitted",
        "retirement_started",
        "retirement_completed",
    ]
    snapshot_id: str
    member_ids: tuple[str, ...]
    pair_fingerprint: str = ""

    @field_validator("pair_fingerprint")
    @classmethod
    def valid_optional_pair(cls, value: str) -> str:
        """Validate a pair identity when the event has one."""
        return "" if not value else _require_fingerprint(value)


class StatelessCurriculumState(BaseModel):
    """Exact-resume state owned by one curriculum controller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    config_fingerprint: str
    generation: int = Field(default=0, ge=0)
    assignment_cursor: int = Field(default=0, ge=0)
    event_cursor: int = Field(default=0, ge=0)
    members: tuple[PfspMember, ...] = ()
    exact_statistics: dict[str, OnlineStatistic] = Field(default_factory=dict)
    pilot_statistics: dict[str, OnlineStatistic] = Field(default_factory=dict)
    inflight: dict[str, CurriculumAssignment] = Field(default_factory=dict)
    lane_coverage: dict[str, int] = Field(default_factory=dict)
    member_coverage: dict[str, int] = Field(default_factory=dict)
    opponent_deck_coverage: dict[str, int] = Field(default_factory=dict)
    seat_coverage: dict[str, int] = Field(default_factory=dict)
    events: tuple[CurriculumEvent, ...] = ()

    @field_validator("config_fingerprint")
    @classmethod
    def valid_config_fingerprint(cls, value: str) -> str:
        """Require exact configuration recovery."""
        return _require_fingerprint(value)

    @field_validator("members")
    @classmethod
    def unique_members(cls, values: tuple[PfspMember, ...]) -> tuple[PfspMember, ...]:
        """Require unique immutable member and bundle identities."""
        ids = tuple(value.member_id for value in values)
        bundles = tuple(value.bundle_fingerprint for value in values)
        if len(set(ids)) != len(ids) or len(set(bundles)) != len(bundles):
            raise ValueError("PFSP member and bundle identities must be unique")
        return values

    @field_validator(
        "lane_coverage",
        "member_coverage",
        "opponent_deck_coverage",
        "seat_coverage",
    )
    @classmethod
    def non_negative_counters(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject corrupt rolling coverage counters."""
        if any(count < 0 for count in value.values()):
            raise ValueError("curriculum coverage cannot be negative")
        return value


class StatelessCurriculumLaneCoverageRebaseAudit(BaseModel):
    """Evidence for one explicit settled lane-ledger epoch transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-curriculum-lane-coverage-rebase-v1"] = (
        "stateless-curriculum-lane-coverage-rebase-v1"
    )
    apportionment: Literal["hamilton_neutral_deficit_v1"] = (
        "hamilton_neutral_deficit_v1"
    )
    source_config_fingerprint: str
    target_config_fingerprint: str
    source_state_fingerprint: str
    rebased_state_fingerprint: str
    assignment_cursor: int = Field(ge=0)
    source_lane_coverage: dict[CurriculumLane, int]
    source_lane_coverage_total: int = Field(ge=0)
    source_assignment_cursor_gap: int = Field(ge=0)
    target_lane_mix: OpponentLaneMix
    target_lane_coverage: dict[CurriculumLane, int]
    target_lane_coverage_total: int = Field(ge=0)
    target_assignment_cursor_gap: int = Field(ge=0)
    preserved_non_lane_state: Literal[True] = True

    @field_validator(
        "source_config_fingerprint",
        "target_config_fingerprint",
        "source_state_fingerprint",
        "rebased_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable full identities throughout the audit chain."""
        return _require_fingerprint(value)

    @field_validator(
        "source_lane_coverage",
        "target_lane_coverage",
        mode="before",
    )
    @classmethod
    def exact_lane_coverage(
        cls,
        value: object,
    ) -> object:
        """Require one non-negative counter for every curriculum lane."""
        if not isinstance(value, Mapping):
            raise ValueError("audit lane coverage must be a mapping")
        return _exact_lane_coverage(value, label="audit lane coverage")

    @model_validator(mode="after")
    def coherent_totals(self) -> Self:
        """Cross-check every reported total and cursor gap."""
        source_total = sum(self.source_lane_coverage.values())
        target_total = sum(self.target_lane_coverage.values())
        if (
            self.source_config_fingerprint == self.target_config_fingerprint
            or source_total > self.assignment_cursor
            or source_total != self.source_lane_coverage_total
            or target_total != self.target_lane_coverage_total
            or self.source_assignment_cursor_gap
            != self.assignment_cursor - source_total
            or self.target_assignment_cursor_gap
            != self.assignment_cursor - target_total
            or self.target_assignment_cursor_gap
            != self.source_assignment_cursor_gap
        ):
            raise ValueError("lane-coverage rebase audit totals are incoherent")
        expected_target = apportion_neutral_deficit_lane_coverage(
            assignment_cursor=self.assignment_cursor,
            coverage_total=source_total,
            lane_mix=self.target_lane_mix,
        )
        if self.target_lane_coverage != expected_target:
            raise ValueError("lane-coverage rebase audit allocation is incoherent")
        return self


class PreparedPolicyRoutes(Protocol):
    """One fully loaded next route generation awaiting atomic publication."""

    def commit(self) -> None:
        """Publish all staged routes atomically."""

    def abort(self) -> None:
        """Discard staged routes after a failed admission."""


RoutePreparer = Callable[[Sequence[PfspMember]], PreparedPolicyRoutes]
RouteUnloader = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class PreparedPastSelfAdmission:
    """Verified policy, loaded routes, and deterministic state transition."""

    base_generation: int
    base_state_fingerprint: str
    members: tuple[PfspMember, ...]
    retire_snapshot_ids: tuple[str, ...]
    routes: PreparedPolicyRoutes


@dataclass(frozen=True)
class _AdmissionTransition:
    state: StatelessCurriculumState
    unload_member_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTerminalOutcomes:
    """Fully validated terminal cohort awaiting one durable publication."""

    base_state: StatelessCurriculumState
    final_state: StatelessCurriculumState
    valid_terminals: tuple[bool, ...]
    unload_member_ids: tuple[str, ...]


class _NoopPreparedRoutes:
    def commit(self) -> None:
        return None

    def abort(self) -> None:
        return None


class StatelessCurriculumController:
    """Single-writer lane scheduler and durable matchup-conditioned PFSP."""

    def __init__(
        self,
        config: StatelessCurriculumConfig,
        *,
        anchors: Sequence[PfspMember],
        scripted: Sequence[ScriptedCurriculumBundle],
        state_path: Path,
        route_preparer: RoutePreparer | None = None,
        route_unloader: RouteUnloader | None = None,
        resume: bool = False,
        initial_state: StatelessCurriculumState | None = None,
        persist_mutations: bool = True,
    ) -> None:
        """Initialize or exactly restore the one-writer curriculum state."""
        if resume and initial_state is not None:
            raise ValueError(
                "resume and initial curriculum state are mutually exclusive"
            )
        if initial_state is not None and persist_mutations:
            raise ValueError(
                "initial curriculum state requires deferred durable publication"
            )
        self.config = config
        self.state_path = state_path
        self._writer_thread = threading.get_ident()
        self._lock = threading.RLock()
        self._defer_persistence = False
        self._persist_mutations = persist_mutations
        self._persisted_state: StatelessCurriculumState | None = None
        self._route_preparer = route_preparer or (
            lambda _members: _NoopPreparedRoutes()
        )
        self._route_unloader = route_unloader or (lambda _member_ids: None)
        self._scripted = tuple(scripted)
        self._validate_resources(anchors)
        if resume:
            self._state = self._load_state()
            self._validate_resumed_anchors(anchors)
            verify_member_artifacts((*anchors, *self._state.members))
            self._persisted_state = self._state
            self._load_initial_routes()
        elif initial_state is not None:
            if state_path.exists():
                raise FileExistsError(state_path)
            if initial_state.config_fingerprint != config.fingerprint:
                raise ValueError(
                    "initial curriculum state configuration fingerprint mismatch"
                )
            self._state = initial_state
            self._validate_resumed_anchors(anchors)
            verify_member_artifacts((*anchors, *self._state.members))
            self._load_initial_routes()
        else:
            if state_path.exists():
                raise FileExistsError(state_path)
            self._state = StatelessCurriculumState(
                config_fingerprint=config.fingerprint,
                members=tuple(anchors),
            )
            verify_member_artifacts(anchors)
            self._load_initial_routes()
            self._persist(self._state)

    @property
    def state(self) -> StatelessCurriculumState:
        """Return the immutable current controller snapshot."""
        with self._lock:
            return self._state

    @property
    def persisted_state(self) -> StatelessCurriculumState | None:
        """Return the state currently represented by the external state file."""
        with self._lock:
            return self._persisted_state

    def persist_current_state(
        self,
        *,
        expected_predecessor_fingerprint: str | None,
    ) -> None:
        """Publish in-memory state only from the exact recorded predecessor."""
        self._persist_explicit_state(
            self.state,
            expected_predecessor_fingerprint=expected_predecessor_fingerprint,
        )

    def persist_settled_state(
        self,
        state: StatelessCurriculumState,
        *,
        expected_predecessor_fingerprint: str | None,
    ) -> None:
        """Publish an explicit idle snapshot without replacing live lease state."""
        self._require_writer()
        if state.config_fingerprint != self.config.fingerprint:
            raise ValueError(
                "settled curriculum state configuration fingerprint mismatch"
            )
        if state.inflight:
            raise ValueError("settled curriculum state has assignments in flight")
        if any(member.leases != 0 for member in state.members):
            raise ValueError("settled curriculum state has PFSP member leases")
        self._persist_explicit_state(
            state,
            expected_predecessor_fingerprint=expected_predecessor_fingerprint,
        )

    def _persist_explicit_state(
        self,
        state: StatelessCurriculumState,
        *,
        expected_predecessor_fingerprint: str | None,
    ) -> None:
        """CAS-publish one supplied snapshot from the exact durable predecessor."""
        self._require_writer()
        if expected_predecessor_fingerprint is not None:
            _require_fingerprint(expected_predecessor_fingerprint)
        with self._lock:
            tracked_fingerprint = (
                None
                if self._persisted_state is None
                else stateless_curriculum_state_fingerprint(self._persisted_state)
            )
            if tracked_fingerprint != expected_predecessor_fingerprint:
                raise RuntimeError(
                    "tracked curriculum predecessor differs from checkpoint"
                )
            if self.state_path.exists():
                external = load_external_stateless_curriculum_state(self.state_path)
                external_fingerprint = stateless_curriculum_state_fingerprint(external)
            else:
                external_fingerprint = None
            if external_fingerprint != expected_predecessor_fingerprint:
                raise RuntimeError(
                    "external curriculum predecessor differs from checkpoint"
                )
            self._write_external_state(state)

    def prepare_past_self_admission(
        self,
        pair: DurablePolicyArtifact | VerifiedPolicyPublication,
        *,
        snapshot_id: str,
        current_version: int | None = None,
        base_weight: float = 1.0,
        sampling_floor: float = 0.0,
    ) -> PreparedPastSelfAdmission:
        """Verify one durable policy and stage every exact route before mutation."""
        self._require_writer()
        if isinstance(pair, VerifiedPolicyPublication):
            pair.verify_unchanged()
            artifact = pair.artifact
        else:
            pair.verify()
            artifact = pair
        if (
            artifact.training_roster_fingerprint
            != self.config.training_roster_fingerprint
        ):
            raise ValueError("past-self pair training roster differs from controller")
        if not snapshot_id.strip():
            raise ValueError("past-self snapshot ID must be non-empty")
        effective_version = (
            artifact.version if current_version is None else current_version
        )
        if effective_version < artifact.version:
            raise ValueError("past-self admission cannot precede its policy artifact")
        with self._lock:
            if any(
                member.snapshot_id == snapshot_id for member in self._state.members
            ) or any(
                event.kind == "admitted" and event.snapshot_id == snapshot_id
                for event in self._state.events
            ):
                raise ValueError("past-self snapshot identity is immutable")
            if any(
                member.pair is not None
                and member.pair.fingerprint == artifact.fingerprint
                for member in self._state.members
            ):
                raise ValueError("past-self policy artifact is already resident")
            generation = self._state.generation + 1
            members = tuple(
                _past_self_member(
                    artifact,
                    snapshot_id=snapshot_id,
                    exact_deck_digest=deck_digest,
                    base_weight=base_weight,
                    sampling_floor=sampling_floor,
                    generation=generation,
                )
                for deck_digest in artifact.active_exact_deck_digests
            )
            retire = self._retirement_plan_for(effective_version)
            routes = self._route_preparer(members)
            return PreparedPastSelfAdmission(
                base_generation=self._state.generation,
                base_state_fingerprint=stateless_curriculum_state_fingerprint(
                    self._state
                ),
                members=members,
                retire_snapshot_ids=retire,
                routes=routes,
            )

    def select_past_self_reentry(
        self,
        candidates: Sequence[DurablePolicyArtifact],
        *,
        current_version: int,
    ) -> DurablePolicyArtifact | None:
        """Choose the historically hardest artifact in an uncovered old-age bucket."""
        self._require_writer()
        if current_version < 0:
            raise ValueError("past-self current version must be non-negative")
        with self._lock:
            snapshots = self._active_replaceable_snapshots()
            if any(snapshot.pair is None for snapshot in snapshots):
                raise RuntimeError("replaceable snapshot omitted its durable policy")
            occupied_buckets = {
                self._age_bucket(
                    current_version=current_version,
                    snapshot_version=_required_pair(snapshot).version,
                )
                for snapshot in snapshots
            }
            resident_policies = {
                member.policy_sha256
                for member in self._state.members
                if member.pair is not None
            }
            admitted_pairs = {
                event.pair_fingerprint
                for event in self._state.events
                if event.kind == "admitted" and event.pair_fingerprint
            }
            eligible: dict[str, tuple[DurablePolicyArtifact, int]] = {}
            for pair in candidates:
                if (
                    pair.training_roster_fingerprint
                    != self.config.training_roster_fingerprint
                ):
                    raise ValueError(
                        "past-self reentry pair training roster differs from controller"
                    )
                if pair.version >= current_version:
                    continue
                if pair.fingerprint not in admitted_pairs:
                    continue
                if pair.policy_sha256 in resident_policies:
                    continue
                bucket = self._age_bucket(
                    current_version=current_version,
                    snapshot_version=pair.version,
                )
                if (
                    bucket < self.config.past_self_retention.reentry_minimum_age_bucket
                    or bucket in occupied_buckets
                ):
                    continue
                eligible.setdefault(pair.fingerprint, (pair, bucket))
            if not eligible:
                return None

            def selection_key(
                item: tuple[DurablePolicyArtifact, int],
            ) -> tuple[float, int, int, str]:
                pair, bucket = item
                statistic = self._state.pilot_statistics.get(
                    pair.policy_model_fingerprint
                )
                score = 0.5 if statistic is None else statistic.score_ema
                return (score, -bucket, pair.version, pair.fingerprint)

            return min(eligible.values(), key=selection_key)[0]

    def next_past_self_reentry_snapshot_id(
        self,
        pair: DurablePolicyArtifact,
    ) -> str:
        """Create a never-reused reentry identity for one immutable source policy."""
        self._require_writer()
        prefix = f"past-self-reentry-v{pair.version}-r"
        with self._lock:
            ordinal = (
                sum(
                    event.kind == "admitted" and event.snapshot_id.startswith(prefix)
                    for event in self._state.events
                )
                + 1
            )
        return f"{prefix}{ordinal}"

    def preview_past_self_admission(
        self,
        prepared: PreparedPastSelfAdmission,
    ) -> StatelessCurriculumState:
        """Return the exact post-commit state without mutating external state."""
        self._require_writer()
        with self._lock:
            return self._admission_transition(prepared).state

    def preview_settled_past_self_admission(
        self,
        prepared: PreparedPastSelfAdmission,
        settled_state: StatelessCurriculumState,
    ) -> StatelessCurriculumState:
        """Apply a live admission to its pre-assignment durable baseline.

        Distributed one-version-lag collection may hold leases while the learner
        publishes a checkpoint. The live transition retains any leased retiring
        routes, while this projection removes them from the idle recovery state:
        recovery aborts and deterministically reissues the speculative window.
        """
        self._require_writer()
        with self._lock:
            try:
                self._require_prepared_admission_base(prepared)
                self._require_speculative_descendant(settled_state)
                return self._admission_transition_from_state(
                    prepared,
                    settled_state,
                ).state
            except BaseException:
                prepared.routes.abort()
                raise

    def commit_past_self_admission(
        self,
        prepared: PreparedPastSelfAdmission,
    ) -> tuple[str, ...]:
        """Atomically publish staged routes and persist admission/retirement."""
        self._require_writer()
        with self._lock:
            transition = self._admission_transition(prepared)
            try:
                prepared.routes.commit()
                self._persist(transition.state)
            except BaseException:
                prepared.routes.abort()
                raise
            self._state = transition.state
            if transition.unload_member_ids:
                self._route_unloader(transition.unload_member_ids)
            return transition.unload_member_ids

    def _admission_transition(
        self,
        prepared: PreparedPastSelfAdmission,
    ) -> _AdmissionTransition:
        self._require_prepared_admission_base(prepared)
        try:
            return self._admission_transition_from_state(prepared, self._state)
        except BaseException:
            prepared.routes.abort()
            raise

    def _require_prepared_admission_base(
        self,
        prepared: PreparedPastSelfAdmission,
    ) -> None:
        if (
            prepared.base_generation != self._state.generation
            or prepared.base_state_fingerprint
            != stateless_curriculum_state_fingerprint(self._state)
        ):
            prepared.routes.abort()
            raise RuntimeError("curriculum changed after admission was prepared")

    def _require_speculative_descendant(
        self,
        settled_state: StatelessCurriculumState,
    ) -> None:
        """Verify live state differs only by one unsettled assignment cohort."""
        if settled_state.config_fingerprint != self.config.fingerprint:
            raise ValueError(
                "settled admission state configuration fingerprint mismatch"
            )
        if settled_state.inflight:
            raise ValueError("settled admission state has assignments in flight")
        if any(member.leases != 0 for member in settled_state.members):
            raise ValueError("settled admission state has PFSP member leases")
        live_state = self._state
        assignment_delta = (
            live_state.assignment_cursor - settled_state.assignment_cursor
        )
        # The cursor consumes the whole reservation span, while inflight holds
        # only reservations that actually started. Sparse gaps are released
        # reservations, not missing leases.
        if assignment_delta <= 0 or assignment_delta < len(live_state.inflight):
            raise RuntimeError(
                "live curriculum is not one speculative assignment descendant"
            )
        inflight_cursors = {
            assignment.cursor for assignment in live_state.inflight.values()
        }
        if len(inflight_cursors) != len(live_state.inflight) or any(
            cursor < settled_state.assignment_cursor
            or cursor >= live_state.assignment_cursor
            for cursor in inflight_cursors
        ):
            raise RuntimeError(
                "live curriculum has assignments outside its speculative "
                "reservation span"
            )
        normalized_live = live_state.model_copy(
            update={
                "assignment_cursor": settled_state.assignment_cursor,
                "members": tuple(
                    member.model_copy(update={"leases": 0})
                    for member in live_state.members
                ),
                "inflight": {},
                "lane_coverage": settled_state.lane_coverage,
                "member_coverage": settled_state.member_coverage,
                "opponent_deck_coverage": (settled_state.opponent_deck_coverage),
                "seat_coverage": settled_state.seat_coverage,
            }
        )
        if normalized_live != settled_state:
            raise RuntimeError(
                "live curriculum contains non-speculative settled-state changes"
            )

    @staticmethod
    def _admission_transition_from_state(
        prepared: PreparedPastSelfAdmission,
        state: StatelessCurriculumState,
    ) -> _AdmissionTransition:
        if prepared.base_generation != state.generation:
            raise RuntimeError("settled admission generation differs from live state")
        generation = state.generation + 1
        existing = list(state.members)
        retirement_ids: list[str] = []
        for index, member in enumerate(existing):
            if member.snapshot_id not in prepared.retire_snapshot_ids:
                continue
            if member.pinned:
                raise RuntimeError("retirement plan selected a pinned anchor")
            existing[index] = member.model_copy(update={"status": "retiring"})
            retirement_ids.append(member.member_id)
        events = list(state.events)
        event_cursor = state.event_cursor
        if retirement_ids:
            events.append(
                CurriculumEvent(
                    sequence=event_cursor,
                    generation=generation,
                    kind="retirement_started",
                    snapshot_id=",".join(prepared.retire_snapshot_ids),
                    member_ids=tuple(sorted(retirement_ids)),
                )
            )
            event_cursor += 1
        events.append(
            CurriculumEvent(
                sequence=event_cursor,
                generation=generation,
                kind="admitted",
                snapshot_id=prepared.members[0].snapshot_id,
                member_ids=tuple(member.member_id for member in prepared.members),
                pair_fingerprint=prepared.members[0].pair.fingerprint
                if prepared.members[0].pair is not None
                else "",
            )
        )
        event_cursor += 1
        proposed = state.model_copy(
            update={
                "generation": generation,
                "event_cursor": event_cursor,
                "members": tuple(existing) + prepared.members,
                "events": tuple(events),
            }
        )
        removable = tuple(
            member
            for member in proposed.members
            if member.status == "retiring" and member.leases == 0
        )
        if not removable:
            return _AdmissionTransition(proposed, ())
        removable_ids = tuple(member.member_id for member in removable)
        snapshot_ids = tuple(sorted({member.snapshot_id for member in removable}))
        completed = CurriculumEvent(
            sequence=proposed.event_cursor,
            generation=proposed.generation,
            kind="retirement_completed",
            snapshot_id=",".join(snapshot_ids),
            member_ids=removable_ids,
        )
        final = proposed.model_copy(
            update={
                "event_cursor": proposed.event_cursor + 1,
                "members": tuple(
                    member
                    for member in proposed.members
                    if member.member_id not in set(removable_ids)
                ),
                "events": proposed.events + (completed,),
            }
        )
        return _AdmissionTransition(final, removable_ids)

    def assign(
        self,
        *,
        candidate_deck_digest: str,
        candidate_seat: int,
        mirror_policy_fingerprint: str,
        mirror_opponent_deck_digest: str,
        _pfsp_artifacts: frozenset[str] | None = None,
    ) -> CurriculumAssignment:
        """Issue one immutable assignment and lease its original PFSP member."""
        self._require_writer()
        candidate = _require_fingerprint(candidate_deck_digest)
        mirror_policy = _require_fingerprint(mirror_policy_fingerprint)
        mirror_deck = _require_fingerprint(mirror_opponent_deck_digest)
        if candidate_seat not in (0, 1):
            raise ValueError("candidate seat must be zero or one")
        with self._lock:
            lane = self._next_lane()
            cursor = self._state.assignment_cursor
            member: PfspMember | None = None
            scripted: ScriptedCurriculumBundle | None = None
            if lane == "pfsp":
                member = self._sample_pfsp_member(
                    candidate,
                    cursor=cursor,
                    allowed_artifacts=_pfsp_artifacts,
                )
            elif lane == "scripted":
                scripted = self._sample_scripted(cursor=cursor)
            assignment = _assignment(
                cursor=cursor,
                generation=self._state.generation,
                lane=lane,
                candidate_deck_digest=candidate,
                candidate_seat=candidate_seat,
                member=member,
                scripted=scripted,
                seed=self.config.assignment_seed,
                mirror_policy_fingerprint=mirror_policy,
                mirror_opponent_deck_digest=mirror_deck,
            )
            members = list(self._state.members)
            if member is not None:
                index = next(
                    index
                    for index, existing in enumerate(members)
                    if existing.member_id == member.member_id
                )
                members[index] = member.model_copy(update={"leases": member.leases + 1})
            inflight = dict(self._state.inflight)
            inflight[assignment.assignment_id] = assignment
            lane_coverage = _increment(self._state.lane_coverage, lane)
            member_coverage = dict(self._state.member_coverage)
            deck_coverage = dict(self._state.opponent_deck_coverage)
            if member is not None:
                member_coverage = _increment(member_coverage, member.member_id)
                deck_coverage = _increment(
                    deck_coverage,
                    member.exact_deck_digest,
                )
            elif scripted is not None:
                member_coverage = _increment(member_coverage, scripted.opponent_id)
                deck_coverage = _increment(
                    deck_coverage,
                    scripted.exact_deck_digest,
                )
            proposed = self._state.model_copy(
                update={
                    "assignment_cursor": cursor + 1,
                    "members": tuple(members),
                    "inflight": inflight,
                    "lane_coverage": lane_coverage,
                    "member_coverage": member_coverage,
                    "opponent_deck_coverage": deck_coverage,
                    "seat_coverage": _increment(
                        self._state.seat_coverage,
                        str(candidate_seat),
                    ),
                }
            )
            self._persist(proposed)
            self._state = proposed
            return assignment

    def assign_many(
        self,
        requests: Sequence[tuple[str, int, str, str]],
        *,
        pfsp_member_ids: Sequence[str | None] | None = None,
        planned_lanes: Sequence[CurriculumLane] | None = None,
    ) -> tuple[CurriculumAssignment, ...]:
        """Issue one cohort durably with the same ordered assignment semantics."""
        self._require_writer()
        if not requests:
            return ()
        planned_members = None if pfsp_member_ids is None else tuple(pfsp_member_ids)
        if planned_members is not None and len(planned_members) != len(requests):
            raise ValueError("planned PFSP member count differs from requests")
        supplied_lanes = None if planned_lanes is None else tuple(planned_lanes)
        if supplied_lanes is not None and len(supplied_lanes) != len(requests):
            raise ValueError("planned curriculum lane count differs from requests")
        if supplied_lanes is not None and any(
            lane not in {"mirror", "pfsp", "scripted"} for lane in supplied_lanes
        ):
            raise ValueError("planned curriculum lane is invalid")
        with self._lock:
            if self._defer_persistence:
                raise RuntimeError("curriculum mutation batch cannot be nested")
            original = self._state
            active_artifacts = (
                self._plan_pfsp_artifacts(requests) if planned_members is None else None
            )
            lanes = (
                self._preview_lanes(len(requests))
                if supplied_lanes is None
                else supplied_lanes
            )
            member_indexes = {
                member.member_id: index for index, member in enumerate(original.members)
            }
            members_by_id = {member.member_id: member for member in original.members}
            member_leases = [member.leases for member in original.members]
            inflight = dict(original.inflight)
            lane_coverage = dict(original.lane_coverage)
            member_coverage = dict(original.member_coverage)
            deck_coverage = dict(original.opponent_deck_coverage)
            seat_coverage = dict(original.seat_coverage)
            assignments: list[CurriculumAssignment] = []
            cursor = original.assignment_cursor
            for request_index, (request, lane) in enumerate(
                zip(requests, lanes, strict=True)
            ):
                (
                    candidate_deck_digest,
                    candidate_seat,
                    mirror_policy_fingerprint,
                    mirror_opponent_deck_digest,
                ) = request
                candidate = _require_fingerprint(candidate_deck_digest)
                mirror_policy = _require_fingerprint(mirror_policy_fingerprint)
                mirror_deck = _require_fingerprint(mirror_opponent_deck_digest)
                if candidate_seat not in (0, 1):
                    raise ValueError("candidate seat must be zero or one")
                member: PfspMember | None = None
                scripted: ScriptedCurriculumBundle | None = None
                if lane == "pfsp":
                    planned_member_id = (
                        None
                        if planned_members is None
                        else planned_members[request_index]
                    )
                    if planned_members is None:
                        member = self._sample_pfsp_member(
                            candidate,
                            cursor=cursor,
                            allowed_artifacts=active_artifacts,
                        )
                    elif planned_member_id is None:
                        raise ValueError("PFSP lane omitted its planned member")
                    else:
                        try:
                            member = members_by_id[planned_member_id]
                        except KeyError as exc:
                            raise ValueError("planned PFSP member is absent") from exc
                        if member.status != "active":
                            raise ValueError("planned PFSP member is not active")
                elif lane == "scripted":
                    scripted = self._sample_scripted(cursor=cursor)
                if (
                    planned_members is not None
                    and lane != "pfsp"
                    and planned_members[request_index] is not None
                ):
                    raise ValueError("non-PFSP lane declared a planned member")
                assignment = _assignment(
                    cursor=cursor,
                    generation=original.generation,
                    lane=lane,
                    candidate_deck_digest=candidate,
                    candidate_seat=candidate_seat,
                    member=member,
                    scripted=scripted,
                    seed=self.config.assignment_seed,
                    mirror_policy_fingerprint=mirror_policy,
                    mirror_opponent_deck_digest=mirror_deck,
                )
                assignments.append(assignment)
                inflight[assignment.assignment_id] = assignment
                lane_coverage[lane] = lane_coverage.get(lane, 0) + 1
                seat_key = str(candidate_seat)
                seat_coverage[seat_key] = seat_coverage.get(seat_key, 0) + 1
                if member is not None:
                    member_index = member_indexes[member.member_id]
                    member_leases[member_index] += 1
                    member_coverage[member.member_id] = (
                        member_coverage.get(member.member_id, 0) + 1
                    )
                    deck_coverage[member.exact_deck_digest] = (
                        deck_coverage.get(member.exact_deck_digest, 0) + 1
                    )
                elif scripted is not None:
                    member_coverage[scripted.opponent_id] = (
                        member_coverage.get(scripted.opponent_id, 0) + 1
                    )
                    deck_coverage[scripted.exact_deck_digest] = (
                        deck_coverage.get(scripted.exact_deck_digest, 0) + 1
                    )
                cursor += 1
            members = tuple(
                member
                if leases == member.leases
                else member.model_copy(update={"leases": leases})
                for member, leases in zip(
                    original.members,
                    member_leases,
                    strict=True,
                )
            )
            proposed = original.model_copy(
                update={
                    "assignment_cursor": cursor,
                    "members": members,
                    "inflight": inflight,
                    "lane_coverage": lane_coverage,
                    "member_coverage": member_coverage,
                    "opponent_deck_coverage": deck_coverage,
                    "seat_coverage": seat_coverage,
                }
            )
            self._persist(proposed)
            self._state = proposed
            return tuple(assignments)

    def materialize_planned_assignments(
        self,
        requests: Sequence[tuple[str, int, str, str]],
        *,
        pfsp_member_ids: Sequence[str | None],
        planned_lanes: Sequence[CurriculumLane],
        start_cursor: int,
        generation: int,
    ) -> tuple[CurriculumAssignment, ...]:
        """Expand explicit recipes without mutating single-writer state."""
        planned_members = tuple(pfsp_member_ids)
        lanes = tuple(planned_lanes)
        if not (len(requests) == len(planned_members) == len(lanes)):
            raise ValueError("planned curriculum recipe lengths differ")
        if start_cursor < 0 or generation < 0:
            raise ValueError("planned curriculum cursor is invalid")
        with self._lock:
            state = self._state
            if generation != state.generation or start_cursor < state.assignment_cursor:
                raise RuntimeError("planned curriculum recipe targets stale state")
            members_by_id = {
                member.member_id: member
                for member in state.members
                if member.status == "active"
            }
            assignments: list[CurriculumAssignment] = []
            for offset, (request, lane, member_id) in enumerate(
                zip(requests, lanes, planned_members, strict=True)
            ):
                if lane not in {"mirror", "pfsp", "scripted"}:
                    raise ValueError("planned curriculum lane is invalid")
                (
                    candidate_deck_digest,
                    candidate_seat,
                    mirror_policy_fingerprint,
                    mirror_opponent_deck_digest,
                ) = request
                candidate = _require_fingerprint(candidate_deck_digest)
                mirror_policy = _require_fingerprint(mirror_policy_fingerprint)
                mirror_deck = _require_fingerprint(mirror_opponent_deck_digest)
                if candidate_seat not in (0, 1):
                    raise ValueError("candidate seat must be zero or one")
                member: PfspMember | None = None
                scripted: ScriptedCurriculumBundle | None = None
                if lane == "pfsp":
                    if member_id is None:
                        raise ValueError("PFSP recipe omitted its member")
                    try:
                        member = members_by_id[member_id]
                    except KeyError as exc:
                        raise ValueError("PFSP recipe member is absent") from exc
                elif member_id is not None:
                    raise ValueError("non-PFSP recipe declared a member")
                elif lane == "scripted":
                    scripted = self._sample_scripted(cursor=start_cursor + offset)
                assignments.append(
                    _assignment(
                        cursor=start_cursor + offset,
                        generation=generation,
                        lane=lane,
                        candidate_deck_digest=candidate,
                        candidate_seat=candidate_seat,
                        member=member,
                        scripted=scripted,
                        seed=self.config.assignment_seed,
                        mirror_policy_fingerprint=mirror_policy,
                        mirror_opponent_deck_digest=mirror_deck,
                    )
                )
            return tuple(assignments)

    def adopt_assignments(
        self,
        assignments: Sequence[CurriculumAssignment],
        *,
        reservation_count: int | None = None,
    ) -> None:
        """Publish canonical leases and consume their full reservation span."""
        self._require_writer()
        cohort = tuple(assignments)
        reserved = len(cohort) if reservation_count is None else reservation_count
        if reserved < len(cohort) or reserved < 0:
            raise ValueError("curriculum reservation count is invalid")
        if not cohort and reserved == 0:
            return
        with self._lock:
            if self._defer_persistence:
                raise RuntimeError("curriculum mutation batch cannot be nested")
            original = self._state
            reserved_until = original.assignment_cursor + reserved
            cohort_cursors = tuple(item.cursor for item in cohort)
            if tuple(sorted(set(cohort_cursors))) != cohort_cursors or any(
                cursor < original.assignment_cursor or cursor >= reserved_until
                for cursor in cohort_cursors
            ):
                raise ValueError("curriculum adoption leaves its reservation span")
            members_by_id = {member.member_id: member for member in original.members}
            member_indexes = {
                member.member_id: index for index, member in enumerate(original.members)
            }
            member_leases = [member.leases for member in original.members]
            inflight = dict(original.inflight)
            lane_coverage = dict(original.lane_coverage)
            member_coverage = dict(original.member_coverage)
            deck_coverage = dict(original.opponent_deck_coverage)
            seat_coverage = dict(original.seat_coverage)
            for assignment in cohort:
                if (
                    assignment.generation != original.generation
                    or assignment.assignment_id in inflight
                ):
                    raise ValueError("curriculum adoption cohort is not canonical")
                member: PfspMember | None = None
                scripted: ScriptedCurriculumBundle | None = None
                mirror_policy = assignment.opponent_pilot_fingerprint
                mirror_deck = assignment.opponent_deck_digest
                if assignment.lane == "pfsp":
                    try:
                        member = members_by_id[assignment.member_id]
                    except KeyError as exc:
                        raise ValueError(
                            "curriculum adoption member is absent"
                        ) from exc
                    if member.status != "active":
                        raise ValueError("curriculum adoption member is not active")
                elif assignment.member_id:
                    raise ValueError("non-PFSP curriculum lease declared a member")
                elif assignment.lane == "scripted":
                    scripted = self._sample_scripted(cursor=assignment.cursor)
                expected = _assignment(
                    cursor=assignment.cursor,
                    generation=assignment.generation,
                    lane=assignment.lane,
                    candidate_deck_digest=assignment.candidate_deck_digest,
                    candidate_seat=assignment.candidate_seat,
                    member=member,
                    scripted=scripted,
                    seed=self.config.assignment_seed,
                    mirror_policy_fingerprint=mirror_policy,
                    mirror_opponent_deck_digest=mirror_deck,
                )
                if assignment != expected:
                    raise ValueError("curriculum adoption lease is not canonical")
                inflight[assignment.assignment_id] = assignment
                lane_coverage = _increment(lane_coverage, assignment.lane)
                seat_coverage = _increment(
                    seat_coverage,
                    str(assignment.candidate_seat),
                )
                if member is not None:
                    index = member_indexes[member.member_id]
                    member_leases[index] += 1
                    member_coverage = _increment(member_coverage, member.member_id)
                    deck_coverage = _increment(
                        deck_coverage,
                        member.exact_deck_digest,
                    )
                elif scripted is not None:
                    member_coverage = _increment(
                        member_coverage,
                        scripted.opponent_id,
                    )
                    deck_coverage = _increment(
                        deck_coverage,
                        scripted.exact_deck_digest,
                    )
            members = tuple(
                member
                if leases == member.leases
                else member.model_copy(update={"leases": leases})
                for member, leases in zip(
                    original.members,
                    member_leases,
                    strict=True,
                )
            )
            proposed = original.model_copy(
                update={
                    "assignment_cursor": reserved_until,
                    "members": members,
                    "inflight": inflight,
                    "lane_coverage": lane_coverage,
                    "member_coverage": member_coverage,
                    "opponent_deck_coverage": deck_coverage,
                    "seat_coverage": seat_coverage,
                }
            )
            self._persist(proposed)
            self._state = proposed

    def preview_lanes(self, count: int) -> tuple[CurriculumLane, ...]:
        """Preview the next immutable lane portfolio without issuing leases."""
        self._require_writer()
        if count < 0:
            raise ValueError("lane preview count must be non-negative")
        with self._lock:
            return self._preview_lanes(count)

    def pfsp_member_priority(self, candidate: str, member_id: str) -> float:
        """Return the current exact-matchup PFSP weight for one active member."""
        return self.pfsp_member_priorities(((candidate, member_id),))[0]

    def pfsp_member_priorities(
        self,
        requests: Sequence[tuple[str, str]],
    ) -> tuple[float, ...]:
        """Compute one matchup-weight cohort under a single state snapshot."""
        self._require_writer()
        normalized = tuple(
            (_require_fingerprint(candidate), member_id)
            for candidate, member_id in requests
        )
        with self._lock:
            members = {
                item.member_id: item
                for item in self._state.members
                if item.status == "active"
            }
            if any(member_id not in members for _candidate, member_id in normalized):
                raise ValueError("PFSP priority member is absent or inactive")
            result: list[float] = []
            for candidate_digest, member_id in normalized:
                member = members[member_id]
                key = _matchup_key(
                    candidate_digest,
                    member.pilot_artifact_fingerprint,
                    member.exact_deck_digest,
                )
                statistic = self._state.exact_statistics.get(key)
                pilot = self._state.pilot_statistics.get(
                    member.pilot_artifact_fingerprint
                )
                fallback = 0.5 if pilot is None else pilot.score_ema
                score = _blended_score(
                    statistic,
                    fallback=fallback,
                    minimum_evidence=self.config.pfsp.minimum_evidence,
                )
                result.append(
                    float(
                        member.base_weight
                        * (
                            (1.0 - score) ** self.config.pfsp.power
                            + self.config.pfsp.epsilon
                        )
                    )
                )
            return tuple(result)

    def pfsp_member_evidence(self, candidate: str, member_id: str) -> int:
        """Return exact terminal evidence for one candidate/member matchup."""
        return self.pfsp_member_evidence_many(((candidate, member_id),))[0]

    def pfsp_member_scores(
        self,
        requests: Sequence[tuple[str, str]],
    ) -> tuple[float, ...]:
        """Return evidence-shrunk candidate scores under one state snapshot."""
        return tuple(
            score for score, _evidence in self.pfsp_member_observations(requests)
        )

    def pfsp_member_observations(
        self,
        requests: Sequence[tuple[str, str]],
    ) -> tuple[tuple[float, int], ...]:
        """Return score and evidence together under one state snapshot."""
        self._require_writer()
        normalized = tuple(
            (_require_fingerprint(candidate), member_id)
            for candidate, member_id in requests
        )
        with self._lock:
            members = {
                item.member_id: item
                for item in self._state.members
                if item.status == "active"
            }
            if any(member_id not in members for _candidate, member_id in normalized):
                raise ValueError("PFSP score member is absent or inactive")
            result: list[tuple[float, int]] = []
            for candidate_digest, member_id in normalized:
                member = members[member_id]
                statistic = self._state.exact_statistics.get(
                    _matchup_key(
                        candidate_digest,
                        member.pilot_artifact_fingerprint,
                        member.exact_deck_digest,
                    )
                )
                pilot = self._state.pilot_statistics.get(
                    member.pilot_artifact_fingerprint
                )
                result.append(
                    (
                        _blended_score(
                            statistic,
                            fallback=0.5 if pilot is None else pilot.score_ema,
                            minimum_evidence=self.config.pfsp.minimum_evidence,
                        ),
                        0 if statistic is None else statistic.games,
                    )
                )
            return tuple(result)

    def pfsp_member_evidence_many(
        self,
        requests: Sequence[tuple[str, str]],
    ) -> tuple[int, ...]:
        """Read an exact-evidence cohort under a single state snapshot."""
        return tuple(
            evidence for _score, evidence in self.pfsp_member_observations(requests)
        )

    def observe_terminal(
        self,
        assignment_id: str,
        *,
        status: TerminalStatus,
        candidate_score: float | None = None,
    ) -> bool:
        """Release one lease and update PFSP only for a valid engine terminal."""
        prepared = self.prepare_terminals(((assignment_id, status, candidate_score),))
        self.commit_prepared_terminals(prepared)
        return prepared.valid_terminals[0]

    def observe_terminals(
        self,
        observations: Sequence[tuple[str, TerminalStatus, float | None]],
    ) -> tuple[bool, ...]:
        """Commit one validated terminal cohort with a single durable publish."""
        prepared = self.prepare_terminals(observations)
        if not observations:
            return ()
        self.commit_prepared_terminals(prepared)
        return prepared.valid_terminals

    def prepare_terminals(
        self,
        observations: Sequence[tuple[str, TerminalStatus, float | None]],
    ) -> PreparedTerminalOutcomes:
        """Derive and fully validate a cohort without mutation or persistence."""
        self._require_writer()
        assignment_ids = tuple(item[0] for item in observations)
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("terminal cohort contains duplicate assignments")
        with self._lock:
            if self._defer_persistence:
                raise RuntimeError("curriculum mutation batch cannot be nested")
            base_state = self._state
            if len(observations) == 0:
                return PreparedTerminalOutcomes(
                    base_state=base_state,
                    final_state=base_state,
                    valid_terminals=(),
                    unload_member_ids=(),
                )
            inflight = dict(base_state.inflight)
            exact_statistics = dict(base_state.exact_statistics)
            pilot_statistics = dict(base_state.pilot_statistics)
            member_indexes = {
                member.member_id: index
                for index, member in enumerate(base_state.members)
            }
            member_leases = [member.leases for member in base_state.members]
            member_present = [True] * len(base_state.members)
            events = list(base_state.events)
            event_cursor = base_state.event_cursor
            valid_terminals: list[bool] = []
            unload_member_ids: list[str] = []
            for observation_index, (
                assignment_id,
                status,
                candidate_score,
            ) in enumerate(observations):
                assignment = inflight.get(assignment_id)
                if assignment is None:
                    raise ValueError("assignment is stale, duplicate, or unknown")
                valid = status == "engine_terminal" or (
                    status == "step_limit" and candidate_score in (0.0, 1.0)
                )
                if valid and (
                    candidate_score is None
                    or not math.isfinite(candidate_score)
                    or candidate_score not in (0.0, 0.5, 1.0)
                ):
                    raise ValueError("scored outcome must be 0, 0.5, or 1")
                if valid and assignment.lane == "pfsp":
                    if candidate_score is None:
                        raise AssertionError("validated terminal score is missing")
                    score = float(candidate_score)
                    exact_key = _matchup_key(
                        assignment.candidate_deck_digest,
                        assignment.opponent_pilot_fingerprint,
                        assignment.opponent_deck_digest,
                    )
                    exact_statistics[exact_key] = _updated_statistic(
                        exact_statistics.get(exact_key),
                        score=score,
                        alpha=self.config.pfsp.ema_alpha,
                        cursor=assignment.cursor,
                    )
                    pilot_key = assignment.opponent_pilot_fingerprint
                    pilot_statistics[pilot_key] = _updated_statistic(
                        pilot_statistics.get(pilot_key),
                        score=score,
                        alpha=self.config.pfsp.ema_alpha,
                        cursor=assignment.cursor,
                    )

                changed_member_index: int | None = None
                if assignment.member_id:
                    changed_member_index = member_indexes.get(assignment.member_id)
                    if (
                        changed_member_index is None
                        or not member_present[changed_member_index]
                        or member_leases[changed_member_index] <= 0
                    ):
                        raise RuntimeError("assignment PFSP lease is missing")
                    member_leases[changed_member_index] -= 1
                del inflight[assignment_id]

                if observation_index == 0:
                    removable_indexes = [
                        index
                        for index, member in enumerate(base_state.members)
                        if (
                            member_present[index]
                            and member.status == "retiring"
                            and member_leases[index] == 0
                        )
                    ]
                elif (
                    changed_member_index is not None
                    and member_present[changed_member_index]
                    and base_state.members[changed_member_index].status == "retiring"
                    and member_leases[changed_member_index] == 0
                ):
                    removable_indexes = [changed_member_index]
                else:
                    removable_indexes = []
                if removable_indexes:
                    removable_ids = tuple(
                        base_state.members[index].member_id
                        for index in removable_indexes
                    )
                    snapshot_ids = tuple(
                        sorted(
                            {
                                base_state.members[index].snapshot_id
                                for index in removable_indexes
                            }
                        )
                    )
                    events.append(
                        CurriculumEvent(
                            sequence=event_cursor,
                            generation=base_state.generation,
                            kind="retirement_completed",
                            snapshot_id=",".join(snapshot_ids),
                            member_ids=removable_ids,
                        )
                    )
                    event_cursor += 1
                    for index in removable_indexes:
                        member_present[index] = False
                    unload_member_ids.extend(removable_ids)
                valid_terminals.append(valid)

            members = tuple(
                member
                if member_leases[index] == member.leases
                else member.model_copy(update={"leases": member_leases[index]})
                for index, member in enumerate(base_state.members)
                if member_present[index]
            )
            final_state = base_state.model_copy(
                update={
                    "event_cursor": event_cursor,
                    "members": members,
                    "exact_statistics": exact_statistics,
                    "pilot_statistics": pilot_statistics,
                    "inflight": inflight,
                    "events": tuple(events),
                }
            )
            return PreparedTerminalOutcomes(
                base_state=base_state,
                final_state=final_state,
                valid_terminals=tuple(valid_terminals),
                unload_member_ids=tuple(unload_member_ids),
            )

    def validate_prepared_terminals(
        self,
        prepared: PreparedTerminalOutcomes,
    ) -> None:
        """Reject a prepared transition after any intervening mutation."""
        self._require_writer()
        with self._lock:
            if self._state is not prepared.base_state:
                raise RuntimeError("curriculum changed after terminals were prepared")

    def persist_prepared_terminals(
        self,
        prepared: PreparedTerminalOutcomes,
    ) -> None:
        """Durably write a validated final state without publishing it in memory."""
        self._require_writer()
        with self._lock:
            if self._state is not prepared.base_state:
                raise RuntimeError("curriculum changed after terminals were prepared")
            self._persist(prepared.final_state)

    def publish_prepared_terminals(
        self,
        prepared: PreparedTerminalOutcomes,
    ) -> None:
        """Publish a durably written cohort and release retired routes."""
        self._require_writer()
        with self._lock:
            if self._state is not prepared.base_state:
                raise RuntimeError("curriculum changed after terminals were prepared")
            self._state = prepared.final_state
            if prepared.unload_member_ids:
                self._route_unloader(prepared.unload_member_ids)

    def commit_prepared_terminals(
        self,
        prepared: PreparedTerminalOutcomes,
    ) -> None:
        """Persist and publish one prepared terminal cohort."""
        self.persist_prepared_terminals(prepared)
        self.publish_prepared_terminals(prepared)

    def pfsp_probabilities(
        self,
        candidate_deck_digest: str,
    ) -> dict[str, float]:
        """Expose exact conditional member probabilities for observability."""
        candidate = _require_fingerprint(candidate_deck_digest)
        with self._lock:
            members, probabilities = self._pfsp_distribution(candidate)
            return {
                member.member_id: probability
                for member, probability in zip(
                    members,
                    probabilities,
                    strict=True,
                )
            }

    def _validate_resources(self, anchors: Sequence[PfspMember]) -> None:
        anchor_ids = tuple(anchor.member_id for anchor in anchors)
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("historical anchor IDs must be unique")
        for anchor in anchors:
            if anchor.source not in {"historical_anchor", "fixed_stateless_anchor"}:
                raise ValueError("initial PFSP resources must be protected anchors")
        if self.config.lane_mix.pfsp > 0.0 and not anchors:
            raise ValueError("PFSP lane has positive mass but no historical anchor")
        if self.config.lane_mix.scripted > 0.0 and not self._scripted:
            raise ValueError("scripted lane has positive mass but no manifest entries")
        scripted_ids = tuple(item.opponent_id for item in self._scripted)
        if len(set(scripted_ids)) != len(scripted_ids):
            raise ValueError("scripted curriculum IDs must be unique")

    def _load_initial_routes(self) -> None:
        inflight_member_ids = {
            assignment.member_id
            for assignment in self._state.inflight.values()
            if assignment.member_id
        }
        required = tuple(
            member
            for member in self._state.members
            if member.status == "active"
            or member.leases > 0
            or member.member_id in inflight_member_ids
        )
        prepared = self._route_preparer(required)
        try:
            prepared.commit()
        except BaseException:
            prepared.abort()
            raise

    def _load_state(self) -> StatelessCurriculumState:
        state = load_external_stateless_curriculum_state(self.state_path)
        if state.config_fingerprint != self.config.fingerprint:
            raise ValueError("curriculum state configuration fingerprint mismatch")
        return state

    def _validate_resumed_anchors(self, anchors: Sequence[PfspMember]) -> None:
        expected = {member.member_id: member.bundle_fingerprint for member in anchors}
        actual = {
            member.member_id: member.bundle_fingerprint
            for member in self._state.members
            if member.pinned
        }
        if actual != expected:
            raise ValueError("exact resume historical anchor identity mismatch")

    def _retirement_plan_for(self, current_version: int) -> tuple[str, ...]:
        if self.config.retain_all_past_self:
            return ()
        snapshots = self._active_replaceable_snapshots()
        if len(snapshots) < self.config.replaceable_snapshot_capacity:
            return ()
        if any(snapshot.pair is None for snapshot in snapshots):
            raise RuntimeError("replaceable snapshot omitted its durable policy")

        immune_snapshot_id = self._difficulty_immune_snapshot(snapshots)
        buckets: dict[int, list[PfspMember]] = {}
        for snapshot in snapshots:
            bucket = self._age_bucket(
                current_version=current_version,
                snapshot_version=_required_pair(snapshot).version,
            )
            buckets.setdefault(bucket, []).append(snapshot)
        selectable_buckets = {
            bucket: members
            for bucket, members in buckets.items()
            if any(member.snapshot_id != immune_snapshot_id for member in members)
        }
        if not selectable_buckets:
            # A capacity of one cannot make its only member immune without
            # violating the hard replaceable-snapshot bound.
            selectable_buckets = buckets
            immune_snapshot_id = None
        maximum_occupancy = max(len(buckets[bucket]) for bucket in selectable_buckets)
        selected_bucket = min(
            bucket
            for bucket in selectable_buckets
            if len(buckets[bucket]) == maximum_occupancy
        )
        candidates = tuple(
            member
            for member in buckets[selected_bucket]
            if member.snapshot_id != immune_snapshot_id
        )
        if not candidates:
            raise RuntimeError("past-self retirement has no selectable snapshot")

        def eviction_key(member: PfspMember) -> tuple[float, int, str]:
            statistic = self._state.pilot_statistics.get(
                member.pilot_artifact_fingerprint
            )
            score = 0.5 if statistic is None else statistic.score_ema
            return (score, _required_pair(member).version, member.snapshot_id)

        return (max(candidates, key=eviction_key).snapshot_id,)

    def _active_replaceable_snapshots(self) -> tuple[PfspMember, ...]:
        snapshots: dict[str, PfspMember] = {}
        for member in self._state.members:
            if member.pinned or member.status != "active":
                continue
            previous = snapshots.setdefault(member.snapshot_id, member)
            if (
                previous.pair != member.pair
                or previous.admission_generation != member.admission_generation
            ):
                raise RuntimeError("past-self snapshot members disagree on identity")
        return tuple(snapshots.values())

    def _difficulty_immune_snapshot(
        self,
        snapshots: Sequence[PfspMember],
    ) -> str | None:
        if (
            not self.config.past_self_retention.difficulty_immunity_enabled
            or len(snapshots) <= 1
        ):
            return None
        eligible: list[tuple[float, int, str]] = []
        for snapshot in snapshots:
            statistic = self._state.pilot_statistics.get(
                snapshot.pilot_artifact_fingerprint
            )
            if statistic is None or statistic.games < self.config.pfsp.minimum_evidence:
                continue
            eligible.append(
                (
                    statistic.score_ema,
                    -statistic.games,
                    snapshot.snapshot_id,
                )
            )
        return None if not eligible else min(eligible)[2]

    def _age_bucket(
        self,
        *,
        current_version: int,
        snapshot_version: int,
    ) -> int:
        age = current_version - snapshot_version
        if age < 0:
            raise ValueError("past-self snapshot is newer than the admission version")
        unit = self.config.past_self_admission_interval_versions
        if age <= unit:
            return 0
        ratio = self.config.past_self_retention.age_bucket_ratio
        bucket = int(math.floor(math.log(age / float(unit), ratio)))
        return min(
            max(bucket, 0),
            self.config.replaceable_snapshot_capacity - 1,
        )

    def _next_lane(self) -> CurriculumLane:
        masses = self.config.lane_mix.as_mapping()
        issued = self._state.assignment_cursor
        coverage = self._state.lane_coverage
        lanes: tuple[CurriculumLane, ...] = ("mirror", "pfsp", "scripted")
        return max(
            lanes,
            key=lambda lane: (
                masses[lane] * (issued + 1) - coverage.get(lane, 0),
                -lanes.index(lane),
            ),
        )

    def _sample_pfsp_member(
        self,
        candidate: str,
        *,
        cursor: int,
        allowed_artifacts: frozenset[str] | None = None,
    ) -> PfspMember:
        members, probabilities = self._pfsp_distribution(candidate)
        if allowed_artifacts is not None:
            selected = tuple(
                (member, probability)
                for member, probability in zip(
                    members,
                    probabilities,
                    strict=True,
                )
                if member.policy_sha256 in allowed_artifacts
            )
            if not selected:
                raise RuntimeError(
                    "PFSP active artifact set contains no eligible member"
                )
            members = tuple(member for member, _probability in selected)
            total = sum(probability for _member, probability in selected)
            probabilities = tuple(
                probability / total for _member, probability in selected
            )
        index = _sample_index(
            probabilities,
            seed=self.config.assignment_seed,
            cursor=cursor,
            domain="pfsp",
        )
        return members[index]

    def _plan_pfsp_artifacts(
        self,
        requests: Sequence[tuple[str, int, str, str]],
    ) -> frozenset[str] | None:
        """Freeze a hierarchical deficit-selected checkpoint set for one cohort."""
        limit = self.config.pfsp.active_artifacts_per_window
        if limit is None:
            return None
        active_members = tuple(
            member for member in self._state.members if member.status == "active"
        )
        artifacts = tuple(sorted({member.policy_sha256 for member in active_members}))
        if limit >= len(artifacts):
            return None

        candidates = tuple(
            _require_fingerprint(request[0])
            for request, lane in zip(
                requests,
                self._preview_lanes(len(requests)),
                strict=True,
            )
            if lane == "pfsp"
        )
        if not candidates:
            return None
        target_mass: dict[str, float] = dict.fromkeys(artifacts, 0.0)
        for candidate in candidates:
            members, probabilities = self._pfsp_distribution(candidate)
            for member, probability in zip(
                members,
                probabilities,
                strict=True,
            ):
                target_mass[member.policy_sha256] += probability
        inverse_windows = 1.0 / len(candidates)
        coverage = dict.fromkeys(artifacts, 0)
        for member in active_members:
            coverage[member.policy_sha256] += self._state.member_coverage.get(
                member.member_id,
                0,
            )
        expected_total = sum(coverage.values()) + len(candidates)
        ordered = sorted(
            artifacts,
            key=lambda artifact: (
                -(
                    target_mass[artifact] * inverse_windows * expected_total
                    - coverage[artifact]
                ),
                artifact,
            ),
        )
        return frozenset(ordered[:limit])

    def _preview_lanes(self, count: int) -> tuple[CurriculumLane, ...]:
        """Preview deterministic lane deficits without mutating durable state."""
        issued = self._state.assignment_cursor
        coverage = dict(self._state.lane_coverage)
        masses = self.config.lane_mix.as_mapping()
        lane_order: tuple[CurriculumLane, ...] = (
            "mirror",
            "pfsp",
            "scripted",
        )
        result: list[CurriculumLane] = []
        for _index in range(count):
            lane = max(
                lane_order,
                key=lambda item: (
                    masses[item] * (issued + 1) - coverage.get(item, 0),
                    -lane_order.index(item),
                ),
            )
            result.append(lane)
            coverage[lane] = coverage.get(lane, 0) + 1
            issued += 1
        return tuple(result)

    def _pfsp_distribution(
        self,
        candidate: str,
    ) -> tuple[tuple[PfspMember, ...], tuple[float, ...]]:
        members = tuple(
            member for member in self._state.members if member.status == "active"
        )
        if not members:
            raise RuntimeError("PFSP lane has no active member")
        weights: list[float] = []
        floors: list[float] = []
        for member in members:
            key = _matchup_key(
                candidate,
                member.pilot_artifact_fingerprint,
                member.exact_deck_digest,
            )
            statistic = self._state.exact_statistics.get(key)
            pilot = self._state.pilot_statistics.get(member.pilot_artifact_fingerprint)
            fallback = 0.5 if pilot is None else pilot.score_ema
            score = _blended_score(
                statistic,
                fallback=fallback,
                minimum_evidence=self.config.pfsp.minimum_evidence,
            )
            weights.append(
                member.base_weight
                * ((1.0 - score) ** self.config.pfsp.power + self.config.pfsp.epsilon)
            )
            floors.append(
                max(member.sampling_floor, self.config.pfsp.probability_floor)
            )
        probabilities = _bounded_distribution(
            weights,
            floors=floors,
            cap=self.config.pfsp.probability_cap,
        )
        return (members, probabilities)

    def _sample_scripted(self, *, cursor: int) -> ScriptedCurriculumBundle:
        weights = tuple(item.base_weight for item in self._scripted)
        total = sum(weights)
        probabilities = tuple(weight / total for weight in weights)
        index = _sample_index(
            probabilities,
            seed=self.config.assignment_seed,
            cursor=cursor,
            domain="scripted",
        )
        return self._scripted[index]

    def _persist(self, state: StatelessCurriculumState) -> None:
        if self._defer_persistence or not self._persist_mutations:
            return
        self._write_external_state(state)

    def _write_external_state(self, state: StatelessCurriculumState) -> None:
        publish_external_stateless_curriculum_state(self.state_path, state)
        self._persisted_state = state

    def _require_writer(self) -> None:
        if threading.get_ident() != self._writer_thread:
            raise RuntimeError("curriculum mutations require the controller writer")


def historical_anchor_member(
    *,
    member_id: str,
    snapshot_id: str,
    pilot_artifact_fingerprint: str,
    bundle_fingerprint: str,
    exact_deck_digest: str,
    policy_path: Path,
    policy_size_bytes: int,
    policy_sha256: str,
    input_contract_fingerprint: str,
    exact_registry_fingerprint: str,
    base_weight: float = 1.0,
    sampling_floor: float = 0.0,
    source: Literal["historical_anchor", "fixed_stateless_anchor"] = (
        "historical_anchor"
    ),
) -> PfspMember:
    """Build a pinned historical bundle with explicit immutable identity."""
    return PfspMember(
        member_id=member_id,
        snapshot_id=snapshot_id,
        source=source,
        pilot_artifact_fingerprint=pilot_artifact_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        exact_deck_digest=exact_deck_digest,
        policy_path=policy_path,
        policy_size_bytes=policy_size_bytes,
        policy_sha256=policy_sha256,
        input_contract_fingerprint=input_contract_fingerprint,
        exact_registry_fingerprint=exact_registry_fingerprint,
        base_weight=base_weight,
        sampling_floor=sampling_floor,
        pinned=True,
    )


def _past_self_member(
    pair: DurablePolicyArtifact,
    *,
    snapshot_id: str,
    exact_deck_digest: str,
    base_weight: float,
    sampling_floor: float,
    generation: int,
) -> PfspMember:
    payload = {
        "pair_fingerprint": pair.fingerprint,
        "exact_deck_digest": exact_deck_digest,
        "input_contract_fingerprint": pair.input_contract_fingerprint,
    }
    bundle_fingerprint = _fingerprint(_MEMBER_DOMAIN, payload)
    return PfspMember(
        member_id=f"{snapshot_id}:{exact_deck_digest}",
        snapshot_id=snapshot_id,
        source="past_self",
        pilot_artifact_fingerprint=pair.policy_model_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        exact_deck_digest=exact_deck_digest,
        policy_path=pair.policy_path,
        policy_size_bytes=pair.policy_size_bytes,
        policy_sha256=pair.policy_sha256,
        input_contract_fingerprint=pair.input_contract_fingerprint,
        exact_registry_fingerprint=pair.exact_registry_fingerprint,
        base_weight=base_weight,
        sampling_floor=sampling_floor,
        admission_generation=generation,
        pair=pair,
    )


def _required_pair(member: PfspMember) -> DurablePolicyArtifact:
    """Return the durable policy required by a replaceable snapshot."""
    if member.pair is None:
        raise RuntimeError("replaceable snapshot omitted its durable policy")
    return member.pair


def _assignment(
    *,
    cursor: int,
    generation: int,
    lane: CurriculumLane,
    candidate_deck_digest: str,
    candidate_seat: int,
    member: PfspMember | None,
    scripted: ScriptedCurriculumBundle | None,
    seed: int,
    mirror_policy_fingerprint: str,
    mirror_opponent_deck_digest: str,
) -> CurriculumAssignment:
    if lane == "pfsp" and member is None:
        raise AssertionError("PFSP assignment has no member")
    if lane == "scripted" and scripted is None:
        raise AssertionError("scripted assignment has no bundle")
    opponent_id = ""
    artifact = ""
    pilot = ""
    deck = ""
    member_id = ""
    if lane == "mirror":
        opponent_id = "current_policy_mirror"
        artifact = mirror_policy_fingerprint
        pilot = mirror_policy_fingerprint
        deck = mirror_opponent_deck_digest
    elif member is not None:
        opponent_id = member.member_id
        artifact = member.bundle_fingerprint
        pilot = member.pilot_artifact_fingerprint
        deck = member.exact_deck_digest
        member_id = member.member_id
    elif scripted is not None:
        opponent_id = scripted.opponent_id
        artifact = scripted.artifact_fingerprint
        pilot = scripted.artifact_fingerprint
        deck = scripted.exact_deck_digest
    identity = _fingerprint(
        _ASSIGNMENT_DOMAIN,
        {
            "seed": seed,
            "cursor": cursor,
            "generation": generation,
            "lane": lane,
            "candidate_deck_digest": candidate_deck_digest,
            "candidate_seat": candidate_seat,
            "opponent_artifact": artifact,
        },
    )
    return CurriculumAssignment(
        assignment_id=identity,
        cursor=cursor,
        generation=generation,
        lane=lane,
        candidate_deck_digest=candidate_deck_digest,
        candidate_seat=0 if candidate_seat == 0 else 1,
        opponent_id=opponent_id,
        opponent_artifact_fingerprint=artifact,
        opponent_pilot_fingerprint=pilot,
        opponent_deck_digest=deck,
        member_id=member_id,
    )


def _updated_statistic(
    previous: OnlineStatistic | None,
    *,
    score: float,
    alpha: float,
    cursor: int,
) -> OnlineStatistic:
    old = 0.5 if previous is None else previous.score_ema
    return OnlineStatistic(
        games=1 if previous is None else previous.games + 1,
        score_ema=(1.0 - alpha) * old + alpha * score,
        last_assignment_cursor=cursor,
    )


def _blended_score(
    statistic: OnlineStatistic | None,
    *,
    fallback: float,
    minimum_evidence: int,
) -> float:
    if statistic is None or statistic.games <= 0:
        return fallback
    confidence = min(1.0, statistic.games / float(minimum_evidence))
    return (1.0 - confidence) * fallback + confidence * statistic.score_ema


def _bounded_distribution(
    weights: Sequence[float],
    *,
    floors: Sequence[float],
    cap: float,
) -> tuple[float, ...]:
    if len(weights) != len(floors) or not weights:
        raise ValueError("bounded probabilities need aligned non-empty inputs")
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("PFSP weights must be finite and positive")
    effective_cap = max(cap, 1.0 / len(weights))
    effective_floors = tuple(min(floor, effective_cap) for floor in floors)
    floor_mass = sum(effective_floors)
    if floor_mass >= 1.0:
        # Per-member floors are sampling preferences, not a correctness
        # invariant.  A growing past-self pool can make their requested mass
        # mathematically impossible even though every member and weight is
        # otherwise usable.  Preserve the relative protection while reserving
        # enough mass for the PFSP weights to keep influencing the result.
        adaptive_mass = max(1.0 / len(weights), 1e-6)
        floor_budget = 1.0 - adaptive_mass
        scale = floor_budget / floor_mass
        effective_floors = tuple(floor * scale for floor in effective_floors)
        _log_floor_relaxation(
            member_count=len(weights),
            requested_floor_mass=floor_mass,
            effective_cap=effective_cap,
        )
    result = list(effective_floors)
    remaining = 1.0 - sum(result)
    active = {
        index
        for index, probability in enumerate(result)
        if effective_cap - probability > 1e-15
    }
    while remaining > 1e-12 and active:
        active_weight = sum(weights[index] for index in active)
        allocations = {
            index: remaining * weights[index] / active_weight for index in active
        }
        saturated = [
            index
            for index, share in allocations.items()
            if share >= effective_cap - result[index]
        ]
        if saturated:
            for index in saturated:
                room = effective_cap - result[index]
                result[index] += room
                remaining -= room
                active.remove(index)
            continue
        for index, share in allocations.items():
            result[index] += share
        remaining = 0.0
    if remaining > 1e-9:
        raise ValueError("PFSP cap cannot allocate all probability")
    correction = 1.0 - sum(result)
    if correction:
        if correction > 0.0:
            index = max(
                range(len(result)),
                key=lambda item: effective_cap - result[item],
            )
        else:
            index = max(range(len(result)), key=result.__getitem__)
        result[index] += correction
    return tuple(result)


def _log_floor_relaxation(
    *,
    member_count: int,
    requested_floor_mass: float,
    effective_cap: float,
) -> None:
    """Report each degraded pool shape once without affecting sampling."""
    key = (
        member_count,
        round(requested_floor_mass, 12),
        round(effective_cap, 12),
    )
    with _FLOOR_RELAXATION_LOG_LOCK:
        if key in _LOGGED_FLOOR_RELAXATIONS:
            return
        _LOGGED_FLOOR_RELAXATIONS.add(key)
    _LOGGER.warning(
        "PFSP sampling floors exceed probability mass; relaxing them "
        "proportionally: members=%d requested_floor_mass=%.6f cap=%.6f",
        member_count,
        requested_floor_mass,
        effective_cap,
    )


def _sample_index(
    probabilities: Sequence[float],
    *,
    seed: int,
    cursor: int,
    domain: str,
) -> int:
    digest = hashlib.sha256(f"{domain}:{seed}:{cursor}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def _increment(values: Mapping[str, int], key: str) -> dict[str, int]:
    updated = dict(values)
    updated[key] = updated.get(key, 0) + 1
    return updated


def _matchup_key(candidate: str, pilot: str, opponent_deck: str) -> str:
    return json.dumps(
        (candidate, pilot, opponent_deck),
        separators=(",", ":"),
    )


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"checkpoint pair {key} mapping is missing")
    return result


def _verify_file(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size_bytes:
        raise ValueError(f"{label} byte size changed")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"{label} fingerprint changed")


def _discard_audit_page_cache(path: Path) -> None:
    """Best-effort release pages populated only by the full artifact audit."""
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dont_need = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dont_need is None:
        return
    try:
        with path.open("rb") as handle:
            posix_fadvise(handle.fileno(), 0, 0, dont_need)
    except OSError:
        # Cache advice is an optimization. Artifact identity was already
        # verified above, so an unsupported filesystem must not change the
        # correctness or portability of curriculum restoration.
        return


def _fingerprint(domain: bytes, payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def stateless_curriculum_state_fingerprint(
    state: StatelessCurriculumState,
) -> str:
    """Return a deterministic content identity for recovery transitions."""
    fingerprint, _payload = _state_encoding_cache(state)
    if fingerprint is not None:
        return fingerprint
    fingerprint = _fingerprint(_STATE_DOMAIN, state.model_dump(mode="json"))
    _update_state_encoding_cache(state, fingerprint=fingerprint)
    return fingerprint


def rebase_settled_stateless_curriculum_lane_coverage(
    source: StatelessCurriculumState,
    *,
    source_config_fingerprint: str,
    target_config_fingerprint: str,
    target_lane_mix: OpponentLaneMix,
    expected_assignment_cursor: int,
    expected_source_lane_coverage: Mapping[CurriculumLane, int],
    expected_target_lane_coverage: Mapping[CurriculumLane, int],
) -> tuple[
    StatelessCurriculumState,
    StatelessCurriculumLaneCoverageRebaseAudit,
]:
    """Start a declared lane epoch without changing any non-lane state.

    Assignment cursors name immutable leases and therefore cannot be reset. The
    lane coverage ledger is instead re-centered with Hamilton's
    largest-remainder method. The historical reservation gap and the total
    number of observed lane assignments remain unchanged, while each lane's
    target deficit becomes equal up to integer rounding. A common deficit does
    not affect max-deficit scheduling, so the new mix takes effect immediately.
    """
    source_config = _require_fingerprint(source_config_fingerprint)
    target_config = _require_fingerprint(target_config_fingerprint)
    if source_config == target_config:
        raise ValueError("lane-coverage rebase requires a new curriculum identity")
    if source.config_fingerprint != source_config:
        raise ValueError("lane-coverage rebase source config fingerprint mismatch")
    if source.inflight:
        raise ValueError("lane-coverage rebase requires zero in-flight assignments")
    if any(member.leases != 0 for member in source.members):
        raise ValueError("lane-coverage rebase requires zero PFSP member leases")
    if source.assignment_cursor != expected_assignment_cursor:
        raise ValueError("lane-coverage rebase assignment cursor mismatch")

    declared_source = _exact_lane_coverage(
        expected_source_lane_coverage,
        label="declared source lane coverage",
    )
    observed_source = _normalized_state_lane_coverage(source.lane_coverage)
    if observed_source != declared_source:
        raise ValueError("lane-coverage rebase source counters mismatch")
    source_total = sum(observed_source.values())
    if source_total > source.assignment_cursor:
        raise ValueError("lane-coverage rebase source counters exceed its cursor")

    declared_target = _exact_lane_coverage(
        expected_target_lane_coverage,
        label="declared target lane coverage",
    )
    target_coverage = apportion_neutral_deficit_lane_coverage(
        assignment_cursor=source.assignment_cursor,
        coverage_total=source_total,
        lane_mix=target_lane_mix,
    )
    if target_coverage != declared_target:
        raise ValueError("lane-coverage rebase target differs from Hamilton allocation")

    rebased = source.model_copy(update={"lane_coverage": target_coverage})
    if rebased.model_copy(update={"lane_coverage": source.lane_coverage}) != source:
        raise RuntimeError("lane-coverage rebase changed non-lane curriculum state")
    audit = StatelessCurriculumLaneCoverageRebaseAudit(
        source_config_fingerprint=source_config,
        target_config_fingerprint=target_config,
        source_state_fingerprint=stateless_curriculum_state_fingerprint(source),
        rebased_state_fingerprint=stateless_curriculum_state_fingerprint(rebased),
        assignment_cursor=source.assignment_cursor,
        source_lane_coverage=observed_source,
        source_lane_coverage_total=source_total,
        source_assignment_cursor_gap=source.assignment_cursor - source_total,
        target_lane_mix=target_lane_mix,
        target_lane_coverage=target_coverage,
        target_lane_coverage_total=sum(target_coverage.values()),
        target_assignment_cursor_gap=(
            source.assignment_cursor - sum(target_coverage.values())
        ),
    )
    return rebased, audit


def apportion_neutral_deficit_lane_coverage(
    *,
    assignment_cursor: int,
    coverage_total: int,
    lane_mix: OpponentLaneMix,
) -> dict[CurriculumLane, int]:
    """Apportion a ledger so target deficits share one common offset."""
    if coverage_total < 0 or coverage_total > assignment_cursor:
        raise ValueError("lane-coverage Hamilton totals are invalid")
    weights = {
        lane: Fraction(str(weight))
        for lane, weight in lane_mix.as_mapping().items()
    }
    weight_total = sum(weights.values(), start=Fraction(0))
    if weight_total <= 0:
        raise ValueError("lane-coverage Hamilton weights must have positive mass")
    reservation_gap = assignment_cursor - coverage_total
    common_deficit = Fraction(reservation_gap, len(_CURRICULUM_LANES))
    ideals = {
        lane: (
            Fraction(assignment_cursor) * weights[lane] / weight_total
            - common_deficit
        )
        for lane in _CURRICULUM_LANES
    }
    if any(ideal < 0 for ideal in ideals.values()):
        raise ValueError(
            "lane-coverage neutral deficit requires non-negative target counters"
        )
    counts = {
        lane: ideal.numerator // ideal.denominator
        for lane, ideal in ideals.items()
    }
    remainder = coverage_total - sum(counts.values())
    if remainder < 0 or remainder > len(_CURRICULUM_LANES):
        raise RuntimeError("lane-coverage Hamilton remainder is invalid")
    order = sorted(
        _CURRICULUM_LANES,
        key=lambda lane: (
            -(ideals[lane] - counts[lane]),
            _CURRICULUM_LANES.index(lane),
        ),
    )
    for lane in order[:remainder]:
        counts[lane] += 1
    if sum(counts.values()) != coverage_total:
        raise RuntimeError("lane-coverage Hamilton allocation lost ledger mass")
    deficits = tuple(
        Fraction(assignment_cursor) * weights[lane] / weight_total - counts[lane]
        for lane in _CURRICULUM_LANES
    )
    if max(deficits) - min(deficits) > 1:
        raise RuntimeError("lane-coverage Hamilton deficits are not neutral")
    return counts


def _exact_lane_coverage(
    value: Mapping[CurriculumLane, int],
    *,
    label: str,
) -> dict[CurriculumLane, int]:
    """Validate a complete, canonical three-lane counter mapping."""
    if set(value) != set(_CURRICULUM_LANES) or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in value.values()
    ):
        raise ValueError(f"{label} must contain exact non-negative lane counters")
    return {lane: int(value[lane]) for lane in _CURRICULUM_LANES}


def _normalized_state_lane_coverage(
    value: Mapping[str, int],
) -> dict[CurriculumLane, int]:
    """Normalize legacy omitted zero counters while rejecting unknown lanes."""
    if set(value) - set(_CURRICULUM_LANES):
        raise ValueError("lane-coverage rebase source contains an unknown lane")
    return _exact_lane_coverage(
        {lane: value.get(lane, 0) for lane in _CURRICULUM_LANES},
        label="source lane coverage",
    )


def compact_stateless_curriculum_state(
    state: StatelessCurriculumState,
) -> bytes:
    """Return one cached lossless compact encoding of an immutable state."""
    _fingerprint_value, payload = _state_encoding_cache(state)
    if payload is not None:
        return payload
    payload = encode_curriculum_payload(state.model_dump(mode="json"))
    _update_state_encoding_cache(state, payload=payload)
    return payload


def _state_encoding_cache(
    state: StatelessCurriculumState,
) -> tuple[str | None, bytes | None]:
    """Return cache values only when the weak identity still names this object."""
    key = id(state)
    with _STATE_ENCODING_CACHE_LOCK:
        entry = _STATE_ENCODING_CACHE.get(key)
        if entry is None or entry[0]() is not state:
            return None, None
        return entry[1], entry[2]


def _update_state_encoding_cache(
    state: StatelessCurriculumState,
    *,
    fingerprint: str | None = None,
    payload: bytes | None = None,
) -> None:
    """Cache immutable derived values without changing Pydantic equality/copies."""
    key = id(state)

    def remove(reference: weakref.ReferenceType[StatelessCurriculumState]) -> None:
        with _STATE_ENCODING_CACHE_LOCK:
            current = _STATE_ENCODING_CACHE.get(key)
            if current is not None and current[0] is reference:
                _STATE_ENCODING_CACHE.pop(key, None)

    with _STATE_ENCODING_CACHE_LOCK:
        current = _STATE_ENCODING_CACHE.get(key)
        current_fingerprint = (
            current[1] if current is not None and current[0]() is state else None
        )
        current_payload = (
            current[2] if current is not None and current[0]() is state else None
        )
        reference = (
            current[0]
            if current is not None and current[0]() is state
            else weakref.ref(state, remove)
        )
        _STATE_ENCODING_CACHE[key] = (
            reference,
            fingerprint if fingerprint is not None else current_fingerprint,
            payload if payload is not None else current_payload,
        )


def load_compact_stateless_curriculum_state(
    payload: bytes,
) -> StatelessCurriculumState:
    """Validate one compact logical state through the authoritative schema."""
    return StatelessCurriculumState.model_validate(decode_curriculum_payload(payload))


def publish_external_stateless_curriculum_state(
    state_path: Path,
    state: StatelessCurriculumState,
) -> None:
    """Publish one compact immutable payload behind a small atomic pointer."""
    fingerprint = stateless_curriculum_state_fingerprint(state)
    payload = compact_stateless_curriculum_state(state)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    payload_dir = state_path.parent / "curriculum_states"
    payload_path = payload_dir / f"state-{fingerprint}.msgpack.zlib"
    if payload_path.exists():
        if payload_path.read_bytes() != payload:
            raise FileExistsError(
                f"compact curriculum payload differs from retry: {payload_path}"
            )
    else:
        atomic_write_bytes(payload_path, payload, overwrite=False)
    pointer = {
        "format": _EXTERNAL_STATE_FORMAT,
        "schema_version": 2,
        "state_fingerprint": fingerprint,
        "payload_path": str(payload_path.relative_to(state_path.parent)),
        "payload_size_bytes": len(payload),
        "payload_sha256": payload_sha256,
    }
    atomic_write_bytes(
        state_path,
        json_payload(pointer),
        overwrite=state_path.exists(),
    )


def load_external_stateless_curriculum_state(
    state_path: Path,
) -> StatelessCurriculumState:
    """Load either a legacy inline JSON state or a compact V2 pointer."""
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("external curriculum state must be a mapping")
    if raw.get("format") != _EXTERNAL_STATE_FORMAT:
        return StatelessCurriculumState.model_validate(raw)
    relative = Path(str(raw.get("payload_path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError("compact curriculum payload path must be relative")
    payload_path = (state_path.parent / relative).resolve()
    if not payload_path.is_relative_to(state_path.parent.resolve()):
        raise ValueError("compact curriculum payload escaped the control directory")
    payload = payload_path.read_bytes()
    expected_size = int(raw.get("payload_size_bytes", -1))
    expected_sha256 = str(raw.get("payload_sha256", ""))
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != (
        expected_sha256
    ):
        raise ValueError("compact curriculum payload identity mismatch")
    state = load_compact_stateless_curriculum_state(payload)
    expected_fingerprint = str(raw.get("state_fingerprint", ""))
    if stateless_curriculum_state_fingerprint(state) != expected_fingerprint:
        raise ValueError("compact curriculum logical fingerprint mismatch")
    return state


def prune_external_stateless_curriculum_states(
    state_path: Path,
    *,
    keep_last: int,
) -> int:
    """Bound superseded compact payloads after the current pointer is durable."""
    if keep_last <= 0:
        raise ValueError("compact curriculum retention must be positive")
    payload_dir = state_path.parent / "curriculum_states"
    if not payload_dir.is_dir():
        return 0
    current: Path | None = None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and raw.get("format") == _EXTERNAL_STATE_FORMAT:
            relative = Path(str(raw.get("payload_path", "")))
            if relative.is_absolute() or not relative.parts:
                return 0
            candidate = (state_path.parent / relative).resolve()
            if not candidate.is_relative_to(state_path.parent.resolve()):
                return 0
            current = candidate
    except (OSError, ValueError, TypeError):
        return 0
    ordered = sorted(
        (item.resolve() for item in payload_dir.glob("state-*.msgpack.zlib")),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    retained = set(ordered[:keep_last])
    if current is not None:
        retained.add(current)
    removed = 0
    for path in ordered:
        if path not in retained:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _require_fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("curriculum identity must be a lowercase SHA-256")
    return normalized


__all__ = [
    "CurriculumAssignment",
    "CurriculumEvent",
    "DurablePolicyArtifact",
    "DurablePolicyPairArtifact",
    "MatchupPfspConfig",
    "OnlineStatistic",
    "OpponentLaneMix",
    "PastSelfRetentionConfig",
    "PfspMember",
    "PreparedPastSelfAdmission",
    "PreparedPolicyRoutes",
    "PreparedTerminalOutcomes",
    "ScriptedCurriculumBundle",
    "StatelessCurriculumConfig",
    "StatelessCurriculumController",
    "StatelessCurriculumLaneCoverageRebaseAudit",
    "StatelessCurriculumState",
    "VerifiedPolicyPublication",
    "apportion_neutral_deficit_lane_coverage",
    "compact_stateless_curriculum_state",
    "historical_anchor_member",
    "load_external_stateless_curriculum_state",
    "load_compact_stateless_curriculum_state",
    "publish_external_stateless_curriculum_state",
    "prune_external_stateless_curriculum_states",
    "rebase_settled_stateless_curriculum_lane_coverage",
    "stateless_curriculum_state_fingerprint",
    "verify_member_artifacts",
]
