"""Single-writer global-window coordinator for native collection."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.rl.native_collection_delivery import discard_compact_assignments
from ptcg_rl.rl.native_distributed.contracts import (
    NativeArtifactExposure,
    NativeAssignedGame,
    NativeCollectionAttempt,
    NativeCollectionCapacityTier,
    NativeCollectionShardLease,
    NativeCollectionShardResult,
    NativeCollectionWindow,
    NativeCollectionWindowReceipt,
    NativeCollectionWorkerManifest,
    NativeRolloutWorkerIdentity,
    native_assignment_fingerprint,
    native_effective_capacity_tier,
    native_retry_capacity_tier,
)
from ptcg_rl.rl.native_distributed.data_plane import DecodedNativeCollectionPart
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessGameOutcome,
)

CommitCallback = Callable[
    [tuple[StatelessAssignedGame, ...], tuple[StatelessGameOutcome, ...]],
    None,
]

_MAX_FAILURE_REASON_CHARACTERS = 512

# Shards far smaller than the worker's full-tier yield carry proportionally
# less throughput evidence, so their EWMA updates are weighted down and a
# first observation below this weight does not seed the estimate at all.
_EWMA_SEED_MINIMUM_WEIGHT = 0.05


def _bounded_failure_reason(reason: str | None) -> str:
    """Return one log-safe bounded failure reason for coordinator diagnostics."""
    if reason is None:
        return "unknown failure"
    normalized = " ".join(reason.split())
    if len(normalized) <= _MAX_FAILURE_REASON_CHARACTERS:
        return normalized
    return normalized[: _MAX_FAILURE_REASON_CHARACTERS - 3] + "..."


def _discard_nonterminal_assignments(
    parts: Sequence[DecodedNativeCollectionPart],
    assignment_ids: frozenset[str],
) -> list[DecodedNativeCollectionPart]:
    """Project ACKed wire parts onto the retained whole-game transaction."""
    retained: list[DecodedNativeCollectionPart] = []
    for decoded in parts:
        part = discard_compact_assignments(decoded.part, tuple(assignment_ids))
        if part is None:
            continue
        if part is decoded.part:
            retained.append(decoded)
            continue
        counts = {
            "fragment_count": part.fragment_count,
            "decision_count": part.decision_count,
        }
        retained.append(
            replace(
                decoded,
                identity=decoded.identity.model_copy(update=counts),
                payload=replace(
                    decoded.payload,
                    identity=decoded.payload.identity.model_copy(update=counts),
                    part=part,
                ),
            )
        )
    return retained


class NativeCoordinatorError(RuntimeError):
    """Base class for fail-closed distributed coordinator errors."""


class NativeCoordinatorProtocolError(NativeCoordinatorError):
    """A worker message violated the active window identity."""


class NativeCoordinatorQuorumError(NativeCoordinatorError):
    """The required startup or degraded quorum is unavailable."""


class NativeCoordinatorStatus(BaseModel):
    """Atomic monitor projection for one coordinator process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str | None = None
    window_sequence_id: int | None = Field(default=None, ge=0)
    window_state: Literal["idle", "collecting", "committed", "aborted"] = "idle"
    connected_workers: tuple[str, ...] = ()
    degraded_workers: tuple[str, ...] = ()
    topology_workers: tuple[str, ...] = ()
    target_decisions: int = Field(default=0, ge=0)
    accepted_decisions: int = Field(default=0, ge=0)
    provisional_decisions: int = Field(default=0, ge=0)
    terminal_provisional_decisions: int = Field(default=0, ge=0)
    inflight_decision_credit: int = Field(default=0, ge=0)
    overshoot_decisions: int = Field(default=0, ge=0)
    shards_issued: int = Field(default=0, ge=0)
    shards_completed: int = Field(default=0, ge=0)
    shards_discarded: int = Field(default=0, ge=0)
    attempts_started: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    parts_accepted: int = Field(default=0, ge=0)
    active_attempt_ids: tuple[str, ...] = ()
    completed_lease_ids: tuple[str, ...] = ()
    fragments_accepted: int = Field(default=0, ge=0)
    payload_bytes_accepted: int = Field(default=0, ge=0)
    decisions_per_second: float = Field(default=0.0, ge=0.0)
    bytes_per_decision: float = Field(default=0.0, ge=0.0)
    estimated_seconds_remaining: float | None = Field(default=None, ge=0.0)
    learner_ready_drain_requested: bool = False
    learner_clocked_high_water_decisions: int | None = Field(default=None, gt=0)
    learner_clocked_high_water_reached: bool = False
    worker_decisions_per_game_ewma: dict[str, float] = Field(default_factory=dict)
    worker_decisions_per_second_ewma: dict[str, float] = Field(default_factory=dict)


@dataclass(slots=True)
class _WorkerState:
    manifest: NativeCollectionWorkerManifest
    last_heartbeat_unix_ns: int
    ready: bool = False
    active_attempt_id: str | None = None
    decisions_per_game_ewma: float | None = None
    decisions_per_second_ewma: float | None = None


@dataclass(slots=True)
class _AttemptState:
    identity: NativeCollectionAttempt
    parts: list[DecodedNativeCollectionPart] = field(default_factory=list)
    payload_bytes: int = 0
    decisions_by_assignment: dict[str, int] = field(default_factory=dict)
    terminal_assignments: set[str] = field(default_factory=set)
    terminal_decisions: int = 0
    completion_pending: bool = False
    finished: bool = False
    failure_reason: str | None = None


@dataclass(slots=True)
class _ShardState:
    lease: NativeCollectionShardLease
    attempts: list[_AttemptState] = field(default_factory=list)
    accepted_attempt_id: str | None = None
    result: NativeCollectionShardResult | None = None
    discarded: bool = False

    @property
    def active_attempt(self) -> _AttemptState | None:
        """Return the unfinished latest attempt, if any."""
        if not self.attempts or self.attempts[-1].finished:
            return None
        return self.attempts[-1]


@dataclass(frozen=True, slots=True)
class NativeAttemptCompletionReservation:
    """Immutable completion evidence reserved by the control I/O thread."""

    lease: NativeCollectionShardLease
    result: NativeCollectionShardResult
    parts: tuple[DecodedNativeCollectionPart, ...]


@dataclass(frozen=True, slots=True)
class NativePreparedAttemptCompletion:
    """Terminal-only compact parts projected outside the control I/O thread."""

    reservation: NativeAttemptCompletionReservation
    retained_parts: tuple[DecodedNativeCollectionPart, ...]


def _shard_artifact_exposures(
    shard: _ShardState,
) -> tuple[NativeArtifactExposure, ...]:
    """Project one accepted shard into per-artifact exposure counters."""
    result = shard.result
    if result is None:
        raise RuntimeError("native artifact exposure requires a completed shard")
    window = shard.lease.window
    manifests = {item.artifact_id: item for item in window.active_artifacts}
    current = next(
        item.artifact_id for item in window.active_artifacts if item.kind == "current"
    )
    members = {member.member_id: member for member in window.pfsp_members}
    outcomes = {item.curriculum_assignment_id: item for item in result.outcomes}
    started_assignments = tuple(
        assignment
        for assignment in shard.lease.assignments
        if assignment.curriculum.assignment_id in outcomes
    )
    assignments_by_artifact: dict[str, list[NativeAssignedGame]] = {
        current: list(started_assignments)
    }
    for assignment in started_assignments:
        member_id = assignment.curriculum.member_id
        if not member_id:
            continue
        member = members[member_id]
        assignments_by_artifact.setdefault(member.policy_sha256, []).append(assignment)
    return tuple(
        NativeArtifactExposure(
            artifact_id=artifact_id,
            kind=(
                manifests[artifact_id].kind
                if artifact_id in manifests
                else "historical_anchor"
            ),
            assigned_games=len(assignments_by_artifact[artifact_id]),
            engine_terminals=sum(
                outcomes[item.curriculum.assignment_id].status == "engine_terminal"
                for item in assignments_by_artifact[artifact_id]
            ),
            candidate_trainable_decisions=sum(
                outcomes[item.curriculum.assignment_id].candidate_decisions
                for item in assignments_by_artifact[artifact_id]
            )
            if artifact_id != current
            else result.decision_count,
        )
        for artifact_id in shard.lease.effective_required_artifact_ids
        if assignments_by_artifact.get(artifact_id)
    )


def _aggregate_artifact_exposures(
    shards: Sequence[_ShardState],
) -> tuple[NativeArtifactExposure, ...]:
    """Aggregate accepted shard evidence once at the window boundary."""
    totals: dict[
        str,
        tuple[Literal["current", "past_self", "historical_anchor"], int, int, int],
    ] = {}
    for shard in shards:
        for exposure in _shard_artifact_exposures(shard):
            kind, assigned, terminals, decisions = totals.get(
                exposure.artifact_id,
                (exposure.kind, 0, 0, 0),
            )
            if kind != exposure.kind:
                raise RuntimeError("native artifact exposure kind changed")
            totals[exposure.artifact_id] = (
                kind,
                assigned + exposure.assigned_games,
                terminals + exposure.engine_terminals,
                decisions + exposure.candidate_trainable_decisions,
            )
    return tuple(
        NativeArtifactExposure(
            artifact_id=artifact_id,
            kind=kind,
            assigned_games=assigned,
            engine_terminals=terminals,
            candidate_trainable_decisions=decisions,
        )
        for artifact_id, (kind, assigned, terminals, decisions) in sorted(
            totals.items()
        )
    )


class NativeCollectionCoordinator:
    """Own assignment leases, retries, ordering, and exactly-once commit."""

    def __init__(
        self,
        *,
        expected_contract: NativeRolloutWorkerIdentity,
        required_worker_ids: Sequence[str],
        maximum_attempts_per_shard: int,
        lease_timeout_seconds: float,
        heartbeat_timeout_seconds: float,
        degrade_grace_seconds: float,
        allow_degraded_after_startup: bool,
        minimum_degraded_workers: int = 1,
        tail_start_fraction: float = 0.65,
        early_inflight_reservation_fraction: float = 0.8,
        tail_inflight_reservation_fraction: float = 0.35,
        tail_target_seconds: float = 45.0,
        worker_yield_ewma_alpha: float = 0.25,
        learner_clocked_max_trainable_decisions: int | None = None,
    ) -> None:
        """Create an idle coordinator with no recoverable half-window state."""
        if (
            maximum_attempts_per_shard <= 0
            or lease_timeout_seconds <= 0.0
            or heartbeat_timeout_seconds <= 0.0
            or degrade_grace_seconds < heartbeat_timeout_seconds
            or minimum_degraded_workers <= 0
            or not 0.0 < tail_start_fraction < 1.0
            or not 0.0 <= early_inflight_reservation_fraction <= 1.0
            or not 0.0 <= tail_inflight_reservation_fraction <= 1.0
            or tail_inflight_reservation_fraction > early_inflight_reservation_fraction
            or tail_target_seconds <= 0.0
            or not 0.0 < worker_yield_ewma_alpha <= 1.0
            or (
                learner_clocked_max_trainable_decisions is not None
                and learner_clocked_max_trainable_decisions <= 0
            )
        ):
            raise ValueError("native coordinator retry/quorum settings are invalid")
        required = tuple(required_worker_ids)
        if not required or len(required) != len(set(required)):
            raise ValueError("native coordinator required workers must be unique")
        self._expected_contract = expected_contract
        self._required_worker_ids = required
        self._maximum_attempts = maximum_attempts_per_shard
        self._lease_timeout_ns = int(lease_timeout_seconds * 1e9)
        self._heartbeat_timeout_ns = int(heartbeat_timeout_seconds * 1e9)
        self._degrade_grace_ns = int(degrade_grace_seconds * 1e9)
        self._allow_degraded = allow_degraded_after_startup
        self._minimum_degraded_workers = minimum_degraded_workers
        self._tail_start_fraction = tail_start_fraction
        self._early_inflight_reservation_fraction = early_inflight_reservation_fraction
        self._tail_inflight_reservation_fraction = tail_inflight_reservation_fraction
        self._tail_target_seconds = tail_target_seconds
        self._worker_yield_ewma_alpha = worker_yield_ewma_alpha
        self._learner_clocked_max_trainable_decisions = (
            learner_clocked_max_trainable_decisions
        )
        self._workers: dict[str, _WorkerState] = {}
        self._pending_workers: dict[str, _WorkerState] = {}
        self._window: NativeCollectionWindow | None = None
        self._topology_workers: tuple[str, ...] = ()
        self._topology_manifests: tuple[NativeCollectionWorkerManifest, ...] = ()
        self._shards: dict[str, _ShardState] = {}
        self._discarded_attempts: dict[
            str,
            tuple[
                NativeCollectionShardLease,
                NativeCollectionAttempt,
                NativeCollectionWorkerManifest,
            ],
        ] = {}
        self._shard_order: list[str] = []
        self._accepted_decisions = 0
        self._attempts_started = 0
        self._retries = 0
        self._receipt: NativeCollectionWindowReceipt | None = None
        self._committed_windows = 0
        self._window_started_unix_ns = 0
        self._learner_clocked_primary_only = False
        self._learner_ready_drain_requested = False
        self._learner_clocked_high_water_reached = False
        self._lock = threading.RLock()

    @property
    def window(self) -> NativeCollectionWindow | None:
        """Return the active or terminal window identity."""
        with self._lock:
            return self._window

    @property
    def receipt(self) -> NativeCollectionWindowReceipt | None:
        """Return the terminal receipt, if the window is settled."""
        with self._lock:
            return self._receipt

    def register_worker(
        self,
        manifest: NativeCollectionWorkerManifest,
        *,
        now_unix_ns: int,
    ) -> None:
        """Register a compatible worker; active topology changes wait a window."""
        self._validate_worker_contract(manifest.identity)
        worker_id = manifest.identity.worker_id
        with self._lock:
            existing = self._workers.get(worker_id)
            pending = self._pending_workers.get(worker_id)
            for candidate in (existing, pending):
                if (
                    candidate is not None
                    and candidate.manifest.identity.session_id
                    == manifest.identity.session_id
                ):
                    if candidate.manifest != manifest:
                        raise NativeCoordinatorProtocolError(
                            "worker session re-registered with a different manifest"
                        )
                    candidate.last_heartbeat_unix_ns = now_unix_ns
                    return
            registered = _WorkerState(
                manifest=manifest,
                last_heartbeat_unix_ns=now_unix_ns,
            )
            collecting = self._window is not None and self._receipt is None
            if collecting:
                if existing is not None:
                    if existing.active_attempt_id is not None:
                        self._fail_attempt_locked(
                            existing.active_attempt_id,
                            reason="worker session replaced",
                        )
                    existing.ready = False
                self._pending_workers[worker_id] = registered
                return
            self._workers[worker_id] = registered
            self._pending_workers.pop(worker_id, None)

    def mark_ready(
        self,
        worker_id: str,
        *,
        session_id: str,
        now_unix_ns: int,
    ) -> None:
        """Mark a registered worker eligible for the next topology freeze."""
        with self._lock:
            worker = self._worker_session(worker_id, session_id)
            worker.ready = True
            worker.last_heartbeat_unix_ns = now_unix_ns

    def heartbeat(
        self,
        worker_id: str,
        *,
        session_id: str,
        now_unix_ns: int,
        active_attempt_id: str | None = None,
    ) -> None:
        """Refresh one exact worker session and its claimed active attempt."""
        with self._lock:
            worker = self._worker_session(worker_id, session_id)
            worker.last_heartbeat_unix_ns = now_unix_ns
            if (
                active_attempt_id is None
                or worker.active_attempt_id != active_attempt_id
            ):
                return
            for shard in self._shards.values():
                attempt = shard.active_attempt
                if (
                    attempt is not None
                    and attempt.identity.attempt_id == active_attempt_id
                ):
                    attempt.identity = attempt.identity.model_copy(
                        update={
                            "expires_at_unix_ns": (now_unix_ns + self._lease_timeout_ns)
                        }
                    )
                    return

    def expire_workers(self, *, now_unix_ns: int) -> tuple[str, ...]:
        """Fail active attempts owned by heartbeat-expired workers."""
        expired: list[str] = []
        with self._lock:
            expired_attempt_ids = tuple(
                shard.active_attempt.identity.attempt_id
                for shard in self._shards.values()
                if (
                    shard.active_attempt is not None
                    and not shard.active_attempt.completion_pending
                    and now_unix_ns > shard.active_attempt.identity.expires_at_unix_ns
                )
            )
            for attempt_id in expired_attempt_ids:
                self._fail_attempt_locked(
                    attempt_id,
                    reason="shard attempt lease expired",
                )
            for worker_id, worker in self._workers.items():
                if (
                    worker.ready
                    and now_unix_ns - worker.last_heartbeat_unix_ns
                    > self._heartbeat_timeout_ns
                ):
                    expired.append(worker_id)
                    worker.ready = False
                    if worker.active_attempt_id is not None and not any(
                        attempt is not None
                        and attempt.identity.attempt_id == worker.active_attempt_id
                        and attempt.completion_pending
                        for shard in self._shards.values()
                        for attempt in (shard.active_attempt,)
                    ):
                        self._fail_attempt_locked(
                            worker.active_attempt_id,
                            reason="worker heartbeat expired",
                        )
        return tuple(expired)

    def begin_window(
        self,
        window: NativeCollectionWindow,
        *,
        now_unix_ns: int,
        learner_clocked_primary_only: bool = False,
    ) -> tuple[str, ...]:
        """Freeze the connected topology for one globally budgeted window."""
        with self._lock:
            if self._window is not None and self._receipt is None:
                raise NativeCoordinatorError("native collection window is still active")
            self._promote_pending_workers_locked()
            self.expire_workers(now_unix_ns=now_unix_ns)
            ready = tuple(
                sorted(
                    worker_id
                    for worker_id, worker in self._workers.items()
                    if worker.ready
                )
            )
            ready_required = tuple(
                worker_id
                for worker_id in self._required_worker_ids
                if worker_id in ready
            )
            if self._committed_windows == 0:
                if (
                    ready_required != self._required_worker_ids
                    and not self._allow_degraded
                ):
                    missing = tuple(sorted(set(self._required_worker_ids) - set(ready)))
                    raise NativeCoordinatorQuorumError(
                        "initial native collection quorum is incomplete: "
                        + ", ".join(missing)
                    )
                if len(ready_required) < self._minimum_degraded_workers:
                    raise NativeCoordinatorQuorumError(
                        "initial native collection degraded quorum is incomplete"
                    )
                topology = ready
            else:
                missing = tuple(
                    worker_id
                    for worker_id in self._required_worker_ids
                    if worker_id not in ready
                )
                if missing:
                    if not self._allow_degraded:
                        raise NativeCoordinatorQuorumError(
                            "native collection degraded mode is disabled"
                        )
                    for worker_id in missing:
                        worker = self._workers.get(worker_id)
                        if (
                            worker is not None
                            and now_unix_ns - worker.last_heartbeat_unix_ns
                            < self._degrade_grace_ns
                        ):
                            raise NativeCoordinatorQuorumError(
                                "native collection worker is still inside grace: "
                                f"{worker_id}"
                            )
                    topology = ready
                    if len(topology) < self._minimum_degraded_workers:
                        raise NativeCoordinatorQuorumError(
                            "native collection degraded quorum is incomplete"
                        )
                else:
                    topology = ready
            if not window.required_worker_ids:
                window = window.model_copy(
                    update={"required_worker_ids": tuple(topology)}
                )
            elif tuple(window.required_worker_ids) != tuple(topology):
                raise NativeCoordinatorProtocolError(
                    "window required workers differ from frozen topology"
                )
            self._window = window
            self._topology_workers = tuple(topology)
            self._topology_manifests = tuple(
                self._workers[worker_id].manifest for worker_id in topology
            )
            self._shards.clear()
            self._shard_order.clear()
            self._accepted_decisions = 0
            self._attempts_started = 0
            self._retries = 0
            self._receipt = None
            self._window_started_unix_ns = now_unix_ns
            self._learner_clocked_primary_only = learner_clocked_primary_only
            self._learner_ready_drain_requested = False
            self._learner_clocked_high_water_reached = False
            for worker in self._workers.values():
                worker.active_attempt_id = None
            return self._topology_workers

    def session_is_in_active_topology(
        self,
        worker_id: str,
        *,
        session_id: str,
    ) -> bool:
        """Return whether an exact session belongs to the frozen window."""
        with self._lock:
            worker = self._workers.get(worker_id)
            return bool(
                self._window is not None
                and self._receipt is None
                and worker_id in self._topology_workers
                and worker is not None
                and worker.ready
                and worker.manifest.identity.session_id == session_id
            )

    def active_worker_manifests(
        self,
    ) -> tuple[NativeCollectionWorkerManifest, ...]:
        """Return the immutable worker inventory frozen for the open window."""
        with self._lock:
            self._require_collecting_window()
            return self._topology_manifests

    def ready_worker_manifests(
        self,
        worker_ids: Sequence[str],
    ) -> tuple[NativeCollectionWorkerManifest, ...]:
        """Return READY manifests in stable requested-topology order."""
        with self._lock:
            return tuple(
                worker.manifest
                for worker_id in worker_ids
                for worker in (self._workers.get(worker_id),)
                if worker is not None and worker.ready
            )

    def primary_only_during_window(self) -> bool:
        """Return whether the active overlap window permits only full tiers."""
        with self._lock:
            self._require_collecting_window()
            return self._learner_clocked_primary_only

    def select_capacity_tier(
        self,
        worker_id: str,
        *,
        maximum_games: int | None = None,
        exact_games: int | None = None,
    ) -> NativeCollectionCapacityTier | None:
        """Select a safe tier for target- or learner-clocked collection."""
        with self._lock:
            window = self._require_collecting_window()
            worker = self._require_topology_worker(worker_id)
            if worker.active_attempt_id is not None:
                return None
            if self._early_drain_requested_locked():
                return None
            tiers = tuple(
                sorted(
                    (
                        tier
                        for tier in worker.manifest.capacity_tiers
                        if maximum_games is None
                        or max(
                            tier.native_engine_shards,
                            tier.native_process_workers,
                        )
                        <= maximum_games
                        if exact_games is None or tier.concurrent_games == exact_games
                    ),
                    key=lambda item: (
                        self._estimated_tier_decisions_locked(worker, item),
                        item.tier_id,
                    ),
                )
            )
            if not tiers:
                return None
            selected = tiers[-1]
            if self._learner_clocked_primary_only:
                return selected
            provisional = self._provisional_decisions_locked()
            remaining = (
                window.target_trainable_decisions
                - self._accepted_decisions
                - provisional
            )
            tail = self._accepted_decisions >= (
                window.target_trainable_decisions * self._tail_start_fraction
            )
            reservation_fraction = (
                self._tail_inflight_reservation_fraction
                if tail
                else self._early_inflight_reservation_fraction
            )
            remaining -= math.ceil(
                self._remaining_inflight_credit_locked() * reservation_fraction
            )
            if remaining <= 0:
                return None
            if exact_games is not None:
                return self._calibrated_tier_locked(worker, selected)
            # Issue the full advertised tier with its static calibrated credit.
            # Over-issuance against the remaining budget is deliberately cheap:
            # every worker keeps full engine/GPU batches until terminal rows
            # cover the window target. Drain hints then cutoff only surplus
            # in-flight games, whose entire trajectories are rolled back.
            # Subdividing the remaining budget into per-worker shares was
            # measured slower: it starved issuance in the tail and made the
            # slowest lease the window's critical path.
            return selected

    def issue_shard(
        self,
        worker_id: str,
        assignments: Sequence[StatelessAssignedGame],
        *,
        capacity_tier_id: str,
        shard_seed: int,
        now_unix_ns: int,
        required_artifact_ids: Sequence[str] | None = None,
        shard_sequence_id: int | None = None,
        estimated_decision_credit: int | None = None,
    ) -> tuple[NativeCollectionShardLease, NativeCollectionAttempt]:
        """Issue one central assignment lease and its first worker attempt."""
        with self._lock:
            window = self._require_collecting_window()
            worker = self._require_topology_worker(worker_id)
            if worker.active_attempt_id is not None:
                raise NativeCoordinatorError("worker already owns an active attempt")
            if self._early_drain_requested_locked():
                raise NativeCoordinatorError("native collection window is draining")
            try:
                tier = next(
                    item
                    for item in worker.manifest.capacity_tiers
                    if item.tier_id == capacity_tier_id
                )
            except StopIteration as exc:
                raise NativeCoordinatorProtocolError(
                    "worker does not advertise the requested capacity tier"
                ) from exc
            wire_assignments = tuple(
                NativeAssignedGame.from_assignment(item) for item in assignments
            )
            effective_tier = native_effective_capacity_tier(
                tier,
                len(wire_assignments),
            )
            if estimated_decision_credit is not None:
                if estimated_decision_credit <= 0:
                    raise ValueError("native shard decision credit must be positive")
                effective_tier = effective_tier.model_copy(
                    update={"estimated_trainable_decisions": estimated_decision_credit}
                )
            resolved_shard_sequence_id = (
                len(self._shard_order)
                if shard_sequence_id is None
                else shard_sequence_id
            )
            if resolved_shard_sequence_id < 0 or any(
                self._shards[lease_id].lease.shard_sequence_id
                == resolved_shard_sequence_id
                for lease_id in self._shard_order
            ):
                raise ValueError("native shard sequence is invalid or duplicated")
            assignment_fingerprint = native_assignment_fingerprint(wire_assignments)
            lease_id = hashlib.sha256(
                (
                    f"{window.identity.window_id}:{resolved_shard_sequence_id}:"
                    f"{assignment_fingerprint}"
                ).encode()
            ).hexdigest()
            lease = NativeCollectionShardLease(
                lease_id=lease_id,
                window=window,
                shard_sequence_id=resolved_shard_sequence_id,
                assignments=wire_assignments,
                assignments_fingerprint=assignment_fingerprint,
                shard_seed=shard_seed,
                capacity_tier_id=tier.tier_id,
                capacity_tier=effective_tier,
                estimated_decision_credit=(
                    effective_tier.estimated_trainable_decisions
                ),
                issued_at_unix_ns=now_unix_ns,
                required_artifact_ids=(
                    None
                    if required_artifact_ids is None
                    else tuple(required_artifact_ids)
                ),
            )
            shard = _ShardState(lease=lease)
            self._shards[lease_id] = shard
            self._shard_order.append(lease_id)
            self._shard_order.sort(
                key=lambda item: self._shards[item].lease.shard_sequence_id
            )
            attempt = self._start_attempt_locked(
                shard,
                worker,
                now_unix_ns=now_unix_ns,
            )
            return lease, attempt

    def retry_shard(
        self,
        lease_id: str,
        worker_id: str,
        *,
        now_unix_ns: int,
    ) -> NativeCollectionAttempt:
        """Retry an immutable assignment lease with a fresh attempt ID."""
        with self._lock:
            shard = self._require_shard(lease_id)
            if shard.result is not None or shard.discarded:
                raise NativeCoordinatorError(
                    "completed or discarded shard cannot be retried"
                )
            if shard.active_attempt is not None:
                raise NativeCoordinatorError("shard still has an active attempt")
            worker = self._require_topology_worker(worker_id)
            if worker.active_attempt_id is not None:
                raise NativeCoordinatorError("worker already owns an active attempt")
            if (
                native_retry_capacity_tier(
                    worker.manifest.capacity_tiers,
                    shard.lease,
                )
                is None
            ):
                raise NativeCoordinatorError(
                    "worker has no compatible capacity tier for shard retry"
                )
            if len(shard.attempts) >= self._maximum_attempts:
                raise NativeCoordinatorError("native shard exhausted retry attempts")
            self._retries += 1
            return self._start_attempt_locked(
                shard,
                worker,
                now_unix_ns=now_unix_ns,
            )

    def retryable_lease(
        self,
        worker_id: str,
    ) -> NativeCollectionShardLease | None:
        """Return the first failed lease compatible with an idle worker.

        A worker session must not immediately spend the entire retry budget on
        the same deterministic local failure while another compatible session
        has not tried the lease. Single-worker topologies may still retry so a
        transient failure remains recoverable.
        """
        with self._lock:
            worker = self._require_topology_worker(worker_id)
            if (
                worker.active_attempt_id is not None
                or self._early_drain_requested_locked()
            ):
                return None
            for lease_id in self._shard_order:
                shard = self._shards[lease_id]
                worker_already_attempted = any(
                    attempt.identity.worker_id == worker_id
                    and attempt.identity.worker_session_id
                    == worker.manifest.identity.session_id
                    for attempt in shard.attempts
                )
                if (
                    shard.result is None
                    and not shard.discarded
                    and shard.active_attempt is None
                    and len(shard.attempts) < self._maximum_attempts
                    and native_retry_capacity_tier(
                        worker.manifest.capacity_tiers,
                        shard.lease,
                    )
                    is not None
                    and (
                        not worker_already_attempted
                        or not self._has_fresh_retry_worker_locked(
                            shard,
                            excluding_worker_id=worker_id,
                        )
                    )
                ):
                    return shard.lease
            return None

    def receipt_recipient_worker_ids(
        self,
        *,
        now_unix_ns: int,
    ) -> tuple[str, ...]:
        """Return live frozen-topology sessions that can receive settlement."""
        with self._lock:
            self.expire_workers(now_unix_ns=now_unix_ns)
            return tuple(
                worker_id
                for worker_id in self._topology_workers
                if (
                    (worker := self._workers.get(worker_id)) is not None
                    and worker.ready
                )
            )

    def attempt_context(
        self,
        lease_id: str,
        attempt_id: str,
    ) -> tuple[
        NativeCollectionShardLease,
        NativeCollectionAttempt,
        NativeCollectionWorkerManifest,
    ]:
        """Resolve exact identities required to decode one data message."""
        with self._lock:
            shard = self._require_shard(lease_id)
            attempt = shard.active_attempt
            if attempt is None or attempt.identity.attempt_id != attempt_id:
                raise NativeCoordinatorProtocolError(
                    "native collection part belongs to an old attempt"
                )
            worker = self._workers[attempt.identity.worker_id]
            return shard.lease, attempt.identity, worker.manifest

    def discarded_attempt_context(
        self,
        lease_id: str,
        attempt_id: str,
    ) -> (
        tuple[
            NativeCollectionShardLease,
            NativeCollectionAttempt,
            NativeCollectionWorkerManifest,
        ]
        | None
    ):
        """Return a cutoff attempt tombstone for safe late-message draining."""
        with self._lock:
            context = self._discarded_attempts.get(attempt_id)
            if context is None or context[0].lease_id != lease_id:
                return None
            return context

    def discarded_attempt_window_id(
        self,
        attempt_id: str | None,
        *,
        worker_id: str,
        session_id: str,
    ) -> str | None:
        """Resolve a worker's cutoff attempt to its original window."""
        if attempt_id is None:
            return None
        with self._lock:
            context = self._discarded_attempts.get(attempt_id)
            if context is None:
                return None
            _lease, attempt, _manifest = context
            if (
                attempt.worker_id != worker_id
                or attempt.worker_session_id != session_id
            ):
                raise NativeCoordinatorProtocolError(
                    "discarded attempt belongs to another worker session"
                )
            return context[0].window.identity.window_id

    def acknowledge_worker_discards(
        self,
        worker_id: str,
        *,
        session_id: str,
    ) -> None:
        """Forget cutoff tombstones after a worker asks for subsequent work."""
        with self._lock:
            self._discarded_attempts = {
                attempt_id: context
                for attempt_id, context in self._discarded_attempts.items()
                if (
                    context[1].worker_id != worker_id
                    or context[1].worker_session_id != session_id
                )
            }

    def accepted_assignment_ids(self) -> frozenset[str]:
        """Return only assignments that actually started in accepted shards."""
        with self._lock:
            return frozenset(
                outcome.curriculum_assignment_id
                for shard in self._shards.values()
                if (result := shard.result) is not None
                for outcome in result.outcomes
            )

    def exhausted_lease_ids(self) -> tuple[str, ...]:
        """Return incomplete shards that cannot start another attempt."""
        with self._lock:
            return tuple(
                lease_id
                for lease_id in self._shard_order
                if (
                    self._shards[lease_id].result is None
                    and not self._shards[lease_id].discarded
                    and self._shards[lease_id].active_attempt is None
                    and len(self._shards[lease_id].attempts) >= self._maximum_attempts
                )
            )

    def exhausted_lease_diagnostics(self) -> tuple[str, ...]:
        """Describe exhausted leases with bounded worker failure evidence."""
        with self._lock:
            diagnostics: list[str] = []
            for lease_id in self.exhausted_lease_ids():
                attempts = self._shards[lease_id].attempts
                failure_evidence = "; ".join(
                    (
                        f"{attempt.identity.worker_id}#"
                        f"{attempt.identity.attempt_sequence}: "
                        f"{_bounded_failure_reason(attempt.failure_reason)}"
                    )
                    for attempt in attempts
                )
                diagnostics.append(f"{lease_id} [{failure_evidence}]")
            return tuple(diagnostics)

    def accept_part(self, decoded: DecodedNativeCollectionPart) -> None:
        """Retain one exact, contiguous part for the current shard attempt."""
        identity = decoded.identity
        with self._lock:
            self._require_collecting_window()
            shard = self._require_shard(identity.lease_id)
            attempt = shard.active_attempt
            if attempt is None or attempt.identity.attempt_id != identity.attempt_id:
                raise NativeCoordinatorProtocolError(
                    "native collection part belongs to an old or inactive attempt"
                )
            if attempt.completion_pending:
                raise NativeCoordinatorProtocolError(
                    "native collection part arrived after shard completion"
                )
            expected_sequence = len(attempt.parts)
            if identity.part_sequence_id != expected_sequence:
                raise NativeCoordinatorProtocolError(
                    "native collection part sequence is not contiguous"
                )
            if any(
                part.identity.part_id == identity.part_id
                for state in self._shards.values()
                for item in state.attempts
                for part in item.parts
            ):
                raise NativeCoordinatorProtocolError(
                    "native collection part ID is duplicated"
                )
            attempt.parts.append(decoded)
            attempt.payload_bytes += decoded.wire_bytes
            self._record_terminal_evidence_locked(attempt, decoded)
            self._maybe_request_high_water_drain_locked()

    def reserve_attempt_completion(
        self,
        result: NativeCollectionShardResult,
    ) -> NativeAttemptCompletionReservation:
        """Reserve immutable ACKed parts without projecting arrays on I/O."""
        with self._lock:
            self._require_collecting_window()
            shard = self._require_shard(result.lease_id)
            attempt = shard.active_attempt
            if attempt is None or attempt.identity.attempt_id != result.attempt_id:
                raise NativeCoordinatorProtocolError(
                    "native shard result belongs to an old or inactive attempt"
                )
            if attempt.completion_pending:
                raise NativeCoordinatorProtocolError(
                    "native shard completion is already pending"
                )
            if result.shard_sequence_id != shard.lease.shard_sequence_id:
                raise NativeCoordinatorProtocolError(
                    "native shard result sequence differs"
                )
            attempt.completion_pending = True
            return NativeAttemptCompletionReservation(
                lease=shard.lease,
                result=result,
                parts=tuple(attempt.parts),
            )

    @staticmethod
    def project_attempt_completion(
        reservation: NativeAttemptCompletionReservation,
    ) -> NativePreparedAttemptCompletion:
        """Validate and project terminal rows without holding coordinator state."""
        result = reservation.result
        cutoff_assignment_ids = frozenset(
            outcome.curriculum_assignment_id
            for outcome in result.outcomes
            if outcome.status == "window_cutoff"
        )
        nonterminal_assignment_ids = frozenset(
            outcome.curriculum_assignment_id
            for outcome in result.outcomes
            if outcome.status != "engine_terminal"
        )
        if len(cutoff_assignment_ids) != result.report.games_window_cutoff or any(
            outcome.candidate_decisions != 0
            for outcome in result.outcomes
            if outcome.status != "engine_terminal"
        ):
            raise NativeCoordinatorProtocolError(
                "native nonterminal outcomes retained trainable decisions"
            )
        retained_parts = (
            tuple(
                _discard_nonterminal_assignments(
                    reservation.parts,
                    nonterminal_assignment_ids,
                )
            )
            if nonterminal_assignment_ids
            else reservation.parts
        )
        if (
            result.part_count != len(retained_parts)
            or result.fragment_count
            != sum(item.identity.fragment_count for item in retained_parts)
            or result.decision_count
            != sum(item.identity.decision_count for item in retained_parts)
        ):
            raise NativeCoordinatorProtocolError(
                "native shard result counts differ from ACKed parts"
            )
        expected = {
            item.curriculum.assignment_id: item
            for item in reservation.lease.assignments
        }
        actual = {item.curriculum_assignment_id: item for item in result.outcomes}
        if (
            len(expected) != len(reservation.lease.assignments)
            or len(actual) != len(result.outcomes)
            or not set(actual) <= set(expected)
            or not actual
        ):
            raise NativeCoordinatorProtocolError(
                "native shard outcomes do not identify started assignments"
            )
        if result.report.assignment_reservations != len(
            reservation.lease.assignments
        ) or result.report.unstarted_reservations_released != len(
            reservation.lease.assignments
        ) - len(actual):
            raise NativeCoordinatorProtocolError(
                "native shard reservation counters differ from its lease"
            )
        for curriculum_id, outcome in actual.items():
            assignment = expected[curriculum_id]
            if outcome.balance_assignment_id != assignment.balance.assignment_id:
                raise NativeCoordinatorProtocolError(
                    "native shard outcome crossed assignment identity"
                )
        return NativePreparedAttemptCompletion(
            reservation=reservation,
            retained_parts=retained_parts,
        )

    def finalize_attempt_completion(
        self,
        prepared: NativePreparedAttemptCompletion,
    ) -> bool:
        """Atomically accept projected rows, or ignore a settled speculative tail."""
        reservation = prepared.reservation
        result = reservation.result
        with self._lock:
            self._require_collecting_window()
            discarded = self._discarded_attempts.get(result.attempt_id)
            if discarded is not None and discarded[0].lease_id == result.lease_id:
                return False
            shard = self._require_shard(result.lease_id)
            attempt = shard.active_attempt
            if (
                attempt is None
                or attempt.identity.attempt_id != result.attempt_id
                or not attempt.completion_pending
            ):
                raise NativeCoordinatorProtocolError(
                    "native prepared completion belongs to an inactive attempt"
                )
            if len(attempt.parts) != len(reservation.parts) or any(
                current is not reserved
                for current, reserved in zip(
                    attempt.parts,
                    reservation.parts,
                    strict=True,
                )
            ):
                raise NativeCoordinatorProtocolError(
                    "native ACKed parts changed during completion projection"
                )
            attempt.parts[:] = prepared.retained_parts
            attempt.completion_pending = False
            attempt.finished = True
            shard.accepted_attempt_id = attempt.identity.attempt_id
            shard.result = result
            self._accepted_decisions += result.decision_count
            worker = self._workers[attempt.identity.worker_id]
            observed_decisions_per_game = result.decision_count / max(
                result.report.games_finished,
                1,
            )
            observed_decisions_per_second = result.decision_count / max(
                result.elapsed_seconds,
                1.0e-9,
            )
            weight = self._shard_evidence_weight_locked(worker, result)
            # Nonterminal games retain no PPO rows, so only engine terminals
            # belong in the yield denominator. A shard containing cancellations
            # is still a lower-bound sample and cannot drag the estimate down.
            previous_decisions_per_game = worker.decisions_per_game_ewma
            if (
                result.report.games_cancelled == 0
                or previous_decisions_per_game is None
                or observed_decisions_per_game > previous_decisions_per_game
            ):
                worker.decisions_per_game_ewma = self._updated_ewma(
                    previous_decisions_per_game,
                    observed_decisions_per_game,
                    weight=weight,
                )
            worker.decisions_per_second_ewma = self._updated_ewma(
                worker.decisions_per_second_ewma,
                observed_decisions_per_second,
                weight=weight,
            )
            worker.active_attempt_id = None
            self._maybe_request_high_water_drain_locked()
            if (
                not self._learner_clocked_primary_only
                and not self._early_drain_requested_locked()
                and self._target_and_exposure_satisfied_locked()
            ):
                self._discard_unsettled_shards_locked()
            return True

    def reject_attempt_completion(
        self,
        reservation: NativeAttemptCompletionReservation,
        *,
        reason: str,
    ) -> None:
        """Fail one reserved completion after asynchronous projection rejects it."""
        if not reason.strip():
            raise ValueError("native completion rejection requires a reason")
        with self._lock:
            discarded = self._discarded_attempts.get(reservation.result.attempt_id)
            if (
                discarded is not None
                and discarded[0].lease_id == reservation.result.lease_id
            ):
                return
            self._fail_attempt_locked(
                reservation.result.attempt_id,
                reason=reason,
            )

    def complete_attempt(self, result: NativeCollectionShardResult) -> None:
        """Synchronously validate and accept one attempt outside server hot paths."""
        reservation = self.reserve_attempt_completion(result)
        try:
            prepared = self.project_attempt_completion(reservation)
            self.finalize_attempt_completion(prepared)
        except BaseException:
            with self._lock:
                shard = self._shards.get(result.lease_id)
                attempt = None if shard is None else shard.active_attempt
                if (
                    attempt is not None
                    and attempt.identity.attempt_id == result.attempt_id
                ):
                    attempt.completion_pending = False
            raise

    def request_learner_ready_drain(self) -> bool:
        """Stop new issuance and drain live games at a safe fragment boundary."""
        with self._lock:
            window = self._require_collecting_window()
            if self._early_drain_requested_locked():
                return True
            if (
                window.shard_protocol_version == 3
                and self._missing_v2_artifact_exposure_ids_locked(
                    include_unsettled=False
                )
            ):
                return False
            if (
                not self._learner_clocked_primary_only
                and self._accepted_decisions >= window.target_trainable_decisions
            ):
                return False
            has_terminal_evidence = any(
                bool(terminal)
                for shard in self._shards.values()
                if shard.result is None and not shard.discarded
                for attempt in (shard.active_attempt,)
                if attempt is not None
                for part in attempt.parts
                for terminal in part.part.arrays["terminal"]
            )
            if self._accepted_decisions == 0 and not has_terminal_evidence:
                return False
            self._learner_ready_drain_requested = True
            self._discard_inactive_unsettled_shards_locked()
            return True

    def fail_attempt(self, attempt_id: str, *, reason: str) -> None:
        """Release every received frame from one failed attempt."""
        if not reason.strip():
            raise ValueError("native attempt failure requires a reason")
        with self._lock:
            self._fail_attempt_locked(attempt_id, reason=reason)

    def ready_to_commit(self) -> bool:
        """Return whether the decision target is complete and the tail is settled."""
        with self._lock:
            if self._window is None or self._receipt is not None:
                return False
            return self._target_and_exposure_satisfied_locked() and all(
                shard.result is not None or shard.discarded
                for shard in self._shards.values()
            )

    def window_drain_hint(self) -> bool:
        """Advise active attempts to cutoff after terminal rows cover the target.

        ACKed parts remain provisional until their shard reports per-game
        outcomes. Draining on those rows would cutoff many games whose entire
        trajectories must then be rolled back, causing a collect/discard/refill
        cycle. Wait for committed, terminal-game rows to reach the target and
        use cutoff only to bound surplus in-flight work.
        """
        with self._lock:
            window = self._window
            if window is None or self._receipt is not None:
                return False
            if self._early_drain_requested_locked():
                return True
            if self._learner_clocked_primary_only:
                return False
            if self._accepted_decisions < window.target_trainable_decisions:
                return False
            return not self._missing_v2_artifact_exposure_ids_locked(
                include_unsettled=(window.shard_protocol_version != 3)
            )

    def missing_v2_artifact_exposure_ids(
        self,
        *,
        include_unsettled: bool = False,
    ) -> tuple[str, ...]:
        """Return frozen artifacts without accepted trainable-decision exposure."""
        with self._lock:
            self._require_collecting_window()
            return self._missing_v2_artifact_exposure_ids_locked(
                include_unsettled=include_unsettled
            )

    def has_unsettled_shards(self) -> bool:
        """Return whether an issued lease can still produce accepted evidence."""
        with self._lock:
            self._require_collecting_window()
            return any(
                shard.result is None and not shard.discarded
                for shard in self._shards.values()
            )

    def ordered_parts(self) -> tuple[DecodedNativeCollectionPart, ...]:
        """Return central shard/part ordering only after the window is complete."""
        with self._lock:
            if not self.ready_to_commit():
                raise NativeCoordinatorError("native collection window is incomplete")
            return tuple(
                part
                for lease_id in self._shard_order
                if self._shards[lease_id].result is not None
                for attempt in self._accepted_attempt(self._shards[lease_id]).parts
                for part in (attempt,)
            )

    def ordered_assignments_and_outcomes(
        self,
    ) -> tuple[
        tuple[StatelessAssignedGame, ...],
        tuple[StatelessGameOutcome, ...],
    ]:
        """Return controller inputs in central lease and assignment order."""
        with self._lock:
            if not self.ready_to_commit():
                raise NativeCoordinatorError("native collection window is incomplete")
            assignments: list[StatelessAssignedGame] = []
            outcomes: list[StatelessGameOutcome] = []
            for lease_id in self._shard_order:
                shard = self._shards[lease_id]
                result = shard.result
                if result is None:
                    continue
                by_curriculum = {
                    item.curriculum_assignment_id: item for item in result.outcomes
                }
                for item in shard.lease.assignments:
                    if item.curriculum.assignment_id not in by_curriculum:
                        continue
                    assignments.append(item.to_assignment())
                    outcomes.append(by_curriculum[item.curriculum.assignment_id])
            return tuple(assignments), tuple(outcomes)

    def completed_shards(
        self,
    ) -> tuple[
        tuple[
            NativeCollectionShardLease,
            NativeCollectionShardResult,
            tuple[DecodedNativeCollectionPart, ...],
        ],
        ...,
    ]:
        """Return complete shard evidence in central sequence order."""
        with self._lock:
            if not self.ready_to_commit():
                raise NativeCoordinatorError("native collection window is incomplete")
            completed: list[
                tuple[
                    NativeCollectionShardLease,
                    NativeCollectionShardResult,
                    tuple[DecodedNativeCollectionPart, ...],
                ]
            ] = []
            for lease_id in self._shard_order:
                shard = self._shards[lease_id]
                result = shard.result
                if result is None:
                    continue
                attempt = self._accepted_attempt(shard)
                completed.append((shard.lease, result, tuple(attempt.parts)))
            return tuple(completed)

    def commit(
        self,
        callback: CommitCallback,
        *,
        now_unix_ns: int,
    ) -> NativeCollectionWindowReceipt:
        """Apply outcomes without holding heartbeat-visible coordinator state."""
        with self._lock:
            if self._receipt is not None:
                if self._receipt.status != "committed":
                    raise NativeCoordinatorError("native collection window was aborted")
                return self._receipt
            window = self._require_collecting_window()
            assignments, outcomes = self.ordered_assignments_and_outcomes()
        # Controller updates and their persistence may be materially slower
        # than one heartbeat. The complete window is immutable here, so keep
        # the lock free while the learner-owned callback performs that work.
        callback(assignments, outcomes)
        with self._lock:
            if self._receipt is not None:
                if self._receipt.status != "committed":
                    raise NativeCoordinatorError("native collection window was aborted")
                return self._receipt
            if self._window != window:
                raise NativeCoordinatorError(
                    "native collection window changed during commit callback"
                )
            receipt = self._build_receipt(
                window,
                status="committed",
                now_unix_ns=now_unix_ns,
            )
            self._receipt = receipt
            self._committed_windows += 1
            return receipt

    def abort(
        self,
        *,
        reason: str,
        now_unix_ns: int,
    ) -> NativeCollectionWindowReceipt:
        """Abort the whole half-window and release all retained frames."""
        if not reason.strip():
            raise ValueError("native window abort requires a reason")
        with self._lock:
            if self._receipt is not None:
                return self._receipt
            window = self._require_collecting_window()
            for shard in self._shards.values():
                for attempt in shard.attempts:
                    attempt.parts.clear()
                    attempt.payload_bytes = 0
                    attempt.decisions_by_assignment.clear()
                    attempt.terminal_assignments.clear()
                    attempt.terminal_decisions = 0
                    attempt.finished = True
            for worker in self._workers.values():
                worker.active_attempt_id = None
            receipt = self._build_receipt(
                window,
                status="aborted",
                now_unix_ns=now_unix_ns,
                abort_reason=reason,
            )
            self._receipt = receipt
            return receipt

    def release_parts(self) -> None:
        """Release coordinator frame ownership after learner consumption."""
        with self._lock:
            if self._receipt is None:
                raise NativeCoordinatorError(
                    "native collection frames cannot release before settlement"
                )
            for shard in self._shards.values():
                for attempt in shard.attempts:
                    attempt.parts.clear()
                    attempt.payload_bytes = 0
                    attempt.decisions_by_assignment.clear()
                    attempt.terminal_assignments.clear()
                    attempt.terminal_decisions = 0

    def status(self, *, now_unix_ns: int) -> NativeCoordinatorStatus:
        """Build one self-consistent monitor snapshot."""
        with self._lock:
            self.expire_workers(now_unix_ns=now_unix_ns)
            connected = tuple(
                sorted(
                    worker_id
                    for worker_id, worker in self._workers.items()
                    if worker.ready
                )
            )
            degraded = tuple(
                worker_id
                for worker_id in self._required_worker_ids
                if worker_id not in connected
            )
            if self._window is None:
                return NativeCoordinatorStatus(
                    connected_workers=connected,
                    degraded_workers=degraded,
                )
            inflight = sum(
                shard.lease.estimated_decision_credit
                for shard in self._shards.values()
                if shard.result is None and shard.active_attempt is not None
            )
            provisional = self._provisional_decisions_locked()
            terminal_provisional = self._terminal_provisional_decisions_locked()
            accepted_attempts = tuple(
                self._accepted_attempt(shard)
                for shard in self._shards.values()
                if shard.result is not None
            )
            parts = tuple(
                part for attempt in accepted_attempts for part in attempt.parts
            )
            active_attempt_ids = tuple(
                shard.active_attempt.identity.attempt_id
                for lease_id in self._shard_order
                if ((shard := self._shards[lease_id]).active_attempt is not None)
            )
            completed_lease_ids = tuple(
                lease_id
                for lease_id in self._shard_order
                if self._shards[lease_id].result is not None
            )
            elapsed = max((now_unix_ns - self._window_started_unix_ns) / 1e9, 0.0)
            payload_bytes = sum(attempt.payload_bytes for attempt in accepted_attempts)
            decisions_per_second = (
                self._accepted_decisions / elapsed if elapsed > 0.0 else 0.0
            )
            bytes_per_decision = (
                payload_bytes / self._accepted_decisions
                if self._accepted_decisions > 0
                else 0.0
            )
            remaining = max(
                self._window.target_trainable_decisions - self._accepted_decisions,
                0,
            )
            eta = (
                remaining / decisions_per_second if decisions_per_second > 0.0 else None
            )
            state: Literal["collecting", "committed", "aborted"]
            state = "collecting" if self._receipt is None else self._receipt.status
            return NativeCoordinatorStatus(
                window_id=self._window.identity.window_id,
                window_sequence_id=self._window.identity.sequence_id,
                window_state=state,
                connected_workers=connected,
                degraded_workers=degraded,
                topology_workers=self._topology_workers,
                target_decisions=self._window.target_trainable_decisions,
                accepted_decisions=self._accepted_decisions,
                provisional_decisions=provisional,
                terminal_provisional_decisions=terminal_provisional,
                inflight_decision_credit=inflight,
                overshoot_decisions=max(
                    self._accepted_decisions - self._window.target_trainable_decisions,
                    0,
                ),
                shards_issued=len(self._shards),
                shards_completed=sum(
                    shard.result is not None for shard in self._shards.values()
                ),
                shards_discarded=sum(
                    shard.discarded for shard in self._shards.values()
                ),
                attempts_started=self._attempts_started,
                retries=self._retries,
                parts_accepted=len(parts),
                active_attempt_ids=active_attempt_ids,
                completed_lease_ids=completed_lease_ids,
                fragments_accepted=sum(part.identity.fragment_count for part in parts),
                payload_bytes_accepted=payload_bytes,
                decisions_per_second=decisions_per_second,
                bytes_per_decision=bytes_per_decision,
                estimated_seconds_remaining=eta,
                learner_ready_drain_requested=(self._learner_ready_drain_requested),
                learner_clocked_high_water_decisions=(
                    self._learner_clocked_max_trainable_decisions
                    if self._learner_clocked_primary_only
                    else None
                ),
                learner_clocked_high_water_reached=(
                    self._learner_clocked_high_water_reached
                ),
                worker_decisions_per_game_ewma={
                    worker_id: value.decisions_per_game_ewma
                    for worker_id, value in sorted(self._workers.items())
                    if value.decisions_per_game_ewma is not None
                },
                worker_decisions_per_second_ewma={
                    worker_id: value.decisions_per_second_ewma
                    for worker_id, value in sorted(self._workers.items())
                    if value.decisions_per_second_ewma is not None
                },
            )

    def _estimated_tier_decisions_locked(
        self,
        worker: _WorkerState,
        tier: NativeCollectionCapacityTier,
    ) -> int:
        """Calibrate a worker's advertised credit with completed local evidence."""
        if worker.decisions_per_game_ewma is None:
            return tier.estimated_trainable_decisions
        return max(
            1,
            math.ceil(worker.decisions_per_game_ewma * tier.concurrent_games),
        )

    def _calibrated_tier_locked(
        self,
        worker: _WorkerState,
        tier: NativeCollectionCapacityTier,
    ) -> NativeCollectionCapacityTier:
        """Return one lease tier carrying its current empirical decision credit."""
        estimated = self._estimated_tier_decisions_locked(worker, tier)
        if estimated == tier.estimated_trainable_decisions:
            return tier
        return tier.model_copy(update={"estimated_trainable_decisions": estimated})

    def _updated_ewma(
        self,
        previous: float | None,
        observed: float,
        *,
        weight: float = 1.0,
    ) -> float | None:
        """Update bounded positive worker evidence after a completed shard."""
        if observed <= 0.0 or not math.isfinite(observed) or weight <= 0.0:
            return previous
        if previous is None:
            if weight < _EWMA_SEED_MINIMUM_WEIGHT:
                return None
            return observed
        alpha = self._worker_yield_ewma_alpha * min(weight, 1.0)
        return alpha * observed + (1.0 - alpha) * previous

    def _shard_evidence_weight_locked(
        self,
        worker: _WorkerState,
        result: NativeCollectionShardResult,
    ) -> float:
        """Scale EWMA updates by shard size against the worker's full tier."""
        reference = max(
            (
                tier.estimated_trainable_decisions
                for tier in worker.manifest.capacity_tiers
            ),
            default=0,
        )
        if reference <= 0:
            return 1.0
        return min(1.0, result.decision_count / reference)

    def _provisional_decisions_locked(self) -> int:
        """Count ACKed parts from active attempts without committing a shard."""
        return sum(
            part.identity.decision_count
            for shard in self._shards.values()
            if shard.result is None and not shard.discarded
            for attempt in (shard.active_attempt,)
            if attempt is not None
            for part in attempt.parts
        )

    def _terminal_provisional_decisions_locked(self) -> int:
        """Count rows from active games already proven engine-terminal."""
        return sum(
            attempt.terminal_decisions
            for shard in self._shards.values()
            if shard.result is None and not shard.discarded
            for attempt in (shard.active_attempt,)
            if attempt is not None
        )

    def _record_terminal_evidence_locked(
        self,
        attempt: _AttemptState,
        decoded: DecodedNativeCollectionPart,
    ) -> None:
        """Incrementally retain exact whole-game row evidence for one attempt."""
        arrays = decoded.part.arrays
        offsets = arrays["fragment_decision_offsets"]
        for index, (raw_assignment_id, raw_terminal) in enumerate(
            zip(arrays["assignment_ids"], arrays["terminal"], strict=True)
        ):
            assignment_id = str(raw_assignment_id)
            decision_count = int(offsets[index + 1]) - int(offsets[index])
            previous = attempt.decisions_by_assignment.get(assignment_id, 0)
            attempt.decisions_by_assignment[assignment_id] = previous + decision_count
            if assignment_id in attempt.terminal_assignments:
                attempt.terminal_decisions += decision_count
            elif bool(raw_terminal):
                attempt.terminal_assignments.add(assignment_id)
                attempt.terminal_decisions += previous + decision_count

    def _maybe_request_high_water_drain_locked(self) -> None:
        """Close an overlap window once retained terminal rows reach its ceiling."""
        maximum = self._learner_clocked_max_trainable_decisions
        if (
            not self._learner_clocked_primary_only
            or maximum is None
            or self._early_drain_requested_locked()
        ):
            return
        observed = (
            self._accepted_decisions + self._terminal_provisional_decisions_locked()
        )
        if observed < maximum:
            return
        self._learner_clocked_high_water_reached = True
        self._discard_inactive_unsettled_shards_locked()

    def _early_drain_requested_locked(self) -> bool:
        """Return whether learner readiness or the sample ceiling closed admission."""
        return bool(
            self._learner_ready_drain_requested
            or self._learner_clocked_high_water_reached
        )

    def _remaining_inflight_credit_locked(self) -> int:
        """Return estimated active credit not already observed in streamed parts."""
        return sum(
            max(
                shard.lease.estimated_decision_credit
                - sum(part.identity.decision_count for part in attempt.parts),
                0,
            )
            for shard in self._shards.values()
            if shard.result is None and not shard.discarded
            for attempt in (shard.active_attempt,)
            if attempt is not None
        )

    def _start_attempt_locked(
        self,
        shard: _ShardState,
        worker: _WorkerState,
        *,
        now_unix_ns: int,
    ) -> NativeCollectionAttempt:
        attempt_sequence = len(shard.attempts)
        attempt_id = hashlib.sha256(
            (
                f"{shard.lease.lease_id}:{attempt_sequence}:"
                f"{worker.manifest.identity.worker_id}:"
                f"{worker.manifest.identity.session_id}"
            ).encode()
        ).hexdigest()
        attempt = NativeCollectionAttempt(
            attempt_id=attempt_id,
            lease_id=shard.lease.lease_id,
            worker_id=worker.manifest.identity.worker_id,
            worker_session_id=worker.manifest.identity.session_id,
            attempt_sequence=attempt_sequence,
            started_at_unix_ns=now_unix_ns,
            expires_at_unix_ns=now_unix_ns + self._lease_timeout_ns,
        )
        shard.attempts.append(_AttemptState(identity=attempt))
        worker.active_attempt_id = attempt_id
        self._attempts_started += 1
        return attempt

    def _fail_attempt_locked(self, attempt_id: str, *, reason: str) -> None:
        for shard in self._shards.values():
            attempt = shard.active_attempt
            if attempt is None or attempt.identity.attempt_id != attempt_id:
                continue
            attempt.parts.clear()
            attempt.payload_bytes = 0
            attempt.decisions_by_assignment.clear()
            attempt.terminal_assignments.clear()
            attempt.terminal_decisions = 0
            attempt.completion_pending = False
            attempt.finished = True
            attempt.failure_reason = reason
            worker = self._workers.get(attempt.identity.worker_id)
            if worker is not None and worker.active_attempt_id == attempt_id:
                worker.active_attempt_id = None
            if self._early_drain_requested_locked():
                self._discard_shard_locked(shard)
            return
        raise NativeCoordinatorProtocolError(
            f"native collection attempt is not active: {attempt_id} ({reason})"
        )

    def _has_fresh_retry_worker_locked(
        self,
        shard: _ShardState,
        *,
        excluding_worker_id: str,
    ) -> bool:
        attempted_sessions = {
            (
                attempt.identity.worker_id,
                attempt.identity.worker_session_id,
            )
            for attempt in shard.attempts
        }
        for worker_id in self._topology_workers:
            if worker_id == excluding_worker_id:
                continue
            worker = self._workers.get(worker_id)
            if worker is None or not worker.ready:
                continue
            worker_session = (
                worker_id,
                worker.manifest.identity.session_id,
            )
            if worker_session in attempted_sessions:
                continue
            if (
                native_retry_capacity_tier(
                    worker.manifest.capacity_tiers,
                    shard.lease,
                )
                is not None
            ):
                return True
        return False

    def _target_and_exposure_satisfied_locked(self) -> bool:
        window = self._window
        if (
            window is not None
            and window.shard_protocol_version == 3
            and self._missing_v2_artifact_exposure_ids_locked(include_unsettled=False)
        ):
            return False
        if self._learner_clocked_primary_only:
            return bool(
                window is not None
                and self._early_drain_requested_locked()
                and self._accepted_decisions > 0
            )
        return bool(
            window is not None
            and (
                self._accepted_decisions >= window.target_trainable_decisions
                or (
                    self._early_drain_requested_locked()
                    and self._accepted_decisions > 0
                )
            )
        )

    def _discard_unsettled_shards_locked(self) -> None:
        """Drop speculative tail shards once accepted evidence is sufficient."""
        for shard in self._shards.values():
            if shard.result is not None or shard.discarded:
                continue
            self._discard_shard_locked(shard)

    def _discard_inactive_unsettled_shards_locked(self) -> None:
        """Discard failed shards at drain without cancelling live attempts."""
        for shard in self._shards.values():
            if (
                shard.result is None
                and not shard.discarded
                and shard.active_attempt is None
            ):
                self._discard_shard_locked(shard)

    def _discard_shard_locked(self, shard: _ShardState) -> None:
        """Discard one shard and retain tombstones for late worker messages."""
        manifests = {
            (
                manifest.identity.worker_id,
                manifest.identity.session_id,
            ): manifest
            for manifest in self._topology_manifests
        }
        shard.discarded = True
        for attempt in shard.attempts:
            manifest = manifests[
                (
                    attempt.identity.worker_id,
                    attempt.identity.worker_session_id,
                )
            ]
            self._discarded_attempts[attempt.identity.attempt_id] = (
                shard.lease,
                attempt.identity,
                manifest,
            )
            attempt.parts.clear()
            attempt.payload_bytes = 0
            attempt.decisions_by_assignment.clear()
            attempt.terminal_assignments.clear()
            attempt.terminal_decisions = 0
            attempt.finished = True
            worker = self._workers.get(attempt.identity.worker_id)
            if (
                worker is not None
                and worker.active_attempt_id == attempt.identity.attempt_id
            ):
                worker.active_attempt_id = None

    def _build_receipt(
        self,
        window: NativeCollectionWindow,
        *,
        status: Literal["committed", "aborted"],
        now_unix_ns: int,
        abort_reason: str | None = None,
    ) -> NativeCollectionWindowReceipt:
        completed = tuple(
            self._shards[lease_id]
            for lease_id in self._shard_order
            if self._shards[lease_id].result is not None
        )
        return NativeCollectionWindowReceipt(
            window_id=window.identity.window_id,
            sequence_id=window.identity.sequence_id,
            status=status,
            target_trainable_decisions=window.target_trainable_decisions,
            accepted_trainable_decisions=self._accepted_decisions,
            overshoot_decisions=max(
                self._accepted_decisions - window.target_trainable_decisions,
                0,
            ),
            shard_lease_ids=tuple(shard.lease.lease_id for shard in completed),
            accepted_attempt_ids=tuple(
                str(shard.accepted_attempt_id) for shard in completed
            ),
            worker_manifests=self._topology_manifests,
            committed_at_unix_ns=now_unix_ns,
            abort_reason=abort_reason,
            early_commit_reason=(
                self._early_drain_reason_locked() if status == "committed" else None
            ),
            shard_protocol_version=window.shard_protocol_version,
            assignment_plan_revision=window.assignment_plan_revision,
            opponent_pool_revision=window.opponent_pool_revision,
            required_exposure_artifact_ids=(
                window.active_pfsp_artifact_ids
                if window.shard_protocol_version == 3
                else ()
            ),
            artifact_exposures=(
                ()
                if window.shard_protocol_version == 1
                else _aggregate_artifact_exposures(completed)
            ),
        )

    def _early_drain_reason_locked(
        self,
    ) -> Literal["learner_ready_drain", "learner_clocked_high_water"] | None:
        """Return the first-class reason for an early learner-clocked close."""
        if self._learner_ready_drain_requested:
            return "learner_ready_drain"
        if self._learner_clocked_high_water_reached:
            return "learner_clocked_high_water"
        return None

    def _missing_v2_artifact_exposure_ids_locked(
        self,
        *,
        include_unsettled: bool,
    ) -> tuple[str, ...]:
        """Find required V2 artifacts absent from accepted decision evidence."""
        window = self._window
        if window is None or window.shard_protocol_version == 1:
            return ()
        required = set(window.active_pfsp_artifact_ids)
        covered: set[str] = set()
        for shard in self._shards.values():
            if shard.result is None:
                if include_unsettled:
                    covered.update(
                        required & set(shard.lease.effective_required_artifact_ids)
                    )
                continue
            covered.update(
                artifact.artifact_id
                for artifact in _shard_artifact_exposures(shard)
                if (
                    artifact.kind != "current"
                    and artifact.candidate_trainable_decisions > 0
                )
            )
        return tuple(sorted(required - covered))

    def _validate_worker_contract(self, actual: NativeRolloutWorkerIdentity) -> None:
        expected = self._expected_contract
        fields = (
            "source_snapshot_fingerprint",
            "native_abi_version",
            "engine_fact_contract_fingerprint",
            "feature_schema_fingerprint",
            "card_catalog_fingerprint",
            "static_features_fingerprint",
            "exact_registry_fingerprint",
            "scripted_opponents_fingerprint",
            "historical_opponents_fingerprint",
            "model_config_fingerprint",
            "resolved_config_fingerprint",
        )
        mismatches = tuple(
            name for name in fields if getattr(expected, name) != getattr(actual, name)
        )
        if mismatches:
            raise NativeCoordinatorProtocolError(
                "native worker compatibility differs: " + ", ".join(mismatches)
            )

    def _worker_session(self, worker_id: str, session_id: str) -> _WorkerState:
        for workers in (self._workers, self._pending_workers):
            worker = workers.get(worker_id)
            if worker is not None and worker.manifest.identity.session_id == session_id:
                return worker
        raise NativeCoordinatorProtocolError("native worker session identity differs")

    def _promote_pending_workers_locked(self) -> None:
        """Make boundary-waiting sessions eligible for the next topology."""
        for worker_id, worker in self._pending_workers.items():
            self._workers[worker_id] = worker
        self._pending_workers.clear()

    def _require_topology_worker(self, worker_id: str) -> _WorkerState:
        if worker_id not in self._topology_workers:
            raise NativeCoordinatorProtocolError(
                "worker is not in the active window topology"
            )
        worker = self._workers[worker_id]
        if not worker.ready:
            raise NativeCoordinatorProtocolError("worker is not READY")
        return worker

    def _require_collecting_window(self) -> NativeCollectionWindow:
        if self._window is None:
            raise NativeCoordinatorError("native collection window is not open")
        if self._receipt is not None:
            raise NativeCoordinatorError("native collection window is settled")
        return self._window

    def _require_shard(self, lease_id: str) -> _ShardState:
        try:
            return self._shards[lease_id]
        except KeyError as exc:
            raise NativeCoordinatorProtocolError(
                "native collection shard lease is unknown"
            ) from exc

    @staticmethod
    def _accepted_attempt(shard: _ShardState) -> _AttemptState:
        attempt_id = shard.accepted_attempt_id
        if attempt_id is None:
            raise NativeCoordinatorError("native shard has no accepted attempt")
        try:
            return next(
                item
                for item in shard.attempts
                if item.identity.attempt_id == attempt_id
            )
        except StopIteration as exc:
            raise NativeCoordinatorError(
                "native shard accepted attempt disappeared"
            ) from exc


__all__ = [
    "NativeCollectionCoordinator",
    "NativeCoordinatorError",
    "NativeCoordinatorProtocolError",
    "NativeCoordinatorQuorumError",
    "NativeCoordinatorStatus",
]
