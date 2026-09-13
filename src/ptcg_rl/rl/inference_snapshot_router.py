"""Bounded immutable policy-snapshot routing for planner inference."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast

from ptcg_rl.rl.recurrent_runtime import (
    PolicyArtifactIdentity,
    RecurrentSequenceIdentity,
)
from ptcg_rl.runtime.model_lease import (
    ModelLease,
    ModelLeaseCapacityError,
    ModelSnapshotPool,
    ModelSnapshotPoolStats,
)

_PolicyT = TypeVar("_PolicyT")

CandidateSnapshotFactory = Callable[[Mapping[str, Any], int, str], Any]


@dataclass(slots=True)
class _RootLeaseHolder:
    """One root decode lease shared by every row handle in its batch."""

    lease: ModelLease[Any]
    remaining_handles: set[str]
    deadline_monotonic: float


@dataclass(frozen=True)
class RecurrentPolicyBinding:
    """One immutable policy selected for a recurrent request batch."""

    policy: Any
    policy_version: int
    artifact: PolicyArtifactIdentity


@dataclass(frozen=True)
class _RecurrentSequenceLeaseKey:
    """Globally names one actor policy's game-seat recurrent sequence."""

    actor_id: str
    actor_incarnation: int
    policy_id: str
    sequence: RecurrentSequenceIdentity


@dataclass(frozen=True)
class _RecurrentSequenceLeaseHolder:
    """Keep one model snapshot resident for one complete game sequence."""

    lease: ModelLease[Any]
    artifact: PolicyArtifactIdentity


class InferencePolicySnapshotRouter:
    """Route planner requests and recurrent games through immutable snapshots.

    A root planner decode acquires the current snapshot before its first model
    call and transfers ownership of that lease to the emitted context handles.
    Bound continuation/value/proposal calls route to the resident version.
    Recurrent game-seat sequences similarly retain their first-bound snapshot
    until an explicit terminal/error release.
    """

    def __init__(
        self,
        initial_policy: Any,
        *,
        max_resident_snapshots: int,
        max_in_flight_leases: int,
        max_recurrent_sequence_leases: int | None = None,
        recurrent_replay_cache_capacity: int = 4096,
    ) -> None:
        version = _policy_version(initial_policy)
        _policy_model_fingerprint(initial_policy)
        _policy_proposal_version(initial_policy)
        if _policy_recurrent_enabled(initial_policy):
            _policy_artifact_identity(initial_policy)
        self._pool = ModelSnapshotPool[Any](
            max_resident_snapshots=max_resident_snapshots,
            max_in_flight_leases=max_in_flight_leases,
        )
        self._pool.publish(version, initial_policy)
        self._lock = threading.RLock()
        self._max_recurrent_sequence_leases = (
            max_in_flight_leases
            if max_recurrent_sequence_leases is None
            else int(max_recurrent_sequence_leases)
        )
        if self._max_recurrent_sequence_leases <= 0:
            raise ValueError("max_recurrent_sequence_leases must be positive")
        if recurrent_replay_cache_capacity <= 0:
            raise ValueError("recurrent_replay_cache_capacity must be positive")
        self._recurrent_replay_cache_capacity = int(recurrent_replay_cache_capacity)
        self._handles: dict[str, _RootLeaseHolder] = {}
        self._active_by_version: dict[int, int] = {}
        self._recurrent_sequences: dict[
            _RecurrentSequenceLeaseKey,
            _RecurrentSequenceLeaseHolder,
        ] = {}
        self._recurrent_actor_incarnations: dict[str, int] = {}
        # Release acknowledgements can be lost after the server has already
        # dropped the live lease.  Keep the exact released ownership identity
        # independently from the (larger) response payload replay cache so a
        # retry with a new request ID remains safe and artifact-checked.
        self._recurrent_release_tombstones: OrderedDict[
            _RecurrentSequenceLeaseKey,
            PolicyArtifactIdentity,
        ] = OrderedDict()
        self._recurrent_replay_cache: OrderedDict[tuple[str, int, str, int], Any] = (
            OrderedDict()
        )

    @property
    def policy_version(self) -> int:
        """Return the current snapshot version for unbound base requests."""
        current = self._pool.stats().current_version
        if current is None:
            raise RuntimeError("policy snapshot router has no current version")
        return int(current)

    @property
    def model_fingerprint(self) -> str:
        """Return the current snapshot's full model-state identity."""
        return _policy_model_fingerprint(self._pool.resident_snapshot())

    @property
    def proposal_version(self) -> int:
        """Return the current snapshot proposal architecture version."""
        return _policy_proposal_version(self._pool.resident_snapshot())

    @property
    def recurrent_enabled(self) -> bool:
        """Return whether the current snapshot requires recurrent transport."""
        return _policy_recurrent_enabled(self._pool.resident_snapshot())

    @property
    def planner_context_enabled(self) -> bool:
        """Return whether current root decode retains planner contexts."""
        return bool(
            getattr(self._pool.resident_snapshot(), "planner_context_enabled", False)
        )

    @property
    def graph_decode_enabled(self) -> bool:
        """Return whether the current unbound snapshot owns a graph runner."""
        return bool(
            getattr(self._pool.resident_snapshot(), "graph_decode_enabled", False)
        )

    @property
    def _model(self) -> Any:
        """Expose the current immutable model for read-only routing metadata."""
        return getattr(self._pool.resident_snapshot(), "_model", None)

    def stats(self) -> ModelSnapshotPoolStats:
        """Return bounded residency and lease telemetry."""
        return self._pool.stats()

    @property
    def active_recurrent_sequences(self) -> int:
        """Return game-seat sequences currently pinning model snapshots."""
        with self._lock:
            return len(self._recurrent_sequences)

    def graph_decode_stats(self) -> Mapping[str, int]:
        """Aggregate graph counters across currently resident snapshots."""
        totals: dict[str, int] = {}
        for version in self._pool.stats().resident_versions:
            with suppress(ModelLeaseCapacityError):
                policy = self._pool.resident_snapshot(version)
                stats = getattr(policy, "graph_decode_stats", None)
                if not callable(stats):
                    continue
                for key, value in cast(Mapping[str, int], stats()).items():
                    totals[str(key)] = totals.get(str(key), 0) + int(value)
        return totals

    def can_publish(self, version: int) -> bool:
        """Return whether a newer snapshot can be admitted now."""
        return self._pool.can_publish(version)

    def publish_snapshot(self, version: int, policy: Any) -> None:
        """Atomically expose one fully constructed immutable snapshot."""
        if _policy_version(policy) != int(version):
            raise ValueError("snapshot policy version differs from publication")
        _policy_model_fingerprint(policy)
        _policy_proposal_version(policy)
        if _policy_recurrent_enabled(policy):
            _policy_artifact_identity(policy)
        with self._lock:
            self._pool.publish(int(version), policy)

    def bind_recurrent_sequences(
        self,
        *,
        actor_id: str,
        actor_incarnation: int = 0,
        policy_id: str,
        sequences: Sequence[RecurrentSequenceIdentity],
        expected_artifact: PolicyArtifactIdentity | None,
    ) -> RecurrentPolicyBinding:
        """Bind new sequences or route existing ones to one exact snapshot.

        An unbound first request acquires the current snapshot once per sequence.
        Every continuation must present the complete artifact identity returned by
        that first bind. The per-sequence leases, rather than a version alias,
        keep retired snapshots resident until explicit terminal/error release.
        """
        keys = _recurrent_sequence_lease_keys(
            actor_id=actor_id,
            actor_incarnation=actor_incarnation,
            policy_id=policy_id,
            sequences=sequences,
        )
        if expected_artifact is not None and not isinstance(
            expected_artifact,
            PolicyArtifactIdentity,
        ):
            raise TypeError(
                "expected recurrent artifact must be PolicyArtifactIdentity"
            )
        with self._lock:
            self.admit_recurrent_actor_incarnation(
                actor_id=actor_id,
                actor_incarnation=actor_incarnation,
            )
            if expected_artifact is None:
                if (
                    len(self._recurrent_sequences) + len(keys)
                    > self._max_recurrent_sequence_leases
                ):
                    raise ModelLeaseCapacityError(
                        "recurrent sequence lease capacity is exhausted"
                    )
                return self._bind_new_recurrent_sequences(keys)
            return self._bound_recurrent_policy(keys, expected_artifact)

    def release_recurrent_sequences(
        self,
        *,
        actor_id: str,
        actor_incarnation: int = 0,
        policy_id: str,
        sequences: Sequence[RecurrentSequenceIdentity],
        expected_artifact: PolicyArtifactIdentity,
    ) -> int:
        """Idempotently release exact sequence leases and validate ownership.

        RPC delivery is at-least-once: an actor can lose the first successful
        acknowledgement and retry under a new request ID.  A bounded exact-key
        tombstone makes that retry a no-op while still rejecting an unknown
        sequence or an artifact mismatch.
        """
        if not isinstance(expected_artifact, PolicyArtifactIdentity):
            raise TypeError(
                "expected recurrent artifact must be PolicyArtifactIdentity"
            )
        keys = _recurrent_sequence_lease_keys(
            actor_id=actor_id,
            actor_incarnation=actor_incarnation,
            policy_id=policy_id,
            sequences=sequences,
        )
        with self._lock:
            self.admit_recurrent_actor_incarnation(
                actor_id=actor_id,
                actor_incarnation=actor_incarnation,
            )
            holders = tuple(self._recurrent_sequences.get(key) for key in keys)
            active = tuple(holder for holder in holders if holder is not None)
            if any(holder.artifact != expected_artifact for holder in active):
                raise RuntimeError("recurrent sequence release artifact mismatch")
            for key, holder in zip(keys, holders, strict=True):
                if holder is not None:
                    continue
                released_artifact = self._recurrent_release_tombstones.get(key)
                if released_artifact is None:
                    raise KeyError(
                        "recurrent sequence lease is not active: "
                        f"{key.sequence.game_id}"
                    )
                if released_artifact != expected_artifact:
                    raise RuntimeError("recurrent sequence release artifact mismatch")
            for key, holder in zip(keys, holders, strict=True):
                if holder is not None:
                    del self._recurrent_sequences[key]
                self._remember_recurrent_release(key, expected_artifact)
        for holder in active:
            holder.lease.release()
        return len(active)

    def abort_recurrent_sequences(
        self,
        *,
        actor_id: str,
        actor_incarnation: int = 0,
        policy_id: str,
        sequences: Sequence[RecurrentSequenceIdentity],
    ) -> tuple[int, PolicyArtifactIdentity | None]:
        """Release a timed-out first bind by its exact actor namespace.

        The actor cannot name the served artifact when its first response was
        lost.  Actor and policy IDs remain part of the lease key, so this path
        cannot claim another client's state. Missing leases are accepted to
        make cleanup safe when decode failed before binding.
        """
        keys = _recurrent_sequence_lease_keys(
            actor_id=actor_id,
            actor_incarnation=actor_incarnation,
            policy_id=policy_id,
            sequences=sequences,
        )
        with self._lock:
            self.admit_recurrent_actor_incarnation(
                actor_id=actor_id,
                actor_incarnation=actor_incarnation,
            )
            holders = tuple(self._recurrent_sequences.get(key) for key in keys)
            active = tuple(holder for holder in holders if holder is not None)
            artifacts = {holder.artifact for holder in active}
            for key, holder in zip(keys, holders, strict=True):
                self._recurrent_sequences.pop(key, None)
                if holder is not None:
                    self._remember_recurrent_release(key, holder.artifact)
        for holder in active:
            holder.lease.release()
        return (
            len(active),
            next(iter(artifacts)) if len(artifacts) == 1 else None,
        )

    def _remember_recurrent_release(
        self,
        key: _RecurrentSequenceLeaseKey,
        artifact: PolicyArtifactIdentity,
    ) -> None:
        """Retain one bounded exact release identity while holding ``_lock``."""
        existing = self._recurrent_release_tombstones.get(key)
        if existing is not None and existing != artifact:
            raise RuntimeError("recurrent release tombstone artifact mismatch")
        self._recurrent_release_tombstones[key] = artifact
        self._recurrent_release_tombstones.move_to_end(key)
        while (
            len(self._recurrent_release_tombstones)
            > self._recurrent_replay_cache_capacity
        ):
            self._recurrent_release_tombstones.popitem(last=False)

    def admit_recurrent_actor_incarnation(
        self,
        *,
        actor_id: str,
        actor_incarnation: int,
    ) -> int:
        """Admit one actor process generation and reclaim older leases.

        The parent supervisor assigns monotonically increasing incarnations.
        Once a replacement actor is observed, queued requests from its dead
        predecessor are rejected and cannot re-pin or release the new actor's
        sequence state.
        """
        cleaned_actor = str(actor_id).strip()
        incarnation = int(actor_incarnation)
        if not cleaned_actor:
            raise ValueError("recurrent actor ID must be non-empty")
        if incarnation < 0:
            raise ValueError("recurrent actor incarnation must be non-negative")
        with self._lock:
            current = self._recurrent_actor_incarnations.get(cleaned_actor)
            if current is not None and incarnation < current:
                raise RuntimeError(
                    "stale recurrent actor incarnation: "
                    f"actor={cleaned_actor}, request={incarnation}, current={current}"
                )
            if current is not None and incarnation == current:
                return 0
            stale = tuple(
                (key, holder)
                for key, holder in self._recurrent_sequences.items()
                if key.actor_id == cleaned_actor
                and key.actor_incarnation != incarnation
            )
            for sequence_key, holder in stale:
                del self._recurrent_sequences[sequence_key]
                self._remember_recurrent_release(sequence_key, holder.artifact)
            stale_replays = tuple(
                replay_key
                for replay_key in self._recurrent_replay_cache
                if replay_key[0] == cleaned_actor and replay_key[1] != incarnation
            )
            for replay_key in stale_replays:
                del self._recurrent_replay_cache[replay_key]
            self._recurrent_actor_incarnations[cleaned_actor] = incarnation
        for _key, holder in stale:
            holder.lease.release()
        return len(stale)

    def recurrent_actor_incarnation(self, actor_id: str) -> int | None:
        """Return the newest admitted process generation for one actor."""
        cleaned_actor = str(actor_id).strip()
        if not cleaned_actor:
            raise ValueError("recurrent actor ID must be non-empty")
        with self._lock:
            return self._recurrent_actor_incarnations.get(cleaned_actor)

    def recurrent_replay_lookup(
        self,
        *,
        actor_id: str,
        actor_incarnation: int = 0,
        policy_id: str,
        request_id: int,
    ) -> Any | None:
        """Return a completed same-ID response without re-running sampling."""
        key = (
            str(actor_id),
            int(actor_incarnation),
            str(policy_id),
            int(request_id),
        )
        with self._lock:
            value = self._recurrent_replay_cache.get(key)
            if value is not None:
                self._recurrent_replay_cache.move_to_end(key)
            return value

    def recurrent_replay_store(
        self,
        response: Any,
        *,
        actor_id: str,
        actor_incarnation: int = 0,
        policy_id: str,
        request_id: int,
    ) -> None:
        """Publish one completed response into the bounded replay cache."""
        key = (
            str(actor_id),
            int(actor_incarnation),
            str(policy_id),
            int(request_id),
        )
        with self._lock:
            existing = self._recurrent_replay_cache.get(key)
            if existing is not None and existing is not response:
                raise RuntimeError("recurrent request ID already completed")
            self._recurrent_replay_cache[key] = response
            self._recurrent_replay_cache.move_to_end(key)
            while (
                len(self._recurrent_replay_cache)
                > self._recurrent_replay_cache_capacity
            ):
                self._recurrent_replay_cache.popitem(last=False)

    def has_resident_version(self, version: int) -> bool:
        """Return whether the exact version can still serve a bound request."""
        return self._pool.has_resident_version(version)

    def supports_model_version_lease(self, version: int) -> bool:
        """Return whether a bound request has a live root lease and snapshot."""
        return self.has_active_root_lease(version) and self.has_resident_version(
            version
        )

    def has_active_root_lease(self, version: int) -> bool:
        """Return whether a root request still owns the named version."""
        with self._lock:
            return self._active_by_version.get(int(version), 0) > 0

    def policy_for_lease(self, version: int) -> Any:
        """Resolve a version already protected by a live root request lease."""
        if not self.has_active_root_lease(version):
            raise ModelLeaseCapacityError(
                "planner model version has no active root lease"
            )
        return self._pool.resident_snapshot(version)

    def sample_decode_with_trace_for_request(
        self,
        states: Any,
        options: Any,
        decks: Any,
        *,
        temperature: float,
        model_version_lease: int | None,
        retain_planner_context: bool,
    ) -> Any:
        """Decode a root or continuation under the exact request lease."""
        if retain_planner_context:
            if model_version_lease is not None:
                raise ValueError("root planner decode must acquire an unbound lease")
            return self._decode_root_with_lease(
                states,
                options,
                decks,
                temperature=temperature,
            )
        if model_version_lease is None:
            with self._pool.acquire() as lease:
                trace = _decode_without_context(
                    lease.snapshot,
                    states,
                    options,
                    decks,
                    temperature=temperature,
                )
                return _bind_trace_identity(trace, lease.snapshot)
        policy = self.policy_for_lease(model_version_lease)
        trace = _decode_without_context(
            policy,
            states,
            options,
            decks,
            temperature=temperature,
        )
        return _bind_trace_identity(trace, policy)

    def sample_decode_with_trace(
        self,
        states: Any,
        options: Any,
        decks: Any,
        *,
        temperature: float = 1.0,
    ) -> Any:
        """Serve an ordinary unbound decode without retaining root state."""
        return self.sample_decode_with_trace_for_request(
            states,
            options,
            decks,
            temperature=temperature,
            model_version_lease=None,
            retain_planner_context=False,
        )

    def sample_decode(
        self,
        states: Any,
        options: Any,
        decks: Any,
        *,
        temperature: float = 1.0,
    ) -> tuple[Any, Any, Any]:
        """Serve the legacy three-tensor decode surface."""
        trace = self.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def predict_values(self, states: Any, decks: Any) -> Any:
        """Serve an ordinary current-snapshot value request."""
        with self._pool.acquire() as lease:
            return lease.snapshot.predict_values(states, decks)

    def evaluate_planner_candidates(
        self,
        states: Any,
        options: Any,
        candidate_actions: Any,
        candidate_features: Any,
        *,
        ordered_rows: Any,
        decks: Any,
        planner_context_handles: Sequence[str],
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> Any:
        """Route candidate evaluation to its leased resident snapshot."""
        policy = self._policy_for_handles(
            planner_context_handles,
            version=model_version_lease,
        )
        return policy.evaluate_planner_candidates(
            states,
            options,
            candidate_actions,
            candidate_features,
            ordered_rows=ordered_rows,
            decks=decks,
            planner_context_handles=planner_context_handles,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )

    def generate_planner_proposals(
        self,
        states: Any,
        options: Any,
        *,
        ordered_rows: Any,
        limits: Any,
        decks: Any,
        planner_context_handles: Sequence[str],
        model_version_lease: int,
        tensor_schema_fingerprint: str,
        deadline_monotonic: float | None = None,
    ) -> Any:
        """Route proposal generation to its leased resident snapshot."""
        policy = self._policy_for_handles(
            planner_context_handles,
            version=model_version_lease,
        )
        return policy.generate_planner_proposals(
            states,
            options,
            ordered_rows=ordered_rows,
            limits=limits,
            decks=decks,
            planner_context_handles=planner_context_handles,
            model_version_lease=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            deadline_monotonic=deadline_monotonic,
        )

    def bind_planner_context_handles(
        self,
        handles: Sequence[str],
        *,
        policy_version: int,
        tensor_schema_fingerprint: str,
        deadline_monotonic: float | None = None,
    ) -> None:
        """Bind root handles only after proving they share the served lease."""
        policy = self._policy_for_handles(handles, version=policy_version)
        policy.bind_planner_context_handles(
            handles,
            policy_version=policy_version,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )
        if deadline_monotonic is not None:
            deadline = float(deadline_monotonic)
            if deadline <= 0.0:
                raise ValueError("planner context deadline must be positive")
            with self._lock:
                for handle in handles:
                    holder = self._handles.get(str(handle))
                    if holder is None:
                        raise KeyError("planner context handle is not leased")
                    holder.deadline_monotonic = min(
                        holder.deadline_monotonic,
                        deadline,
                    )

    def expire_planner_context_handles(self, now_monotonic: float) -> int:
        """Release root leases whose bounded caller-liveness TTL elapsed."""
        now = float(now_monotonic)
        with self._lock:
            holders = {
                id(holder): holder
                for holder in self._handles.values()
                if holder.deadline_monotonic <= now
            }
            handles = tuple(
                handle
                for holder in holders.values()
                for handle in tuple(holder.remaining_handles)
            )
        if not handles:
            return 0
        return self._release_planner_context_handles(handles, missing_ok=True)

    def release_planner_context_handles(self, handles: Sequence[str]) -> int:
        """Release contexts and their shared root model leases exactly once."""
        return self._release_planner_context_handles(handles, missing_ok=False)

    def _release_planner_context_handles(
        self,
        handles: Sequence[str],
        *,
        missing_ok: bool,
    ) -> int:
        """Atomically claim handles before invoking snapshot cleanup."""
        frozen = tuple(str(handle) for handle in handles)
        if not frozen:
            return 0
        with self._lock:
            if len(set(frozen)) != len(frozen):
                raise ValueError("planner context release contains duplicate handles")
            grouped: dict[int, tuple[_RootLeaseHolder, list[str]]] = {}
            for handle in frozen:
                holder = self._handles.get(handle)
                if holder is None:
                    if missing_ok:
                        continue
                    raise KeyError(f"planner context handle is not leased: {handle}")
                key = id(holder)
                grouped.setdefault(key, (holder, []))[1].append(handle)
            lease_releases: set[int] = set()
            for holder, holder_handles in grouped.values():
                for handle in holder_handles:
                    if self._handles.get(handle) is not holder:
                        raise AssertionError("planner handle claim changed under lock")
                    del self._handles[handle]
                    holder.remaining_handles.remove(handle)
                if not holder.remaining_handles:
                    version = int(holder.lease.version)
                    active = self._active_by_version.get(version, 0)
                    if active <= 0:
                        raise AssertionError("planner root lease count underflow")
                    if active == 1:
                        self._active_by_version.pop(version)
                    else:
                        self._active_by_version[version] = active - 1
                    lease_releases.add(id(holder))

        released = 0
        for holder, holder_handles in grouped.values():
            policy = holder.lease.snapshot
            try:
                count = int(policy.release_planner_context_handles(holder_handles))
                if count != len(holder_handles):
                    raise RuntimeError("snapshot released an incomplete handle group")
                released += count
            finally:
                if id(holder) in lease_releases:
                    holder.lease.release()
        return released

    def close(self) -> None:
        """Best-effort cleanup for shutdown and failed serving startup."""
        with self._lock:
            handles = tuple(self._handles)
            recurrent = tuple(self._recurrent_sequences.values())
            self._recurrent_sequences.clear()
            self._recurrent_actor_incarnations.clear()
            self._recurrent_release_tombstones.clear()
            self._recurrent_replay_cache.clear()
        if handles:
            with suppress(Exception):
                self._release_planner_context_handles(handles, missing_ok=True)
        for holder in recurrent:
            with suppress(Exception):
                holder.lease.release()

    def _bind_new_recurrent_sequences(
        self,
        keys: tuple[_RecurrentSequenceLeaseKey, ...],
    ) -> RecurrentPolicyBinding:
        """Acquire one current-snapshot lease per previously unseen sequence."""
        if any(key in self._recurrent_sequences for key in keys):
            raise RuntimeError(
                "active recurrent sequence cannot be rebound without an artifact"
            )
        leases: list[ModelLease[Any]] = []
        try:
            for _key in keys:
                leases.append(self._pool.acquire())
            versions = {lease.version for lease in leases}
            policies = {id(lease.snapshot) for lease in leases}
            if len(versions) != 1 or len(policies) != 1:
                raise RuntimeError("recurrent first bind crossed model snapshots")
            policy = leases[0].snapshot
            artifact = _policy_artifact_identity(policy)
            binding = RecurrentPolicyBinding(
                policy=policy,
                policy_version=leases[0].version,
                artifact=artifact,
            )
            holders = tuple(
                _RecurrentSequenceLeaseHolder(lease=lease, artifact=artifact)
                for lease in leases
            )
            for key in keys:
                self._recurrent_release_tombstones.pop(key, None)
            self._recurrent_sequences.update(zip(keys, holders, strict=True))
            return binding
        except Exception:
            for lease in leases:
                lease.release()
            raise

    def _bound_recurrent_policy(
        self,
        keys: tuple[_RecurrentSequenceLeaseKey, ...],
        expected_artifact: PolicyArtifactIdentity,
    ) -> RecurrentPolicyBinding:
        """Resolve continuations only when every sequence owns one snapshot."""
        holders = tuple(self._recurrent_sequences.get(key) for key in keys)
        missing = tuple(
            key for key, holder in zip(keys, holders, strict=True) if holder is None
        )
        if missing:
            raise KeyError(
                f"recurrent sequence lease is not active: {missing[0].sequence.game_id}"
            )
        active = cast(tuple[_RecurrentSequenceLeaseHolder, ...], holders)
        if any(holder.artifact != expected_artifact for holder in active):
            raise RuntimeError("recurrent sequence artifact mismatch")
        versions = {holder.lease.version for holder in active}
        policies = {id(holder.lease.snapshot) for holder in active}
        if len(versions) != 1 or len(policies) != 1:
            raise RuntimeError("recurrent request mixed immutable snapshot leases")
        policy = active[0].lease.snapshot
        if _policy_artifact_identity(policy) != expected_artifact:
            raise RuntimeError("resident recurrent policy artifact changed")
        return RecurrentPolicyBinding(
            policy=policy,
            policy_version=active[0].lease.version,
            artifact=expected_artifact,
        )

    def _decode_root_with_lease(
        self,
        states: Any,
        options: Any,
        decks: Any,
        *,
        temperature: float,
    ) -> Any:
        try:
            lease = self._pool.acquire()
        except ModelLeaseCapacityError:
            # The current snapshot itself cannot be evicted by publication.
            # Keep this strong local reference for the complete non-retaining
            # base decode and return an explicit schema-9 fallback signal.
            policy = self._pool.resident_snapshot()
            base_trace = _decode_without_context(
                policy,
                states,
                options,
                decks,
                temperature=temperature,
            )
            return replace(
                _bind_trace_identity(base_trace, policy),
                planner_fallback_reason="model_lease_capacity",
            )
        policy = lease.snapshot
        trace: Any | None = None
        try:
            decoder = getattr(
                policy,
                "sample_decode_with_trace_for_request",
                None,
            )
            if not callable(decoder):
                raise RuntimeError("snapshot has no lease-aware root decode surface")
            trace = decoder(
                states,
                options,
                decks,
                temperature=temperature,
                model_version_lease=None,
                retain_planner_context=True,
            )
            handles = tuple(str(value) for value in trace.planner_context_handles)
            fallback_reason = str(getattr(trace, "planner_fallback_reason", ""))
            if fallback_reason:
                if fallback_reason != "model_lease_capacity" or handles:
                    raise RuntimeError(
                        "root decode returned invalid context-capacity metadata"
                    )
                lease.release()
                return _bind_trace_identity(trace, policy)
            batch_size = int(states.card_ids.shape[0])
            if len(handles) != batch_size or len(set(handles)) != batch_size:
                raise RuntimeError(
                    "root planner decode did not emit one unique context per row"
                )
            holder = _RootLeaseHolder(
                lease=lease,
                remaining_handles=set(handles),
                deadline_monotonic=float("inf"),
            )
            with self._lock:
                if any(handle in self._handles for handle in handles):
                    raise RuntimeError("planner context handle was reused")
                for handle in handles:
                    self._handles[handle] = holder
                version = int(lease.version)
                self._active_by_version[version] = (
                    self._active_by_version.get(version, 0) + 1
                )
            return _bind_trace_identity(trace, policy)
        except Exception:
            if trace is not None:
                handles = tuple(
                    str(value)
                    for value in getattr(trace, "planner_context_handles", ())
                )
                if handles:
                    with suppress(Exception):
                        policy.release_planner_context_handles(handles)
            lease.release()
            raise

    def _policy_for_handles(
        self,
        handles: Sequence[str],
        *,
        version: int,
    ) -> Any:
        frozen = tuple(str(handle) for handle in handles)
        if not frozen or len(set(frozen)) != len(frozen):
            raise ValueError("planner call requires unique leased context handles")
        with self._lock:
            resolved = tuple(self._handles.get(handle) for handle in frozen)
            if any(holder is None for holder in resolved):
                raise KeyError("planner context handle is not leased")
            active_holders = tuple(
                cast(_RootLeaseHolder, holder) for holder in resolved
            )
            holders = {id(holder) for holder in active_holders}
            policies = {id(holder.lease.snapshot) for holder in active_holders}
            versions = {holder.lease.version for holder in active_holders}
        if len(holders) == 0 or len(policies) != 1 or versions != {int(version)}:
            raise RuntimeError("planner contexts mixed immutable model leases")
        return self._pool.resident_snapshot(version)


def _decode_without_context(
    policy: Any,
    states: Any,
    options: Any,
    decks: Any,
    *,
    temperature: float,
) -> Any:
    request_decoder = getattr(
        policy,
        "sample_decode_with_trace_for_request",
        None,
    )
    if callable(request_decoder):
        trace = request_decoder(
            states,
            options,
            decks,
            temperature=temperature,
            model_version_lease=_policy_version(policy),
            retain_planner_context=False,
        )
    else:
        trace = policy.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
        )
    if getattr(trace, "planner_context_handles", ()):
        raise RuntimeError("non-retaining decode unexpectedly emitted contexts")
    return trace


def _bind_trace_identity(trace: Any, policy: Any) -> Any:
    """Attach the identity of the snapshot that actually produced a trace."""
    fields = getattr(trace, "__dataclass_fields__", {})
    required = {
        "served_policy_version",
        "served_model_fingerprint",
        "served_proposal_version",
    }
    if not required.issubset(fields):
        raise RuntimeError("decode trace cannot carry immutable snapshot identity")
    return replace(
        trace,
        served_policy_version=_policy_version(policy),
        served_model_fingerprint=_policy_model_fingerprint(policy),
        served_proposal_version=_policy_proposal_version(policy),
    )


def _recurrent_sequence_lease_keys(
    *,
    actor_id: str,
    actor_incarnation: int,
    policy_id: str,
    sequences: Sequence[RecurrentSequenceIdentity],
) -> tuple[_RecurrentSequenceLeaseKey, ...]:
    """Validate and globally namespace recurrent sequence identities."""
    cleaned_actor = str(actor_id).strip()
    cleaned_policy = str(policy_id).strip()
    if not cleaned_actor or not cleaned_policy:
        raise ValueError("recurrent sequence actor and policy IDs must be non-empty")
    cleaned_incarnation = int(actor_incarnation)
    if cleaned_incarnation < 0:
        raise ValueError("recurrent actor incarnation must be non-negative")
    frozen = tuple(sequences)
    if not frozen:
        raise ValueError("recurrent sequence lease batch must not be empty")
    if any(not isinstance(sequence, RecurrentSequenceIdentity) for sequence in frozen):
        raise TypeError("recurrent sequence lease requires validated identities")
    keys = tuple(
        _RecurrentSequenceLeaseKey(
            actor_id=cleaned_actor,
            actor_incarnation=cleaned_incarnation,
            policy_id=cleaned_policy,
            sequence=sequence,
        )
        for sequence in frozen
    )
    if len(set(keys)) != len(keys):
        raise ValueError("recurrent sequence lease batch contains duplicates")
    return keys


def _policy_artifact_identity(policy: Any) -> PolicyArtifactIdentity:
    """Return and cross-check one snapshot's complete recurrent identity."""
    value = getattr(policy, "policy_artifact_identity", None)
    if callable(value):
        value = value()
    if not isinstance(value, PolicyArtifactIdentity):
        raise ValueError("snapshot policy has no complete artifact identity")
    if not value.compatibility.public_event_schema_fingerprint:
        raise ValueError("snapshot policy artifact is not recurrent")
    if value.model_fingerprint != _policy_model_fingerprint(policy):
        raise ValueError("snapshot policy artifact differs from its model state")
    return value


def _policy_recurrent_enabled(policy: Any) -> bool:
    """Resolve recurrent capability without hiding a published artifact."""
    value = getattr(policy, "recurrent_enabled", None)
    if callable(value):
        value = value()
    if value is not None:
        if not isinstance(value, bool):
            raise TypeError("snapshot policy recurrent_enabled must be boolean")
        return value

    artifact = getattr(policy, "policy_artifact_identity", None)
    if callable(artifact):
        artifact = artifact()
    if artifact is None:
        return False
    if not isinstance(artifact, PolicyArtifactIdentity):
        raise TypeError("snapshot policy artifact identity has an invalid type")
    return bool(artifact.compatibility.public_event_schema_fingerprint)


def _policy_version(policy: Any) -> int:
    value = int(getattr(policy, "policy_version", -1))
    if value < 0:
        raise ValueError("snapshot policy version must be non-negative")
    return value


def _policy_model_fingerprint(policy: Any) -> str:
    value = getattr(policy, "model_fingerprint", None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("snapshot policy has no full model-state fingerprint")
    return value


def _policy_proposal_version(policy: Any) -> int:
    value = int(getattr(policy, "proposal_version", -1))
    if value < 0:
        raise ValueError("snapshot proposal version must be non-negative")
    return value


__all__ = [
    "CandidateSnapshotFactory",
    "InferencePolicySnapshotRouter",
    "RecurrentPolicyBinding",
]
