"""Central batched H200 inference for parallel stateless engine actors."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

import torch

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.policy_inputs import CanonicalPolicyInput
from ptcg_rl.decks import DeckBatch
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactConfig
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.stateless_actor import (
    SimpleStatelessActorPolicy,
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_training_config import StatelessHistoricalAnchorConfig

CURRENT_POLICY_ROUTE = "current"
_MAX_BATCH_WAIT_MULTIPLIER = 4.0


@dataclass(frozen=True)
class StatelessInferenceRequest:
    """One actor-local batch awaiting central policy sampling."""

    actor_index: int
    request_id: int
    rows: tuple[SimpleStatelessActorRow, ...]
    temperature: float
    policy_route: str = CURRENT_POLICY_ROUTE


@dataclass(frozen=True)
class StatelessInferenceResponse:
    """One ordered response or a fail-closed server error."""

    request_id: int
    trace: StatelessActorBatchTrace | None = None
    error: str | None = None
    requires_sequence_finalize: bool = False


@dataclass(frozen=True)
class SequenceInferenceFinalizeRequest:
    """Commit or abort one remote batch after engine acceptance is known."""

    actor_index: int
    request_id: int
    policy_route: str
    committed: tuple[bool, ...]


@dataclass(frozen=True)
class SequenceInferenceFinalizeResponse:
    """Acknowledge that all staged rows reached a terminal transaction state."""

    request_id: int
    error: str | None = None


@dataclass(frozen=True)
class SequenceInferenceReleaseRequest:
    """Release one terminal game-seat cache in the central sequence owner."""

    actor_index: int
    request_id: int
    policy_route: str
    game_id: str
    seat: int


@dataclass(frozen=True)
class SequenceInferenceReleaseResponse:
    """Acknowledge central terminal-cache release."""

    request_id: int
    error: str | None = None


@dataclass(frozen=True)
class HistoricalInferenceRow:
    """One worker-tensorized legacy policy input."""

    member_id: str
    policy_input: CanonicalPolicyInput


@dataclass(frozen=True)
class HistoricalInferenceRequest:
    """One worker-local legacy batch awaiting the central GPU owner."""

    actor_index: int
    request_id: int
    rows: tuple[HistoricalInferenceRow, ...]


@dataclass(frozen=True)
class HistoricalInferenceResponse:
    """Ordered legacy actions or a fail-closed server error."""

    request_id: int
    actions: tuple[tuple[int, ...], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class StatelessInferenceFailure:
    """Generic fatal response understood by every remote policy client."""

    error: str


class CentralHistoricalPolicyExecutor:
    """Own one legacy GPU policy per immutable historical member."""

    def __init__(
        self,
        resources: Mapping[
            str,
            tuple[StatelessHistoricalAnchorConfig, Sequence[int]],
        ],
    ) -> None:
        """Register immutable artifacts while deferring expensive model loads."""
        self._resources = {
            member_id: (config, canonicalize_deck(deck))
            for member_id, (config, deck) in resources.items()
        }
        for member_id, (config, deck) in self._resources.items():
            if member_id != config.member_id:
                raise ValueError("historical resource key differs from member ID")
            if deck.deck_digest != config.exact_deck_digest:
                raise ValueError("historical inference deck fingerprint mismatch")
        self._policies: dict[str, CheckpointPolicy] = {}

    @property
    def loaded_policy_count(self) -> int:
        """Return unique member policies resident in the central process."""
        return len(self._policies)

    def preload(self) -> None:
        """Load every pinned policy once before worker hot-path requests."""
        for member_id in self._resources:
            self._policy(member_id)

    def select_actions(
        self,
        member_id: str,
        inputs: Sequence[CanonicalPolicyInput],
    ) -> tuple[tuple[int, ...], ...]:
        """Run one artifact-homogeneous legacy batch."""
        if not inputs:
            return ()
        policy = self._policy(member_id)
        _config, deck = self._resources[member_id]
        states = collate_state_tokens(
            [policy_input.state for policy_input in inputs],
            device=policy.device,
        )
        options = collate_encoded_options(
            [policy_input.options for policy_input in inputs],
            min_counts=[policy_input.min_count for policy_input in inputs],
            max_counts=[policy_input.max_count for policy_input in inputs],
            device=policy.device,
        )
        decks = DeckBatch.from_decks(
            tuple(deck for _input in inputs),
            device=policy.device,
        )
        return policy.select_preencoded_actions(states, options, decks)

    def _policy(self, member_id: str) -> CheckpointPolicy:
        existing = self._policies.get(member_id)
        if existing is not None:
            return existing
        try:
            config, deck = self._resources[member_id]
        except KeyError as error:
            raise KeyError(f"unknown central historical member: {member_id}") from error
        policy = CheckpointPolicy(
            config.checkpoint_path,
            device=config.device,
            own_deck=deck.card_ids,
        )
        if policy.checkpoint_sha256 != config.checkpoint_sha256:
            raise ValueError("historical checkpoint fingerprint changed")
        if policy.recurrent_enabled:
            raise ValueError("central historical policy must be stateless")
        actual_registry = policy.checkpoint_registry_sha256
        if (
            actual_registry is not None
            and actual_registry != config.exact_registry_fingerprint
        ):
            raise ValueError("historical checkpoint registry changed")
        policy.configure_inference_cache(enabled=False)
        self._policies[member_id] = policy
        return policy


class RemoteStatelessActorPolicy:
    """Actor-process client that keeps engine and public trackers local."""

    def __init__(
        self,
        *,
        actor_index: int,
        identity: StatelessFragmentIdentity,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
        policy_route: str = CURRENT_POLICY_ROUTE,
        engine_fact_config: ProspectiveEngineFactConfig | None = None,
    ) -> None:
        """Bind one response lane and immutable behavior publication."""
        if timeout_seconds <= 0.0:
            raise ValueError("inference timeout must be positive")
        self.actor_index = actor_index
        self.identity = identity
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self.policy_route = policy_route.strip()
        self.engine_fact_config = engine_fact_config
        if not self.policy_route:
            raise ValueError("remote stateless policy route cannot be empty")
        self._request_id = 0
        self._control_request_id = 0
        self._pending_sequence_request_id: int | None = None
        self._pending_sequence_rows: dict[str, int] = {}
        self._pending_sequence_resolutions: list[bool | None] = []

    @property
    def uses_generalist_sequence(self) -> bool:
        """Return whether the remote route requires transactional KV control."""
        return self.identity.schema_version == 2

    def sample(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> StatelessActorBatchTrace:
        """Send compact rows to the central H200 owner and await exact traces."""
        if generator is not None:
            raise ValueError("remote stateless inference owns its RNG stream")
        if not rows:
            raise ValueError("remote stateless inference requires at least one row")
        if self._pending_sequence_request_id is not None:
            raise RuntimeError("remote sequence batch was not finalized")
        request_id = self._request_id
        self._request_id += 1
        self.request_queue.put(
            StatelessInferenceRequest(
                actor_index=self.actor_index,
                request_id=request_id,
                rows=tuple(rows),
                temperature=float(temperature),
                policy_route=self.policy_route,
            ),
            timeout=self.timeout_seconds,
        )
        try:
            response = self.response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError("central stateless inference timed out") from error
        if isinstance(response, StatelessInferenceFailure):
            raise RuntimeError(f"central stateless inference failed: {response.error}")
        if not isinstance(response, StatelessInferenceResponse):
            raise TypeError("central stateless inference returned an invalid response")
        if response.request_id != request_id:
            raise RuntimeError("central stateless inference response crossed requests")
        if response.error is not None:
            raise RuntimeError(f"central stateless inference failed: {response.error}")
        if response.trace is None:
            raise RuntimeError("central stateless inference omitted its trace")
        if (
            response.trace.behavior_policy_version
            != self.identity.behavior_policy_version
            or response.trace.behavior_policy_fingerprint
            != self.identity.behavior_policy_fingerprint
            or response.trace.input_contract_fingerprint
            != self.identity.input_contract_fingerprint
        ):
            raise RuntimeError("central stateless inference identity changed")
        if response.requires_sequence_finalize:
            if not self.uses_generalist_sequence:
                raise RuntimeError("stateless remote route requested sequence finalize")
            request_rows: dict[str, int] = {}
            for index, row in enumerate(rows):
                identity = row.sequence_identity
                if identity is None:
                    raise ValueError("remote sequence row has no request identity")
                if identity.request_id in request_rows:
                    raise ValueError("remote sequence batch repeats a request")
                request_rows[identity.request_id] = index
            self._pending_sequence_request_id = request_id
            self._pending_sequence_rows = request_rows
            self._pending_sequence_resolutions = [None] * len(rows)
        elif self.uses_generalist_sequence:
            raise RuntimeError("sequence remote route omitted finalize ownership")
        return response.trace

    def commit_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Mark one engine-accepted proposal and finalize when the batch closes."""
        self._resolve_sequence_decision(row, trace, committed=True)

    def abort_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
    ) -> None:
        """Mark one rejected proposal and finalize when the batch closes."""
        self._resolve_sequence_decision(row, trace, committed=False)

    def release_game(self, *, game_id: str, seat: int) -> None:
        """Release one terminal cache in the central sequence actor."""
        if not self.uses_generalist_sequence:
            return
        if self._pending_sequence_request_id is not None:
            raise RuntimeError("cannot release a remote sequence with pending rows")
        request_id = self._next_control_request_id()
        self.request_queue.put(
            SequenceInferenceReleaseRequest(
                actor_index=self.actor_index,
                request_id=request_id,
                policy_route=self.policy_route,
                game_id=game_id,
                seat=seat,
            ),
            timeout=self.timeout_seconds,
        )
        response = self._receive_control_response()
        if not isinstance(response, SequenceInferenceReleaseResponse):
            raise TypeError("central sequence release returned an invalid response")
        if response.request_id != request_id:
            raise RuntimeError("central sequence release response crossed requests")
        if response.error is not None:
            raise RuntimeError(f"central sequence release failed: {response.error}")

    def _resolve_sequence_decision(
        self,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
        *,
        committed: bool,
    ) -> None:
        if not self.uses_generalist_sequence:
            raise RuntimeError("stateless remote route cannot finalize sequence rows")
        request_id = self._pending_sequence_request_id
        identity = row.sequence_identity
        if request_id is None or identity is None:
            raise RuntimeError("remote sequence proposal is not pending")
        if trace.sequence_request_id != identity.request_id:
            raise ValueError("remote sequence trace differs from its row")
        try:
            index = self._pending_sequence_rows[identity.request_id]
        except KeyError as error:
            raise KeyError("unknown remote sequence request") from error
        if self._pending_sequence_resolutions[index] is not None:
            raise RuntimeError("remote sequence proposal was already finalized")
        self._pending_sequence_resolutions[index] = committed
        if any(value is None for value in self._pending_sequence_resolutions):
            return
        resolutions = tuple(
            bool(value) for value in self._pending_sequence_resolutions
        )
        self.request_queue.put(
            SequenceInferenceFinalizeRequest(
                actor_index=self.actor_index,
                request_id=request_id,
                policy_route=self.policy_route,
                committed=resolutions,
            ),
            timeout=self.timeout_seconds,
        )
        response = self._receive_control_response()
        if not isinstance(response, SequenceInferenceFinalizeResponse):
            raise TypeError("central sequence finalize returned an invalid response")
        if response.request_id != request_id:
            raise RuntimeError("central sequence finalize response crossed requests")
        if response.error is not None:
            raise RuntimeError(f"central sequence finalize failed: {response.error}")
        self._pending_sequence_request_id = None
        self._pending_sequence_rows = {}
        self._pending_sequence_resolutions = []

    def _next_control_request_id(self) -> int:
        request_id = self._control_request_id
        self._control_request_id += 1
        return request_id

    def _receive_control_response(self) -> object:
        try:
            response = self.response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError("central sequence control timed out") from error
        if isinstance(response, StatelessInferenceFailure):
            raise RuntimeError(f"central stateless inference failed: {response.error}")
        return response


class StatelessInferenceBroker:
    """Merge actor requests into large single-owner H200 forwards."""

    def __init__(
        self,
        *,
        actor: SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
        request_queue: Any,
        response_queues: Mapping[int, Any],
        max_batch_rows: int,
        batch_wait_seconds: float,
        routed_actors: Mapping[
            str,
            SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
        ]
        | None = None,
        historical_executor: CentralHistoricalPolicyExecutor | None = None,
    ) -> None:
        """Create a bounded inference thread for one behavior publication."""
        if max_batch_rows <= 0:
            raise ValueError("inference max batch rows must be positive")
        if batch_wait_seconds < 0.0:
            raise ValueError("inference batch wait cannot be negative")
        self.actors = {CURRENT_POLICY_ROUTE: actor}
        for route, routed_actor in (routed_actors or {}).items():
            normalized = route.strip()
            if not normalized or normalized == CURRENT_POLICY_ROUTE:
                raise ValueError("routed actor has an invalid policy route")
            existing = self.actors.get(normalized)
            if existing is not None and existing is not routed_actor:
                raise ValueError("policy route resolves to multiple actors")
            self.actors[normalized] = routed_actor
        self.historical_executor = historical_executor
        self.request_queue = request_queue
        self.response_queues = dict(response_queues)
        self.max_batch_rows = int(max_batch_rows)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self._closed = False
        self._fatal_error: BaseException | None = None
        self.batches = 0
        self.rows = 0
        self.inference_seconds = 0.0
        self.past_self_batches = 0
        self.past_self_rows = 0
        self.past_self_inference_seconds = 0.0
        self.historical_batches = 0
        self.historical_rows = 0
        self.historical_inference_seconds = 0.0
        self._pending_sequence: dict[
            tuple[int, int, str],
            tuple[
                Any,
                tuple[SimpleStatelessActorRow, ...],
                tuple[StatelessActorDecisionTrace, ...],
            ],
        ] = {}
        self._thread = threading.Thread(
            target=self._serve,
            name="stateless-inference-broker",
            daemon=True,
        )

    @property
    def fatal_error(self) -> BaseException | None:
        """Return a server-side failure observed by the broker thread."""
        return self._fatal_error

    def start(self) -> None:
        """Start serving actor requests."""
        self._thread.start()

    def close(self) -> None:
        """Stop after all already-enqueued requests and join the thread."""
        if self._closed:
            return
        self._closed = True
        self.request_queue.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise TimeoutError("stateless inference broker did not stop")

    def _serve(self) -> None:
        try:
            stop_after_batch = False
            while not stop_after_batch:
                first = self.request_queue.get()
                if first is None:
                    if self._pending_sequence:
                        raise RuntimeError(
                            "central sequence broker closed with pending proposals"
                        )
                    return
                if not isinstance(
                    first,
                    (
                        StatelessInferenceRequest,
                        HistoricalInferenceRequest,
                        SequenceInferenceFinalizeRequest,
                        SequenceInferenceReleaseRequest,
                    ),
                ):
                    raise TypeError("invalid stateless inference request")
                if isinstance(
                    first,
                    (
                        SequenceInferenceFinalizeRequest,
                        SequenceInferenceReleaseRequest,
                    ),
                ):
                    self._serve_batch((first,))
                    continue
                requests: list[
                    StatelessInferenceRequest
                    | HistoricalInferenceRequest
                    | SequenceInferenceFinalizeRequest
                    | SequenceInferenceReleaseRequest
                ] = [first]
                rows = len(first.rows)
                batch_started_at = time.monotonic()
                idle_deadline = batch_started_at + self.batch_wait_seconds
                hard_deadline = batch_started_at + (
                    self.batch_wait_seconds * _MAX_BATCH_WAIT_MULTIPLIER
                )
                while rows < self.max_batch_rows:
                    remaining = min(idle_deadline, hard_deadline) - time.monotonic()
                    if remaining <= 0.0:
                        break
                    try:
                        item = self.request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        stop_after_batch = True
                        break
                    if not isinstance(
                        item,
                        (
                            StatelessInferenceRequest,
                            HistoricalInferenceRequest,
                            SequenceInferenceFinalizeRequest,
                            SequenceInferenceReleaseRequest,
                        ),
                    ):
                        raise TypeError("invalid stateless inference request")
                    requests.append(item)
                    if isinstance(
                        item,
                        (StatelessInferenceRequest, HistoricalInferenceRequest),
                    ):
                        rows += len(item.rows)
                    idle_deadline = time.monotonic() + self.batch_wait_seconds
                self._serve_batch(requests)
        except BaseException as error:
            self._fatal_error = error
            message = f"{type(error).__name__}: {error}"
            for response_queue in self.response_queues.values():
                with suppress(queue.Full):
                    response_queue.put_nowait(StatelessInferenceFailure(error=message))

    def _serve_batch(
        self,
        requests: Sequence[
            StatelessInferenceRequest
            | HistoricalInferenceRequest
            | SequenceInferenceFinalizeRequest
            | SequenceInferenceReleaseRequest
        ],
    ) -> None:
        actor_requests: defaultdict[
            str,
            list[StatelessInferenceRequest],
        ] = defaultdict(list)
        historical_requests: list[HistoricalInferenceRequest] = []
        finalize_requests: list[SequenceInferenceFinalizeRequest] = []
        release_requests: list[SequenceInferenceReleaseRequest] = []
        for request in requests:
            if isinstance(request, StatelessInferenceRequest):
                actor_requests[request.policy_route].append(request)
            elif isinstance(request, HistoricalInferenceRequest):
                historical_requests.append(request)
            elif isinstance(request, SequenceInferenceFinalizeRequest):
                finalize_requests.append(request)
            else:
                release_requests.append(request)
        for policy_route, routed_requests in actor_requests.items():
            self._serve_actor_batch(policy_route, routed_requests)
        if historical_requests:
            self._serve_historical_batch(historical_requests)
        for request in finalize_requests:
            self._serve_sequence_finalize(request)
        for request in release_requests:
            self._serve_sequence_release(request)

    def _serve_actor_batch(
        self,
        policy_route: str,
        requests: Sequence[StatelessInferenceRequest],
    ) -> None:
        if any(request.temperature != requests[0].temperature for request in requests):
            raise ValueError("central batch mixes behavior temperatures")
        try:
            actor = self.actors[policy_route]
        except KeyError as error:
            raise KeyError(f"unknown stateless policy route: {policy_route}") from error
        offsets = [0]
        merged_rows: list[SimpleStatelessActorRow] = []
        for request in requests:
            merged_rows.extend(request.rows)
            offsets.append(len(merged_rows))
        started_at = time.perf_counter()
        trace = actor.sample(
            tuple(merged_rows),
            temperature=requests[0].temperature,
        )
        elapsed = time.perf_counter() - started_at
        if policy_route == CURRENT_POLICY_ROUTE:
            self.inference_seconds += elapsed
            self.batches += 1
            self.rows += len(merged_rows)
        else:
            self.past_self_inference_seconds += elapsed
            self.past_self_batches += 1
            self.past_self_rows += len(merged_rows)
        sequence_actor = (
            actor
            if bool(getattr(actor, "uses_generalist_sequence", False))
            else None
        )
        for index, request in enumerate(requests):
            start = offsets[index]
            end = offsets[index + 1]
            if sequence_actor is not None:
                key = (
                    request.actor_index,
                    request.request_id,
                    request.policy_route,
                )
                if key in self._pending_sequence:
                    raise RuntimeError("central sequence request is already pending")
                self._pending_sequence[key] = (
                    sequence_actor,
                    tuple(merged_rows[start:end]),
                    tuple(trace.decisions[start:end]),
                )
            response_queue = self.response_queues[request.actor_index]
            response_queue.put(
                StatelessInferenceResponse(
                    request_id=request.request_id,
                    trace=StatelessActorBatchTrace(
                        behavior_policy_version=(trace.behavior_policy_version),
                        behavior_policy_fingerprint=(trace.behavior_policy_fingerprint),
                        input_contract_fingerprint=(trace.input_contract_fingerprint),
                        decisions=trace.decisions[start:end],
                    ),
                    requires_sequence_finalize=sequence_actor is not None,
                )
            )

    def _serve_sequence_finalize(
        self,
        request: SequenceInferenceFinalizeRequest,
    ) -> None:
        key = (request.actor_index, request.request_id, request.policy_route)
        try:
            actor, rows, decisions = self._pending_sequence.pop(key)
        except KeyError as error:
            raise KeyError("unknown central sequence proposal batch") from error
        if len(request.committed) != len(rows):
            raise ValueError("central sequence finalize mask is misaligned")
        for row, decision, committed in zip(
            rows,
            decisions,
            request.committed,
            strict=True,
        ):
            if committed:
                actor.commit_decision(row, decision)
            else:
                actor.abort_decision(row, decision)
        self.response_queues[request.actor_index].put(
            SequenceInferenceFinalizeResponse(request_id=request.request_id)
        )

    def _serve_sequence_release(
        self,
        request: SequenceInferenceReleaseRequest,
    ) -> None:
        try:
            actor = self.actors[request.policy_route]
        except KeyError as error:
            raise KeyError("unknown central sequence release route") from error
        if not bool(getattr(actor, "uses_generalist_sequence", False)):
            raise TypeError("central sequence release targeted a stateless actor")
        cast(Any, actor).release_game(game_id=request.game_id, seat=request.seat)
        self.response_queues[request.actor_index].put(
            SequenceInferenceReleaseResponse(request_id=request.request_id)
        )

    def _serve_historical_batch(
        self,
        requests: Sequence[HistoricalInferenceRequest],
    ) -> None:
        executor = self.historical_executor
        if executor is None:
            raise RuntimeError("central historical inference is not configured")
        merged = tuple(row for request in requests for row in request.rows)
        grouped: defaultdict[
            str,
            list[tuple[int, CanonicalPolicyInput]],
        ] = defaultdict(list)
        for index, row in enumerate(merged):
            grouped[row.member_id].append((index, row.policy_input))
        actions: list[tuple[int, ...] | None] = [None] * len(merged)
        started_at = time.perf_counter()
        for member_id, rows in grouped.items():
            decoded = executor.select_actions(
                member_id,
                tuple(policy_input for _index, policy_input in rows),
            )
            for (index, _policy_input), action in zip(rows, decoded, strict=True):
                actions[index] = action
        self.historical_inference_seconds += time.perf_counter() - started_at
        self.historical_batches += len(grouped)
        self.historical_rows += len(merged)
        if any(action is None for action in actions):
            raise RuntimeError("central historical inference omitted an action")
        offset = 0
        for request in requests:
            stop = offset + len(request.rows)
            response_queue = self.response_queues[request.actor_index]
            response_queue.put(
                HistoricalInferenceResponse(
                    request_id=request.request_id,
                    actions=tuple(
                        action for action in actions[offset:stop] if action is not None
                    ),
                )
            )
            offset = stop


__all__ = [
    "CURRENT_POLICY_ROUTE",
    "CentralHistoricalPolicyExecutor",
    "HistoricalInferenceRequest",
    "HistoricalInferenceResponse",
    "HistoricalInferenceRow",
    "RemoteStatelessActorPolicy",
    "SequenceInferenceFinalizeRequest",
    "SequenceInferenceFinalizeResponse",
    "SequenceInferenceReleaseRequest",
    "SequenceInferenceReleaseResponse",
    "StatelessInferenceBroker",
    "StatelessInferenceFailure",
    "StatelessInferenceRequest",
    "StatelessInferenceResponse",
]
