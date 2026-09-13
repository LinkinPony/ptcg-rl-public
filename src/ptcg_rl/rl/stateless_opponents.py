"""Immutable historical and past-self runtimes for stateless collection."""

from __future__ import annotations

import gc
import queue
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import torch

from ptcg_rl.actions.selection import normalize_action_order
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import ContextBeliefTracker, OpponentBeliefFeatureConfig
from ptcg_rl.engine.prospective_facts import (
    ProspectiveEngineFactConfig,
    ProspectiveEngineFactProducer,
)
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    materialize_simple_stateless_checkpoint_model,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.policy_inputs import SimpleStatelessPublicInputAdapter
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceRolloutPrecision,
)
from ptcg_rl.rl.stateless_actor import (
    SimpleStatelessActorPolicy,
    StatelessActorPolicy,
)
from ptcg_rl.rl.stateless_checkpoint import (
    LoadedStatelessPolicyCheckpoint,
    load_stateless_policy_checkpoint,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember, PreparedPolicyRoutes
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_inference import (
    HistoricalInferenceRequest,
    HistoricalInferenceResponse,
    HistoricalInferenceRow,
    RemoteStatelessActorPolicy,
    StatelessInferenceFailure,
)
from ptcg_rl.rl.stateless_training_config import StatelessHistoricalAnchorConfig


class PastSelfPolicyRuntime(Protocol):
    """Actor lookup contract consumed by the engine collector."""

    def actor(
        self,
        member_id: str,
    ) -> StatelessActorPolicy | GeneralistSequenceActorPolicy:
        """Resolve one exact past-self route."""


@dataclass(frozen=True, slots=True)
class PastSelfArchivedSequenceSource:
    """Verified CPU FP32 master used to build one window-local native actor."""

    model: SimpleStatelessPolicyValueNet
    identity: StatelessFragmentIdentity
    device: torch.device


PastSelfNativeSource: TypeAlias = (
    SimpleStatelessActorPolicy
    | GeneralistSequenceActorPolicy
    | PastSelfArchivedSequenceSource
)


class HistoricalPolicyRuntime(Protocol):
    """Tracker-bound historical action contract consumed by collection."""

    def start_game(
        self,
        member_id: str,
        *,
        game_id: str,
        seat: int,
    ) -> None:
        """Lease one game-local public context tracker."""

    def act_many(
        self,
        requests: Sequence[HistoricalPolicyRequest],
    ) -> tuple[tuple[int, ...], ...]:
        """Return actions aligned with tracker-bound requests."""

    def release_game(self, member_id: str, game_id: str) -> None:
        """Release one game-local tracker."""


@dataclass
class _PastSelfPreparedRoutes:
    """Staged clean policy routes awaiting curriculum publication."""

    owner: PastSelfPolicyPool
    actors: dict[
        str,
        SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
    ]
    archived_sequence_sources: dict[str, PastSelfArchivedSequenceSource]
    committed: bool = False

    def commit(self) -> None:
        """Publish all staged member-to-actor bindings at once."""
        if self.committed:
            raise RuntimeError("past-self routes were already committed")
        staged_ids = set(self.actors) | set(self.archived_sequence_sources)
        resident_ids = set(self.owner._actors) | set(
            self.owner._archived_sequence_sources
        )
        overlap = staged_ids & resident_ids
        if overlap:
            raise ValueError("past-self member route is already loaded")
        self.owner._actors.update(self.actors)
        self.owner._archived_sequence_sources.update(self.archived_sequence_sources)
        self.committed = True

    def abort(self) -> None:
        """Release a staged generation that was not published."""
        if self.committed:
            return
        self.actors.clear()
        self.archived_sequence_sources.clear()
        gc.collect()


class PastSelfPolicyPool:
    """Load a durable policy before exposing any past-self member route."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        fragment_horizon: int,
        non_sequence_rollout_precision: SequenceRolloutPrecision = "fp32",
        archive_sequence_models_on_cpu: bool = False,
    ) -> None:
        """Create an empty bounded pool."""
        if non_sequence_rollout_precision not in {"fp32", "bf16"}:
            raise ValueError("non-sequence rollout precision must be fp32 or bf16")
        self.device = torch.device(device)
        self.fragment_horizon = fragment_horizon
        self.non_sequence_rollout_precision = non_sequence_rollout_precision
        self.archive_sequence_models_on_cpu = bool(archive_sequence_models_on_cpu)
        self._actors: dict[
            str,
            SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
        ] = {}
        self._archived_sequence_sources: dict[
            str,
            PastSelfArchivedSequenceSource,
        ] = {}

    def prepare(
        self,
        members: Sequence[PfspMember],
        *,
        preloaded: Mapping[Path, LoadedStatelessPolicyCheckpoint] | None = None,
    ) -> PreparedPolicyRoutes:
        """Strict-load each unique durable policy and stage declared routes."""
        if not members:
            return _PastSelfPreparedRoutes(self, {}, {})
        by_policy: dict[Path, list[PfspMember]] = {}
        for member in members:
            if member.pair is None:
                # Historical anchors have a separate archive-native runtime.
                continue
            by_policy.setdefault(member.pair.policy_path, []).append(member)
        staged: dict[
            str,
            SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
        ] = {}
        staged_archives: dict[str, PastSelfArchivedSequenceSource] = {}
        for policy_path, grouped in by_policy.items():
            expected = grouped[0].pair
            if expected is None:
                raise RuntimeError("past-self policy artifact disappeared")
            if any(member.pair != expected for member in grouped):
                raise ValueError("past-self members disagree on policy artifact")
            loaded = None if preloaded is None else preloaded.get(policy_path)
            if loaded is None:
                loaded = load_stateless_policy_checkpoint(
                    policy_path,
                    expected_artifact=expected,
                )
            elif loaded.artifact != expected:
                raise ValueError("preloaded past-self artifact differs from member")
            model = materialize_simple_stateless_checkpoint_model(
                loaded.model_config_value,
                loaded.model_state,
            )
            if (
                model.sequence is None
                and self.non_sequence_rollout_precision == "bf16"
            ):
                _materialize_resident_bfloat16_model(
                    model,
                    expected_fingerprint=(
                        loaded.artifact.policy_model_fingerprint
                    ),
                    device=self.device,
                )
            identity = _fragment_identity(
                loaded.artifact.version,
                loaded.artifact.policy_model_fingerprint,
                loaded.identity,
                horizon=self.fragment_horizon,
            )
            actor: SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy
            if model.sequence is None:
                actor = SimpleStatelessActorPolicy(
                    model,
                    identity=identity,
                    device=self.device,
                    # Exact loading already canonical-hashed the strict CPU state.
                    verify_model_state=False,
                )
                archived_source = None
            elif self.archive_sequence_models_on_cpu:
                _freeze_archived_sequence_model(
                    model,
                    expected_fingerprint=(loaded.artifact.policy_model_fingerprint),
                )
                archived_source = PastSelfArchivedSequenceSource(
                    model=model,
                    identity=identity,
                    device=self.device,
                )
            else:
                actor = GeneralistSequenceActorPolicy(
                    model,
                    identity=identity,
                    device=self.device,
                    verify_model_state=False,
                )
                archived_source = None
            for member in grouped:
                if member.exact_deck_digest not in (
                    route.deck_digest
                    for route in loaded.model_config_value.exact_routes
                ):
                    raise ValueError("past-self member is absent from loaded model")
                if archived_source is None:
                    staged[member.member_id] = actor
                else:
                    staged_archives[member.member_id] = archived_source
        return _PastSelfPreparedRoutes(self, staged, staged_archives)

    def actor(
        self,
        member_id: str,
    ) -> SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy:
        """Resolve one atomically published past-self route."""
        if member_id in self._archived_sequence_sources:
            raise RuntimeError("CPU-archived sequence routes require native_source()")
        try:
            return self._actors[member_id]
        except KeyError as error:
            raise KeyError(f"past-self route is not loaded: {member_id}") from error

    def native_source(self, member_id: str) -> PastSelfNativeSource:
        """Resolve an actor or immutable CPU master for native collection."""
        archived = self._archived_sequence_sources.get(member_id)
        if archived is not None:
            return archived
        try:
            return self._actors[member_id]
        except KeyError as error:
            raise KeyError(f"past-self route is not loaded: {member_id}") from error

    @property
    def loaded_member_ids(self) -> frozenset[str]:
        """Return atomically published member routes in this process."""
        return frozenset(set(self._actors) | set(self._archived_sequence_sources))

    def unload(self, member_ids: Sequence[str]) -> None:
        """Drop retired member bindings after all leases drain."""
        for member_id in member_ids:
            self._actors.pop(member_id, None)
            self._archived_sequence_sources.pop(member_id, None)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _freeze_archived_sequence_model(
    model: SimpleStatelessPolicyValueNet,
    *,
    expected_fingerprint: str,
) -> None:
    """Verify and freeze a sequence master without creating CUDA residency."""
    non_cpu = tuple(
        name
        for name, tensor in model.state_dict().items()
        if tensor.device.type != "cpu"
    )
    if non_cpu:
        preview = ", ".join(non_cpu[:4])
        raise ValueError(f"archived sequence state must remain on CPU: {preview}")
    non_fp32 = tuple(
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    )
    if non_fp32:
        preview = ", ".join(non_fp32[:4])
        raise TypeError(
            "archived sequence loading requires an FP32 artifact; "
            f"non-FP32 state: {preview}"
        )
    actual_fingerprint = canonical_model_state_fingerprint(model)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "constructed archived sequence model differs from its verified artifact"
        )
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None


def _materialize_resident_bfloat16_model(
    model: SimpleStatelessPolicyValueNet,
    *,
    expected_fingerprint: str,
    device: torch.device,
) -> None:
    """Verify one immutable FP32 artifact before its sole BF16 conversion."""
    non_fp32 = tuple(
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    )
    if non_fp32:
        preview = ", ".join(non_fp32[:4])
        raise TypeError(
            "resident BF16 past-self loading requires an FP32 artifact; "
            f"non-FP32 state: {preview}"
        )
    actual_fingerprint = canonical_model_state_fingerprint(model)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "constructed past-self model differs from its verified FP32 artifact"
        )
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None


@dataclass(frozen=True)
class _HistoricalResource:
    config: StatelessHistoricalAnchorConfig
    own_deck: tuple[int, ...]


@dataclass
class _HistoricalRuntime:
    policy: CheckpointPolicy
    trackers: dict[str, ContextBeliefTracker]


@dataclass(frozen=True)
class HistoricalPolicyRequest:
    """One tracker-bound historical callback awaiting grouped inference."""

    member_id: str
    game_id: str
    observation: Mapping[str, Any]
    forced_action: tuple[int, ...] | None


class HistoricalPolicyPool:
    """Share each stateless legacy checkpoint while isolating game trackers."""

    def __init__(
        self,
        resources: Mapping[str, tuple[StatelessHistoricalAnchorConfig, Sequence[int]]],
    ) -> None:
        """Register verified member resources without loading model weights yet."""
        self._resources = {
            member_id: _HistoricalResource(
                config=config,
                own_deck=tuple(int(card_id) for card_id in deck),
            )
            for member_id, (config, deck) in resources.items()
        }
        self._runtimes: dict[str, _HistoricalRuntime] = {}

    def start_game(
        self,
        member_id: str,
        *,
        game_id: str,
        seat: int,
    ) -> None:
        """Lease an archive-native context tracker for one game."""
        runtime = self._runtime(member_id)
        if game_id in runtime.trackers:
            raise ValueError("historical game tracker is already leased")
        resource = self._resources[member_id]
        belief = _belief_config(resource.config)
        tracker = ContextBeliefTracker(belief=belief)
        tracker.begin_game(player_index=seat, own_deck=resource.own_deck)
        runtime.trackers[game_id] = tracker

    def act(
        self,
        member_id: str,
        *,
        game_id: str,
        observation: Mapping[str, object],
        forced_action: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Update public context, then use forced or archive-native policy action."""
        return self.act_many(
            (
                HistoricalPolicyRequest(
                    member_id=member_id,
                    game_id=game_id,
                    observation=observation,
                    forced_action=forced_action,
                ),
            )
        )[0]

    def act_many(
        self,
        requests: Sequence[HistoricalPolicyRequest],
    ) -> tuple[tuple[int, ...], ...]:
        """Update every private tracker, then batch non-forced rows per artifact."""
        if not requests:
            return ()
        actions: list[tuple[int, ...] | None] = [None] * len(requests)
        grouped: defaultdict[
            str,
            list[tuple[int, Mapping[str, Any]]],
        ] = defaultdict(list)
        for index, request in enumerate(requests):
            runtime = self._runtime(request.member_id)
            try:
                tracker = runtime.trackers[request.game_id]
            except KeyError as error:
                raise KeyError("historical game tracker is not leased") from error
            adapted = tracker.observation_with_context(request.observation)
            if request.forced_action is not None:
                actions[index] = request.forced_action
            else:
                grouped[request.member_id].append((index, adapted))
        for member_id, rows in grouped.items():
            runtime = self._runtime(member_id)
            decoded = runtime.policy.select_actions(
                tuple(observation for _index, observation in rows)
            )
            for (index, _observation), action in zip(
                rows,
                decoded,
                strict=True,
            ):
                actions[index] = action
        if any(action is None for action in actions):
            raise RuntimeError("historical batch omitted a requested action")
        return tuple(action for action in actions if action is not None)

    def release_game(self, member_id: str, game_id: str) -> None:
        """Release a terminal/cancelled archive-native tracker."""
        runtime = self._runtimes.get(member_id)
        if runtime is not None:
            runtime.trackers.pop(game_id, None)

    def _runtime(self, member_id: str) -> _HistoricalRuntime:
        existing = self._runtimes.get(member_id)
        if existing is not None:
            return existing
        try:
            resource = self._resources[member_id]
        except KeyError as error:
            raise KeyError(f"unknown historical PFSP member: {member_id}") from error
        policy = CheckpointPolicy(
            resource.config.checkpoint_path,
            device=resource.config.device,
            own_deck=resource.own_deck,
        )
        if policy.recurrent_enabled:
            raise ValueError(
                "shared historical stateless pool cannot load recurrent checkpoints"
            )
        policy.configure_inference_cache(enabled=False)
        runtime = _HistoricalRuntime(policy=policy, trackers={})
        self._runtimes[member_id] = runtime
        return runtime


class RemoteHistoricalPolicyPool:
    """Keep public trackers local while centralizing legacy GPU inference."""

    def __init__(
        self,
        resources: Mapping[
            str,
            tuple[StatelessHistoricalAnchorConfig, Sequence[int]],
        ],
        *,
        actor_index: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
    ) -> None:
        """Bind immutable resources without loading worker-local model copies."""
        if timeout_seconds <= 0.0:
            raise ValueError("historical inference timeout must be positive")
        self._resources = {
            member_id: _HistoricalResource(
                config=config,
                own_deck=tuple(int(card_id) for card_id in deck),
            )
            for member_id, (config, deck) in resources.items()
        }
        self._trackers: dict[str, dict[str, ContextBeliefTracker]] = defaultdict(dict)
        self.actor_index = int(actor_index)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self._request_id = 0

    def start_game(
        self,
        member_id: str,
        *,
        game_id: str,
        seat: int,
    ) -> None:
        """Lease an isolated tracker without touching CUDA in this worker."""
        try:
            resource = self._resources[member_id]
        except KeyError as error:
            raise KeyError(f"unknown historical PFSP member: {member_id}") from error
        trackers = self._trackers[member_id]
        if game_id in trackers:
            raise ValueError("historical game tracker is already leased")
        tracker = ContextBeliefTracker(belief=_belief_config(resource.config))
        tracker.begin_game(player_index=seat, own_deck=resource.own_deck)
        trackers[game_id] = tracker

    def act_many(
        self,
        requests: Sequence[HistoricalPolicyRequest],
    ) -> tuple[tuple[int, ...], ...]:
        """Update trackers locally and batch every learned row through one queue."""
        if not requests:
            return ()
        actions: list[tuple[int, ...] | None] = [None] * len(requests)
        remote_rows: list[HistoricalInferenceRow] = []
        remote_indices: list[int] = []
        for index, request in enumerate(requests):
            try:
                tracker = self._trackers[request.member_id][request.game_id]
            except KeyError as error:
                raise KeyError("historical game tracker is not leased") from error
            adapted = tracker.observation_with_context(request.observation)
            if request.forced_action is not None:
                actions[index] = request.forced_action
                continue
            policy_input = build_canonical_policy_input(adapted)
            if policy_input is None:
                raise ValueError("historical observation could not be tensorized")
            remote_indices.append(index)
            remote_rows.append(
                HistoricalInferenceRow(
                    member_id=request.member_id,
                    policy_input=policy_input,
                )
            )
        if remote_rows:
            request_id = self._request_id
            self._request_id += 1
            self.request_queue.put(
                HistoricalInferenceRequest(
                    actor_index=self.actor_index,
                    request_id=request_id,
                    rows=tuple(remote_rows),
                ),
                timeout=self.timeout_seconds,
            )
            try:
                response = self.response_queue.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                raise TimeoutError("central historical inference timed out") from error
            if isinstance(response, StatelessInferenceFailure):
                raise RuntimeError(
                    f"central historical inference failed: {response.error}"
                )
            if not isinstance(response, HistoricalInferenceResponse):
                raise TypeError(
                    "central historical inference returned an invalid response"
                )
            if response.request_id != request_id:
                raise RuntimeError("central historical response crossed requests")
            if response.error is not None:
                raise RuntimeError(
                    f"central historical inference failed: {response.error}"
                )
            if len(response.actions) != len(remote_indices):
                raise RuntimeError("central historical response changed row count")
            for index, action in zip(
                remote_indices,
                response.actions,
                strict=True,
            ):
                actions[index] = normalize_action_order(
                    _observation_select(requests[index].observation),
                    action,
                )
        if any(action is None for action in actions):
            raise RuntimeError("remote historical inference omitted an action")
        return tuple(action for action in actions if action is not None)

    def release_game(self, member_id: str, game_id: str) -> None:
        """Release one terminal/cancelled worker-local tracker."""
        trackers = self._trackers.get(member_id)
        if trackers is not None:
            trackers.pop(game_id, None)


class RemotePastSelfPolicyPool:
    """Resolve past-self routes to the same central multi-policy broker."""

    def __init__(
        self,
        *,
        actor_index: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
        engine_fact_config: ProspectiveEngineFactConfig | None = None,
    ) -> None:
        """Create an empty worker-local route table with no model weights."""
        self.actor_index = int(actor_index)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self.engine_fact_config = engine_fact_config
        self._actors: dict[str, RemoteStatelessActorPolicy] = {}

    def sync(
        self,
        members: Sequence[PfspMember],
        *,
        current_identity: StatelessFragmentIdentity,
    ) -> None:
        """Rebuild lightweight routes from the one-writer curriculum snapshot."""
        routes: dict[str, RemoteStatelessActorPolicy] = {}
        actors: dict[str, RemoteStatelessActorPolicy] = {}
        for member in members:
            if member.source != "past_self":
                continue
            pair = member.pair
            if pair is None:
                raise ValueError("past-self member omitted its policy artifact")
            if (
                pair.input_contract_fingerprint
                != current_identity.input_contract_fingerprint
            ):
                raise ValueError("past-self input contract differs from current policy")
            if member.exact_deck_digest not in pair.active_exact_deck_digests:
                raise ValueError("past-self exact route is absent from its artifact")
            route = past_self_policy_route(member)
            actor = routes.get(route)
            if actor is None:
                sequence_route = (
                    current_identity.schema_version == 2
                    and pair.model_config_fingerprint
                    == current_identity.model_config_fingerprint
                )
                identity = current_identity.model_copy(
                    update={
                        "schema_version": 2 if sequence_route else 1,
                        "behavior_policy_version": pair.version,
                        "behavior_policy_fingerprint": (pair.policy_model_fingerprint),
                        "model_config_fingerprint": (
                            pair.model_config_fingerprint
                        ),
                        "exact_registry_fingerprint": (
                            pair.exact_registry_fingerprint
                        ),
                        "sequence_contract_fingerprint": (
                            current_identity.sequence_contract_fingerprint
                            if sequence_route
                            else None
                        ),
                    }
                )
                actor = RemoteStatelessActorPolicy(
                    actor_index=self.actor_index,
                    identity=identity,
                    request_queue=self.request_queue,
                    response_queue=self.response_queue,
                    timeout_seconds=self.timeout_seconds,
                    policy_route=route,
                    engine_fact_config=(
                        self.engine_fact_config if sequence_route else None
                    ),
                )
                routes[route] = actor
            actors[member.member_id] = actor
        self._actors = actors

    def actor(self, member_id: str) -> StatelessActorPolicy:
        """Resolve one lightweight remote route."""
        try:
            return self._actors[member_id]
        except KeyError as error:
            raise KeyError(f"past-self route is not loaded: {member_id}") from error


def past_self_policy_route(member: PfspMember) -> str:
    """Return the cross-process route shared by one immutable policy artifact."""
    if member.source != "past_self" or member.pair is None:
        raise ValueError("only past-self members have central policy routes")
    return f"past-self:{member.policy_sha256}"


def past_self_input_adapter(
    actor: StatelessActorPolicy | GeneralistSequenceActorPolicy,
    *,
    catalog: object,
    player_index: int,
    own_deck: Sequence[int],
) -> SimpleStatelessPublicInputAdapter:
    """Construct an isolated public adapter from an admitted actor contract."""
    from ptcg_rl.belief.public_catalog import PublicDeckCatalog
    from ptcg_rl.rl.policy_inputs import (
        SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
        simple_stateless_input_contract,
    )

    if not isinstance(catalog, PublicDeckCatalog):
        raise TypeError("past-self adapter requires a public deck catalog")
    identity = actor.identity
    contract = simple_stateless_input_contract(
        public_catalog_fingerprint=identity.public_deck_catalog_fingerprint,
        card_catalog_fingerprint=identity.card_catalog_fingerprint,
        public_context_fingerprint=identity.public_context_fingerprint,
        wrapper_runtime_fingerprint=SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    )
    if contract.fingerprint != identity.input_contract_fingerprint:
        raise ValueError("past-self policy input contract cannot be reconstructed")
    producer = None
    if isinstance(actor, GeneralistSequenceActorPolicy):
        sequence = actor.model.config.sequence
        if sequence is None:
            raise RuntimeError("sequence past-self actor lost its model contract")
        if sequence.engine_facts.enabled:
            producer = ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence.engine_facts.sampler),
                config=sequence.engine_facts,
            )
    elif bool(getattr(actor, "uses_generalist_sequence", False)):
        engine_fact_config = getattr(actor, "engine_fact_config", None)
        if not isinstance(engine_fact_config, ProspectiveEngineFactConfig):
            raise RuntimeError("remote sequence actor lost its engine-fact contract")
        if engine_fact_config.enabled:
            producer = ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=engine_fact_config.sampler),
                config=engine_fact_config,
            )
    return SimpleStatelessPublicInputAdapter(
        catalog,
        contract=contract,
        player_index=player_index,
        own_deck=own_deck,
        engine_fact_producer=producer,
    )


def _fragment_identity(
    version: int,
    model_fingerprint: str,
    identity: object,
    *,
    horizon: int,
) -> StatelessFragmentIdentity:
    from ptcg_rl.rl.stateless_checkpoint import StatelessPolicyIdentity

    if not isinstance(identity, StatelessPolicyIdentity):
        raise TypeError("past-self pair has no stateless policy identity")
    return StatelessFragmentIdentity(
        schema_version=(
            2 if identity.sequence_contract_fingerprint is not None else 1
        ),
        horizon=horizon,
        behavior_policy_version=version,
        behavior_policy_fingerprint=model_fingerprint,
        model_config_fingerprint=identity.model_config_fingerprint,
        action_schema_fingerprint=identity.action_schema_fingerprint,
        public_context_fingerprint=identity.public_context_fingerprint,
        card_catalog_fingerprint=identity.card_catalog_fingerprint,
        public_deck_catalog_fingerprint=identity.public_deck_catalog_fingerprint,
        exact_registry_fingerprint=identity.exact_registry_fingerprint,
        belief_target_semantics_fingerprint=(
            identity.belief_target_semantics_fingerprint
        ),
        input_contract_fingerprint=identity.input_contract_fingerprint,
        resolved_config_fingerprint=identity.resolved_config_fingerprint,
        sequence_contract_fingerprint=identity.sequence_contract_fingerprint,
    )


def _belief_config(
    config: StatelessHistoricalAnchorConfig,
) -> OpponentBeliefFeatureConfig | None:
    if config.belief_summary_path is None:
        return None
    return OpponentBeliefFeatureConfig(
        enabled=True,
        deck_signature_summary_path=config.belief_summary_path,
        deck_signature_summary_sha256=config.belief_summary_sha256,
    )


def _observation_select(observation: Mapping[str, Any]) -> Any:
    select = observation.get("select")
    if select is None:
        raise ValueError("historical observation omitted select")
    return select


__all__ = [
    "HistoricalPolicyRequest",
    "HistoricalPolicyRuntime",
    "HistoricalPolicyPool",
    "PastSelfArchivedSequenceSource",
    "PastSelfNativeSource",
    "PastSelfPolicyRuntime",
    "PastSelfPolicyPool",
    "RemoteHistoricalPolicyPool",
    "RemotePastSelfPolicyPool",
    "past_self_policy_route",
    "past_self_input_adapter",
]
