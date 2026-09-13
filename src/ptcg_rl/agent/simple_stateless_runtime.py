"""CPU/GPU deployment runtime for a fixed-deck clean stateless policy."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import orjson
import torch

from ptcg_rl.actions.selection import (
    ENGINE_PROVEN_UNORDERED_SET_CONTEXTS,
    forced_action,
    is_legal_action,
)
from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import PublicEventDecisionToken
from ptcg_rl.context.public_event_arrays import collate_public_event_deltas
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.sequence.action import (
    build_accepted_action_record,
    collate_accepted_actions,
)
from ptcg_rl.model.sequence.core import TemporalKvCache
from ptcg_rl.model.sequence.network import TemporalPreparedDecision
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessBackboneOutput,
    SimpleStatelessPolicyValueNet,
    materialize_simple_stateless_checkpoint_model,
)
from ptcg_rl.model.simple_stateless.rollout_cache import (
    mark_immutable_rollout_cache_generation,
)
from ptcg_rl.model.simple_stateless.routing import (
    SimpleExactRoutePlan,
    resolve_simple_exact_routes,
)
from ptcg_rl.rl.policy_inputs import (
    SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    SimpleStatelessActorRow,
    SimpleStatelessPublicInputAdapter,
    collate_simple_stateless_actor_rows,
    simple_stateless_input_contract,
)
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_checkpoint import load_stateless_policy_checkpoint
from ptcg_rl.rl.stateless_export import load_fixed_deck_checkpoint


@dataclass(frozen=True, slots=True)
class PreparedSimpleStatelessDecision:
    """One game-local decision frozen until a batched action is accepted."""

    policy: FixedDeckStatelessPolicy
    observation: Any
    select: Any
    request_fingerprint: str
    row: SimpleStatelessActorRow
    event_token: PublicEventDecisionToken


@dataclass(frozen=True, slots=True)
class SimpleStatelessBatchResult:
    """One action plus operational evidence from its shared model call."""

    action: tuple[int, ...]
    batch_size: int
    queue_seconds: float
    service_seconds: float


class SimpleStatelessInferenceBatcher(Protocol):
    """Scheduling boundary implemented by the evaluation CUDA owner."""

    def submit(
        self,
        request: PreparedSimpleStatelessDecision,
    ) -> SimpleStatelessBatchResult:
        """Return one action after model-coherent, route-aware inference."""


class FixedDeckStatelessPolicy:
    """Temperature-controlled deployment with a public-only game tracker."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        public_catalog_manifest_path: Path,
        device: str = "cpu",
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Load and cross-check immutable deployment assets."""
        self._device = _resolve_device(device)
        model, payload = load_fixed_deck_checkpoint(checkpoint_path)
        catalog, catalog_manifest = load_public_deck_catalog(
            public_catalog_manifest_path
        )
        expected_catalog = str(payload["public_deck_catalog_fingerprint"])
        if (
            catalog.fingerprint != expected_catalog
            or catalog_manifest.catalog_fingerprint != expected_catalog
        ):
            raise ValueError("deployment checkpoint and public catalog differ")
        assets = _asset_fingerprints(payload)
        self._contract = simple_stateless_input_contract(
            public_catalog_fingerprint=catalog.fingerprint,
            card_catalog_fingerprint=_asset(assets, "card_catalog_static"),
            public_context_fingerprint=_asset(
                assets,
                "public_context_schema",
            ),
            wrapper_runtime_fingerprint=(SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT),
        )
        if self._contract.fingerprint != _asset(assets, "input_contract"):
            raise ValueError("deployment input contract fingerprint changed")
        if self._contract.action_schema_fingerprint != _asset(
            assets,
            "action_schema",
        ):
            raise ValueError("deployment action schema fingerprint changed")

        self._checkpoint_path = checkpoint_path.resolve()
        self._catalog_manifest_path = public_catalog_manifest_path.resolve()
        self._catalog = catalog
        self._payload = payload
        self._model = model.float() if self._device.type == "cpu" else model
        self._model = self._model.to(self._device).eval()
        self._autocast_dtype = torch.float16
        self._policy_temperature = 0.0
        self._sampling_seed = 0
        self._sampling_generator = torch.Generator(device=self._device).manual_seed(0)
        self._route = self._model.config.exact_routes[0]
        self._own_deck = canonicalize_deck(self._route.canonical_card_ids)
        self._adapter: SimpleStatelessPublicInputAdapter | None = None
        # Keep the callback alive: Python may recycle a bare ``id`` between turns.
        self._prepared_observation: Any | None = None
        self._prepared_request_fingerprint: str | None = None
        self._prepared_row: SimpleStatelessActorRow | None = None
        self._served_observation: Any | None = None
        self._served_request_fingerprint: str | None = None
        self._served_action: tuple[int, ...] | None = None
        self._sequence_cache: TemporalKvCache | None = None
        self._sequence_block_index = 0
        self._episode_id = uuid.uuid4().hex
        self._inference_batcher: SimpleStatelessInferenceBatcher | None = None
        self._last_inference_batch_size = 0
        self._last_inference_queue_seconds = 0.0
        self._last_inference_service_seconds = 0.0
        sequence_config = self._model.config.sequence
        self._engine_fact_producer = (
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence_config.engine_facts.sampler),
                config=sequence_config.engine_facts,
            )
            if sequence_config is not None and sequence_config.engine_facts.enabled
            else None
        )
        if own_deck is not None:
            self.bind_own_deck(own_deck)

    @property
    def simple_stateless_fixed_deck(self) -> bool:
        """Identify this deployment surface without relying on class names."""
        return True

    @property
    def simple_stateless_runtime(self) -> bool:
        """Request the public-only runtime path without belief/search augmentation."""
        return True

    @property
    def fixed_deck_digest(self) -> str:
        """Return the only exact route retained by the exporter."""
        return self._route.deck_digest

    @property
    def public_deck_catalog_fingerprint(self) -> str:
        """Return the verified immutable public catalog identity."""
        return self._catalog.fingerprint

    @property
    def own_deck_signature(self) -> str:
        """Expose the bound exact deck for runtime validation."""
        return self._own_deck.signature

    def bind_own_deck(self, own_deck: Sequence[int]) -> None:
        """Reject a packaged deck that differs from the fixed route."""
        canonical = canonicalize_deck(own_deck)
        if canonical.deck_digest != self._route.deck_digest:
            raise ValueError("packaged deck differs from fixed deployment route")
        self._own_deck = canonical
        self.reset_runtime_episode()

    def configure_deployment_sampling(self, *, temperature: float, seed: int) -> None:
        """Bind a reproducible local RNG stream for positive-temperature decode."""
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("deployment policy temperature must be non-negative")
        self._policy_temperature = float(temperature)
        self._sampling_seed = int(seed)
        self._sampling_generator = torch.Generator(device=self._device).manual_seed(
            self._sampling_seed
        )

    def reset_runtime_episode(self) -> None:
        """Discard deterministic public history at a game boundary."""
        self._adapter = None
        self._prepared_observation = None
        self._prepared_request_fingerprint = None
        self._prepared_row = None
        self._served_observation = None
        self._served_request_fingerprint = None
        self._served_action = None
        self._sequence_cache = None
        self._sequence_block_index = 0
        self._episode_id = uuid.uuid4().hex
        self._last_inference_batch_size = 0
        self._last_inference_queue_seconds = 0.0
        self._last_inference_service_seconds = 0.0
        model = getattr(self, "_model", None)
        sequence_config = getattr(getattr(model, "config", None), "sequence", None)
        self._engine_fact_producer = (
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence_config.engine_facts.sampler),
                config=sequence_config.engine_facts,
            )
            if sequence_config is not None and sequence_config.engine_facts.enabled
            else None
        )

    def observe_public_observation(self, observation: Any) -> None:
        """Advance public history on every callback, including forced prompts."""
        self._prepare_row(observation)

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Decode one complete engine-legal option sequence."""
        request_fingerprint = _request_fingerprint(observation)
        if (
            observation is self._served_observation
            or request_fingerprint == self._served_request_fingerprint
        ) and self._served_action is not None:
            return self._served_action
        select = _field(observation, "select")
        action = forced_action(select)
        if action is not None:
            self._prepare_row(observation)
            self._last_inference_batch_size = 0
            self._last_inference_queue_seconds = 0.0
            self._last_inference_service_seconds = 0.0
            self._served_observation = observation
            self._served_request_fingerprint = request_fingerprint
            self._served_action = action
            return action
        request = self.prepare_batched_decision(
            observation,
            request_fingerprint=request_fingerprint,
        )
        batcher: SimpleStatelessInferenceBatcher | None = getattr(
            self,
            "_inference_batcher",
            None,
        )
        result: SimpleStatelessBatchResult
        if batcher is not None:
            result = batcher.submit(request)
        else:
            started = time.perf_counter()
            try:
                (action,) = self.execute_batched_decisions((request,))
            except BaseException:
                with suppress(BaseException):
                    self.abort_batched_decision(request)
                raise
            result = SimpleStatelessBatchResult(
                action=action,
                batch_size=1,
                queue_seconds=0.0,
                service_seconds=time.perf_counter() - started,
            )
        self._last_inference_batch_size = result.batch_size
        self._last_inference_queue_seconds = result.queue_seconds
        self._last_inference_service_seconds = result.service_seconds
        return result.action

    @property
    def inference_batch_key(self) -> tuple[int, str, float]:
        """Identify rows sharing both one model and one decode temperature."""
        return (id(self._model), str(self._device), self._policy_temperature)

    @property
    def inference_model_batch_key(self) -> tuple[int, str]:
        """Identify rows that may share one expensive model encoding."""
        return (id(self._model), str(self._device))

    def bind_inference_batcher(
        self,
        batcher: SimpleStatelessInferenceBatcher | None,
    ) -> None:
        """Attach an operational batch scheduler without changing policy state."""
        self._inference_batcher = batcher

    def last_inference_batch_telemetry(self) -> dict[str, float | int]:
        """Return scheduling evidence for the most recent model decision."""
        return {
            "policy_batch_size": self._last_inference_batch_size,
            "policy_batch_queue_seconds": self._last_inference_queue_seconds,
            "policy_batch_service_seconds": self._last_inference_service_seconds,
        }

    def prepare_batched_decision(
        self,
        observation: Any,
        *,
        request_fingerprint: str | None = None,
    ) -> PreparedSimpleStatelessDecision:
        """Freeze one non-forced callback for shared greedy inference."""
        fingerprint = request_fingerprint or _request_fingerprint(observation)
        row = self._prepare_row(observation)
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError("deployment adapter is not initialized")
        event_token = adapter.prepare_decision()
        select = _field(observation, "select")
        if forced_action(select) is not None:
            adapter.abort_decision(event_token)
            raise ValueError("forced actions cannot enter policy inference batching")
        if self._model.sequence is not None:
            player_index = int(_field(_field(observation, "current"), "yourIndex", -1))
            row = replace(
                row,
                sequence_identity=SequenceDecisionIdentity(
                    game_id=self._episode_id,
                    seat=player_index,  # type: ignore[arg-type]
                    decision_index=self._sequence_block_index,
                    request_id=(
                        f"{self._episode_id}:{player_index}:"
                        f"{self._sequence_block_index}"
                    ),
                ),
            )
        return PreparedSimpleStatelessDecision(
            policy=self,
            observation=observation,
            select=select,
            request_fingerprint=fingerprint,
            row=row,
            event_token=event_token,
        )

    @staticmethod
    def execute_batched_decisions(
        requests: Sequence[PreparedSimpleStatelessDecision],
    ) -> tuple[tuple[int, ...], ...]:
        """Run one model-coherent batch and commit each decision."""
        if not requests:
            raise ValueError("batched policy inference requires decisions")
        policy = requests[0].policy
        model = policy._model
        device = policy._device
        if any(
            request.policy._model is not model
            or request.policy._device != device
            or request.policy._autocast_dtype != policy._autocast_dtype
            for request in requests
        ):
            raise ValueError("batched decisions cross resident policy models")
        rows = tuple(request.row for request in requests)
        batch = collate_simple_stateless_actor_rows(
            rows,
            device=device,
        )
        routes = resolve_simple_exact_routes(
            batch.deck_signatures,
            model.config,
            device=device,
        )
        use_autocast = device.type == "cuda"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=policy._autocast_dtype,
                enabled=use_autocast,
            ),
        ):
            state = model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            prepared: tuple[TemporalPreparedDecision, ...] = ()
            if model.sequence is not None:
                event_batch = collate_public_event_deltas(
                    tuple(row.public_event_delta for row in rows),
                    device=device,
                )
                if len(requests) == 1:
                    prepared = (
                        model.prepare_sequence_incremental(
                            state,
                            event_batch,
                            block_index=policy._sequence_block_index,
                            cache=policy._sequence_cache,
                        ),
                    )
                else:
                    prepared = model.prepare_sequence_incremental_many(
                        state,
                        event_batch,
                        block_indices=tuple(
                            request.policy._sequence_block_index for request in requests
                        ),
                        caches=tuple(
                            request.policy._sequence_cache for request in requests
                        ),
                    )
                state = model.condition_sequence(
                    state,
                    torch.cat(tuple(item.context for item in prepared), dim=0),
                )
            options = model.encode_legal_options(
                state,
                batch.options,
                route_plan=routes,
            )
            actions = _decode_temperature_groups(
                model=model,
                state=state,
                option_embeddings=options,
                option_batch=batch.options,
                deck_signatures=batch.deck_signatures,
                route_plan=routes,
                requests=requests,
                device=device,
            )
            if any(
                not is_legal_action(request.select, action)
                for request, action in zip(requests, actions, strict=True)
            ):
                raise RuntimeError("deployment policy emitted an illegal action")
            committed_caches: tuple[TemporalKvCache | None, ...]
            if prepared:
                accepted = tuple(
                    build_accepted_action_record(
                        state=row.state,
                        options=row.options,
                        action=action,
                        min_count=row.min_count,
                        max_count=row.max_count,
                        stop_sampled=_greedy_stop_sampled(row, action),
                    )
                    for row, action in zip(rows, actions, strict=True)
                )
                accepted_batch = collate_accepted_actions(
                    accepted,
                    device=device,
                )
                if len(requests) == 1:
                    committed_caches = (
                        model.commit_sequence_incremental(
                            prepared[0],
                            accepted_batch,
                        ),
                    )
                else:
                    committed_caches = model.commit_sequence_incremental_many(
                        prepared,
                        accepted_batch,
                    )
            else:
                committed_caches = (None,) * len(requests)
        for request, action, committed_cache in zip(
            requests,
            actions,
            committed_caches,
            strict=True,
        ):
            request.policy._commit_batched_decision(
                request,
                action=action,
                committed_cache=committed_cache,
            )
        return actions

    def _commit_batched_decision(
        self,
        request: PreparedSimpleStatelessDecision,
        *,
        action: tuple[int, ...],
        committed_cache: TemporalKvCache | None,
    ) -> None:
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError("deployment adapter disappeared during inference")
        adapter.commit_decision(request.event_token)
        if self._model.sequence is not None:
            if committed_cache is None:
                raise RuntimeError("temporal inference returned no committed cache")
            self._sequence_cache = committed_cache
            self._sequence_block_index += 1
        self._served_observation = request.observation
        self._served_request_fingerprint = request.request_fingerprint
        self._served_action = action

    def abort_batched_decision(
        self,
        request: PreparedSimpleStatelessDecision,
    ) -> None:
        """Retain public events after a failed or rejected batch proposal."""
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError("deployment adapter disappeared during inference")
        adapter.abort_decision(request.event_token)

    def prewarm(self) -> None:
        """Weights and immutable assets are already loaded by construction."""

    def close(self) -> None:
        """Release per-game public state without mutating immutable assets."""
        self.reset_runtime_episode()

    def _prepare_row(self, observation: Any) -> SimpleStatelessActorRow:
        request_fingerprint = _request_fingerprint(observation)
        if (
            observation is self._prepared_observation
            or request_fingerprint == self._prepared_request_fingerprint
        ) and self._prepared_row is not None:
            return self._prepared_row
        if self._adapter is None:
            player_index = int(_field(_field(observation, "current"), "yourIndex", -1))
            if player_index not in (0, 1):
                raise ValueError("deployment observation has an invalid player index")
            self._adapter = SimpleStatelessPublicInputAdapter(
                self._catalog,
                contract=self._contract,
                player_index=player_index,
                own_deck=self._own_deck.card_ids,
                engine_fact_producer=self._engine_fact_producer,
            )
        row = self._adapter.prepare(_mapping(observation))
        self._prepared_observation = observation
        self._prepared_request_fingerprint = request_fingerprint
        self._prepared_row = row
        return row


class RoutedDeckStatelessPolicy(FixedDeckStatelessPolicy):
    """Temperature-controlled diagnostic view of a routed learner checkpoint.

    Unlike a release bundle, this surface deliberately keeps the full routed
    training topology. It is therefore diagnostic-only and requires an exact
    deck binding before every game.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        public_catalog_manifest_path: Path,
        device: str = "cpu",
        own_deck: Sequence[int] | None = None,
        resident_precision: Literal["source", "bfloat16"] = "source",
        rollout_inductor: bool = False,
    ) -> None:
        self._device = _resolve_device(device)
        if resident_precision == "bfloat16" and self._device.type != "cuda":
            raise ValueError("bfloat16 resident checkpoint inference requires CUDA")
        if rollout_inductor and resident_precision != "bfloat16":
            raise ValueError(
                "checkpoint rollout Inductor requires bfloat16 resident weights"
            )
        loaded = load_stateless_policy_checkpoint(checkpoint_path)
        catalog, catalog_manifest = load_public_deck_catalog(
            public_catalog_manifest_path
        )
        identity = loaded.identity
        if (
            catalog.fingerprint != identity.public_deck_catalog_fingerprint
            or catalog_manifest.catalog_fingerprint
            != identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("routed checkpoint and public catalog differ")
        self._contract = simple_stateless_input_contract(
            public_catalog_fingerprint=catalog.fingerprint,
            card_catalog_fingerprint=identity.card_catalog_fingerprint,
            public_context_fingerprint=identity.public_context_fingerprint,
            wrapper_runtime_fingerprint=(SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT),
        )
        if self._contract.fingerprint != identity.input_contract_fingerprint:
            raise ValueError("routed checkpoint input contract changed")
        if (
            self._contract.action_schema_fingerprint
            != identity.action_schema_fingerprint
        ):
            raise ValueError("routed checkpoint action schema changed")
        model = materialize_simple_stateless_checkpoint_model(
            loaded.model_config_value,
            loaded.model_state,
        )
        self._checkpoint_path = checkpoint_path.resolve()
        self._catalog_manifest_path = public_catalog_manifest_path.resolve()
        self._catalog = catalog
        self._payload = identity.model_dump(mode="json")
        if resident_precision == "bfloat16":
            model = model.to(device=self._device, dtype=torch.bfloat16).eval()
            model.requires_grad_(False)
            for parameter in model.parameters():
                parameter.grad = None
            mark_immutable_rollout_cache_generation(model)
            if rollout_inductor:
                if not model.supports_bfloat16_rollout_inductor:
                    raise RuntimeError(
                        "checkpoint rollout Inductor requires CUDA varlen_attn"
                    )
                model.enable_bfloat16_rollout_inductor()
        else:
            model = model.float() if self._device.type == "cpu" else model
            model = model.to(self._device).eval()
        self._model = model
        self._autocast_dtype = torch.bfloat16
        self._policy_temperature = 0.0
        self._sampling_seed = 0
        self._sampling_generator = torch.Generator(device=self._device).manual_seed(0)
        self._own_deck = canonicalize_deck(
            self._model.config.exact_routes[0].canonical_card_ids
        )
        self._adapter = None
        self._prepared_observation = None
        self._prepared_request_fingerprint = None
        self._prepared_row = None
        self._served_observation = None
        self._served_request_fingerprint = None
        self._served_action = None
        self._sequence_cache = None
        self._sequence_block_index = 0
        self._episode_id = uuid.uuid4().hex
        self._inference_batcher = None
        self._last_inference_batch_size = 0
        self._last_inference_queue_seconds = 0.0
        self._last_inference_service_seconds = 0.0
        sequence_config = self._model.config.sequence
        self._engine_fact_producer = (
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence_config.engine_facts.sampler),
                config=sequence_config.engine_facts,
            )
            if sequence_config is not None and sequence_config.engine_facts.enabled
            else None
        )
        if own_deck is not None:
            self.bind_own_deck(own_deck)

    @property
    def simple_stateless_fixed_deck(self) -> bool:
        """Identify this as a full routed checkpoint, not a deployment export."""
        return False

    @property
    def fixed_deck_digest(self) -> str:
        """Expose the currently bound exact route for diagnostic telemetry."""
        return self._own_deck.deck_digest

    def bind_own_deck(self, own_deck: Sequence[int]) -> None:
        """Bind one exact route and reject generic or absent deck identities."""
        canonical = canonicalize_deck(own_deck)
        route_digests = {route.deck_digest for route in self._model.config.exact_routes}
        if canonical.deck_digest not in route_digests:
            raise ValueError("diagnostic deck is absent from routed checkpoint")
        self._own_deck = canonical
        self.reset_runtime_episode()

    def fork(self) -> RoutedDeckStatelessPolicy:
        """Share immutable model weights while isolating game-local public state."""
        clone = object.__new__(RoutedDeckStatelessPolicy)
        clone._device = self._device
        clone._contract = self._contract
        clone._checkpoint_path = self._checkpoint_path
        clone._catalog_manifest_path = self._catalog_manifest_path
        clone._catalog = self._catalog
        clone._payload = self._payload
        clone._model = self._model
        clone._autocast_dtype = self._autocast_dtype
        clone._policy_temperature = getattr(self, "_policy_temperature", 0.0)
        clone._sampling_seed = getattr(self, "_sampling_seed", 0)
        clone._sampling_generator = torch.Generator(device=self._device).manual_seed(
            clone._sampling_seed
        )
        clone._own_deck = self._own_deck
        clone._adapter = None
        clone._prepared_observation = None
        clone._prepared_request_fingerprint = None
        clone._prepared_row = None
        clone._served_observation = None
        clone._served_request_fingerprint = None
        clone._served_action = None
        clone._sequence_cache = None
        clone._sequence_block_index = 0
        clone._episode_id = uuid.uuid4().hex
        clone._inference_batcher = getattr(self, "_inference_batcher", None)
        clone._last_inference_batch_size = 0
        clone._last_inference_queue_seconds = 0.0
        clone._last_inference_service_seconds = 0.0
        sequence_config = getattr(
            getattr(clone._model, "config", None),
            "sequence",
            None,
        )
        clone._engine_fact_producer = (
            ProspectiveEngineFactProducer(
                sampler=BeliefSampler(config=sequence_config.engine_facts.sampler),
                config=sequence_config.engine_facts,
            )
            if sequence_config is not None and sequence_config.engine_facts.enabled
            else None
        )
        return clone


_ROUTED_PROTOTYPES: dict[tuple[str, str, str], RoutedDeckStatelessPolicy] = {}


def routed_stateless_policy(
    checkpoint_path: Path,
    *,
    public_catalog_manifest_path: Path,
    device: str = "cpu",
) -> RoutedDeckStatelessPolicy:
    """Return one state-isolated wrapper over cached routed model weights."""
    key = (
        str(checkpoint_path.resolve()),
        str(public_catalog_manifest_path.resolve()),
        device.strip().lower(),
    )
    prototype = _ROUTED_PROTOTYPES.get(key)
    if prototype is None:
        prototype = RoutedDeckStatelessPolicy(
            checkpoint_path,
            public_catalog_manifest_path=public_catalog_manifest_path,
            device=device,
        )
        _ROUTED_PROTOTYPES[key] = prototype
    return prototype.fork()


def _decode_temperature_groups(
    *,
    model: SimpleStatelessPolicyValueNet,
    state: SimpleStatelessBackboneOutput,
    option_embeddings: torch.Tensor,
    option_batch: OptionBatch,
    deck_signatures: Sequence[str],
    route_plan: SimpleExactRoutePlan,
    requests: Sequence[PreparedSimpleStatelessDecision],
    device: torch.device,
) -> tuple[tuple[int, ...], ...]:
    """Decode temperature subgroups after one shared expensive encoding."""
    grouped_rows: dict[float, list[int]] = {}
    for row_index, request in enumerate(requests):
        grouped_rows.setdefault(request.policy._policy_temperature, []).append(
            row_index
        )
    if len(grouped_rows) == 1:
        temperature = next(iter(grouped_rows))
        return _decode_uniform_temperature(
            model=model,
            policy_tokens=state.policy,
            opponent_belief=state.opponent_belief,
            option_embeddings=option_embeddings,
            option_batch=option_batch,
            route_plan=route_plan,
            requests=requests,
            temperature=temperature,
            device=device,
        )
    actions: list[tuple[int, ...] | None] = [None] * len(requests)
    for temperature, host_indices in grouped_rows.items():
        indices = torch.tensor(host_indices, dtype=torch.long, device=device)
        group_options = _select_option_batch_rows(
            option_batch,
            indices=indices,
            host_indices=host_indices,
        )
        group_routes = resolve_simple_exact_routes(
            tuple(deck_signatures[index] for index in host_indices),
            model.config,
            device=device,
        )
        policy_tokens = state.policy.index_select(0, indices)
        opponent_belief = state.opponent_belief.index_select(0, indices)
        group_embeddings = option_embeddings.index_select(0, indices)
        decoded = _decode_uniform_temperature(
            model=model,
            policy_tokens=policy_tokens,
            opponent_belief=opponent_belief,
            option_embeddings=group_embeddings,
            option_batch=group_options,
            route_plan=group_routes,
            requests=tuple(requests[index] for index in host_indices),
            temperature=temperature,
            device=device,
        )
        for row_index, action in zip(host_indices, decoded, strict=True):
            actions[row_index] = action
    if any(action is None for action in actions):
        raise RuntimeError("temperature-group decoding omitted a request row")
    return tuple(cast(tuple[int, ...], action) for action in actions)


def _decode_uniform_temperature(
    *,
    model: SimpleStatelessPolicyValueNet,
    policy_tokens: torch.Tensor,
    opponent_belief: torch.Tensor,
    option_embeddings: torch.Tensor,
    option_batch: OptionBatch,
    route_plan: SimpleExactRoutePlan,
    requests: Sequence[PreparedSimpleStatelessDecision],
    temperature: float,
    device: torch.device,
) -> tuple[tuple[int, ...], ...]:
    """Decode one temperature-homogeneous view of an encoded batch."""
    if temperature == 0.0:
        return model.heads.greedy_decode(
            policy_tokens,
            opponent_belief,
            option_embeddings,
            option_batch,
            route_plan=route_plan,
        )
    maximum_steps = (
        max(option_batch.maximum_counts, default=0)
        if option_batch.maximum_counts
        else int(option_batch.valid_options.shape[1]) + 1
    )
    sampling_uniforms = torch.cat(
        tuple(
            torch.rand(
                (1, maximum_steps + 1),
                dtype=torch.float32,
                device=device,
                generator=request.policy._sampling_generator,
            )
            for request in requests
        ),
        dim=0,
    )
    sampled = model.heads.sample_decode_actions(
        policy_tokens,
        opponent_belief,
        option_embeddings,
        option_batch,
        route_plan=route_plan,
        temperature=temperature,
        sampling_uniforms=sampling_uniforms,
    )
    return _materialize_action_sequences(sampled)


def _select_option_batch_rows(
    options: OptionBatch,
    *,
    indices: torch.Tensor,
    host_indices: Sequence[int],
) -> OptionBatch:
    """Select aligned option rows while preserving host decode bounds."""

    def select(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, indices.to(device=value.device))

    return OptionBatch(
        option_types=select(options.option_types),
        contexts=select(options.contexts),
        entity_slots=select(options.entity_slots),
        entity_slot_mask=select(options.entity_slot_mask),
        attack_ids=select(options.attack_ids),
        card_ids=select(options.card_ids),
        scalars=select(options.scalars),
        dynamic_effect_features=select(options.dynamic_effect_features),
        dynamic_effect_masks=select(options.dynamic_effect_masks),
        valid_options=select(options.valid_options),
        min_counts=select(options.min_counts),
        max_counts=select(options.max_counts),
        option_lengths=(
            tuple(options.option_lengths[index] for index in host_indices)
            if options.option_lengths
            else ()
        ),
        maximum_counts=(
            tuple(options.maximum_counts[index] for index in host_indices)
            if options.maximum_counts
            else ()
        ),
    )


def _greedy_stop_sampled(
    row: SimpleStatelessActorRow,
    action: Sequence[int],
) -> bool:
    """Match the deployed greedy STOP record used by temporal cache commits."""
    count_first = row.min_count < row.max_count and any(
        int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        for context in row.options.contexts
    )
    return not count_first and len(action) < row.max_count


def _materialize_action_sequences(actions: Any) -> tuple[tuple[int, ...], ...]:
    """Transfer one packed sampled-action batch through the deployment boundary."""
    packed = (
        torch.cat(
            (
                actions.choice_indices,
                actions.lengths.to(dtype=actions.choice_indices.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        .detach()
        .cpu()
    )
    choice_width = int(actions.choice_indices.shape[1])
    return tuple(
        tuple(int(value) for value in packed[row, : int(packed[row, choice_width])])
        for row in range(int(packed.shape[0]))
    )


def _resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA deployment device is unavailable")
    return resolved


def _asset_fingerprints(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    assets = payload.get("asset_manifest_fingerprints")
    if not isinstance(assets, Mapping):
        raise ValueError("deployment checkpoint has no asset fingerprints")
    return assets


def _asset(assets: Mapping[str, Any], name: str) -> str:
    value = assets.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"deployment asset fingerprint is missing: {name}")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("deployment observation must be a mapping")
    return value


def _request_fingerprint(observation: Any) -> str:
    """Identify one actor-visible callback without timing or protected state."""
    mapping = _mapping(observation)
    visible = {
        str(key): value
        for key, value in mapping.items()
        if str(key) not in {"remainingOverageTime", "search_begin_input"}
    }
    try:
        payload = orjson.dumps(visible, option=orjson.OPT_SORT_KEYS)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "deployment observation is not canonical-JSON compatible"
        ) from exc
    return hashlib.sha256(
        b"ptcg-rl/simple-stateless-visible-request/v1\x00" + payload
    ).hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "FixedDeckStatelessPolicy",
    "PreparedSimpleStatelessDecision",
    "RoutedDeckStatelessPolicy",
    "SimpleStatelessBatchResult",
    "SimpleStatelessInferenceBatcher",
    "routed_stateless_policy",
]
