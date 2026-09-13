"""Per-policy public input contracts and clean stateless actor tensorization."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from ptcg_rl.actions.encoding import (
    EncodedOptionArrayFeatures,
    StateTokenLayout,
    encode_option_arrays,
)
from ptcg_rl.agent.probe import observation_with_probe_features
from ptcg_rl.belief.public_catalog import (
    PublicDeckCatalog,
    PublicDeckPosterior,
    PublicDeckPosteriorArrays,
)
from ptcg_rl.context import (
    ContextBeliefTracker,
    GameContextFeatures,
    OpponentBeliefFeatureConfig,
    PublicCatalogContext,
    PublicCatalogTracker,
    PublicEventDecisionToken,
    PublicEventDelta,
)
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.model.input_schema import POLICY_INPUT_SCHEMA_FINGERPRINT
from ptcg_rl.model.policy import OptionBatch, collate_encoded_options
from ptcg_rl.model.simple_stateless import (
    PublicBeliefSummaryBatch,
    collate_public_belief_summaries,
)
from ptcg_rl.model.state_encoder import (
    StateBatch,
    StateTokenArrayFeatures,
    collate_state_tokens,
    encode_observation_token_arrays,
)
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_POLICY_INPUT_CONTRACT_DOMAIN = b"ptcg-rl/per-policy-input-contract/v1\x00"
_FROZEN_BUNDLE_DOMAIN = b"ptcg-rl/frozen-policy-bundle/v1\x00"
_SIMPLE_INPUT_SCHEMA_DOMAIN = b"ptcg-rl/simple-stateless-input-schema/v1\x00"
_SIMPLE_WRAPPER_RUNTIME_DOMAIN = b"ptcg-rl/simple-stateless-wrapper-runtime/v1\x00"

SIMPLE_STATELESS_INPUT_SCHEMA_VERSION = 1
SIMPLE_STATELESS_INPUT_SCHEMA_FINGERPRINT = hashlib.sha256(
    _SIMPLE_INPUT_SCHEMA_DOMAIN
    + json.dumps(
        {
            "version": SIMPLE_STATELESS_INPUT_SCHEMA_VERSION,
            "state": "public_state_token_arrays_v2",
            "deck": "canonical_exact_multiset_v1",
            "belief": "complete_public_catalog_raw_summary_v1",
            "action": POLICY_INPUT_SCHEMA_FINGERPRINT,
            "private_fields": (),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT = hashlib.sha256(
    _SIMPLE_WRAPPER_RUNTIME_DOMAIN
    + b"public-catalog-tracker+state-v2+complete-action-v2+stateless"
).hexdigest()


class PolicyInputContract(BaseModel):
    """Immutable observation/belief contract owned by one policy artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_kind: Literal["simple_public_catalog", "legacy_archive_native"]
    schema_version: int = Field(ge=1)
    input_schema_fingerprint: str
    action_schema_fingerprint: str
    public_context_fingerprint: str
    card_catalog_fingerprint: str
    belief_runtime_fingerprint: str
    wrapper_runtime_fingerprint: str
    recurrent_contract: str = "stateless"

    @field_validator(
        "input_schema_fingerprint",
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "belief_runtime_fingerprint",
        "wrapper_runtime_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require every input identity component to be a full SHA-256."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("policy input identity fields must be SHA-256")
        return normalized

    @field_validator("recurrent_contract")
    @classmethod
    def non_empty_recurrent_contract(cls, value: str) -> str:
        """Reject ambiguous recurrent runtime declarations."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("recurrent contract must be non-empty")
        return normalized

    @property
    def fingerprint(self) -> str:
        """Return the canonical input adapter identity."""
        return _canonical_fingerprint(
            _POLICY_INPUT_CONTRACT_DOMAIN,
            self.model_dump(mode="json"),
        )


class FrozenPolicyBundleIdentity(BaseModel):
    """Complete immutable pilot-plus-exact-deck rollout identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pilot_id: str
    checkpoint_size_bytes: int = Field(ge=1)
    checkpoint_sha256: str
    original_model_config_fingerprint: str
    exact_deck_digest: str
    frozen_registry_fingerprint: str
    input_contract: PolicyInputContract
    archive_native_belief_prior_fingerprint: str
    wrapper_runtime_fingerprint: str
    recurrent_runtime_contract: str

    @field_validator("pilot_id", "recurrent_runtime_contract")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Normalize required human/runtime identifiers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("frozen bundle identity text must be non-empty")
        return normalized

    @field_validator(
        "checkpoint_sha256",
        "original_model_config_fingerprint",
        "exact_deck_digest",
        "frozen_registry_fingerprint",
        "archive_native_belief_prior_fingerprint",
        "wrapper_runtime_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require full content fingerprints for every frozen component."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("frozen bundle identity fields must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_runtime(self) -> FrozenPolicyBundleIdentity:
        """Bind the adapter and wrapper runtime declarations."""
        if (
            self.wrapper_runtime_fingerprint
            != self.input_contract.wrapper_runtime_fingerprint
        ):
            raise ValueError("frozen wrapper and input contract fingerprints differ")
        if (
            self.recurrent_runtime_contract
            != self.input_contract.recurrent_contract
        ):
            raise ValueError("frozen recurrent and input contracts differ")
        if self.input_contract.adapter_kind != "legacy_archive_native":
            raise ValueError("historical frozen bundles need archive-native adapters")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the exact opponent bundle fingerprint used by PFSP."""
        return _canonical_fingerprint(
            _FROZEN_BUNDLE_DOMAIN,
            self.model_dump(mode="json"),
        )

    def verify_checkpoint(self, path: Path) -> None:
        """Stream-verify the configured immutable checkpoint artifact."""
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != self.checkpoint_size_bytes:
            raise ValueError("frozen checkpoint byte size changed")
        if _file_sha256(path) != self.checkpoint_sha256:
            raise ValueError("frozen checkpoint fingerprint changed")


@dataclass(frozen=True)
class SimpleStatelessActorRow:
    """One public-only decision row before device collation."""

    state: StateTokenArrayFeatures
    options: EncodedOptionArrayFeatures
    min_count: int
    max_count: int
    own_deck: CanonicalDeck
    belief_summary: PublicDeckPosterior | PublicDeckPosteriorArrays
    catalog_fingerprint: str
    input_contract_fingerprint: str
    public_event_delta: PublicEventDelta = PublicEventDelta()
    engine_fact_producer_fingerprint: str | None = None
    sequence_identity: SequenceDecisionIdentity | None = None


@dataclass(frozen=True)
class SimpleStatelessTensorizeRequest:
    """Pickle-safe public observation payload for CPU tensorization workers."""

    observation: Mapping[str, Any]
    context_features: GameContextFeatures


@dataclass(frozen=True)
class SimpleStatelessTensorizedInput:
    """Tracker-independent numeric features returned by a CPU worker."""

    state: StateTokenArrayFeatures
    options: EncodedOptionArrayFeatures
    min_count: int
    max_count: int


@dataclass(frozen=True)
class SimpleStatelessPolicyInputBatch:
    """Complete clean stateless actor/learner model inputs."""

    states: StateBatch
    options: OptionBatch
    unique_deck_card_ids: Tensor
    deck_counts: Tensor
    deck_valid_mask: Tensor
    deck_signatures: tuple[str, ...]
    belief_summary: PublicBeliefSummaryBatch
    state_rows: tuple[StateTokenArrayFeatures, ...]
    option_rows: tuple[EncodedOptionArrayFeatures, ...]
    min_counts: tuple[int, ...]
    max_counts: tuple[int, ...]
    input_contract_fingerprint: str

    @property
    def batch_size(self) -> int:
        """Return the number of aligned policy decisions."""
        return len(self.deck_signatures)


@dataclass(frozen=True)
class SimpleStatelessObservationBatch:
    """State/deck/belief inputs without unused legal-option collation."""

    states: StateBatch
    unique_deck_card_ids: Tensor
    deck_counts: Tensor
    deck_valid_mask: Tensor
    deck_signatures: tuple[str, ...]
    belief_summary: PublicBeliefSummaryBatch
    input_contract_fingerprint: str

    @property
    def batch_size(self) -> int:
        """Return the number of aligned observation rows."""
        return len(self.deck_signatures)


@dataclass(frozen=True)
class NativePolicyActorRow:
    """Observation prepared by one historical policy's private adapter."""

    observation: Mapping[str, Any]
    own_deck: CanonicalDeck
    input_contract_fingerprint: str


class SimpleStatelessPublicInputAdapter:
    """Per-game candidate adapter that consumes only public information."""

    def __init__(
        self,
        catalog: PublicDeckCatalog,
        *,
        contract: PolicyInputContract,
        player_index: int,
        own_deck: Sequence[int],
        engine_fact_producer: ProspectiveEngineFactProducer | None = None,
    ) -> None:
        """Bind one immutable catalog and exact acting deck to one game seat."""
        if contract.adapter_kind != "simple_public_catalog":
            raise ValueError("candidate adapter requires a public-catalog contract")
        if contract.belief_runtime_fingerprint != catalog.fingerprint:
            raise ValueError("candidate input contract uses a different catalog")
        self.contract = contract
        self.engine_fact_producer = engine_fact_producer
        self.own_deck = canonicalize_deck(own_deck)
        self._tracker = PublicCatalogTracker(catalog)
        self._tracker.begin_game(
            player_index=player_index,
            own_deck=self.own_deck.card_ids,
        )

    def prepare(self, observation: Mapping[str, Any]) -> SimpleStatelessActorRow:
        """Tensorize one callback without accepting controller-private identity."""
        context = self.observe(observation)
        return self.tensorize_observed(observation, context=context)

    def observe(self, observation: Mapping[str, Any]) -> PublicCatalogContext:
        """Advance public evidence without materializing unused policy tensors."""
        return self._tracker.update(observation)

    def tensorize_observed(
        self,
        observation: Mapping[str, Any],
        *,
        context: PublicCatalogContext,
    ) -> SimpleStatelessActorRow:
        """Tensorize an observation already applied exactly once to this tracker."""
        if context.catalog_fingerprint != self._tracker.catalog.fingerprint:
            raise ValueError("observed context uses a different public catalog")
        return self.complete_tensorized(
            tensorize_simple_stateless_request(
                self.tensorize_request(observation, context=context)
            ),
            context=context,
        )

    def tensorize_request(
        self,
        observation: Mapping[str, Any],
        *,
        context: PublicCatalogContext,
    ) -> SimpleStatelessTensorizeRequest:
        """Build a worker-safe request after advancing this adapter's tracker."""
        if context.catalog_fingerprint != self._tracker.catalog.fingerprint:
            raise ValueError("observed context uses a different public catalog")
        prepared_observation: Mapping[str, Any] = observation
        if self.engine_fact_producer is not None:
            result = self.engine_fact_producer.run(
                dict(observation),
                context.features,
                your_deck=self.own_deck.card_ids,
            )
            if result is not None:
                prepared_observation = observation_with_probe_features(
                    observation,
                    result.as_probe_result(),
                )
        return SimpleStatelessTensorizeRequest(
            observation=prepared_observation,
            context_features=context.features,
        )

    def complete_tensorized(
        self,
        tensorized: SimpleStatelessTensorizedInput,
        *,
        context: PublicCatalogContext,
    ) -> SimpleStatelessActorRow:
        """Attach parent-owned deck and posterior data to numeric worker output."""
        if context.catalog_fingerprint != self._tracker.catalog.fingerprint:
            raise ValueError("tensorized context uses a different public catalog")
        return SimpleStatelessActorRow(
            state=tensorized.state.without_layout(),
            options=tensorized.options,
            min_count=tensorized.min_count,
            max_count=tensorized.max_count,
            own_deck=self.own_deck,
            belief_summary=PublicDeckPosteriorArrays.from_posterior(
                context.posterior
            ),
            catalog_fingerprint=context.catalog_fingerprint,
            input_contract_fingerprint=self.contract.fingerprint,
            public_event_delta=context.features.public_event_delta,
            engine_fact_producer_fingerprint=(
                None
                if self.engine_fact_producer is None
                else self.engine_fact_producer.fingerprint
            ),
        )

    @property
    def known_opponent_counts(self) -> tuple[tuple[int, int], ...]:
        """Expose only deduplicated public evidence for learner target subtraction."""
        return self._tracker.known_opponent_counts

    def prepare_decision(self) -> PublicEventDecisionToken:
        """Freeze current public events until the engine accepts an action."""
        return self._tracker.context.prepare_decision()

    def commit_decision(
        self,
        token: PublicEventDecisionToken,
    ) -> PublicEventDelta:
        """Consume exactly one accepted policy decision's public events."""
        return self._tracker.context.commit_decision(token)

    def abort_decision(self, token: PublicEventDecisionToken) -> None:
        """Keep public events after a rejected or failed proposal."""
        self._tracker.context.abort_decision(token)


def tensorize_simple_stateless_request(
    request: SimpleStatelessTensorizeRequest,
) -> SimpleStatelessTensorizedInput:
    """Materialize one public actor row without mutable tracker state."""
    observation = request.observation
    public_observation = dict(observation)
    public_observation["gameContext"] = (
        request.context_features.as_observation_dict()
    )
    layout = StateTokenLayout.from_observation(
        public_observation,
        context_features=request.context_features,
    )
    select = _field(public_observation, "select")
    if select is None:
        raise ValueError("candidate observation has no select prompt")
    options = encode_option_arrays(select, layout)
    _apply_probe_features(options, public_observation)
    if len(options) <= 0:
        raise ValueError("candidate observation has no legal options")
    minimum = min(len(options), max(0, _int_field(select, "minCount", 0)))
    maximum = min(
        len(options),
        max(minimum, _int_field(select, "maxCount", len(options))),
    )
    return SimpleStatelessTensorizedInput(
        state=encode_observation_token_arrays(
            public_observation,
            layout=layout,
        ),
        options=options,
        min_count=minimum,
        max_count=maximum,
    )


class ArchiveNativePolicyInputAdapter:
    """Per-game historical adapter with its own archived belief producer."""

    def __init__(
        self,
        *,
        contract: PolicyInputContract,
        belief: OpponentBeliefFeatureConfig | None,
        player_index: int,
        own_deck: Sequence[int],
    ) -> None:
        """Construct an isolated legacy tracker for one frozen artifact."""
        if contract.adapter_kind != "legacy_archive_native":
            raise ValueError("historical adapter requires archive-native contract")
        self.contract = contract
        self.own_deck = canonicalize_deck(own_deck)
        self._tracker = ContextBeliefTracker(belief=belief)
        self._tracker.begin_game(
            player_index=player_index,
            own_deck=self.own_deck.card_ids,
        )

    def prepare(self, observation: Mapping[str, Any]) -> NativePolicyActorRow:
        """Apply only the frozen artifact's private native augmentation."""
        adapted = self._tracker.observation_with_context(observation)
        if not isinstance(adapted, Mapping):
            raise TypeError("archive-native adapter returned a non-mapping observation")
        return NativePolicyActorRow(
            observation=cast(Mapping[str, Any], adapted),
            own_deck=self.own_deck,
            input_contract_fingerprint=self.contract.fingerprint,
        )


PreparedInput = TypeVar("PreparedInput", covariant=True)


class PolicyInputAdapter(Protocol[PreparedInput]):
    """Minimal per-game adapter interface owned by a policy artifact."""

    contract: PolicyInputContract

    def prepare(self, observation: Mapping[str, Any]) -> PreparedInput:
        """Return this policy's native public model inputs."""


PolicyInputAdapterFactory = Callable[
    [int, Sequence[int]],
    PolicyInputAdapter[Any],
]


class PolicyInputAdapterRegistry:
    """Lease isolated game-seat adapters by immutable policy fingerprint."""

    def __init__(self) -> None:
        """Initialize an empty single-process adapter registry."""
        self._factories: dict[
            str,
            tuple[PolicyInputContract, PolicyInputAdapterFactory],
        ] = {}
        self._leases: dict[tuple[str, int, str], PolicyInputAdapter[Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        artifact_fingerprint: str,
        contract: PolicyInputContract,
        factory: PolicyInputAdapterFactory,
    ) -> None:
        """Register one immutable artifact-to-adapter binding."""
        artifact = _require_fingerprint(artifact_fingerprint)
        with self._lock:
            existing = self._factories.get(artifact)
            if existing is not None:
                if existing[0] != contract or existing[1] is not factory:
                    raise ValueError("policy input artifact cannot be rebound")
                return
            self._factories[artifact] = (contract, factory)

    def lease(
        self,
        *,
        game_id: str,
        seat: int,
        artifact_fingerprint: str,
        own_deck: Sequence[int],
    ) -> PolicyInputAdapter[Any]:
        """Return one policy-native tracker isolated to a game and seat."""
        if not game_id:
            raise ValueError("input adapter lease requires a game ID")
        if seat not in (0, 1):
            raise ValueError("input adapter seat must be zero or one")
        artifact = _require_fingerprint(artifact_fingerprint)
        key = (game_id, seat, artifact)
        with self._lock:
            existing = self._leases.get(key)
            if existing is not None:
                return existing
            try:
                _contract, factory = self._factories[artifact]
            except KeyError as error:
                raise KeyError(
                    f"policy input adapter is not registered: {artifact}"
                ) from error
            adapter = factory(seat, own_deck)
            self._leases[key] = adapter
            return adapter

    def release_game(self, game_id: str) -> int:
        """Release all per-seat trackers for one finished or cancelled game."""
        with self._lock:
            keys = tuple(key for key in self._leases if key[0] == game_id)
            for key in keys:
                del self._leases[key]
            return len(keys)


def simple_stateless_input_contract(
    *,
    public_catalog_fingerprint: str,
    card_catalog_fingerprint: str,
    public_context_fingerprint: str,
    wrapper_runtime_fingerprint: str,
) -> PolicyInputContract:
    """Build the candidate's fixed public-only stateless input identity."""
    return PolicyInputContract(
        adapter_kind="simple_public_catalog",
        schema_version=SIMPLE_STATELESS_INPUT_SCHEMA_VERSION,
        input_schema_fingerprint=SIMPLE_STATELESS_INPUT_SCHEMA_FINGERPRINT,
        action_schema_fingerprint=POLICY_INPUT_SCHEMA_FINGERPRINT,
        public_context_fingerprint=public_context_fingerprint,
        card_catalog_fingerprint=card_catalog_fingerprint,
        belief_runtime_fingerprint=public_catalog_fingerprint,
        wrapper_runtime_fingerprint=wrapper_runtime_fingerprint,
        recurrent_contract="stateless",
    )


def collate_simple_stateless_actor_rows(
    rows: Sequence[SimpleStatelessActorRow],
    *,
    device: torch.device | str | None = None,
    deduplicate_belief: bool = False,
) -> SimpleStatelessPolicyInputBatch:
    """Collate public-only rows into one model-ready homogeneous batch."""
    observation = collate_simple_stateless_observation_rows(
        rows,
        device=device,
        deduplicate_belief=deduplicate_belief,
    )
    options = collate_encoded_options(
        tuple(row.options for row in rows),
        min_counts=tuple(row.min_count for row in rows),
        max_counts=tuple(row.max_count for row in rows),
        device=device,
    )
    normalized_mins = tuple(
        min(len(row.options), max(0, int(row.min_count))) for row in rows
    )
    normalized_maxes = tuple(
        min(
            len(row.options),
            max(normalized_min, int(row.max_count)),
        )
        for row, normalized_min in zip(
            rows,
            normalized_mins,
            strict=True,
        )
    )
    return SimpleStatelessPolicyInputBatch(
        states=observation.states,
        options=options,
        unique_deck_card_ids=observation.unique_deck_card_ids,
        deck_counts=observation.deck_counts,
        deck_valid_mask=observation.deck_valid_mask,
        deck_signatures=observation.deck_signatures,
        belief_summary=observation.belief_summary,
        state_rows=tuple(row.state for row in rows),
        option_rows=tuple(row.options for row in rows),
        min_counts=normalized_mins,
        max_counts=normalized_maxes,
        input_contract_fingerprint=observation.input_contract_fingerprint,
    )


def collate_simple_stateless_observation_rows(
    rows: Sequence[SimpleStatelessActorRow],
    *,
    device: torch.device | str | None = None,
    deduplicate_belief: bool = False,
) -> SimpleStatelessObservationBatch:
    """Collate only snapshot-backbone inputs for temporal context rows."""
    if not rows:
        raise ValueError("at least one stateless observation row is required")
    contract = rows[0].input_contract_fingerprint
    catalog = rows[0].catalog_fingerprint
    if any(row.input_contract_fingerprint != contract for row in rows):
        raise ValueError("cannot mix policy input contracts in one batch")
    if any(row.catalog_fingerprint != catalog for row in rows):
        raise ValueError("cannot mix public catalog identities in one batch")
    unique_ids, counts, valid = _collate_deck_multisets(
        tuple(row.own_deck for row in rows),
        device=device,
    )
    return SimpleStatelessObservationBatch(
        states=collate_state_tokens(
            tuple(row.state for row in rows),
            device=device,
        ),
        unique_deck_card_ids=unique_ids,
        deck_counts=counts,
        deck_valid_mask=valid,
        deck_signatures=tuple(row.own_deck.signature for row in rows),
        belief_summary=collate_public_belief_summaries(
            tuple(row.belief_summary for row in rows),
            catalog_fingerprint=catalog,
            device=device,
            deduplicate=deduplicate_belief,
        ),
        input_contract_fingerprint=contract,
    )


def _collate_deck_multisets(
    decks: Sequence[CanonicalDeck],
    *,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor, Tensor]:
    counter_rows = tuple(Counter(deck.card_ids) for deck in decks)
    width = max(len(counter) for counter in counter_rows)
    card_id_rows = np.zeros((len(decks), width), dtype=np.int64)
    count_rows = np.zeros((len(decks), width), dtype=np.float32)
    valid_rows = np.zeros((len(decks), width), dtype=np.bool_)
    for row, counter in enumerate(counter_rows):
        pairs = tuple(sorted(counter.items()))
        card_id_rows[row, : len(pairs)] = [
            card_id for card_id, _count in pairs
        ]
        count_rows[row, : len(pairs)] = [
            count for _card_id, count in pairs
        ]
        valid_rows[row, : len(pairs)] = True
    return (
        torch.as_tensor(card_id_rows, device=device),
        torch.as_tensor(count_rows, device=device),
        torch.as_tensor(valid_rows, device=device),
    )


def _apply_probe_features(
    options: EncodedOptionArrayFeatures,
    observation: Mapping[str, Any],
) -> None:
    """Persist the exact prospective tensors actually shown to the actor."""
    raw_features = observation.get("probeEffectFeatures")
    raw_masks = observation.get("probeEffectMasks")
    if not isinstance(raw_features, Sequence) or not isinstance(
        raw_masks,
        Sequence,
    ):
        return
    for index in range(min(len(options), len(raw_features), len(raw_masks))):
        if not bool(raw_masks[index]):
            continue
        row = raw_features[index]
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != DYNAMIC_EFFECT_FEATURE_SIZE
        ):
            raise ValueError("prospective engine fact row has invalid width")
        values = np.asarray(row, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("prospective engine facts must be finite")
        options.dynamic_effect_features[index, :] = values
        options.dynamic_effect_masks[index] = True


def _canonical_fingerprint(domain: bytes, payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _require_fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("artifact fingerprint must be SHA-256")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    try:
        return int(_field(value, name, default))
    except (TypeError, ValueError):
        return default


__all__ = [
    "ArchiveNativePolicyInputAdapter",
    "FrozenPolicyBundleIdentity",
    "NativePolicyActorRow",
    "PolicyInputAdapter",
    "PolicyInputAdapterRegistry",
    "PolicyInputContract",
    "SIMPLE_STATELESS_INPUT_SCHEMA_FINGERPRINT",
    "SIMPLE_STATELESS_INPUT_SCHEMA_VERSION",
    "SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT",
    "SimpleStatelessActorRow",
    "SimpleStatelessObservationBatch",
    "SimpleStatelessPolicyInputBatch",
    "SimpleStatelessPublicInputAdapter",
    "SimpleStatelessTensorizedInput",
    "SimpleStatelessTensorizeRequest",
    "collate_simple_stateless_actor_rows",
    "collate_simple_stateless_observation_rows",
    "simple_stateless_input_contract",
    "tensorize_simple_stateless_request",
]
