"""Vectorized rollout stepping loop."""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeGuard, cast

import numpy as np
import torch
from pydantic import field_validator
from torch import Tensor

from ptcg_rl.actions.encoding import (
    EncodedOptionArrayFeatures,
    EncodedOptionInput,
    StateTokenLayout,
    encode_option_arrays,
)
from ptcg_rl.actions.selection import (
    forced_action,
    is_legal_action,
    normalize_action_order,
)
from ptcg_rl.agent.probe import ActTimeSearchConfig, observation_with_probe_features
from ptcg_rl.agent.search.context import public_search_observation
from ptcg_rl.agent.search.executed_endpoint import build_executed_endpoint_leaf
from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.agent.search.root_information import RootInformationLeaf
from ptcg_rl.agent.search.root_information_producer import (
    root_information_belief_summary,
)
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    ExpectedCardCount,
    GameContext,
    GameContextFeatures,
    GameContextSnapshot,
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    PublicEventBatch,
    PublicEventDecisionToken,
    PublicEventDelta,
    collate_public_event_deltas,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.probe_resolution import ProbeTransition
from ptcg_rl.engine.search_api_probe import SearchApiProbeBackend
from ptcg_rl.engine.vector_battle import (
    DeckPair,
    FinishedGame,
    VectorGame,
)
from ptcg_rl.model import (
    OptionBatch,
    RecurrentPolicyState,
    StateBatch,
    StateTokenArrayFeatures,
    StateTokenInput,
    collate_encoded_options,
    collate_state_tokens,
    concatenate_recurrent_policy_states,
    encode_observation_token_arrays,
    select_recurrent_policy_state_rows,
)
from ptcg_rl.model.network import SampleDecodeTrace
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.opponents import BattleAgent
from ptcg_rl.profiling import StageTimer, time_stage
from ptcg_rl.rl.amortized_policy_iteration.contracts import BehaviorKind
from ptcg_rl.rl.curriculum import GameAssignment
from ptcg_rl.rl.engine_teacher import (
    AsyncEngineTeacherProducer,
    EngineTeacherBatchCompletion,
    EngineTeacherProducer,
    EngineTeacherRequest,
    EngineTeacherTarget,
    validate_engine_teacher_target,
)
from ptcg_rl.rl.factual import (
    FactualLaneConfig,
    FactualSuccessor,
    FactualTransitionTarget,
    build_factual_transition_target,
    factual_successor,
)
from ptcg_rl.rl.macro_credit import MacroCreditConfig
from ptcg_rl.rl.macro_teacher import MacroTeacherRequest
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerPolicyDecision
from ptcg_rl.rl.planner_evidence import PlannerBehaviorEvidence
from ptcg_rl.rl.recurrent_runtime import (
    PolicyArtifactIdentity,
    RecurrentDecodeResult,
    RecurrentInferenceBatch,
    RecurrentSequenceIdentity,
    validate_policy_artifact_identity,
)
from ptcg_rl.runtime.planner_telemetry import PlannerTelemetryAccumulator

if TYPE_CHECKING:
    from ptcg_rl.rl.collection import (
        RolloutConfig,
        RolloutPolicyFactory,
        RolloutPoolFactory,
    )

RolloutMode = Literal["self_play", "frozen", "scripted", "curriculum"]
PolicyRole = Literal["candidate", "frozen"]
RolloutProbeBackend = Literal["auto", "native", "search_api"]
_DECK_PATH_ENV = "POKEMON_TCG_DECK_PATH"
_SCRIPTED_DECK_CACHE_DIR = Path(tempfile.gettempdir()) / "ptcg_rl_scripted_decks"
_OLDEST_LIVE_GAMES_LIMIT = 4


class RolloutProbeConfig(ActTimeSearchConfig):
    """Configuration for training-rollout dynamic effect probes."""

    enabled: bool = False
    conservative_override_enabled: bool = False
    dropout: float = 0.05
    backend: RolloutProbeBackend = "auto"

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Reject invalid rollout probe dropout probabilities."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value


class RolloutBeliefConfig(OpponentBeliefFeatureConfig):
    """Configuration for rollout-side belief state tokens."""

    enabled: bool = False
    cache_max_entries: int = 32_768

    @field_validator("cache_max_entries")
    @classmethod
    def valid_cache_max_entries(cls, value: int) -> int:
        """Reject non-positive rollout belief cache bounds."""
        if value <= 0:
            raise ValueError("cache_max_entries must be positive")
        return value


class RolloutPolicy(Protocol):
    """Policy object used by vectorized rollout."""

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample actions, sequence log-probs, and value estimates."""


class PipelinedRolloutPolicy(RolloutPolicy, Protocol):
    """Rollout policy that supports explicit request/receive pipelining."""

    def submit_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> object:
        """Submit a decode request and return an opaque handle."""

    def receive_decode(
        self,
        handle: object,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Receive the decode result for one handle."""


class RecurrentRolloutPolicy(RolloutPolicy, Protocol):
    """Local policy exposing one pure recurrent transition per decision."""

    recurrent_enabled: bool

    def initial_recurrent_state(self, batch_size: int) -> RecurrentPolicyState:
        """Return explicit zero states for newly registered sequences."""

    def sample_decode_with_recurrent_state(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        public_events: PublicEventBatch,
        previous_state: RecurrentPolicyState,
        *,
        temperature: float,
        retain_planner_context: bool,
    ) -> tuple[SampleDecodeTrace, RecurrentPolicyState]:
        """Decode one complete action and return only a proposed next state."""


class PipelinedRecurrentRolloutPolicy(PipelinedRolloutPolicy, Protocol):
    """Remote policy carrying recurrent state through its request protocol."""

    recurrent_enabled: bool

    def initial_recurrent_state(self, batch_size: int) -> RecurrentPolicyState:
        """Return canonical zero states for newly registered sequences."""

    def submit_recurrent_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        recurrent: RecurrentInferenceBatch,
        *,
        temperature: float,
        retain_planner_context: bool,
    ) -> object:
        """Submit one complete-action recurrent decode."""

    def receive_recurrent_decode(self, handle: object) -> RecurrentDecodeResult:
        """Receive the proposed state and authoritative served artifact."""

    def release_recurrent_sequences(
        self,
        sequences: Sequence[RecurrentSequenceIdentity],
        expected_artifact: PolicyArtifactIdentity,
    ) -> int:
        """Release terminal or failed sequence leases from the server."""

    def abort_recurrent_sequences(
        self,
        sequences: Sequence[RecurrentSequenceIdentity],
    ) -> int:
        """Release a first-bind lease whose response was never observed."""


@dataclass(frozen=True)
class RolloutPlannerRowContext:
    """Root-visible actor row aligned with one base decode result."""

    game_id: str
    seat: int
    policy_role: PolicyRole
    should_record: bool
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    context_snapshot: GameContextSnapshot
    deck_pair: DeckPair
    model_deck: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RolloutPlannerBatch:
    """One cross-game pre-action batch sharing a behavior model lease."""

    policy: RolloutPolicy
    rows: tuple[RolloutPlannerRowContext, ...]
    states: StateBatch
    options: OptionBatch
    decks: DeckBatch
    base_actions: tuple[tuple[int, ...], ...]
    base_logprobs: tuple[float, ...]
    base_values: tuple[float, ...]
    policy_version: int
    model_fingerprint: str
    proposal_version: int
    planner_context_handles: tuple[str, ...]
    root_fallback_reason: PlannerFallbackReason | None = None


class RolloutPlannerBehaviorService(Protocol):
    """Synchronous behavior dependency that may batch across rollout games."""

    def plan_batch(
        self,
        batch: RolloutPlannerBatch,
    ) -> Sequence[PlannerPolicyDecision | None]:
        """Return one explicit planner/fallback branch per requested row."""


class VectorPoolLike(Protocol):
    """Subset of ``VectorBattlePool`` used by the rollout loop."""

    def pending(self) -> list[VectorGame]:
        """Return non-terminal games waiting for a select response."""

    def submit(self, game_id: str, action: Sequence[int]) -> None:
        """Submit one select response."""

    def finished(self) -> list[FinishedGame]:
        """Return terminal games and refill the pool."""

    def recycle(self, game_id: str) -> None:
        """Discard one non-terminal game and immediately refill the pool."""


class RolloutRecorder(Protocol):
    """Recorder hook used by future trajectory storage."""

    def record(self, decision: RolloutDecision) -> None:
        """Record one non-forced policy decision."""

    def finalize(self, finished: FinishedGame) -> None:
        """Finalize one finished game."""

    def discard(self, game_id: str, *, reason: str) -> None:
        """Drop one unfinished game's buffered evidence without finalizing it."""


@dataclass(frozen=True)
class RolloutActors:
    """Policies and optional scripted opponent used by a rollout loop."""

    mode: RolloutMode
    candidate_policy: RolloutPolicy
    frozen_policy: RolloutPolicy | None = None
    frozen_policies: Mapping[str, RolloutPolicy] = field(default_factory=dict)
    scripted_agent: BattleAgent | None = None
    scripted_agents: Mapping[str, BattleAgent] = field(default_factory=dict)
    # Stateful scripted opponents (module-level per-game state) must be
    # instantiated per game because vectorized rollouts interleave games.
    scripted_agent_factories: Mapping[str, Callable[[], BattleAgent]] = field(
        default_factory=dict
    )
    scripted_tiers: Mapping[str, int] = field(default_factory=dict)
    curriculum_assignments: Mapping[str, GameAssignment] = field(default_factory=dict)
    candidate_seat: int = 0

    def __post_init__(self) -> None:
        """Validate mode-specific actor wiring."""
        if self.candidate_seat not in (0, 1):
            raise ValueError("candidate_seat must be 0 or 1")
        if (
            self.mode == "frozen"
            and self.frozen_policy is None
            and not self.frozen_policies
        ):
            raise ValueError("frozen rollout mode requires frozen_policy")
        if (
            self.mode == "scripted"
            and self.scripted_agent is None
            and not self.scripted_agents
        ):
            raise ValueError("scripted rollout mode requires scripted_agent")
        if self.mode == "curriculum" and not self.curriculum_assignments:
            raise ValueError("curriculum rollout mode requires assignments")


@dataclass(frozen=True)
class RolloutDecision:
    """Frozen behavior evidence with an optional aligned auxiliary target."""

    game_id: str
    seat: int
    deck_pair: DeckPair
    action: tuple[int, ...]
    action_logprob: float
    value_pred: float
    sampling_temperature: float
    policy_role: PolicyRole
    observation: Mapping[str, Any]
    behavior_kind: BehaviorKind = "policy_sample"
    opponent_name: str = ""
    opponent_tier: int = -1
    encoded_state: StateTokenInput | None = None
    encoded_options: EncodedOptionInput = ()
    min_count: int | None = None
    max_count: int | None = None
    policy_version: int = 0
    token_logprobs: tuple[float, ...] | None = None
    prefix_value_preds: tuple[float, ...] | None = None
    stop_sampled: bool | None = None
    engine_teacher_target: EngineTeacherTarget | None = None
    factual_target: FactualTransitionTarget | None = None
    factual_transition_steps: int | None = None
    macro_credit_enabled: bool = False
    planner_behavior: PlannerBehaviorEvidence | None = None
    executed_endpoint_value_leaf: RootInformationLeaf | None = None
    training_metadata: Mapping[str, str] = field(default_factory=dict)
    reanalysis_observation: Mapping[str, Any] | None = None
    reanalysis_context_snapshots: (
        tuple[GameContextSnapshot, GameContextSnapshot] | None
    ) = None
    public_event_delta: PublicEventDelta | None = None
    policy_artifact_identity: PolicyArtifactIdentity | None = None


@dataclass(frozen=True)
class RolloutStepStats:
    """Counters from one rollout step."""

    pending_games: int = 0
    forced_actions: int = 0
    scripted_actions: int = 0
    policy_actions: int = 0
    recorded_decisions: int = 0
    finished_games: int = 0

    @property
    def actions(self) -> int:
        """Total engine submissions made by this step."""
        return self.forced_actions + self.scripted_actions + self.policy_actions


@dataclass(frozen=True)
class RecurrentGameRecycleResult:
    """One quiescent stale-game recycling decision."""

    recycled_game_ids: tuple[str, ...] = ()
    deferred_game_ids: tuple[str, ...] = ()
    released_sequences: int = 0
    max_candidate_version_age: int = 0


@dataclass(frozen=True)
class RolloutRunSummary:
    """Counters accumulated across ``RolloutStepper.run_until``."""

    iterations: int
    forced_actions: int
    scripted_actions: int
    policy_actions: int
    recorded_decisions: int
    finished_games: int


@dataclass
class ListRolloutRecorder:
    """In-memory recorder useful for tests and early smoke checks."""

    decisions: list[RolloutDecision] = field(default_factory=list)
    finished_games: list[FinishedGame] = field(default_factory=list)
    discarded_games: list[tuple[str, str]] = field(default_factory=list)

    def record(self, decision: RolloutDecision) -> None:
        """Append one decision."""
        self.decisions.append(decision)

    def finalize(self, finished: FinishedGame) -> None:
        """Append one finished game."""
        self.finished_games.append(finished)

    def discard(self, game_id: str, *, reason: str) -> None:
        """Drop decisions for one unfinished game and record the reason."""
        self.decisions = [
            decision for decision in self.decisions if decision.game_id != game_id
        ]
        self.discarded_games.append((game_id, reason))


def run_rollout(
    config: RolloutConfig,
    *,
    pool_factory: RolloutPoolFactory | None = None,
    policy_factory: RolloutPolicyFactory | None = None,
) -> dict[str, Any]:
    """Run configured trajectory collection via the collection module."""
    from ptcg_rl.rl.collection import run_rollout as collection_run_rollout

    return collection_run_rollout(
        config,
        pool_factory=pool_factory,
        policy_factory=policy_factory,
    )


@dataclass
class _NoopRolloutRecorder:
    def record(self, decision: RolloutDecision) -> None:
        del decision

    def finalize(self, finished: FinishedGame) -> None:
        del finished

    def discard(self, game_id: str, *, reason: str) -> None:
        del game_id, reason


@dataclass(frozen=True)
class _PolicyTurn:
    game: VectorGame
    seat: int
    role: PolicyRole
    policy_id: str
    should_record: bool
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    public_event_token: PublicEventDecisionToken
    recurrent_lease: _RecurrentStateLease | None


@dataclass(frozen=True)
class _PolicyObservation:
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    own_deck: CanonicalDeck


@dataclass(frozen=True)
class _PolicyInputBatch:
    states: StateBatch
    options: OptionBatch
    decks: DeckBatch
    state_features: tuple[StateTokenArrayFeatures, ...]
    encoded_options: tuple[EncodedOptionArrayFeatures, ...]
    min_counts: tuple[int, ...]
    max_counts: tuple[int, ...]


@dataclass(frozen=True)
class _PolicyDecodeResult:
    """Sampled policy batch with optional token-level behavior evidence."""

    actions: tuple[tuple[int, ...], ...]
    action_logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor | None = None
    prefix_values: Tensor | None = None
    token_mask: Tensor | None = None
    stop_sampled: Tensor | None = None
    planner_context_handles: tuple[str, ...] = ()
    served_policy_version: int | None = None
    served_model_fingerprint: str = ""
    served_proposal_version: int | None = None
    planner_fallback_reason: str = ""
    proposed_recurrent_states: tuple[RecurrentPolicyState, ...] = ()
    served_policy_artifact_fingerprint: str = ""
    served_policy_artifact: PolicyArtifactIdentity | None = None


@dataclass(frozen=True)
class _RecurrentStateKey:
    """Exact actor-side owner of one recurrent state sequence."""

    game_id: str
    seat: int
    policy_role: PolicyRole
    policy_id: str
    exact_deck_digest: str
    exact_deck_signature: str
    policy_artifact_fingerprint: str | None


@dataclass(frozen=True)
class _CommittedRecurrentState:
    """Last engine-accepted recurrent state and aligned event generation."""

    state: RecurrentPolicyState
    generation: int
    artifact: PolicyArtifactIdentity | None
    policy_version: int | None


@dataclass(frozen=True)
class _RecurrentStateLease:
    """Immutable old-state lease used by one pure decision preparation."""

    key: _RecurrentStateKey
    state: RecurrentPolicyState
    generation: int
    artifact: PolicyArtifactIdentity | None
    policy_version: int | None


@dataclass(frozen=True)
class _PreparedRecurrentTransition:
    """Candidate state that remains uncommitted until engine acceptance."""

    lease: _RecurrentStateLease
    proposed_state: RecurrentPolicyState


@dataclass(frozen=True)
class _DecisionIdentity:
    """Stable identity for joining independently resolved decision evidence."""

    sequence_id: int
    game_id: str
    seat: int


@dataclass(frozen=True)
class _FrozenPolicyDecision:
    """Behavior evidence detached from mutable policy and engine state."""

    order_index: int
    turn: _PolicyTurn
    select: Any
    deck_pair: DeckPair
    action: tuple[int, ...]
    action_logprob: float
    value_pred: float
    encoded_state: StateTokenInput
    encoded_options: EncodedOptionInput
    min_count: int
    max_count: int
    policy_version: int
    behavior_kind: BehaviorKind
    reanalysis_observation: Mapping[str, Any] | None
    reanalysis_context_snapshots: tuple[GameContextSnapshot, GameContextSnapshot] | None
    token_trace: tuple[tuple[float, ...], tuple[float, ...], bool] | None
    teacher_request: EngineTeacherRequest | MacroTeacherRequest | None
    recurrent_transition: _PreparedRecurrentTransition | None
    planner_behavior: PlannerBehaviorEvidence | None = None
    identity: _DecisionIdentity | None = None


@dataclass
class _PendingDecisionEvidence:
    """One recorded behavior row waiting for its configured auxiliaries."""

    identity: _DecisionIdentity
    decision: _FrozenPolicyDecision
    factual_resolved: bool
    teacher_resolved: bool
    factual_target: FactualTransitionTarget | None = None
    factual_transition_steps: int | None = None
    executed_endpoint_value_leaf: RootInformationLeaf | None = None
    teacher_target: EngineTeacherTarget | None = None


@dataclass
class _PendingFactualDecision:
    """One behavior decision awaiting its actual semantic successor."""

    decision: _FrozenPolicyDecision
    before_observation: Mapping[str, Any]
    transitions: list[ProbeTransition] = field(default_factory=list)


@dataclass
class _SubmissionReceipt:
    """Marks the irreversible boundary after an engine submit returns."""

    engine_accepted: bool = False
    context_committed: bool = False
    recurrent_committed: bool = False


_BeliefCacheValue = tuple[tuple[ExpectedCardCount, ...], float, bool]


class _RolloutFeatureAugmenter:
    """Attach optional belief tokens and dynamic effect probes before collation."""

    def __init__(
        self,
        *,
        probe_config: RolloutProbeConfig,
        belief_config: RolloutBeliefConfig,
        seed: int,
    ) -> None:
        self._probe_config = probe_config
        self._belief_config = belief_config
        self._rng = random.Random(seed)
        self._belief_producer = (
            OpponentBeliefFeatureProducer.from_config(belief_config)
            if belief_config.enabled
            else None
        )
        self._belief_cache: OrderedDict[tuple[Any, ...], _BeliefCacheValue] = (
            OrderedDict()
        )
        self._belief_cache_max_entries = belief_config.cache_max_entries
        self._belief_cache_hits = 0
        self._belief_cache_misses = 0
        self._belief_cache_evictions = 0
        self._sampler = (
            BeliefSampler(config=probe_config.sampler) if probe_config.enabled else None
        )
        self._probe_backend = _make_probe_backend(probe_config)
        self._configured_probe_backend = probe_config.backend
        self._probe_backend_counts: dict[str, int] = {}
        self._probe_calls = 0
        self._eligible_options = 0
        self._probed_options = 0
        self._worlds = 0
        self._native_batch_calls = 0
        self._native_transitions = 0
        self._native_errors = 0
        self._unresolved_options = 0
        self._unresolved_worlds = 0

    @property
    def enabled(self) -> bool:
        """Return whether any rollout feature augmentation is active."""
        return self._belief_producer is not None or self._probe_backend is not None

    def summary(self) -> Mapping[str, Any]:
        """Return rollout feature augmentation counters."""
        return {
            "belief_enabled": self._belief_producer is not None,
            "belief_cache_entries": len(self._belief_cache),
            "belief_cache_max_entries": self._belief_cache_max_entries,
            "belief_cache_hits": self._belief_cache_hits,
            "belief_cache_misses": self._belief_cache_misses,
            "belief_cache_evictions": self._belief_cache_evictions,
            "probe_enabled": self._probe_backend is not None,
            "probe_configured_backend": self._configured_probe_backend,
            "probe_backend_counts": dict(sorted(self._probe_backend_counts.items())),
            "probe_calls": self._probe_calls,
            "probe_eligible_options": self._eligible_options,
            "probe_probed_options": self._probed_options,
            "probe_worlds": self._worlds,
            "probe_native_batch_calls": self._native_batch_calls,
            "probe_native_transitions": self._native_transitions,
            "probe_native_errors": self._native_errors,
            "probe_unresolved_options": self._unresolved_options,
            "probe_unresolved_worlds": self._unresolved_worlds,
        }

    def augment_turn(self, turn: _PolicyTurn) -> _PolicyObservation:
        """Return a policy observation carrying optional belief/probe features."""
        observation = turn.observation
        context_features = turn.context_features
        own_deck = _own_deck_for_turn(turn)
        if self._belief_producer is not None:
            context_features = self._belief_features(
                observation,
                context_features,
            )
            observation = dict(observation)
            observation["gameContext"] = context_features.as_observation_dict()
        if self._probe_backend is None or self._sampler is None:
            return _PolicyObservation(observation, context_features, own_deck)
        if self._rng.random() < self._probe_config.dropout:
            return _PolicyObservation(observation, context_features, own_deck)
        result = self._probe_backend.run(
            observation,
            context_features,
            your_deck=turn.game.deck_pair[turn.seat],
            sampler=self._sampler,
            worlds=self._probe_config.worlds,
            rng=self._rng,
        )
        self._record_probe_stats(result.stats)
        if result.probe is None:
            return _PolicyObservation(observation, context_features, own_deck)
        return _PolicyObservation(
            cast(
                Mapping[str, Any],
                observation_with_probe_features(observation, result.probe),
            ),
            context_features,
            own_deck,
        )

    def context_with_belief(
        self,
        observation: Mapping[str, Any],
        context_features: GameContextFeatures,
    ) -> GameContextFeatures:
        """Apply the same public belief producer used by behavior tensorization."""
        return self._belief_features(observation, context_features)

    def _record_probe_stats(self, stats: Any) -> None:
        backend = str(getattr(stats, "backend", "unknown"))
        self._probe_backend_counts[backend] = (
            self._probe_backend_counts.get(backend, 0) + 1
        )
        self._probe_calls += 1
        self._eligible_options += int(getattr(stats, "eligible_options", 0))
        self._probed_options += int(getattr(stats, "probed_options", 0))
        self._worlds += int(getattr(stats, "worlds", 0))
        self._native_batch_calls += int(getattr(stats, "native_batch_calls", 0))
        self._native_transitions += int(getattr(stats, "native_transitions", 0))
        self._native_errors += int(getattr(stats, "native_errors", 0))
        self._unresolved_options += int(getattr(stats, "unresolved_options", 0))
        self._unresolved_worlds += int(getattr(stats, "unresolved_worlds", 0))

    def _belief_features(
        self,
        observation: Mapping[str, Any],
        context_features: GameContextFeatures,
    ) -> GameContextFeatures:
        if self._belief_producer is None:
            return context_features
        key = _belief_cache_key(observation, context_features)
        cached = self._belief_cache.get(key)
        if cached is None:
            self._belief_cache_misses += 1
            cached = self._belief_producer.features(observation, context_features)
            self._belief_cache[key] = cached
            if len(self._belief_cache) > self._belief_cache_max_entries:
                self._belief_cache.popitem(last=False)
                self._belief_cache_evictions += 1
        else:
            self._belief_cache_hits += 1
            self._belief_cache.move_to_end(key)
        belief, entropy, is_empty = cached
        return replace(
            context_features,
            opponent_belief=belief,
            opponent_belief_entropy=entropy,
            opponent_belief_empty=is_empty,
        )


@dataclass(frozen=True)
class _PendingPolicyBatch:
    policy: PipelinedRolloutPolicy
    turns: tuple[_PolicyTurn, ...]
    observations: tuple[_PolicyObservation, ...]
    policy_inputs: _PolicyInputBatch
    handle: object
    order_indices: tuple[int, ...]
    macro_teacher_sample_mask: tuple[bool, ...]
    recurrent_request: RecurrentInferenceBatch | None = None


def _policy_routes(turns: Sequence[_PolicyTurn]) -> tuple[tuple[PolicyRole, str], ...]:
    routes: list[tuple[PolicyRole, str]] = []
    seen: set[tuple[PolicyRole, str]] = set()
    for turn in turns:
        route = (turn.role, turn.policy_id)
        if route in seen:
            continue
        seen.add(route)
        routes.append(route)
    return tuple(routes)


def _as_pipelined_policy(policy: RolloutPolicy) -> TypeGuard[PipelinedRolloutPolicy]:
    return callable(getattr(policy, "submit_decode", None)) and callable(
        getattr(policy, "receive_decode", None)
    )


def _policy_recurrent_enabled(policy: RolloutPolicy) -> bool:
    """Return an explicit recurrent capability flag without guessing by failure."""
    value = getattr(policy, "recurrent_enabled", False)
    if callable(value):
        value = value()
    if not isinstance(value, bool):
        raise TypeError("rollout policy recurrent_enabled must be boolean")
    return value


def _as_local_recurrent_policy(
    policy: RolloutPolicy,
) -> TypeGuard[RecurrentRolloutPolicy]:
    """Return whether the policy exposes the complete local recurrent contract."""
    return bool(
        _policy_recurrent_enabled(policy)
        and callable(getattr(policy, "initial_recurrent_state", None))
        and callable(getattr(policy, "sample_decode_with_recurrent_state", None))
    )


def _as_pipelined_recurrent_policy(
    policy: RolloutPolicy,
) -> TypeGuard[PipelinedRecurrentRolloutPolicy]:
    """Return whether a remote policy exposes the recurrent wire contract."""
    return bool(
        _policy_recurrent_enabled(policy)
        and _as_pipelined_policy(policy)
        and callable(getattr(policy, "initial_recurrent_state", None))
        and callable(getattr(policy, "submit_recurrent_decode", None))
        and callable(getattr(policy, "receive_recurrent_decode", None))
        and callable(getattr(policy, "release_recurrent_sequences", None))
        and callable(getattr(policy, "abort_recurrent_sequences", None))
    )


def _sample_policy_decode(
    policy: RolloutPolicy,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    temperature: float,
    retain_planner_context: bool = False,
) -> _PolicyDecodeResult:
    improvement = getattr(policy, "behavior_kind", "policy_sample") == "improvement"
    request_sampler = getattr(policy, "sample_decode_with_trace_for_request", None)
    if callable(request_sampler) and not improvement:
        trace = cast(
            SampleDecodeTrace,
            request_sampler(
                states,
                options,
                decks,
                temperature=temperature,
                model_version_lease=None,
                retain_planner_context=retain_planner_context,
            ),
        )
        return _policy_decode_result_from_trace(trace)
    trace_sampler = getattr(policy, "sample_decode_with_trace", None)
    if callable(trace_sampler) and not improvement:
        trace = cast(
            SampleDecodeTrace,
            trace_sampler(states, options, decks, temperature=temperature),
        )
        return _policy_decode_result_from_trace(trace)
    actions, logprobs, values = policy.sample_decode(
        states,
        options,
        decks,
        temperature=temperature,
    )
    return _PolicyDecodeResult(
        actions=tuple(tuple(int(index) for index in action) for action in actions),
        action_logprobs=logprobs,
        values=values,
    )


def _sample_recurrent_policy_decode(
    policy: RecurrentRolloutPolicy,
    turns: Sequence[_PolicyTurn],
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    temperature: float,
    retain_planner_context: bool,
) -> _PolicyDecodeResult:
    """Prepare one recurrent transition and reuse it for the complete decode."""
    leases = tuple(turn.recurrent_lease for turn in turns)
    if any(lease is None for lease in leases):
        raise RuntimeError("recurrent rollout batch is missing an old-state lease")
    concrete_leases = cast(tuple[_RecurrentStateLease, ...], leases)
    artifact_fingerprint = _policy_artifact_fingerprint(policy)
    if any(
        lease.key.policy_artifact_fingerprint != artifact_fingerprint
        for lease in concrete_leases
    ):
        raise RuntimeError("recurrent rollout batch crossed a policy artifact")
    previous_state = concatenate_recurrent_policy_states(
        tuple(lease.state for lease in concrete_leases)
    )
    public_events = collate_public_event_deltas(
        tuple(turn.public_event_token.delta for turn in turns),
        device=states.card_ids.device,
    )
    trace, proposed_state = policy.sample_decode_with_recurrent_state(
        states,
        options,
        decks,
        public_events,
        previous_state,
        temperature=temperature,
        retain_planner_context=retain_planner_context,
    )
    if _policy_artifact_fingerprint(policy) != artifact_fingerprint:
        raise RuntimeError("policy artifact changed during recurrent decode")
    if proposed_state.batch_size != len(turns):
        raise RuntimeError("recurrent decode returned the wrong state batch size")
    result = _policy_decode_result_from_trace(trace)
    return replace(
        result,
        proposed_recurrent_states=tuple(
            select_recurrent_policy_state_rows(
                proposed_state,
                torch.tensor(
                    (index,),
                    dtype=torch.long,
                    device=proposed_state.hidden.device,
                ),
            ).detach()
            for index in range(len(turns))
        ),
        served_policy_artifact_fingerprint=artifact_fingerprint,
        served_policy_artifact=_policy_artifact_identity(policy),
    )


def _recurrent_inference_batch(
    turns: Sequence[_PolicyTurn],
    states: StateBatch,
) -> RecurrentInferenceBatch:
    """Build one ownership- and generation-bound remote recurrent request."""
    leases = tuple(turn.recurrent_lease for turn in turns)
    if any(lease is None for lease in leases):
        raise RuntimeError("remote recurrent batch is missing an old-state lease")
    concrete = cast(tuple[_RecurrentStateLease, ...], leases)
    artifacts = tuple(lease.artifact for lease in concrete)
    expected_artifact = artifacts[0]
    if any(artifact != expected_artifact for artifact in artifacts[1:]):
        raise RuntimeError("remote recurrent batch crossed artifact leases")
    return RecurrentInferenceBatch(
        public_events=collate_public_event_deltas(
            tuple(turn.public_event_token.delta for turn in turns),
            device=states.card_ids.device,
        ),
        previous_state=concatenate_recurrent_policy_states(
            tuple(lease.state for lease in concrete)
        ),
        event_generations=tuple(lease.generation for lease in concrete),
        sequences=tuple(
            RecurrentSequenceIdentity(
                game_id=turn.game.game_id,
                seat=turn.seat,
                exact_deck_signature=canonicalize_deck(
                    turn.game.deck_pair[turn.seat]
                ).signature,
            )
            for turn in turns
        ),
        expected_artifact=expected_artifact,
    )


def _policy_decode_result_from_recurrent_response(
    request: RecurrentInferenceBatch,
    response: RecurrentDecodeResult,
) -> _PolicyDecodeResult:
    """Validate response ownership before exposing actions or proposed state."""
    if response.event_generations != request.event_generations:
        raise RuntimeError("recurrent response changed event generations")
    if response.sequences != request.sequences:
        raise RuntimeError("recurrent response changed sequence ownership")
    if request.expected_artifact is not None:
        validate_policy_artifact_identity(
            request.expected_artifact,
            response.served_artifact,
        )
    if (
        response.trace.served_model_fingerprint
        != response.served_artifact.model_fingerprint
    ):
        raise RuntimeError("recurrent trace differs from its served artifact")
    result = _policy_decode_result_from_trace(response.trace)
    return replace(
        result,
        proposed_recurrent_states=tuple(
            select_recurrent_policy_state_rows(
                response.proposed_state,
                torch.tensor(
                    (index,),
                    dtype=torch.long,
                    device=response.proposed_state.hidden.device,
                ),
            ).detach()
            for index in range(request.batch_size)
        ),
        served_policy_artifact_fingerprint=response.served_artifact.fingerprint,
        served_policy_artifact=response.served_artifact,
    )


def _receive_policy_decode(
    policy: PipelinedRolloutPolicy,
    handle: object,
) -> _PolicyDecodeResult:
    trace_receiver = (
        None
        if getattr(policy, "behavior_kind", "policy_sample") == "improvement"
        else getattr(policy, "receive_decode_with_trace", None)
    )
    if callable(trace_receiver):
        trace = cast(SampleDecodeTrace, trace_receiver(handle))
        return _policy_decode_result_from_trace(trace)
    actions, logprobs, values = policy.receive_decode(handle)
    return _PolicyDecodeResult(
        actions=tuple(tuple(int(index) for index in action) for action in actions),
        action_logprobs=logprobs,
        values=values,
    )


def _submit_policy_decode(
    policy: PipelinedRolloutPolicy,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    temperature: float,
    retain_planner_context: bool,
) -> object:
    """Submit one root decode with explicit planner-context ownership."""
    request_submitter = getattr(policy, "submit_decode_for_request", None)
    if callable(request_submitter):
        return request_submitter(
            states,
            options,
            decks,
            temperature=temperature,
            retain_planner_context=retain_planner_context,
        )
    return policy.submit_decode(
        states,
        options,
        decks,
        temperature=temperature,
    )


def _policy_decode_result_from_trace(trace: SampleDecodeTrace) -> _PolicyDecodeResult:
    return _PolicyDecodeResult(
        actions=trace.actions,
        action_logprobs=trace.action_logprobs,
        values=trace.values,
        token_logprobs=trace.token_logprobs,
        prefix_values=trace.prefix_values,
        token_mask=trace.token_mask,
        stop_sampled=trace.stop_sampled,
        planner_context_handles=trace.planner_context_handles,
        served_policy_version=trace.served_policy_version,
        served_model_fingerprint=trace.served_model_fingerprint,
        served_proposal_version=trace.served_proposal_version,
        planner_fallback_reason=trace.planner_fallback_reason,
    )


def _validate_token_trace_batch(
    result: _PolicyDecodeResult,
    *,
    batch_size: int,
) -> None:
    fields = (
        result.token_logprobs,
        result.prefix_values,
        result.token_mask,
        result.stop_sampled,
    )
    if all(field is None for field in fields):
        return
    if any(field is None for field in fields):
        raise RuntimeError("sample_decode returned an incomplete token trace")
    token_logprobs = cast(Tensor, result.token_logprobs)
    prefix_values = cast(Tensor, result.prefix_values)
    token_mask = cast(Tensor, result.token_mask)
    stop_sampled = cast(Tensor, result.stop_sampled)
    if (
        token_logprobs.ndim != 2
        or prefix_values.shape != token_logprobs.shape
        or token_mask.shape != token_logprobs.shape
        or int(token_logprobs.shape[0]) != batch_size
    ):
        raise RuntimeError("sample_decode returned misaligned token trace tensors")
    if stop_sampled.ndim != 1 or int(stop_sampled.shape[0]) != batch_size:
        raise RuntimeError("sample_decode returned the wrong STOP trace batch size")
    if token_mask.dtype != torch.bool or stop_sampled.dtype != torch.bool:
        raise RuntimeError("sample_decode token masks must be boolean")


def _active_token_trace(
    result: _PolicyDecodeResult,
    *,
    index: int,
) -> tuple[tuple[float, ...], tuple[float, ...], bool] | None:
    if result.token_logprobs is None:
        return None
    token_logprobs = result.token_logprobs
    prefix_values = cast(Tensor, result.prefix_values)
    token_mask = cast(Tensor, result.token_mask)
    stop_sampled = cast(Tensor, result.stop_sampled)
    mask = token_mask[index]
    active_logprobs = _tensor_values(token_logprobs[index].masked_select(mask))
    active_prefix_values = _tensor_values(prefix_values[index].masked_select(mask))
    if not active_logprobs:
        raise RuntimeError("sample_decode returned an empty active token trace")
    return (
        tuple(active_logprobs),
        tuple(active_prefix_values),
        bool(stop_sampled[index].item()),
    )


def _policy_turn_batches(
    group: Sequence[_PolicyTurn],
    *,
    pipelined: bool,
    max_inflight_batch_decisions: int = 0,
    partition_planner_fast_path: bool = False,
    partition_recurrent_artifact: bool = False,
) -> tuple[tuple[_PolicyTurn, ...], ...]:
    if max_inflight_batch_decisions < 0:
        raise ValueError("max_inflight_batch_decisions must be non-negative")
    artifact_batches = (
        _partition_recurrent_artifact_leases(group)
        if partition_recurrent_artifact
        else (tuple(group),)
    )
    semantic_batches: tuple[tuple[_PolicyTurn, ...], ...]
    if partition_planner_fast_path:
        semantic_batches = tuple(
            batch
            for artifact_batch in artifact_batches
            for batch in (
                tuple(
                    turn
                    for turn in artifact_batch
                    if not _turn_requires_planner_context(turn)
                ),
                tuple(
                    turn
                    for turn in artifact_batch
                    if _turn_requires_planner_context(turn)
                ),
            )
            if batch
        )
    else:
        semantic_batches = artifact_batches
    if not pipelined or max_inflight_batch_decisions == 0:
        return semantic_batches
    return tuple(
        tuple(batch[offset : offset + max_inflight_batch_decisions])
        for batch in semantic_batches
        for offset in range(0, len(batch), max_inflight_batch_decisions)
    )


def _partition_recurrent_artifact_leases(
    group: Sequence[_PolicyTurn],
) -> tuple[tuple[_PolicyTurn, ...], ...]:
    """Keep each remote request on one bound artifact or first-bind cohort."""
    partitions: list[tuple[PolicyArtifactIdentity | None, list[_PolicyTurn]]] = []
    for turn in group:
        lease = turn.recurrent_lease
        if lease is None:
            raise RuntimeError("recurrent rollout turn is missing its state lease")
        for artifact, rows in partitions:
            if artifact == lease.artifact:
                rows.append(turn)
                break
        else:
            partitions.append((lease.artifact, [turn]))
    return tuple(tuple(rows) for _artifact, rows in partitions)


def _turn_requires_planner_context(turn: _PolicyTurn) -> bool:
    """Keep exact-single rows out of retained-context planner work.

    Fixed single-select prompts with multiple legal options deliberately return
    true: those shapes feed the reusable engine-equivalence probe.
    """
    select = turn.observation.get("select")
    return describe_prompt_action_space(select).legal_action_count > 1


def _make_probe_backend(probe_config: RolloutProbeConfig) -> Any | None:
    if not probe_config.enabled:
        return None
    if probe_config.backend in {"auto", "native"}:
        raise RuntimeError(
            "per-decision native probe was removed; use batched engine facts "
            "or explicitly select the diagnostic search_api backend"
        )
    if probe_config.backend == "search_api":
        return SearchApiProbeBackend(manual_coin=probe_config.manual_coin)
    raise ValueError(f"unsupported rollout probe backend: {probe_config.backend}")


def _summarize_live_game_steps(
    games: Sequence[VectorGame],
) -> dict[str, Any]:
    """Summarize one bounded snapshot without retaining per-game history."""
    ordered_games = sorted(
        ((max(0, int(game.steps)), str(game.game_id)) for game in games),
        key=lambda row: (row[0], row[1]),
    )
    if not ordered_games:
        return {
            "live_game_count": 0,
            "live_game_steps_min": 0,
            "live_game_steps_mean": 0.0,
            "live_game_steps_p50": 0,
            "live_game_steps_p90": 0,
            "live_game_steps_p99": 0,
            "live_game_steps_max": 0,
            "oldest_live_games": [],
        }
    ordered_steps = tuple(steps for steps, _game_id in ordered_games)
    oldest = sorted(
        ordered_games,
        key=lambda row: (-row[0], row[1]),
    )[:_OLDEST_LIVE_GAMES_LIMIT]
    return {
        "live_game_count": len(ordered_steps),
        "live_game_steps_min": ordered_steps[0],
        "live_game_steps_mean": sum(ordered_steps) / len(ordered_steps),
        "live_game_steps_p50": _live_game_step_percentile(ordered_steps, 0.50),
        "live_game_steps_p90": _live_game_step_percentile(ordered_steps, 0.90),
        "live_game_steps_p99": _live_game_step_percentile(ordered_steps, 0.99),
        "live_game_steps_max": ordered_steps[-1],
        "oldest_live_games": [
            {"game_id": game_id, "steps": steps} for steps, game_id in oldest
        ],
    }


def _live_game_step_percentile(
    ordered_steps: Sequence[int],
    quantile: float,
) -> int:
    """Return a nearest-rank percentile from a non-empty ordered sequence."""
    index = min(
        len(ordered_steps) - 1,
        max(0, math.ceil(quantile * len(ordered_steps)) - 1),
    )
    return int(ordered_steps[index])


class RolloutStepper:
    """Drive one ``VectorBattlePool`` with batched policy inference."""

    def __init__(
        self,
        *,
        pool: VectorPoolLike,
        actors: RolloutActors,
        recorder: RolloutRecorder | None = None,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        timer: StageTimer | None = None,
        probe_config: RolloutProbeConfig | None = None,
        belief_config: RolloutBeliefConfig | None = None,
        factual_config: FactualLaneConfig | None = None,
        macro_credit_config: MacroCreditConfig | None = None,
        engine_teacher_producer: EngineTeacherProducer | None = None,
        planner_behavior_service: RolloutPlannerBehaviorService | None = None,
        seed: int = 0,
        record_policy_decisions: bool = True,
        record_public_event_deltas: bool = False,
        reanalysis_root_probability: float = 0.0,
    ) -> None:
        """Initialize rollout state."""
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if (
            not math.isfinite(reanalysis_root_probability)
            or reanalysis_root_probability < 0.0
            or reanalysis_root_probability > 1.0
        ):
            raise ValueError("reanalysis_root_probability must be in [0, 1]")
        self._pool = pool
        self._actors = actors
        self._recorder = recorder if recorder is not None else _NoopRolloutRecorder()
        self._temperature = float(temperature)
        self._device = device
        self._timer = timer
        self._record_policy_decisions = bool(record_policy_decisions)
        self._record_public_event_deltas = bool(record_public_event_deltas)
        self._reanalysis_root_probability = float(reanalysis_root_probability)
        self._reanalysis_rng = random.Random(seed + 7_919)
        self._factual_config = factual_config or FactualLaneConfig()
        self._macro_credit_config = macro_credit_config or MacroCreditConfig()
        if self._macro_credit_config.enabled and not self._factual_config.enabled:
            raise ValueError("macro credit requires decision-local factual evidence")
        self._macro_teacher_enabled = bool(
            self._macro_credit_config.native_teacher_enabled
        )
        self._macro_teacher_sample_eligible = 0
        self._macro_teacher_sampled = 0
        self._pending_factual: dict[str, _PendingFactualDecision] = {}
        self._factual_started = 0
        self._factual_completed = 0
        self._factual_transition_steps = 0
        self._factual_actor_relation_counts: Counter[str] = Counter()
        self._factual_next_context_counts: Counter[int] = Counter()
        self._recorded_policy_evidence = 0
        if planner_behavior_service is not None and engine_teacher_producer is not None:
            raise ValueError(
                "pre-action planner behavior cannot run with legacy engine teacher"
            )
        if self._macro_teacher_enabled and planner_behavior_service is not None:
            raise ValueError(
                "post-behavior macro teaching cannot change planner behavior"
            )
        if self._macro_teacher_enabled and engine_teacher_producer is None:
            raise ValueError("native macro teaching requires an async producer")
        self._planner_behavior_service = planner_behavior_service
        self._planner_telemetry = (
            None
            if planner_behavior_service is None
            else PlannerTelemetryAccumulator(latency_window=4_096)
        )
        macro_tensorizer = self._macro_credit_config.root_information_tensorizer
        self._planner_belief_summary_width = int(
            macro_tensorizer.belief_summary_dim
            if macro_tensorizer is not None
            else getattr(planner_behavior_service, "belief_summary_width", 0)
        )
        if self._planner_belief_summary_width < 0:
            raise ValueError("planner belief summary width must be non-negative")
        self._engine_teacher_producer = engine_teacher_producer
        self._engine_teacher_requests = 0
        self._engine_teacher_batches = 0
        self._engine_teacher_targets = 0
        self._engine_teacher_search_targets = 0
        self._engine_teacher_behavior_matches = 0
        self._engine_teacher_confidence_sum = 0.0
        self._engine_teacher_weight_sum = 0.0
        self._pending_teacher_batches: dict[
            int,
            tuple[_FrozenPolicyDecision, ...],
        ] = {}
        self._pending_teacher_games: Counter[str] = Counter()
        self._next_decision_sequence = 0
        self._pending_decision_evidence: dict[int, _PendingDecisionEvidence] = {}
        self._pending_record_order: dict[str, deque[int]] = {}
        self._pending_record_games: Counter[str] = Counter()
        self._deferred_finished: dict[str, FinishedGame] = {}
        self._contexts: dict[tuple[str, int], GameContext] = {}
        self._recurrent_states: dict[
            _RecurrentStateKey,
            _CommittedRecurrentState,
        ] = {}
        self._recurrent_key_by_seat: dict[
            tuple[str, int],
            _RecurrentStateKey,
        ] = {}
        self._recurrent_state_initializations = 0
        self._recurrent_prepares = 0
        self._recurrent_commits = 0
        self._recurrent_drain_game_ids: frozenset[str] | None = None
        self._scripted_game_agents: dict[str, BattleAgent] = {}
        self._live_game_step_summary = _summarize_live_game_steps(())
        self._max_live_game_steps_seen = 0
        self._stale_recurrent_recycle_polls = 0
        self._stale_recurrent_candidate_sequences_examined = 0
        self._stale_recurrent_games_recycled = 0
        self._stale_recurrent_games_deferred_pending_evidence = 0
        self._stale_recurrent_sequences_released = 0
        self._stale_recurrent_last_learner_version = -1
        self._stale_recurrent_last_served_version = -1
        self._stale_recurrent_oldest_candidate_policy_version = -1
        self._stale_recurrent_max_candidate_version_age = 0
        self._feature_augmenter = _RolloutFeatureAugmenter(
            probe_config=probe_config or RolloutProbeConfig(),
            belief_config=belief_config or RolloutBeliefConfig(),
            seed=seed,
        )

    @property
    def context_count(self) -> int:
        """Return active per-game, per-seat contexts."""
        return len(self._contexts)

    @property
    def recurrent_state_count(self) -> int:
        """Return active game-seat-policy recurrent sequences."""
        return len(self._recurrent_states)

    @property
    def has_active_recurrent_sequences(self) -> bool:
        """Return whether an in-place local weight load needs a game barrier."""
        return bool(self._recurrent_states)

    @property
    def recurrent_drain_active(self) -> bool:
        """Return whether rollout is draining a fixed pre-publication cohort."""
        return self._recurrent_drain_game_ids is not None

    def begin_recurrent_drain(self) -> None:
        """Freeze the currently bound games and stop admitting refill games."""
        if self._recurrent_drain_game_ids is not None:
            raise RuntimeError("recurrent rollout drain is already active")
        game_ids = frozenset(key.game_id for key in self._recurrent_states)
        if not game_ids:
            raise RuntimeError("cannot drain an empty recurrent rollout cohort")
        self._recurrent_drain_game_ids = game_ids

    def finish_recurrent_drain(self) -> None:
        """Reopen admission only after every bound recurrent sequence ended."""
        if self._recurrent_drain_game_ids is None:
            raise RuntimeError("recurrent rollout drain is not active")
        if self._recurrent_states:
            raise RuntimeError("cannot finish a live recurrent rollout drain")
        self._recurrent_drain_game_ids = None

    def close(self) -> None:
        """Release all actor-local game state at shutdown or fatal abort."""
        release_errors: list[Exception] = []
        release_groups: dict[
            tuple[int, str],
            tuple[
                PipelinedRecurrentRolloutPolicy,
                PolicyArtifactIdentity | None,
                list[RecurrentSequenceIdentity],
            ],
        ] = {}
        for recurrent_key, committed in tuple(self._recurrent_states.items()):
            try:
                policy = self._policy_for_route(
                    recurrent_key.policy_role,
                    recurrent_key.policy_id,
                )
                remote_policy = (
                    policy if _as_pipelined_recurrent_policy(policy) else None
                )
                if remote_policy is None:
                    continue
                artifact = committed.artifact
                artifact_key = "" if artifact is None else artifact.fingerprint
                group_key = (id(remote_policy), artifact_key)
                group = release_groups.get(group_key)
                if group is None:
                    group = (remote_policy, artifact, [])
                    release_groups[group_key] = group
                elif group[1] != artifact:
                    raise RuntimeError("recurrent cleanup artifact key collision")
                group[2].append(
                    RecurrentSequenceIdentity(
                        game_id=recurrent_key.game_id,
                        seat=recurrent_key.seat,
                        exact_deck_signature=(recurrent_key.exact_deck_signature),
                    )
                )
            except Exception as exc:
                release_errors.append(exc)

        try:
            # One request per remote policy/artifact bounds shutdown latency by
            # route count rather than by every live game-seat sequence.
            for remote_policy, artifact, sequences in release_groups.values():
                try:
                    frozen = tuple(sequences)
                    released = (
                        remote_policy.abort_recurrent_sequences(frozen)
                        if artifact is None
                        else remote_policy.release_recurrent_sequences(
                            frozen,
                            artifact,
                        )
                    )
                    if not 0 <= released <= len(frozen):
                        raise RuntimeError(
                            "remote recurrent release returned an invalid "
                            "sequence count"
                        )
                except Exception as exc:  # Preserve cleanup for other routes.
                    release_errors.append(exc)
        finally:
            # Actor-local state is dead after close even when the remote service
            # is unavailable. Retaining it only permits duplicate cleanup calls
            # to mask the original actor failure.
            self._recurrent_states.clear()
            self._recurrent_key_by_seat.clear()
            self._contexts.clear()
            self._recurrent_drain_game_ids = None
            self._scripted_game_agents.clear()
        if release_errors:
            raise RuntimeError(
                "failed to release one or more remote recurrent sequences"
            ) from release_errors[0]

    @property
    def feature_summary(self) -> Mapping[str, Any]:
        """Return feature augmentation counters."""
        summary = dict(self._feature_augmenter.summary())
        if self._planner_telemetry is not None:
            summary["planner_telemetry"] = asdict(self._planner_telemetry.summary())
        service_stats = getattr(self._planner_behavior_service, "stats", None)
        if callable(service_stats):
            summary["planner_service"] = asdict(service_stats())
        producer_summary = getattr(self._engine_teacher_producer, "summary", None)
        if callable(producer_summary):
            summary.update(dict(producer_summary()))
        targets = self._engine_teacher_targets
        requests = self._engine_teacher_requests
        summary.update(
            {
                "factual_enabled": self._factual_config.enabled,
                "factual_started": self._factual_started,
                "factual_completed": self._factual_completed,
                "factual_coverage": (
                    self._factual_completed / self._factual_started
                    if self._factual_started
                    else 0.0
                ),
                "factual_closed_rate": (
                    self._factual_completed / self._factual_started
                    if self._factual_started
                    else 0.0
                ),
                "factual_transition_steps": self._factual_transition_steps,
                "factual_mean_transition_steps": (
                    self._factual_transition_steps / self._factual_completed
                    if self._factual_completed
                    else 0.0
                ),
                "factual_pending_decisions": len(self._pending_factual),
                "factual_actor_relation_counts": dict(
                    sorted(self._factual_actor_relation_counts.items())
                ),
                "factual_next_context_counts": dict(
                    sorted(self._factual_next_context_counts.items())
                ),
                "engine_teacher_enabled": self._engine_teacher_producer is not None,
                "macro_teacher_sample_eligible": (self._macro_teacher_sample_eligible),
                "macro_teacher_sampled": self._macro_teacher_sampled,
                "macro_teacher_sample_rate": (
                    self._macro_teacher_sampled / self._macro_teacher_sample_eligible
                    if self._macro_teacher_sample_eligible
                    else 0.0
                ),
                "engine_teacher_requests": requests,
                "engine_teacher_batches": self._engine_teacher_batches,
                "engine_teacher_targets": targets,
                "engine_teacher_target_rate": (targets / requests if requests else 0.0),
                "engine_teacher_search_targets": self._engine_teacher_search_targets,
                "engine_teacher_search_target_rate": (
                    self._engine_teacher_search_targets / targets if targets else 0.0
                ),
                "engine_teacher_behavior_matches": (
                    self._engine_teacher_behavior_matches
                ),
                "engine_teacher_behavior_match_rate": (
                    self._engine_teacher_behavior_matches / targets if targets else 0.0
                ),
                "engine_teacher_mean_confidence": (
                    self._engine_teacher_confidence_sum / targets if targets else 0.0
                ),
                "engine_teacher_mean_weight": (
                    self._engine_teacher_weight_sum / targets if targets else 0.0
                ),
                "engine_teacher_pending_batches": len(self._pending_teacher_batches),
                "engine_teacher_pending_decisions": sum(
                    self._pending_teacher_games.values()
                ),
                "engine_teacher_deferred_finished_games": len(self._deferred_finished),
                "recurrent_active_sequences": self.recurrent_state_count,
                "recurrent_state_initializations": (
                    self._recurrent_state_initializations
                ),
                "recurrent_prepares": self._recurrent_prepares,
                "recurrent_commits": self._recurrent_commits,
                "recurrent_drain_active": self.recurrent_drain_active,
                "recurrent_drain_games": (
                    0
                    if self._recurrent_drain_game_ids is None
                    else len(self._recurrent_drain_game_ids)
                ),
                "stale_recurrent_recycle_polls": (self._stale_recurrent_recycle_polls),
                "stale_recurrent_candidate_sequences_examined": (
                    self._stale_recurrent_candidate_sequences_examined
                ),
                "stale_recurrent_games_recycled": (
                    self._stale_recurrent_games_recycled
                ),
                "stale_recurrent_games_deferred_pending_evidence": (
                    self._stale_recurrent_games_deferred_pending_evidence
                ),
                "stale_recurrent_sequences_released": (
                    self._stale_recurrent_sequences_released
                ),
                "stale_recurrent_last_learner_version": (
                    self._stale_recurrent_last_learner_version
                ),
                "stale_recurrent_last_served_version": (
                    self._stale_recurrent_last_served_version
                ),
                "stale_recurrent_oldest_candidate_policy_version": (
                    self._stale_recurrent_oldest_candidate_policy_version
                ),
                "stale_recurrent_max_candidate_version_age": (
                    self._stale_recurrent_max_candidate_version_age
                ),
            }
        )
        summary.update(self._live_game_step_summary)
        summary["max_live_game_steps_seen"] = self._max_live_game_steps_seen
        return summary

    def step(self) -> RolloutStepStats:
        """Advance every currently pending game by at most one select."""
        self._collect_engine_teacher_results()
        with time_stage(self._timer, "rollout_pending"):
            pending = self._pool.pending()
        self._observe_live_game_steps(pending)
        if self._recurrent_drain_game_ids is not None:
            pending = [
                game
                for game in pending
                if game.game_id in self._recurrent_drain_game_ids
            ]
        forced_count = 0
        scripted_count = 0
        policy_turns: list[_PolicyTurn] = []

        for game in pending:
            seat = _player_index(game.observation)
            if seat not in (0, 1):
                raise ValueError(f"invalid rollout seat: {seat}")
            with time_stage(self._timer, "rollout_context"):
                context_observation, context_features = self._observation_with_context(
                    game,
                    seat,
                )
            select = _select(context_observation)
            with time_stage(self._timer, "rollout_forced_check"):
                forced = forced_action(select)
            route = self._policy_route_for_seat(game.game_id, seat)
            context = self._context_for(game, seat)
            if forced is not None:
                self._ensure_recurrent_state(
                    game,
                    seat,
                    route,
                    context=context,
                )
                with time_stage(self._timer, "rollout_submit"):
                    self._submit_checked(game, select, forced)
                forced_count += 1
                continue

            if route.role is None:
                self._reject_live_recurrent_route_change(game.game_id, seat)
                scripted_count += 1
                with time_stage(self._timer, "rollout_scripted"):
                    self._submit_scripted(game, context_observation)
                continue
            public_event_token = context.prepare_decision()
            policy_turns.append(
                _PolicyTurn(
                    game=game,
                    seat=seat,
                    role=route.role,
                    policy_id=route.policy_id,
                    should_record=self._should_record(game.game_id, seat),
                    observation=context_observation,
                    context_features=context_features,
                    public_event_token=public_event_token,
                    recurrent_lease=self._recurrent_lease_for(
                        game,
                        seat,
                        route,
                        context=context,
                        token=public_event_token,
                    ),
                )
            )

        policy_count, recorded_count = self._submit_policy_turns(policy_turns)
        with time_stage(self._timer, "rollout_finished"):
            finished = self._pool.finished()
        if finished:
            self._max_live_game_steps_seen = max(
                self._max_live_game_steps_seen,
                *(max(0, int(done.steps)) for done in finished),
            )
        for done in finished:
            self._close_finished_factual(done)
            self._discard_contexts(done.game_id)
            self._scripted_game_agents.pop(done.game_id, None)
            if self._pending_factual.get(done.game_id):
                raise RuntimeError(
                    "finished rollout game has incomplete factual decisions: "
                    f"{done.game_id}"
                )
            self._pending_factual.pop(done.game_id, None)
            if self._pending_record_games.get(done.game_id, 0) > 0:
                if done.game_id in self._deferred_finished:
                    raise RuntimeError("rollout game finished more than once")
                self._deferred_finished[done.game_id] = done
            else:
                self._finalize_finished_game(done)
        return RolloutStepStats(
            pending_games=len(pending),
            forced_actions=forced_count,
            scripted_actions=scripted_count,
            policy_actions=policy_count,
            recorded_decisions=recorded_count,
            finished_games=len(finished),
        )

    def recycle_stale_recurrent_games(
        self,
        *,
        learner_latest_version: int | None,
        inference_served_version: int | None,
        max_staleness: int,
        lifecycle_discard_callback: (
            Callable[[Sequence[str], str], None] | None
        ) = None,
    ) -> RecurrentGameRecycleResult:
        """Discard stale unfinished games at a complete-step boundary."""
        if learner_latest_version is not None and learner_latest_version < 0:
            raise ValueError("recurrent recycling versions must be non-negative")
        if inference_served_version is not None and inference_served_version < 0:
            raise ValueError("recurrent recycling versions must be non-negative")
        if max_staleness < 0:
            raise ValueError("recurrent recycling max_staleness must be non-negative")
        self._stale_recurrent_recycle_polls += 1
        self._stale_recurrent_last_learner_version = (
            -1 if learner_latest_version is None else learner_latest_version
        )
        self._stale_recurrent_last_served_version = (
            -1 if inference_served_version is None else inference_served_version
        )
        if learner_latest_version is None or inference_served_version is None:
            return RecurrentGameRecycleResult()

        candidate_versions_by_game: dict[str, list[int]] = {}
        for key, committed in self._recurrent_states.items():
            if key.policy_role != "candidate" or committed.policy_version is None:
                continue
            candidate_versions_by_game.setdefault(key.game_id, []).append(
                committed.policy_version
            )
        candidate_versions = tuple(
            version
            for versions in candidate_versions_by_game.values()
            for version in versions
        )
        self._stale_recurrent_candidate_sequences_examined += len(candidate_versions)
        if candidate_versions:
            oldest_version = min(candidate_versions)
            max_age = max(0, learner_latest_version - oldest_version)
            self._stale_recurrent_oldest_candidate_policy_version = oldest_version
            self._stale_recurrent_max_candidate_version_age = max(
                self._stale_recurrent_max_candidate_version_age,
                max_age,
            )
        else:
            max_age = 0
            self._stale_recurrent_oldest_candidate_policy_version = -1

        stale_game_ids = tuple(
            sorted(
                game_id
                for game_id, versions in candidate_versions_by_game.items()
                if any(
                    learner_latest_version - version > max_staleness
                    and inference_served_version > version
                    for version in versions
                )
            )
        )
        deferred_game_ids = tuple(
            game_id
            for game_id in stale_game_ids
            if self._game_has_pending_evidence(game_id)
        )
        deferred_set = set(deferred_game_ids)
        recycled_game_ids = tuple(
            game_id for game_id in stale_game_ids if game_id not in deferred_set
        )
        self._stale_recurrent_games_deferred_pending_evidence += len(deferred_game_ids)
        if not recycled_game_ids:
            return RecurrentGameRecycleResult(
                deferred_game_ids=deferred_game_ids,
                max_candidate_version_age=max_age,
            )

        pending_game_ids = {game.game_id for game in self._pool.pending()}
        missing = [
            game_id for game_id in recycled_game_ids if game_id not in pending_game_ids
        ]
        if missing:
            raise RuntimeError(
                "stale recurrent games are no longer pending: " + ", ".join(missing)
            )

        discard_many = getattr(self._recorder, "discard_many", None)
        if callable(discard_many):
            discard_many(
                recycled_game_ids,
                reason="stale_recurrent_policy",
            )
        else:
            for game_id in recycled_game_ids:
                self._recorder.discard(
                    game_id,
                    reason="stale_recurrent_policy",
                )

        released_sequences = 0
        for game_id in recycled_game_ids:
            released_sequences += self._discard_contexts(game_id)
            self._scripted_game_agents.pop(game_id, None)
        if lifecycle_discard_callback is not None:
            lifecycle_discard_callback(
                recycled_game_ids,
                "stale_recurrent_policy",
            )
        for game_id in recycled_game_ids:
            self._pool.recycle(game_id)

        self._stale_recurrent_games_recycled += len(recycled_game_ids)
        self._stale_recurrent_sequences_released += released_sequences
        return RecurrentGameRecycleResult(
            recycled_game_ids=recycled_game_ids,
            deferred_game_ids=deferred_game_ids,
            released_sequences=released_sequences,
            max_candidate_version_age=max_age,
        )

    def _game_has_pending_evidence(self, game_id: str) -> bool:
        """Return whether asynchronous evidence still owns this game."""
        return bool(
            self._pending_record_games.get(game_id, 0)
            or self._pending_teacher_games.get(game_id, 0)
            or game_id in self._pending_factual
            or game_id in self._deferred_finished
        )

    def _observe_live_game_steps(self, games: Sequence[VectorGame]) -> None:
        """Capture one bounded, engine-call-free live-game step snapshot."""
        self._live_game_step_summary = _summarize_live_game_steps(games)
        self._max_live_game_steps_seen = max(
            self._max_live_game_steps_seen,
            int(self._live_game_step_summary["live_game_steps_max"]),
        )

    def flush_engine_teacher(self) -> None:
        """Attach all accepted asynchronous targets at an orderly boundary."""
        producer = self._async_engine_teacher_producer()
        if producer is not None:
            self._consume_engine_teacher_completions(producer.drain())
        if self._pending_teacher_batches:
            raise RuntimeError("engine teacher drain left pending batches")
        unresolved_teacher = sum(
            not evidence.teacher_resolved
            for evidence in self._pending_decision_evidence.values()
        )
        if unresolved_teacher:
            raise RuntimeError(
                "engine teacher drain left unresolved decision evidence: "
                f"{unresolved_teacher}"
            )
        if self._deferred_finished:
            unresolved = ", ".join(sorted(self._deferred_finished))
            raise RuntimeError(
                f"engine teacher drain left unfinished trajectories: {unresolved}"
            )

    def run_until(
        self,
        *,
        total_finished_games: int | None = None,
        total_recorded_decisions: int | None = None,
        max_iterations: int = 100_000,
    ) -> RolloutRunSummary:
        """Run rollout steps until one configured budget is reached."""
        if total_finished_games is None and total_recorded_decisions is None:
            raise ValueError("at least one rollout budget must be set")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        iterations = 0
        forced_actions = 0
        scripted_actions = 0
        policy_actions = 0
        recorded_decisions = 0
        finished_games = 0
        while iterations < max_iterations:
            if (
                total_finished_games is not None
                and finished_games >= total_finished_games
            ):
                break
            if (
                total_recorded_decisions is not None
                and recorded_decisions >= total_recorded_decisions
            ):
                break
            stats = self.step()
            iterations += 1
            forced_actions += stats.forced_actions
            scripted_actions += stats.scripted_actions
            policy_actions += stats.policy_actions
            recorded_decisions += stats.recorded_decisions
            finished_games += stats.finished_games
            if stats.pending_games == 0 and stats.finished_games == 0:
                break
        return RolloutRunSummary(
            iterations=iterations,
            forced_actions=forced_actions,
            scripted_actions=scripted_actions,
            policy_actions=policy_actions,
            recorded_decisions=recorded_decisions,
            finished_games=finished_games,
        )

    def _submit_policy_turns(
        self,
        turns: Sequence[_PolicyTurn],
    ) -> tuple[int, int]:
        turn_order = {id(turn): index for index, turn in enumerate(turns)}
        frozen_decisions: list[_FrozenPolicyDecision] = []
        pending_batches: list[_PendingPolicyBatch] = []
        for route in _policy_routes(turns):
            role, policy_id = route
            group = tuple(
                turn
                for turn in turns
                if turn.role == role and turn.policy_id == policy_id
            )
            if not group:
                continue
            policy = self._policy_for_route(role, policy_id)
            pipelined_policy = policy if _as_pipelined_policy(policy) else None
            pipelined_recurrent_policy = (
                policy if _as_pipelined_recurrent_policy(policy) else None
            )
            local_recurrent_policy = (
                policy
                if pipelined_policy is None and _as_local_recurrent_policy(policy)
                else None
            )
            if _policy_recurrent_enabled(policy):
                if pipelined_policy is not None and pipelined_recurrent_policy is None:
                    raise RuntimeError(
                        "recurrent pipelined rollout policy lacks its recurrent "
                        "transport contract"
                    )
                if pipelined_policy is None and local_recurrent_policy is None:
                    raise RuntimeError(
                        "recurrent rollout policy lacks the local recurrent contract"
                    )
            elif any(turn.recurrent_lease is not None for turn in group):
                raise RuntimeError("stateless policy received a recurrent state lease")
            for batch_group in _policy_turn_batches(
                group,
                pipelined=pipelined_policy is not None,
                max_inflight_batch_decisions=(
                    0
                    if pipelined_policy is None
                    else int(
                        getattr(
                            pipelined_policy,
                            "preferred_inflight_batch_decisions",
                            0,
                        )
                    )
                ),
                partition_planner_fast_path=(
                    (
                        self._planner_behavior_service is not None
                        or self._macro_teacher_enabled
                    )
                    and role == "candidate"
                ),
                partition_recurrent_artifact=(pipelined_recurrent_policy is not None),
            ):
                observations = self._policy_input_observations(batch_group)
                with time_stage(self._timer, "rollout_encode_collate"):
                    policy_inputs = _collate_policy_inputs(
                        observations,
                        device=self._device,
                    )
                macro_teacher_sample_mask = self._macro_teacher_sample_mask(
                    batch_group,
                    policy_inputs,
                )
                retain_planner_context = bool(
                    (
                        self._planner_behavior_service is not None
                        and role == "candidate"
                        and any(
                            _turn_requires_planner_context(turn) for turn in batch_group
                        )
                    )
                    or any(macro_teacher_sample_mask)
                )
                if pipelined_policy is not None:
                    recurrent_request = (
                        None
                        if pipelined_recurrent_policy is None
                        else _recurrent_inference_batch(
                            batch_group,
                            policy_inputs.states,
                        )
                    )
                    with time_stage(self._timer, "rollout_policy_send"):
                        if recurrent_request is None:
                            handle = _submit_policy_decode(
                                pipelined_policy,
                                policy_inputs.states,
                                policy_inputs.options,
                                policy_inputs.decks,
                                temperature=self._temperature,
                                retain_planner_context=retain_planner_context,
                            )
                        else:
                            if pipelined_recurrent_policy is None:
                                raise AssertionError(
                                    "remote recurrent policy contract is missing"
                                )
                            handle = pipelined_recurrent_policy.submit_recurrent_decode(
                                policy_inputs.states,
                                policy_inputs.options,
                                policy_inputs.decks,
                                recurrent_request,
                                temperature=self._temperature,
                                retain_planner_context=retain_planner_context,
                            )
                    self._recurrent_prepares += int(
                        recurrent_request is not None
                    ) * len(batch_group)
                    pending_batches.append(
                        _PendingPolicyBatch(
                            policy=pipelined_policy,
                            turns=batch_group,
                            observations=observations,
                            policy_inputs=policy_inputs,
                            handle=handle,
                            order_indices=tuple(
                                turn_order[id(turn)] for turn in batch_group
                            ),
                            macro_teacher_sample_mask=macro_teacher_sample_mask,
                            recurrent_request=recurrent_request,
                        )
                    )
                    continue
                with time_stage(self._timer, "rollout_policy_sample"):
                    result = (
                        _sample_policy_decode(
                            policy,
                            policy_inputs.states,
                            policy_inputs.options,
                            policy_inputs.decks,
                            temperature=self._temperature,
                            retain_planner_context=retain_planner_context,
                        )
                        if local_recurrent_policy is None
                        else _sample_recurrent_policy_decode(
                            local_recurrent_policy,
                            batch_group,
                            policy_inputs.states,
                            policy_inputs.options,
                            policy_inputs.decks,
                            temperature=self._temperature,
                            retain_planner_context=retain_planner_context,
                        )
                    )
                self._recurrent_prepares += int(
                    local_recurrent_policy is not None
                ) * len(batch_group)
                frozen_decisions.extend(
                    self._freeze_policy_results(
                        policy=policy,
                        group=batch_group,
                        policy_observations=observations,
                        policy_inputs=policy_inputs,
                        result=result,
                        order_indices=tuple(
                            turn_order[id(turn)] for turn in batch_group
                        ),
                        macro_teacher_sample_mask=macro_teacher_sample_mask,
                    )
                )
        for pending in pending_batches:
            with time_stage(self._timer, "rollout_policy_wait"):
                result = (
                    _receive_policy_decode(
                        pending.policy,
                        pending.handle,
                    )
                    if pending.recurrent_request is None
                    else _policy_decode_result_from_recurrent_response(
                        pending.recurrent_request,
                        cast(
                            PipelinedRecurrentRolloutPolicy,
                            pending.policy,
                        ).receive_recurrent_decode(pending.handle),
                    )
                )
            frozen_decisions.extend(
                self._freeze_policy_results(
                    policy=pending.policy,
                    group=pending.turns,
                    policy_observations=pending.observations,
                    policy_inputs=pending.policy_inputs,
                    result=result,
                    order_indices=pending.order_indices,
                    macro_teacher_sample_mask=pending.macro_teacher_sample_mask,
                )
            )
        frozen_decisions.sort(key=lambda decision: decision.order_index)
        submitted: list[_FrozenPolicyDecision] = []
        try:
            for raw_decision in frozen_decisions:
                decision = (
                    self._register_recorded_decision(raw_decision)
                    if raw_decision.turn.should_record
                    else raw_decision
                )
                if decision.turn.should_record and self._factual_config.enabled:
                    self._start_factual_decision(decision)
                receipt = _SubmissionReceipt()
                try:
                    with time_stage(self._timer, "rollout_submit"):
                        self._submit_checked(
                            decision.turn.game,
                            decision.select,
                            decision.action,
                            on_accepted=partial(
                                self._accept_policy_turn,
                                decision.turn,
                                decision.recurrent_transition,
                                receipt,
                            ),
                        )
                except Exception:
                    commit_complete = bool(
                        receipt.context_committed
                        and (
                            decision.recurrent_transition is None
                            or receipt.recurrent_committed
                        )
                    )
                    if commit_complete:
                        submitted.append(decision)
                    else:
                        self._discard_uncommitted_decision(decision)
                        if receipt.engine_accepted:
                            self._discard_contexts(decision.turn.game.game_id)
                    raise
                submitted.append(decision)
        except Exception:
            self._finish_policy_submissions(submitted)
            raise
        recorded_count = self._finish_policy_submissions(submitted)
        return (len(submitted), recorded_count)

    def _finish_policy_submissions(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
    ) -> int:
        """Resolve post-submit evidence for the accepted batch prefix."""
        recorded = tuple(
            decision for decision in decisions if decision.turn.should_record
        )
        if self._engine_teacher_producer is None:
            self._emit_ready_decisions(recorded)
        elif not self._submit_engine_teacher_async(recorded):
            targets = self._engine_teacher_targets_for(recorded)
            self._record_engine_teacher_results(recorded, targets)
        return len(recorded)

    def _discard_uncommitted_decision(
        self,
        decision: _FrozenPolicyDecision,
    ) -> None:
        """Remove bookkeeping for a decision whose context did not commit."""
        if self._factual_config.enabled and decision.turn.should_record:
            root = self._pending_factual.pop(decision.turn.game.game_id, None)
            if root is not None:
                if root.decision is not decision:
                    raise RuntimeError("unaccepted factual decision identity changed")
                self._factual_started -= 1
        if not decision.turn.should_record:
            return
        identity = self._decision_identity(decision)
        evidence = self._pending_decision_evidence.pop(identity.sequence_id, None)
        if evidence is None:
            raise RuntimeError("unaccepted rollout decision evidence is missing")
        pending_order = self._pending_record_order.get(identity.game_id)
        if pending_order is None or not pending_order:
            raise RuntimeError("unaccepted rollout decision order is missing")
        if pending_order.pop() != identity.sequence_id:
            raise RuntimeError("unaccepted rollout decision is not the newest row")
        if not pending_order:
            del self._pending_record_order[identity.game_id]
        pending_count = self._pending_record_games.get(identity.game_id, 0)
        if pending_count <= 0:
            raise RuntimeError("unaccepted rollout decision count is invalid")
        if pending_count == 1:
            del self._pending_record_games[identity.game_id]
        else:
            self._pending_record_games[identity.game_id] = pending_count - 1

    def _policy_input_observations(
        self,
        turns: Sequence[_PolicyTurn],
    ) -> tuple[_PolicyObservation, ...]:
        if not self._feature_augmenter.enabled:
            return tuple(
                _PolicyObservation(
                    turn.observation,
                    turn.context_features,
                    _own_deck_for_turn(turn),
                )
                for turn in turns
            )
        with time_stage(self._timer, "rollout_probe"):
            return tuple(self._feature_augmenter.augment_turn(turn) for turn in turns)

    def _freeze_policy_results(
        self,
        *,
        policy: RolloutPolicy,
        group: Sequence[_PolicyTurn],
        policy_observations: Sequence[_PolicyObservation],
        policy_inputs: _PolicyInputBatch,
        result: _PolicyDecodeResult,
        order_indices: Sequence[int],
        macro_teacher_sample_mask: Sequence[bool],
    ) -> tuple[_FrozenPolicyDecision, ...]:
        # Remote policies expose the response version through mutable state.
        # Freeze it before validation or any auxiliary policy inference.
        behavior_policy_version = (
            _policy_version(policy)
            if result.served_policy_version is None
            else result.served_policy_version
        )
        if len(result.actions) != len(group):
            raise RuntimeError("sample_decode returned the wrong batch size")
        if len(order_indices) != len(group):
            raise RuntimeError("rollout turn ordering is misaligned")
        if len(policy_observations) != len(group):
            raise RuntimeError("rollout policy observations are misaligned")
        if len(macro_teacher_sample_mask) != len(group):
            raise RuntimeError("macro teacher sampling mask is misaligned")
        logprob_values = _tensor_values(result.action_logprobs)
        value_preds = _tensor_values(result.values)
        if len(logprob_values) != len(group) or len(value_preds) != len(group):
            raise RuntimeError("sample_decode returned the wrong tensor batch size")
        _validate_token_trace_batch(result, batch_size=len(group))
        if len(policy_inputs.decks) != len(group):
            raise RuntimeError("policy deck batch is misaligned with rollout turns")
        recurrent_leases = tuple(turn.recurrent_lease for turn in group)
        recurrent_batch = any(lease is not None for lease in recurrent_leases)
        bound_recurrent_leases: tuple[_RecurrentStateLease, ...] = ()
        if recurrent_batch:
            if any(lease is None for lease in recurrent_leases):
                raise RuntimeError("recurrent state leases are partially populated")
            if len(result.proposed_recurrent_states) != len(group):
                raise RuntimeError(
                    "recurrent decode returned the wrong proposed-state count"
                )
            bound_recurrent_leases = self._bind_recurrent_leases(
                policy=policy,
                turns=group,
                leases=cast(tuple[_RecurrentStateLease, ...], recurrent_leases),
                served_policy_version=behavior_policy_version,
                served_fingerprint=result.served_policy_artifact_fingerprint,
                served_artifact=result.served_policy_artifact,
            )
        elif result.proposed_recurrent_states:
            raise RuntimeError("stateless decode returned recurrent state")

        planner_decisions: tuple[PlannerPolicyDecision | None, ...]
        candidate_indices = tuple(
            index for index, turn in enumerate(group) if turn.role == "candidate"
        )
        macro_teacher_batch: RolloutPlannerBatch | None = None
        if self._macro_teacher_enabled and any(macro_teacher_sample_mask):
            macro_teacher_batch = self._planner_batch_from_decode(
                policy=policy,
                group=group,
                policy_observations=policy_observations,
                policy_inputs=policy_inputs,
                result=result,
                base_actions=tuple(
                    tuple(int(option_index) for option_index in action)
                    for action in result.actions
                ),
                base_logprobs=tuple(logprob_values),
                base_values=tuple(value_preds),
                behavior_policy_version=behavior_policy_version,
                macro_teacher_sample_mask=macro_teacher_sample_mask,
            )
        if self._planner_behavior_service is None or not candidate_indices:
            planner_decisions = tuple(None for _ in group)
        else:
            if not _is_sha256(result.served_model_fingerprint):
                raise RuntimeError(
                    "planner-enabled decode omitted its exact model fingerprint"
                )
            if result.served_proposal_version is None:
                raise RuntimeError(
                    "planner-enabled decode omitted its proposal version"
                )
            root_fallback_reason = _root_decode_fallback_reason(
                result.planner_fallback_reason
            )
            context_required = any(
                _turn_requires_planner_context(turn) for turn in group
            )
            if root_fallback_reason is None:
                if context_required and (
                    len(result.planner_context_handles) != len(group)
                ):
                    raise RuntimeError(
                        "planner-enabled decode did not retain leased contexts"
                    )
                if not context_required and result.planner_context_handles:
                    raise RuntimeError(
                        "planner base-fast-path decode retained root contexts"
                    )
            if root_fallback_reason is not None and result.planner_context_handles:
                raise RuntimeError("planner root fallback retained contexts")
            with time_stage(self._timer, "rollout_planner_behavior"):
                service_result = self._planner_behavior_service.plan_batch(
                    RolloutPlannerBatch(
                        policy=policy,
                        rows=tuple(
                            RolloutPlannerRowContext(
                                game_id=turn.game.game_id,
                                seat=turn.seat,
                                policy_role=turn.role,
                                should_record=turn.should_record,
                                observation=policy_observations[index].observation,
                                context_features=(
                                    policy_observations[index].context_features
                                ),
                                context_snapshot=self._context_for(
                                    turn.game,
                                    turn.seat,
                                ).snapshot(),
                                deck_pair=turn.game.deck_pair,
                            )
                            for index, turn in enumerate(group)
                        ),
                        states=policy_inputs.states,
                        options=policy_inputs.options,
                        decks=policy_inputs.decks,
                        base_actions=result.actions,
                        base_logprobs=tuple(logprob_values),
                        base_values=tuple(value_preds),
                        policy_version=behavior_policy_version,
                        model_fingerprint=result.served_model_fingerprint,
                        proposal_version=result.served_proposal_version,
                        planner_context_handles=result.planner_context_handles,
                        root_fallback_reason=root_fallback_reason,
                    )
                )
            planner_decisions = tuple(service_result)
            if len(planner_decisions) != len(group):
                raise RuntimeError(
                    "planner behavior service returned the wrong batch size"
                )
            if self._planner_telemetry is None:
                raise AssertionError("planner telemetry accumulator is missing")
            service_stats_fn = getattr(
                self._planner_behavior_service,
                "stats",
                None,
            )
            native_occupancy = (
                int(getattr(service_stats_fn(), "peak_active_rows", 0))
                if callable(service_stats_fn)
                else 0
            )
            for row_plan in planner_decisions:
                if row_plan is None:
                    continue
                runtime_stats = row_plan.runtime_stats
                fallback_reason = row_plan.planner_behavior.fallback_reason
                self._planner_telemetry.record_decision(
                    row_plan.telemetry_events,
                    planner_used=not row_plan.used_base_trace,
                    fallback_reason=(
                        fallback_reason.name.lower()
                        if row_plan.used_base_trace
                        else None
                    ),
                    cache_hits={
                        "root_context": runtime_stats.root_context_hits,
                        "engine_reuse": runtime_stats.engine_reuse_hits,
                        "leaf_reuse": runtime_stats.leaf_reuse_hits,
                        "prefix_reuse": runtime_stats.prefix_reuse_count,
                    },
                    cache_misses={
                        "root_context": runtime_stats.root_context_misses,
                        "engine_reuse": runtime_stats.engine_reuse_misses,
                        "leaf_reuse": runtime_stats.leaf_reuse_misses,
                    },
                    native_occupancy=native_occupancy,
                )

        frozen: list[_FrozenPolicyDecision] = []
        behavior_kind = cast(
            BehaviorKind,
            str(getattr(policy, "behavior_kind", "policy_sample")),
        )
        if behavior_kind not in ("policy_sample", "improvement"):
            raise ValueError("rollout policy has an invalid behavior_kind")
        for index, (turn, action) in enumerate(zip(group, result.actions, strict=True)):
            canonical_observation = policy_observations[index]
            archive_reanalysis_root = bool(
                turn.should_record
                and self._reanalysis_root_probability > 0.0
                and self._reanalysis_rng.random() < self._reanalysis_root_probability
            )
            select = _select(canonical_observation.observation)
            sampled_action = tuple(int(option_index) for option_index in action)
            normalized = normalize_action_order(select, sampled_action)
            if normalized != sampled_action:
                raise RuntimeError(
                    "sample_decode returned a non-canonical unordered action"
                )
            if not is_legal_action(select, sampled_action):
                raise RuntimeError("sample_decode returned an illegal action")
            row_plan = planner_decisions[index]
            if row_plan is None:
                if (
                    self._planner_behavior_service is not None
                    and turn.should_record
                    and turn.role == "candidate"
                ):
                    raise RuntimeError(
                        "recorded schema-9 row is missing planner branch evidence"
                    )
                behavior_action = sampled_action
                behavior_logprob = logprob_values[index]
                planner_behavior = None
                token_trace = _active_token_trace(result, index=index)
            else:
                if row_plan.planner_behavior.policy_version != behavior_policy_version:
                    raise RuntimeError(
                        "planner evidence differs from the behavior model lease"
                    )
                behavior_action = row_plan.action
                behavior_logprob = row_plan.old_logprob
                planner_behavior = row_plan.planner_behavior
                normalized_planner = normalize_action_order(select, behavior_action)
                if normalized_planner != behavior_action or not is_legal_action(
                    select,
                    behavior_action,
                ):
                    raise RuntimeError(
                        "planner behavior returned a non-canonical illegal action"
                    )
                if row_plan.used_base_trace:
                    if behavior_action != sampled_action or not math.isclose(
                        behavior_logprob,
                        logprob_values[index],
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise RuntimeError(
                            "base fallback changed the sampled base behavior"
                        )
                    token_trace = _active_token_trace(result, index=index)
                else:
                    token_trace = None
            teacher_request: EngineTeacherRequest | MacroTeacherRequest | None = None
            if (
                turn.should_record
                and self._macro_teacher_enabled
                and macro_teacher_sample_mask[index]
            ):
                if macro_teacher_batch is None:
                    raise RuntimeError("sampled macro teacher batch is unavailable")
                teacher_request = MacroTeacherRequest(
                    planner_batch=macro_teacher_batch,
                    row_index=index,
                )
            elif (
                turn.should_record
                and self._engine_teacher_producer is not None
                and not self._macro_teacher_enabled
            ):
                teacher_request = EngineTeacherRequest(
                    game_id=turn.game.game_id,
                    seat=turn.seat,
                    observation=turn.observation,
                    context_features=turn.context_features,
                    context_snapshot=self._context_for(
                        turn.game,
                        turn.seat,
                    ).snapshot(),
                    counterparty_context_snapshot=replace(
                        self._context_for(
                            turn.game,
                            1 - turn.seat,
                        ).snapshot(),
                        own_deck_counts=(),
                    ),
                    own_deck=turn.game.deck_pair[turn.seat],
                    behavior_action=behavior_action,
                )
            frozen.append(
                _FrozenPolicyDecision(
                    order_index=int(order_indices[index]),
                    turn=turn,
                    select=select,
                    deck_pair=turn.game.deck_pair,
                    action=behavior_action,
                    action_logprob=behavior_logprob,
                    value_pred=value_preds[index],
                    encoded_state=policy_inputs.state_features[index],
                    encoded_options=policy_inputs.encoded_options[index],
                    min_count=policy_inputs.min_counts[index],
                    max_count=policy_inputs.max_counts[index],
                    policy_version=behavior_policy_version,
                    behavior_kind=behavior_kind,
                    reanalysis_observation=(
                        canonical_observation.observation
                        if archive_reanalysis_root
                        else None
                    ),
                    reanalysis_context_snapshots=(
                        (
                            self._context_for(turn.game, 0).snapshot(),
                            self._context_for(turn.game, 1).snapshot(),
                        )
                        if archive_reanalysis_root
                        else None
                    ),
                    token_trace=token_trace,
                    teacher_request=teacher_request,
                    recurrent_transition=(
                        None
                        if not bound_recurrent_leases
                        else _PreparedRecurrentTransition(
                            lease=bound_recurrent_leases[index],
                            proposed_state=result.proposed_recurrent_states[index],
                        )
                    ),
                    planner_behavior=planner_behavior,
                )
            )
        return tuple(frozen)

    def _planner_batch_from_decode(
        self,
        *,
        policy: RolloutPolicy,
        group: Sequence[_PolicyTurn],
        policy_observations: Sequence[_PolicyObservation],
        policy_inputs: _PolicyInputBatch,
        result: _PolicyDecodeResult,
        base_actions: tuple[tuple[int, ...], ...],
        base_logprobs: tuple[float, ...],
        base_values: tuple[float, ...],
        behavior_policy_version: int,
        macro_teacher_sample_mask: Sequence[bool],
    ) -> RolloutPlannerBatch:
        """Freeze one retained root batch for post-behavior native teaching."""
        if not _is_sha256(result.served_model_fingerprint):
            raise RuntimeError("macro teacher decode omitted its model fingerprint")
        if result.served_proposal_version is None:
            raise RuntimeError("macro teacher decode omitted its proposal version")
        root_fallback_reason = _root_decode_fallback_reason(
            result.planner_fallback_reason
        )
        if len(macro_teacher_sample_mask) != len(group):
            raise RuntimeError("macro teacher sampling mask is misaligned")
        context_required = any(macro_teacher_sample_mask)
        if root_fallback_reason is None:
            if context_required and len(result.planner_context_handles) != len(group):
                raise RuntimeError("macro teacher decode did not retain root contexts")
            if not context_required and result.planner_context_handles:
                raise RuntimeError("macro teacher fast path retained root contexts")
        elif result.planner_context_handles:
            raise RuntimeError("macro teacher root fallback retained contexts")
        return RolloutPlannerBatch(
            policy=policy,
            rows=tuple(
                RolloutPlannerRowContext(
                    game_id=turn.game.game_id,
                    seat=turn.seat,
                    policy_role=(
                        "candidate" if macro_teacher_sample_mask[index] else "frozen"
                    ),
                    should_record=turn.should_record,
                    observation=policy_observations[index].observation,
                    context_features=policy_observations[index].context_features,
                    context_snapshot=self._context_for(
                        turn.game,
                        turn.seat,
                    ).snapshot(),
                    deck_pair=turn.game.deck_pair,
                )
                for index, turn in enumerate(group)
            ),
            states=policy_inputs.states,
            options=policy_inputs.options,
            decks=policy_inputs.decks,
            base_actions=base_actions,
            base_logprobs=base_logprobs,
            base_values=base_values,
            policy_version=behavior_policy_version,
            model_fingerprint=result.served_model_fingerprint,
            proposal_version=result.served_proposal_version,
            planner_context_handles=result.planner_context_handles,
            root_fallback_reason=root_fallback_reason,
        )

    def _macro_teacher_sample_mask(
        self,
        turns: Sequence[_PolicyTurn],
        policy_inputs: _PolicyInputBatch,
    ) -> tuple[bool, ...]:
        """Sample root-visible rows deterministically before queue admission."""
        if not self._macro_teacher_enabled:
            return (False,) * len(turns)
        fingerprints = policy_inputs.states.root_input_fingerprints
        if fingerprints is None or len(fingerprints) != len(turns):
            raise RuntimeError("macro teacher sampling requires root fingerprints")
        probability = self._macro_credit_config.native_teacher_sample_probability
        threshold = int(probability * (1 << 64))
        sampled: list[bool] = []
        for turn, root_fingerprint in zip(turns, fingerprints, strict=True):
            eligible = bool(
                turn.role == "candidate"
                and turn.should_record
                and _turn_requires_planner_context(turn)
            )
            self._macro_teacher_sample_eligible += int(eligible)
            digest = hashlib.sha256(b"ptcg-rl/macro-teacher-sample/v1\x00")
            digest.update(
                str(self._macro_credit_config.native_teacher_sampling_seed).encode(
                    "ascii"
                )
            )
            digest.update(b"\x00")
            digest.update(turn.game.game_id.encode("utf-8"))
            digest.update(int(turn.seat).to_bytes(1, "big", signed=False))
            digest.update(bytes.fromhex(root_fingerprint))
            selected = (
                eligible and int.from_bytes(digest.digest()[:8], "big") < threshold
            )
            sampled.append(selected)
            self._macro_teacher_sampled += int(selected)
        return tuple(sampled)

    def _register_recorded_decision(
        self,
        decision: _FrozenPolicyDecision,
    ) -> _FrozenPolicyDecision:
        """Assign a stable join key before either auxiliary lane can resolve."""
        if decision.identity is not None:
            raise RuntimeError("rollout decision was registered more than once")
        identity = _DecisionIdentity(
            sequence_id=self._next_decision_sequence,
            game_id=decision.turn.game.game_id,
            seat=decision.turn.seat,
        )
        self._next_decision_sequence += 1
        registered = replace(decision, identity=identity)
        self._pending_decision_evidence[identity.sequence_id] = (
            _PendingDecisionEvidence(
                identity=identity,
                decision=registered,
                factual_resolved=not self._factual_config.enabled,
                teacher_resolved=self._engine_teacher_producer is None,
            )
        )
        self._pending_record_order.setdefault(identity.game_id, deque()).append(
            identity.sequence_id
        )
        self._pending_record_games[identity.game_id] += 1
        return registered

    def _emit_ready_decisions(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
    ) -> None:
        """Emit registered rows whose configured evidence is already resolved."""
        for decision in decisions:
            identity = self._decision_identity(decision)
            evidence = self._pending_decision_evidence.get(identity.sequence_id)
            if evidence is not None:
                self._maybe_emit_decision(evidence)

    def _resolve_factual_decision(
        self,
        decision: _FrozenPolicyDecision,
        target: FactualTransitionTarget | None,
        factual_transition_steps: int | None = None,
        executed_endpoint_value_leaf: RootInformationLeaf | None = None,
    ) -> None:
        """Resolve factual evidence; ``None`` remains an explicit resolution."""
        evidence = self._decision_evidence(decision)
        if evidence.factual_resolved:
            raise RuntimeError("factual decision evidence resolved more than once")
        if executed_endpoint_value_leaf is not None and target is None:
            raise ValueError("executed endpoint leaf requires a factual target")
        if (
            self._macro_credit_config.enabled
            and target is not None
            and (factual_transition_steps is None or factual_transition_steps <= 0)
        ):
            raise ValueError("factual target requires a positive transition count")
        evidence.factual_target = target
        evidence.factual_transition_steps = factual_transition_steps
        evidence.executed_endpoint_value_leaf = executed_endpoint_value_leaf
        evidence.factual_resolved = True
        self._maybe_emit_decision(evidence)

    def _resolve_teacher_decision(
        self,
        decision: _FrozenPolicyDecision,
        target: EngineTeacherTarget | None,
    ) -> None:
        """Resolve teacher evidence; queue drops are represented by ``None``."""
        evidence = self._decision_evidence(decision)
        if evidence.teacher_resolved:
            raise RuntimeError("teacher decision evidence resolved more than once")
        evidence.teacher_target = target
        evidence.teacher_resolved = True
        self._maybe_emit_decision(evidence)

    def _decision_identity(
        self,
        decision: _FrozenPolicyDecision,
    ) -> _DecisionIdentity:
        identity = decision.identity
        if identity is None:
            raise RuntimeError("recorded rollout decision has no join identity")
        if (
            identity.game_id != decision.turn.game.game_id
            or identity.seat != decision.turn.seat
        ):
            raise RuntimeError("rollout decision identity does not match its turn")
        return identity

    def _decision_evidence(
        self,
        decision: _FrozenPolicyDecision,
    ) -> _PendingDecisionEvidence:
        identity = self._decision_identity(decision)
        evidence = self._pending_decision_evidence.get(identity.sequence_id)
        if evidence is None:
            raise RuntimeError(
                "rollout decision evidence is unknown or already emitted"
            )
        if (
            evidence.identity.sequence_id != identity.sequence_id
            or evidence.identity.game_id != identity.game_id
            or evidence.identity.seat != identity.seat
        ):
            raise RuntimeError("rollout decision evidence identity is misaligned")
        return evidence

    def _maybe_emit_decision(self, evidence: _PendingDecisionEvidence) -> None:
        """Record the ready prefix for one game in behavior sequence order."""
        sequence_id = evidence.identity.sequence_id
        if self._pending_decision_evidence.get(sequence_id) is not evidence:
            raise RuntimeError("rollout decision evidence state is no longer pending")
        game_id = evidence.identity.game_id
        pending_order = self._pending_record_order.get(game_id)
        if pending_order is None or sequence_id not in pending_order:
            raise RuntimeError("rollout decision is absent from its game order")
        while pending_order:
            next_sequence_id = pending_order[0]
            next_evidence = self._pending_decision_evidence.get(next_sequence_id)
            if next_evidence is None:
                raise RuntimeError("ordered rollout decision evidence is missing")
            if not next_evidence.factual_resolved or not next_evidence.teacher_resolved:
                return
            self._record_policy_decision(
                next_evidence.decision,
                engine_teacher_target=next_evidence.teacher_target,
                factual_target=next_evidence.factual_target,
                factual_transition_steps=next_evidence.factual_transition_steps,
                executed_endpoint_value_leaf=(
                    next_evidence.executed_endpoint_value_leaf
                ),
            )
            del self._pending_decision_evidence[next_sequence_id]
            pending_order.popleft()

            pending_count = self._pending_record_games.get(game_id, 0)
            if pending_count <= 0:
                raise RuntimeError("rollout decision game count became non-positive")
            if pending_count > 1:
                self._pending_record_games[game_id] = pending_count - 1
                continue
            del self._pending_record_games[game_id]
        del self._pending_record_order[game_id]
        finished = self._deferred_finished.pop(game_id, None)
        if finished is not None:
            self._finalize_finished_game(finished)

    def _engine_teacher_targets_for(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
    ) -> tuple[EngineTeacherTarget | None, ...]:
        """Dispatch one actor-step teacher batch after behavior submission."""
        producer = self._engine_teacher_producer
        if producer is None or not decisions:
            return (None,) * len(decisions)
        optional_requests = tuple(decision.teacher_request for decision in decisions)
        if any(request is None for request in optional_requests):
            raise RuntimeError("recorded teacher requests are incomplete")
        requests = cast(tuple[EngineTeacherRequest, ...], optional_requests)
        self._engine_teacher_requests += len(requests)
        self._engine_teacher_batches += 1
        with time_stage(self._timer, "rollout_engine_teacher"):
            produce_batch = getattr(producer, "produce_batch", None)
            if callable(produce_batch):
                targets = tuple(produce_batch(requests))
            else:
                targets = tuple(producer.produce(request) for request in requests)
        if len(targets) != len(decisions):
            raise RuntimeError("engine teacher returned the wrong batch size")

        return tuple(targets)

    def _submit_engine_teacher_async(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
    ) -> bool:
        """Queue auxiliary work while preserving immediate behavior submission."""
        producer = self._async_engine_teacher_producer()
        if producer is None:
            return False
        if not decisions:
            return True
        requested = tuple(
            decision for decision in decisions if decision.teacher_request is not None
        )
        unrequested = tuple(
            decision for decision in decisions if decision.teacher_request is None
        )
        if unrequested:
            self._record_engine_teacher_results(
                unrequested,
                (None,) * len(unrequested),
            )
        if not requested:
            return True
        requests = self._teacher_requests(requested)
        self._engine_teacher_requests += len(requests)
        self._engine_teacher_batches += 1
        with time_stage(self._timer, "rollout_engine_teacher_submit"):
            batch_id = cast(Any, producer).submit_async(requests)
        if batch_id is None:
            self._record_engine_teacher_results(
                requested,
                (None,) * len(requested),
            )
            return True
        if batch_id in self._pending_teacher_batches:
            raise RuntimeError("engine teacher reused an in-flight batch identity")
        frozen = requested
        self._pending_teacher_batches[batch_id] = frozen
        self._pending_teacher_games.update(
            decision.turn.game.game_id for decision in frozen
        )
        return True

    def _collect_engine_teacher_results(self) -> None:
        producer = self._async_engine_teacher_producer()
        if producer is None:
            return
        with time_stage(self._timer, "rollout_engine_teacher_collect"):
            completed = producer.poll_completed()
        self._consume_engine_teacher_completions(completed)

    def _consume_engine_teacher_completions(
        self,
        completions: Sequence[EngineTeacherBatchCompletion],
    ) -> None:
        for completion in completions:
            decisions = self._pending_teacher_batches.pop(
                completion.batch_id,
                None,
            )
            if decisions is None:
                raise RuntimeError("engine teacher completed an unknown batch")
            validated = self._validated_engine_teacher_targets(
                decisions,
                completion.targets,
            )
            touched_games: set[str] = set()
            for decision in decisions:
                game_id = decision.turn.game.game_id
                touched_games.add(game_id)
                self._pending_teacher_games[game_id] -= 1
                if self._pending_teacher_games[game_id] < 0:
                    raise RuntimeError("engine teacher game count became negative")
            for game_id in touched_games:
                if self._pending_teacher_games[game_id] > 0:
                    continue
                del self._pending_teacher_games[game_id]
            for decision, target in zip(decisions, validated, strict=True):
                self._resolve_teacher_decision(decision, target)

    def _record_engine_teacher_results(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
        targets: Sequence[EngineTeacherTarget | None],
    ) -> None:
        validated = self._validated_engine_teacher_targets(decisions, targets)
        for decision, target in zip(decisions, validated, strict=True):
            self._resolve_teacher_decision(decision, target)

    def _validated_engine_teacher_targets(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
        targets: Sequence[EngineTeacherTarget | None],
    ) -> tuple[EngineTeacherTarget | None, ...]:
        if len(targets) != len(decisions):
            raise RuntimeError("engine teacher returned the wrong batch size")
        validated_targets: list[EngineTeacherTarget | None] = []
        for decision, target in zip(decisions, targets, strict=True):
            if target is None:
                validated_targets.append(None)
                continue
            validated = validate_engine_teacher_target(decision.select, target)
            self._engine_teacher_targets += 1
            self._engine_teacher_search_targets += int(
                validated.search_evidence is not None
            )
            self._engine_teacher_behavior_matches += int(
                validated.action == decision.action
            )
            self._engine_teacher_confidence_sum += validated.confidence
            self._engine_teacher_weight_sum += validated.weight
            validated_targets.append(validated)
        return tuple(validated_targets)

    def _teacher_requests(
        self,
        decisions: Sequence[_FrozenPolicyDecision],
    ) -> tuple[EngineTeacherRequest | MacroTeacherRequest, ...]:
        optional = tuple(decision.teacher_request for decision in decisions)
        if any(request is None for request in optional):
            raise RuntimeError("recorded teacher requests are incomplete")
        return cast(
            tuple[EngineTeacherRequest | MacroTeacherRequest, ...],
            optional,
        )

    def _async_engine_teacher_producer(
        self,
    ) -> AsyncEngineTeacherProducer | None:
        producer = self._engine_teacher_producer
        return producer if isinstance(producer, AsyncEngineTeacherProducer) else None

    def _finalize_finished_game(self, finished: FinishedGame) -> None:
        with time_stage(self._timer, "rollout_finalize"):
            self._recorder.finalize(finished)

    def _record_policy_decision(
        self,
        decision: _FrozenPolicyDecision,
        *,
        engine_teacher_target: EngineTeacherTarget | None,
        factual_target: FactualTransitionTarget | None = None,
        factual_transition_steps: int | None = None,
        executed_endpoint_value_leaf: RootInformationLeaf | None = None,
    ) -> None:
        """Record immutable behavior evidence with aligned auxiliary targets."""
        turn = decision.turn
        opponent_name, opponent_tier = self._opponent_metadata(turn.game.game_id)
        token_trace = decision.token_trace
        with time_stage(self._timer, "rollout_record"):
            self._recorder.record(
                RolloutDecision(
                    game_id=turn.game.game_id,
                    seat=turn.seat,
                    deck_pair=decision.deck_pair,
                    action=decision.action,
                    action_logprob=decision.action_logprob,
                    value_pred=decision.value_pred,
                    sampling_temperature=self._temperature,
                    policy_role=turn.role,
                    opponent_name=opponent_name,
                    opponent_tier=opponent_tier,
                    observation=turn.observation,
                    behavior_kind=decision.behavior_kind,
                    encoded_state=decision.encoded_state,
                    encoded_options=decision.encoded_options,
                    min_count=decision.min_count,
                    max_count=decision.max_count,
                    policy_version=decision.policy_version,
                    token_logprobs=(None if token_trace is None else token_trace[0]),
                    prefix_value_preds=(
                        None if token_trace is None else token_trace[1]
                    ),
                    stop_sampled=(None if token_trace is None else token_trace[2]),
                    engine_teacher_target=engine_teacher_target,
                    factual_target=factual_target,
                    factual_transition_steps=(
                        factual_transition_steps
                        if self._macro_credit_config.enabled
                        else None
                    ),
                    macro_credit_enabled=self._macro_credit_config.enabled,
                    planner_behavior=decision.planner_behavior,
                    executed_endpoint_value_leaf=executed_endpoint_value_leaf,
                    training_metadata=self._training_metadata(turn.game.game_id),
                    reanalysis_observation=decision.reanalysis_observation,
                    reanalysis_context_snapshots=(
                        decision.reanalysis_context_snapshots
                    ),
                    public_event_delta=(
                        turn.public_event_token.delta
                        if self._record_public_event_deltas
                        else None
                    ),
                    policy_artifact_identity=(
                        None
                        if decision.recurrent_transition is None
                        else decision.recurrent_transition.lease.artifact
                    ),
                )
            )
        self._recorded_policy_evidence += 1

    def _start_factual_decision(self, decision: _FrozenPolicyDecision) -> None:
        """Open one actual-transition root immediately before behavior submit."""
        game = decision.turn.game
        if game.game_id in self._pending_factual:
            raise RuntimeError(
                f"decision-local factual roots cannot overlap: {game.game_id}"
            )
        self._pending_factual[game.game_id] = _PendingFactualDecision(
            decision=decision,
            before_observation=_factual_observation_snapshot(game.observation),
        )
        self._factual_started += 1

    def _advance_factual_decisions(
        self,
        game: VectorGame,
        *,
        before_observation: Mapping[str, Any],
    ) -> None:
        """Append one engine step and close at the next real decision boundary."""
        root = self._pending_factual.get(game.game_id)
        if root is None:
            return
        after_observation = _factual_observation_snapshot(game.observation)
        transition = ProbeTransition(
            before_observation=before_observation,
            after_observation=after_observation,
            logs=tuple(_sequence(after_observation.get("logs", ()))),
        )
        root.transitions.append(transition)
        successor = factual_successor(
            after_observation,
            root_player_index=root.decision.turn.seat,
        )
        if successor is None:
            return
        self._complete_factual_decision(
            root,
            after_observation=after_observation,
            successor=successor,
        )

    def _close_finished_factual(self, finished: FinishedGame) -> None:
        """Resolve a live factual root from the pool's terminal snapshot."""
        root = self._pending_factual.get(finished.game_id)
        if root is None:
            return
        after_observation = _factual_observation_snapshot(finished.observation)
        current = _field(after_observation, "current")
        if _int_field(current, "result", -1) < 0:
            raise RuntimeError(
                "finished rollout game has a non-terminal factual snapshot: "
                f"{finished.game_id}"
            )
        before_observation = (
            root.transitions[-1].after_observation
            if root.transitions
            else root.before_observation
        )
        root.transitions.append(
            ProbeTransition(
                before_observation=before_observation,
                after_observation=after_observation,
                logs=tuple(_sequence(after_observation.get("logs", ()))),
            )
        )
        successor = factual_successor(
            after_observation,
            root_player_index=root.decision.turn.seat,
        )
        if successor is None:
            raise RuntimeError(
                "finished rollout game did not resolve its factual successor: "
                f"{finished.game_id}"
            )
        self._complete_factual_decision(
            root,
            after_observation=after_observation,
            successor=successor,
        )

    def _complete_factual_decision(
        self,
        root: _PendingFactualDecision,
        *,
        after_observation: Mapping[str, Any],
        successor: FactualSuccessor,
    ) -> None:
        """Build and resolve one factual target at its decision boundary."""
        target = build_factual_transition_target(
            root_action=root.decision.action,
            before_observation=root.before_observation,
            after_observation=after_observation,
            transitions=tuple(root.transitions),
            successor=successor,
            perspective_player=root.decision.turn.seat,
        )
        root_player = root.decision.turn.seat
        root_observation = public_search_observation(
            after_observation,
            perspective_player_index=root_player,
        )
        context_features = self._context_for(
            root.decision.turn.game,
            root_player,
        ).update(root_observation)
        context_features = self._feature_augmenter.context_with_belief(
            root_observation,
            context_features,
        )
        endpoint_leaf = build_executed_endpoint_leaf(
            after_observation,
            root_player=root_player,
            context_features=context_features,
            actor_relation=successor.actor_relation,
            next_context=successor.next_context,
            exact_effect=target.effect_features,
            belief_summary=root_information_belief_summary(
                context_features,
                width=self._planner_belief_summary_width,
            ),
        )
        if (
            root.decision.planner_behavior is None
            and not self._macro_credit_config.enabled
        ):
            endpoint_leaf = None
        game_id = root.decision.turn.game.game_id
        if self._pending_factual.pop(game_id, None) is not root:
            raise RuntimeError("factual decision root is no longer pending")
        self._resolve_factual_decision(
            root.decision,
            target,
            factual_transition_steps=len(root.transitions),
            executed_endpoint_value_leaf=endpoint_leaf,
        )
        self._factual_completed += 1
        self._factual_transition_steps += len(root.transitions)
        self._factual_actor_relation_counts[successor.actor_relation.name.lower()] += 1
        self._factual_next_context_counts[successor.next_context] += 1

    def _submit_scripted(
        self,
        game: VectorGame,
        observation: Mapping[str, Any],
    ) -> None:
        assignment = self._assignment_for_game(game.game_id)
        agent: BattleAgent | None = None
        if assignment is not None and assignment.opponent_kind == "scripted":
            agent = self._scripted_agent_for_game(
                game.game_id,
                assignment.opponent_id,
            )
        agent = agent or self._actors.scripted_agent
        if agent is None:
            raise RuntimeError("scripted opponent is not configured")
        select = _select(observation)
        deck = _scripted_deck_for_game(
            game,
            assignment=assignment,
            candidate_seat=self._actors.candidate_seat,
        )
        with _scripted_deck_path_env(deck):
            action = normalize_action_order(select, agent.act(observation))
        seat = _player_index(observation)
        context = self._context_for(game, seat)
        token = context.prepare_decision()
        receipt = _SubmissionReceipt()
        try:
            self._submit_checked(
                game,
                select,
                action,
                on_accepted=partial(
                    self._commit_context_decision,
                    context,
                    token,
                    receipt,
                ),
            )
        except Exception:
            if receipt.engine_accepted and not receipt.context_committed:
                self._discard_contexts(game.game_id)
            raise

    def _submit_checked(
        self,
        game: VectorGame,
        select: Any,
        action: Sequence[int],
        *,
        on_accepted: Callable[[], Any] | None = None,
    ) -> None:
        normalized = normalize_action_order(select, action)
        if not is_legal_action(select, normalized):
            raise ValueError(f"illegal rollout action for {game.game_id}: {normalized}")
        before_observation = (
            _factual_observation_snapshot(game.observation)
            if self._factual_config.enabled
            else None
        )
        self._pool.submit(game.game_id, normalized)
        if on_accepted is not None:
            on_accepted()
        if before_observation is not None:
            self._advance_factual_decisions(
                game,
                before_observation=before_observation,
            )

    def _scripted_agent_for_game(
        self,
        game_id: str,
        opponent_id: str,
    ) -> BattleAgent | None:
        cached = self._scripted_game_agents.get(game_id)
        if cached is not None:
            return cached
        factory = self._actors.scripted_agent_factories.get(opponent_id)
        if factory is None:
            return self._actors.scripted_agents.get(opponent_id)
        agent = factory()
        agent.reset()
        self._scripted_game_agents[game_id] = agent
        return agent

    def _observation_with_context(
        self,
        game: VectorGame,
        seat: int,
    ) -> tuple[Mapping[str, Any], GameContextFeatures]:
        context = self._context_for(game, seat)
        features = context.update(game.observation)
        counterparty_context = self._context_for(game, 1 - seat)
        counterparty_context.update(
            public_search_observation(
                game.observation,
                perspective_player_index=1 - seat,
            ),
            accumulate_public_events=False,
        )
        observation = dict(game.observation)
        observation["gameContext"] = features.as_observation_dict()
        return observation, features

    def _context_for(self, game: VectorGame, seat: int) -> GameContext:
        key = (game.game_id, seat)
        context = self._contexts.get(key)
        if context is None:
            context = GameContext(player_index=seat)
            context.reset(player_index=seat, own_deck=game.deck_pair[seat])
            self._contexts[key] = context
        return context

    def _ensure_recurrent_state(
        self,
        game: VectorGame,
        seat: int,
        route: _PolicyRoute,
        *,
        context: GameContext,
    ) -> _RecurrentStateLease | None:
        """Create or validate one explicit game-seat-policy state boundary."""
        seat_key = (game.game_id, seat)
        if route.role is None:
            self._reject_live_recurrent_route_change(game.game_id, seat)
            return None
        policy = self._policy_for_route(route.role, route.policy_id)
        if not _policy_recurrent_enabled(policy):
            self._reject_live_recurrent_route_change(game.game_id, seat)
            return None
        pipelined_policy = policy if _as_pipelined_policy(policy) else None
        remote_recurrent_policy = (
            policy if _as_pipelined_recurrent_policy(policy) else None
        )
        local_recurrent_policy = (
            policy
            if pipelined_policy is None and _as_local_recurrent_policy(policy)
            else None
        )
        if pipelined_policy is not None and remote_recurrent_policy is None:
            raise RuntimeError(
                "recurrent pipelined rollout policy lacks its recurrent transport "
                "contract"
            )
        if pipelined_policy is None and local_recurrent_policy is None:
            raise RuntimeError(
                "recurrent rollout policy lacks the local recurrent contract"
            )
        deck = canonicalize_deck(game.deck_pair[seat])
        bound_key = self._recurrent_key_by_seat.get(seat_key)
        initial_artifact: PolicyArtifactIdentity | None = None
        if bound_key is not None:
            if (
                bound_key.policy_role != route.role
                or bound_key.policy_id != route.policy_id
                or bound_key.exact_deck_digest != deck.deck_digest
                or bound_key.exact_deck_signature != deck.signature
            ):
                raise RuntimeError(
                    "active recurrent rollout sequence cannot switch route or deck"
                )
            if (
                local_recurrent_policy is not None
                and bound_key.policy_artifact_fingerprint
                != _policy_artifact_fingerprint(local_recurrent_policy)
            ):
                raise RuntimeError(
                    "active recurrent rollout sequence cannot switch policy artifact"
                )
            state_key = bound_key
        else:
            initial_artifact = (
                None
                if remote_recurrent_policy is not None
                else _policy_artifact_identity(policy)
            )
            state_key = _RecurrentStateKey(
                game_id=game.game_id,
                seat=seat,
                policy_role=route.role,
                policy_id=route.policy_id,
                exact_deck_digest=deck.deck_digest,
                exact_deck_signature=deck.signature,
                policy_artifact_fingerprint=(
                    None
                    if remote_recurrent_policy is not None
                    else _policy_artifact_fingerprint(policy)
                ),
            )
        committed = self._recurrent_states.get(state_key)
        if committed is None:
            if bound_key is not None:
                raise RuntimeError("active recurrent rollout state is missing")
            recurrent_policy = (
                remote_recurrent_policy
                if remote_recurrent_policy is not None
                else local_recurrent_policy
            )
            if recurrent_policy is None:
                raise AssertionError("validated recurrent policy is missing")
            initial_state = recurrent_policy.initial_recurrent_state(1).detach()
            if initial_state.batch_size != 1:
                raise RuntimeError(
                    "recurrent policy returned the wrong initial state batch size"
                )
            if (
                torch.count_nonzero(initial_state.hidden).item()
                or torch.count_nonzero(initial_state.cell).item()
            ):
                raise RuntimeError(
                    "recurrent policy initial state must be canonical exact zero"
                )
            committed = _CommittedRecurrentState(
                state=initial_state,
                generation=context.public_event_generation,
                artifact=initial_artifact,
                policy_version=(
                    None
                    if remote_recurrent_policy is not None
                    else _policy_version(policy)
                ),
            )
            self._recurrent_states[state_key] = committed
            self._recurrent_key_by_seat[seat_key] = state_key
            self._recurrent_state_initializations += 1
        if committed.generation != context.public_event_generation:
            raise RuntimeError("recurrent state and public-event generations diverged")
        return _RecurrentStateLease(
            key=state_key,
            state=committed.state,
            generation=committed.generation,
            artifact=committed.artifact,
            policy_version=committed.policy_version,
        )

    def _recurrent_lease_for(
        self,
        game: VectorGame,
        seat: int,
        route: _PolicyRoute,
        *,
        context: GameContext,
        token: PublicEventDecisionToken,
    ) -> _RecurrentStateLease | None:
        """Return the exact old state aligned with a prepared event token."""
        lease = self._ensure_recurrent_state(
            game,
            seat,
            route,
            context=context,
        )
        if lease is not None and lease.generation != token.generation:
            raise RuntimeError(
                "recurrent state lease and public-event token generations differ"
            )
        return lease

    def _reject_live_recurrent_route_change(self, game_id: str, seat: int) -> None:
        """Fail closed if an active recurrent sequence loses its policy route."""
        if (game_id, seat) in self._recurrent_key_by_seat:
            raise RuntimeError(
                "active recurrent rollout sequence cannot switch to stateless route"
            )

    def _bind_recurrent_leases(
        self,
        *,
        policy: RolloutPolicy,
        turns: Sequence[_PolicyTurn],
        leases: tuple[_RecurrentStateLease, ...],
        served_policy_version: int,
        served_fingerprint: str,
        served_artifact: PolicyArtifactIdentity | None,
    ) -> tuple[_RecurrentStateLease, ...]:
        """Atomically bind first-use remote leases before engine submission."""
        if len(turns) != len(leases):
            raise RuntimeError("recurrent lease rows are misaligned")
        if served_artifact is not None:
            if served_fingerprint != served_artifact.fingerprint:
                raise RuntimeError(
                    "recurrent served fingerprint differs from its full artifact"
                )
        elif not _is_sha256(served_fingerprint):
            raise RuntimeError("recurrent decode omitted its policy artifact")
        if _as_pipelined_recurrent_policy(policy) and served_artifact is None:
            raise RuntimeError("remote recurrent decode omitted its full artifact")

        prepared: list[
            tuple[
                _RecurrentStateLease,
                _RecurrentStateKey,
                _CommittedRecurrentState,
                PolicyArtifactIdentity | None,
            ]
        ] = []
        for turn, lease in zip(turns, leases, strict=True):
            if lease.key.game_id != turn.game.game_id or lease.key.seat != turn.seat:
                raise RuntimeError("recurrent lease owner differs from policy turn")
            if (
                lease.key.policy_role != turn.role
                or lease.key.policy_id != turn.policy_id
            ):
                raise RuntimeError("recurrent lease route differs from policy turn")
            active_key = self._recurrent_key_by_seat.get(
                (lease.key.game_id, lease.key.seat)
            )
            if active_key != lease.key:
                raise RuntimeError("recurrent lease is no longer active")
            current = self._recurrent_states.get(lease.key)
            if (
                current is None
                or current.generation != lease.generation
                or current.state is not lease.state
                or current.artifact != lease.artifact
                or current.policy_version != lease.policy_version
            ):
                raise RuntimeError("recurrent first-bind lease is stale")
            if (
                lease.policy_version is not None
                and lease.policy_version != served_policy_version
            ):
                raise RuntimeError(
                    "recurrent continuation changed served policy version"
                )
            if lease.artifact is not None:
                if served_artifact is None:
                    raise RuntimeError(
                        "bound recurrent lease lost its full artifact identity"
                    )
                validate_policy_artifact_identity(
                    lease.artifact,
                    served_artifact,
                )
            if lease.key.policy_artifact_fingerprint is None:
                if served_artifact is None:
                    raise RuntimeError(
                        "unbound recurrent lease requires a full served artifact"
                    )
                if (
                    torch.count_nonzero(lease.state.hidden).item()
                    or torch.count_nonzero(lease.state.cell).item()
                ):
                    raise RuntimeError(
                        "unbound recurrent lease no longer has exact zero state"
                    )
                bound_key = replace(
                    lease.key,
                    policy_artifact_fingerprint=served_artifact.fingerprint,
                )
            else:
                if lease.key.policy_artifact_fingerprint != served_fingerprint:
                    raise RuntimeError(
                        "recurrent decode artifact identity is misaligned"
                    )
                bound_key = lease.key
            collision = self._recurrent_states.get(bound_key)
            if bound_key != lease.key and collision is not None:
                raise RuntimeError("recurrent first-bind artifact key collided")
            prepared.append(
                (
                    lease,
                    bound_key,
                    current,
                    served_artifact if served_artifact is not None else lease.artifact,
                )
            )

        bound: list[_RecurrentStateLease] = []
        for lease, bound_key, current, artifact in prepared:
            if bound_key != lease.key:
                del self._recurrent_states[lease.key]
            policy_version = (
                served_policy_version
                if lease.policy_version is None
                else lease.policy_version
            )
            committed = replace(
                current,
                artifact=artifact,
                policy_version=policy_version,
            )
            self._recurrent_states[bound_key] = committed
            self._recurrent_key_by_seat[(bound_key.game_id, bound_key.seat)] = bound_key
            bound.append(
                _RecurrentStateLease(
                    key=bound_key,
                    state=committed.state,
                    generation=committed.generation,
                    artifact=artifact,
                    policy_version=policy_version,
                )
            )
        return tuple(bound)

    def _accept_policy_turn(
        self,
        turn: _PolicyTurn,
        transition: _PreparedRecurrentTransition | None,
        receipt: _SubmissionReceipt,
    ) -> None:
        """Commit event generation and proposed state at one accepted boundary."""
        context = self._context_for(turn.game, turn.seat)
        receipt.engine_accepted = True
        if (transition is None) != (turn.recurrent_lease is None):
            raise RuntimeError("policy turn lost its recurrent transition")
        context.abort_decision(turn.public_event_token)
        committed_recurrent: _CommittedRecurrentState | None = None
        if transition is not None:
            committed_recurrent = self._validate_recurrent_transition(
                transition,
                turn=turn,
            )
        context.commit_decision(turn.public_event_token)
        receipt.context_committed = True
        if transition is not None:
            if committed_recurrent is None:
                raise AssertionError("validated recurrent state is missing")
            self._recurrent_states[transition.lease.key] = committed_recurrent
            receipt.recurrent_committed = True
            self._recurrent_commits += 1

    def _validate_recurrent_transition(
        self,
        transition: _PreparedRecurrentTransition,
        *,
        turn: _PolicyTurn,
    ) -> _CommittedRecurrentState:
        """Validate a candidate fully before the public cursor is mutated."""
        lease = transition.lease
        if lease.key.game_id != turn.game.game_id or lease.key.seat != turn.seat:
            raise RuntimeError("recurrent transition owner differs from policy turn")
        active_key = self._recurrent_key_by_seat.get(
            (lease.key.game_id, lease.key.seat)
        )
        if active_key != lease.key:
            raise RuntimeError(
                "recurrent transition policy artifact is no longer active"
            )
        policy = self._policy_for_route(turn.role, turn.policy_id)
        if not _policy_recurrent_enabled(policy):
            raise RuntimeError("recurrent policy became stateless before commit")
        if (
            not _as_pipelined_recurrent_policy(policy)
            and _policy_artifact_fingerprint(policy)
            != lease.key.policy_artifact_fingerprint
        ):
            raise RuntimeError("policy artifact changed before recurrent commit")
        current = self._recurrent_states.get(lease.key)
        if (
            current is None
            or current.generation != lease.generation
            or current.state is not lease.state
            or current.artifact != lease.artifact
            or current.policy_version != lease.policy_version
        ):
            raise RuntimeError("recurrent transition lease is stale or committed")
        proposed = transition.proposed_state.detach()
        if proposed.batch_size != 1:
            raise RuntimeError("committed recurrent state must have batch size one")
        if (
            proposed.hidden.shape != current.state.hidden.shape
            or proposed.hidden.device != current.state.hidden.device
            or proposed.hidden.dtype != current.state.hidden.dtype
            or proposed.cell.shape != current.state.cell.shape
            or proposed.cell.device != current.state.cell.device
            or proposed.cell.dtype != current.state.cell.dtype
        ):
            raise RuntimeError("proposed recurrent state layout changed")
        return _CommittedRecurrentState(
            state=proposed,
            generation=lease.generation + 1,
            artifact=lease.artifact,
            policy_version=lease.policy_version,
        )

    @staticmethod
    def _commit_context_decision(
        context: GameContext,
        token: PublicEventDecisionToken,
        receipt: _SubmissionReceipt,
    ) -> None:
        """Commit one accepted callback and expose partial-failure state."""
        receipt.engine_accepted = True
        context.commit_decision(token)
        receipt.context_committed = True

    def _discard_contexts(self, game_id: str) -> int:
        recurrent_keys = tuple(
            key for key in self._recurrent_states if key.game_id == game_id
        )
        for recurrent_key in recurrent_keys:
            self._release_recurrent_state(recurrent_key)
        for context_key in [key for key in self._contexts if key[0] == game_id]:
            del self._contexts[context_key]
        return len(recurrent_keys)

    def _release_recurrent_state(self, key: _RecurrentStateKey) -> None:
        """Release one remote snapshot lease, then forget its actor state."""
        committed = self._recurrent_states.get(key)
        if committed is None:
            return
        policy = self._policy_for_route(key.policy_role, key.policy_id)
        remote_policy = policy if _as_pipelined_recurrent_policy(policy) else None
        if remote_policy is not None:
            sequences = (
                RecurrentSequenceIdentity(
                    game_id=key.game_id,
                    seat=key.seat,
                    exact_deck_signature=key.exact_deck_signature,
                ),
            )
            released = (
                remote_policy.abort_recurrent_sequences(sequences)
                if committed.artifact is None
                else remote_policy.release_recurrent_sequences(
                    sequences,
                    committed.artifact,
                )
            )
            if released not in (0, 1):
                raise RuntimeError(
                    "remote recurrent release returned the wrong sequence count"
                )
        del self._recurrent_states[key]
        seat_key = (key.game_id, key.seat)
        if self._recurrent_key_by_seat.get(seat_key) == key:
            del self._recurrent_key_by_seat[seat_key]

    def _policy_for_route(self, role: PolicyRole, policy_id: str) -> RolloutPolicy:
        if role == "candidate":
            return self._actors.candidate_policy
        policy = self._actors.frozen_policies.get(policy_id)
        if policy is not None:
            return policy
        if self._actors.frozen_policy is None:
            raise RuntimeError("frozen policy is not configured")
        return self._actors.frozen_policy

    def _policy_route_for_seat(self, game_id: str, seat: int) -> _PolicyRoute:
        assignment = self._assignment_for_game(game_id)
        if assignment is not None:
            if assignment.opponent_kind == "self_play":
                return _PolicyRoute(role="candidate", policy_id="candidate")
            if seat == self._actors.candidate_seat:
                return _PolicyRoute(role="candidate", policy_id="candidate")
            if assignment.opponent_kind == "frozen":
                return _PolicyRoute(role="frozen", policy_id=assignment.opponent_id)
            return _PolicyRoute(role=None, policy_id=assignment.opponent_id)

        if self._actors.mode == "self_play":
            return _PolicyRoute(role="candidate", policy_id="candidate")
        if seat == self._actors.candidate_seat:
            return _PolicyRoute(role="candidate", policy_id="candidate")
        if self._actors.mode == "frozen":
            return _PolicyRoute(role="frozen", policy_id="frozen")
        return _PolicyRoute(role=None, policy_id="scripted")

    def _should_record(self, game_id: str, seat: int) -> bool:
        if not self._record_policy_decisions:
            return False
        assignment = self._assignment_for_game(game_id)
        if assignment is not None:
            if seat == self._actors.candidate_seat:
                return True
            return (
                assignment.opponent_kind == "self_play"
                and assignment.train_opponent_seat
            )
        return self._actors.mode == "self_play" or seat == self._actors.candidate_seat

    def _assignment_for_game(self, game_id: str) -> GameAssignment | None:
        return self._actors.curriculum_assignments.get(game_id)

    def _opponent_metadata(self, game_id: str) -> tuple[str, int]:
        assignment = self._assignment_for_game(game_id)
        if assignment is None:
            return ("", -1)
        if assignment.opponent_kind == "self_play":
            return ("self_play", -1)
        if assignment.opponent_kind == "scripted":
            return (
                assignment.opponent_id,
                self._actors.scripted_tiers.get(
                    assignment.opponent_id,
                    -1,
                ),
            )
        return (assignment.opponent_id, -1)

    def _training_metadata(self, game_id: str) -> Mapping[str, str]:
        assignment = self._assignment_for_game(game_id)
        if assignment is None:
            return {}
        metadata = {
            "candidate_lane": assignment.candidate_lane,
            "candidate_deck_label": assignment.candidate_deck_label,
            "candidate_seat": str(self._actors.candidate_seat),
            "opponent_deck_label": assignment.opponent_deck_label,
            "opponent_kind": assignment.opponent_kind,
            "opponent_id": assignment.opponent_id,
        }
        if assignment.frozen_sampling_lane is not None:
            metadata["frozen_sampling_lane"] = assignment.frozen_sampling_lane
        return metadata


@dataclass(frozen=True)
class _PolicyRoute:
    role: PolicyRole | None
    policy_id: str


def _collate_policy_inputs(
    observations: Sequence[_PolicyObservation],
    *,
    device: torch.device | str | None,
) -> _PolicyInputBatch:
    state_features: list[StateTokenArrayFeatures] = []
    encoded_options: list[EncodedOptionArrayFeatures] = []
    min_counts: list[int] = []
    max_counts: list[int] = []
    for item in observations:
        observation = item.observation
        select = _select(observation)
        if select is None:
            raise ValueError("policy rollout observation has no select prompt")
        layout = StateTokenLayout.from_observation(
            observation,
            context_features=item.context_features,
        )
        options = encode_option_arrays(select, layout)
        _apply_observation_probe_features(options, observation)
        if len(options) == 0:
            raise ValueError("policy rollout observation has no legal options")
        state_features.append(
            encode_observation_token_arrays(observation, layout=layout)
        )
        encoded_options.append(options)
        min_counts.append(_int_field(select, "minCount", 0))
        max_counts.append(_int_field(select, "maxCount", len(options)))
    option_batch = collate_encoded_options(
        encoded_options,
        min_counts=min_counts,
        max_counts=max_counts,
        device=device,
    )
    normalized_mins = tuple(
        min(len(options), max(0, int(min_count)))
        for options, min_count in zip(encoded_options, min_counts, strict=True)
    )
    normalized_maxes = tuple(
        min(len(options), max(normalized_min, int(max_count)))
        for options, normalized_min, max_count in zip(
            encoded_options,
            normalized_mins,
            max_counts,
            strict=True,
        )
    )
    states = replace(
        collate_state_tokens(state_features, device=device),
        root_input_fingerprints=tuple(
            canonical_planner_root_input_fingerprint(
                state,
                options,
                min_count=min_count,
                max_count=max_count,
            )
            for state, options, min_count, max_count in zip(
                state_features,
                encoded_options,
                normalized_mins,
                normalized_maxes,
                strict=True,
            )
        ),
    )
    return _PolicyInputBatch(
        states=states,
        options=option_batch,
        decks=DeckBatch.from_decks(
            tuple(item.own_deck for item in observations),
            device=device,
        ),
        state_features=tuple(state_features),
        encoded_options=tuple(encoded_options),
        min_counts=tuple(
            int(value) for value in option_batch.min_counts.detach().cpu().tolist()
        ),
        max_counts=tuple(
            int(value) for value in option_batch.max_counts.detach().cpu().tolist()
        ),
    )


def _own_deck_for_turn(turn: _PolicyTurn) -> CanonicalDeck:
    """Resolve only the acting seat's immutable deck for model input."""
    if turn.seat not in (0, 1):
        raise ValueError(f"invalid rollout seat: {turn.seat}")
    if len(turn.game.deck_pair) != 2:
        raise ValueError("rollout game must contain exactly two decks")
    return canonicalize_deck(turn.game.deck_pair[turn.seat])


def _apply_observation_probe_features(
    options: EncodedOptionArrayFeatures,
    observation: Mapping[str, Any],
) -> None:
    features = _sequence(_field(observation, "probeEffectFeatures", ()))
    masks = _sequence(_field(observation, "probeEffectMasks", ()))
    if not features or not masks:
        return
    option_count = len(options)
    limit = min(option_count, len(features), len(masks))
    for index in range(limit):
        if not bool(masks[index]):
            continue
        row = _sequence(features[index])
        if len(row) != DYNAMIC_EFFECT_FEATURE_SIZE:
            continue
        options.dynamic_effect_features[index, :] = np.asarray(row, dtype=np.float32)
        options.dynamic_effect_masks[index] = True


def _scripted_deck_for_game(
    game: VectorGame,
    *,
    assignment: GameAssignment | None,
    candidate_seat: int,
) -> tuple[int, ...]:
    if assignment is not None:
        return tuple(int(card_id) for card_id in assignment.opponent_deck)
    opponent_seat = 1 - candidate_seat
    return tuple(int(card_id) for card_id in game.deck_pair[opponent_seat])


@contextmanager
def _scripted_deck_path_env(deck: Sequence[int]) -> Iterator[None]:
    path = _cached_scripted_deck_path(deck)
    previous = os.environ.get(_DECK_PATH_ENV)
    os.environ[_DECK_PATH_ENV] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_DECK_PATH_ENV, None)
        else:
            os.environ[_DECK_PATH_ENV] = previous


def _cached_scripted_deck_path(deck: Sequence[int]) -> Path:
    normalized = tuple(int(card_id) for card_id in deck)
    if len(normalized) != 60:
        raise ValueError(
            f"scripted opponent deck must contain 60 cards: {len(normalized)}"
        )
    payload = "\n".join(str(card_id) for card_id in normalized) + "\n"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    path = _SCRIPTED_DECK_CACHE_DIR / f"deck_{digest}.csv"
    if path.exists():
        return path
    _SCRIPTED_DECK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
    return path


def _select(observation: Any) -> Any:
    return _field(observation, "select")


def _player_index(observation: Any) -> int:
    return _int_field(_field(observation, "current"), "yourIndex", -1)


def _factual_observation_snapshot(
    observation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Keep only engine fields required to derive a non-leaking target."""
    current = observation.get("current")
    select = observation.get("select")
    if current is None:
        raise ValueError("factual transition observation has no current state")
    return {
        "current": current,
        "select": select,
        "logs": tuple(_sequence(observation.get("logs", ()))),
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default


def _belief_cache_key(
    observation: Mapping[str, Any],
    context_features: GameContextFeatures,
) -> tuple[Any, ...]:
    current = _field(observation, "current")
    your_index = _int_field(current, "yourIndex", 0)
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    return (
        your_index,
        _visible_player_card_signature(current, opponent_index),
        tuple(context_features.opponent_revealed),
    )


def _visible_player_card_signature(current: Any, player_index: int) -> tuple[int, ...]:
    players = _sequence(_field(current, "players", ()))
    if player_index < 0 or player_index >= len(players):
        return ()
    card_ids: list[int] = []
    _add_visible_player_card_ids(players[player_index], card_ids)
    for card in _sequence(_field(current, "stadium", ())):
        if _int_field(card, "playerIndex", -1) == player_index:
            _add_card_id(card, card_ids)
    for card in _sequence(_field(current, "looking", ())):
        if card is not None and _int_field(card, "playerIndex", -1) == player_index:
            _add_card_id(card, card_ids)
    return tuple(sorted(card_ids))


def _add_visible_player_card_ids(player: Any, card_ids: list[int]) -> None:
    for pokemon in _sequence(_field(player, "active", ())):
        if pokemon is not None:
            _add_pokemon_card_ids(pokemon, card_ids)
    for pokemon in _sequence(_field(player, "bench", ())):
        _add_pokemon_card_ids(pokemon, card_ids)
    for field_name in ("discard", "prize", "hand"):
        for card in _sequence(_field(player, field_name, ())):
            if card is not None:
                _add_card_id(card, card_ids)


def _add_pokemon_card_ids(pokemon: Any, card_ids: list[int]) -> None:
    _add_card_id(pokemon, card_ids)
    for field_name in ("energyCards", "tools", "preEvolution"):
        for card in _sequence(_field(pokemon, field_name, ())):
            _add_card_id(card, card_ids)


def _add_card_id(card: Any, card_ids: list[int]) -> None:
    card_id = _int_field(card, "id", 0)
    if card_id > 0:
        card_ids.append(card_id)


def _tensor_values(values: Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().float().cpu().tolist())


def _policy_artifact_fingerprint(policy: RolloutPolicy) -> str:
    """Resolve the centralized immutable recurrent identity with a legacy bridge."""
    identity = _policy_artifact_identity(policy)
    if identity is not None:
        return identity.fingerprint
    value = getattr(policy, "policy_artifact_fingerprint", None)
    if callable(value):
        value = value()
    if value is None:
        # Temporary compatibility bridge until every publisher exposes the
        # complete config/schema-bound artifact identity through the property
        # above. Keep the fallback centralized so it cannot become a key schema.
        value = getattr(policy, "model_fingerprint", None)
        if callable(value):
            value = value()
    if not isinstance(value, str) or not _is_sha256(value):
        raise RuntimeError(
            "recurrent rollout policy requires an immutable artifact fingerprint"
        )
    return value


def _policy_artifact_identity(
    policy: RolloutPolicy,
) -> PolicyArtifactIdentity | None:
    """Return a full local/remote artifact contract when already published."""
    value = getattr(policy, "policy_artifact_identity", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    if isinstance(value, PolicyArtifactIdentity):
        return value
    if isinstance(value, Mapping):
        return PolicyArtifactIdentity.model_validate(value)
    raise TypeError("policy_artifact_identity has an invalid runtime type")


def _policy_version(policy: RolloutPolicy) -> int:
    value = getattr(policy, "policy_version", 0)
    if callable(value):
        value = value()
    return int(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _root_decode_fallback_reason(
    value: str,
) -> PlannerFallbackReason | None:
    if not value:
        return None
    if value == "model_lease_capacity":
        return PlannerFallbackReason.MODEL_LEASE_CAPACITY
    raise RuntimeError(f"planner root decode returned unknown fallback: {value}")
