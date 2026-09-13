"""Tensor-ready experience records passed from rollout actors to learners."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ptcg_rl.actions.encoding import (
    EncodedOptionArrayFeatures,
    EncodedOptionInput,
)
from ptcg_rl.actions.selection import is_unordered_set_selection
from ptcg_rl.agent.search.endpoint_value_rows import (
    ExecutedEndpointValueTableBuilder,
)
from ptcg_rl.agent.search.root_information import RootInformationLeaf
from ptcg_rl.context import (
    PublicEventArrayBlock,
    PublicEventDelta,
    build_public_event_array_block,
    validate_public_event_array_block,
)
from ptcg_rl.decks.identity import (
    DECK_SIZE,
    CanonicalDeck,
    canonicalize_deck,
)
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.factual_schema import (
    FACTUAL_NEXT_CONTEXT_COUNT,
    FactualActorRelation,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.model.state_encoder import StateTokenInput
from ptcg_rl.rl.amortized_policy_iteration.contracts import BehaviorKind
from ptcg_rl.rl.endpoint_value_arrays import (
    ExecutedEndpointValueArrayBlock,
    build_executed_endpoint_value_array_block,
    validate_executed_endpoint_value_array_block,
)
from ptcg_rl.rl.engine_teacher import EngineTeacherTarget, optional_target_arrays
from ptcg_rl.rl.factual import FactualTransitionTarget
from ptcg_rl.rl.macro_credit_arrays import (
    ExecutedMacroArrayBlock,
    build_executed_macro_array_block,
    validate_executed_macro_array_block,
)
from ptcg_rl.rl.macro_teacher import MacroTeacherEvidence
from ptcg_rl.rl.macro_teacher_arrays import (
    MacroTeacherArrayBlock,
    build_macro_teacher_array_block,
    validate_macro_teacher_array_block,
)
from ptcg_rl.rl.planner_evidence import (
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
)
from ptcg_rl.rl.planner_evidence_arrays import (
    PlannerEvidenceArrayBlock,
    build_planner_evidence_array_block,
    validate_planner_evidence_array_block,
)
from ptcg_rl.rl.recurrent_runtime import PolicyArtifactIdentity
from ptcg_rl.rl.search_evidence_arrays import (
    SearchEvidenceArrayBlock,
    build_search_evidence_array_block,
    validate_search_evidence_array_block,
)
from ptcg_rl.rl.trajectory import CompletedTrajectory

if TYPE_CHECKING:
    from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
        ReanalysisRoot,
    )


@dataclass(frozen=True)
class DecisionRecord:
    """One trainable policy decision with already-encoded model inputs."""

    seat: int
    decision_index: int
    state: StateTokenInput
    options: EncodedOptionInput
    min_count: int
    max_count: int
    action: tuple[int, ...]
    action_logprob: float
    value_pred: float
    policy_version: int
    behavior_kind: BehaviorKind = "policy_sample"
    sampling_temperature: float = 1.0
    token_logprobs: tuple[float, ...] | None = None
    prefix_value_preds: tuple[float, ...] | None = None
    stop_sampled: bool | None = None
    engine_teacher_target: EngineTeacherTarget | None = None
    factual_target: FactualTransitionTarget | None = None
    factual_transition_steps: int | None = None
    macro_credit_enabled: bool = False
    planner_behavior: PlannerBehaviorEvidence | None = None
    executed_endpoint_value_leaf: RootInformationLeaf | None = None
    reanalysis_root: ReanalysisRoot | None = None
    public_event_delta: PublicEventDelta | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent behavior traces."""
        if self.behavior_kind not in ("policy_sample", "improvement"):
            raise ValueError("decision behavior_kind is invalid")
        _validate_decision_token_trace(self)
        _validate_decision_engine_teacher_target(self)
        _validate_decision_macro_contract(self)
        _validate_decision_planner_behavior(self)


@dataclass(frozen=True)
class StateArrayBlock:
    """Padded numpy state features for a trajectory decision block."""

    card_ids: np.ndarray
    areas: np.ndarray
    owner_roles: np.ndarray
    token_kinds: np.ndarray
    scalars: np.ndarray
    last_attack_ids: np.ndarray
    padding_mask: np.ndarray
    attachment_card_ids: np.ndarray | None = None
    attachment_parent_indices: np.ndarray | None = None
    attachment_kinds: np.ndarray | None = None
    entity_slots: np.ndarray | None = None


@dataclass(frozen=True)
class OptionArrayBlock:
    """Padded numpy option features for a trajectory decision block."""

    option_types: np.ndarray
    contexts: np.ndarray
    entity_slots: np.ndarray
    entity_slot_mask: np.ndarray
    attack_ids: np.ndarray
    card_ids: np.ndarray
    scalars: np.ndarray
    dynamic_effect_features: np.ndarray
    dynamic_effect_masks: np.ndarray
    valid_options: np.ndarray
    min_counts: np.ndarray
    max_counts: np.ndarray


@dataclass(frozen=True)
class TrajectoryArrayBlock:
    """Compact per-decision arrays for learner queue transport."""

    states: StateArrayBlock
    options: OptionArrayBlock
    seats: np.ndarray
    decision_indices: np.ndarray
    action_offsets: np.ndarray
    action_indices: np.ndarray
    action_logprobs: np.ndarray
    value_preds: np.ndarray
    policy_versions: np.ndarray
    behavior_kinds: np.ndarray | None = None
    sampling_temperatures: np.ndarray | None = None
    token_offsets: np.ndarray | None = None
    token_logprobs: np.ndarray | None = None
    prefix_value_preds: np.ndarray | None = None
    stop_sampled: np.ndarray | None = None
    engine_teacher_action_offsets: np.ndarray | None = None
    engine_teacher_action_indices: np.ndarray | None = None
    engine_teacher_confidences: np.ndarray | None = None
    engine_teacher_weights: np.ndarray | None = None
    engine_teacher_masks: np.ndarray | None = None
    engine_teacher_search: SearchEvidenceArrayBlock | None = None
    factual_effect_targets: np.ndarray | None = None
    factual_actor_relations: np.ndarray | None = None
    factual_next_contexts: np.ndarray | None = None
    factual_transition_steps: np.ndarray | None = None
    executed_macros: ExecutedMacroArrayBlock | None = None
    macro_teacher: MacroTeacherArrayBlock | None = None
    planner_behavior: PlannerEvidenceArrayBlock | None = None
    executed_endpoint_value_indices: np.ndarray | None = None
    executed_endpoint_values: ExecutedEndpointValueArrayBlock | None = None
    public_events: PublicEventArrayBlock | None = None

    @property
    def decision_count(self) -> int:
        """Return number of decisions represented by this block."""
        return int(self.seats.shape[0])

    @property
    def has_public_events(self) -> bool:
        """Return whether every row carries a bounded public event delta."""
        if self.public_events is None:
            return False
        validate_public_event_array_block(self.public_events)
        if self.public_events.decision_count != self.decision_count:
            raise ValueError("public event rows differ from trajectory decisions")
        return True

    def action_at(self, index: int) -> tuple[int, ...]:
        """Return the variable-length selected action for one decision."""
        start = int(self.action_offsets[index])
        stop = int(self.action_offsets[index + 1])
        return tuple(int(value) for value in self.action_indices[start:stop])

    @property
    def has_token_trace(self) -> bool:
        """Return whether the block carries complete token behavior evidence."""
        fields = (
            self.token_offsets,
            self.token_logprobs,
            self.prefix_value_preds,
            self.stop_sampled,
        )
        if all(field is None for field in fields):
            return False
        if any(field is None for field in fields):
            raise ValueError("trajectory block has an incomplete token trace")
        return True

    def token_logprobs_at(self, index: int) -> tuple[float, ...] | None:
        """Return active behavior token log-probabilities for one decision."""
        if not self.has_token_trace:
            return None
        offsets = cast(np.ndarray, self.token_offsets)
        values = cast(np.ndarray, self.token_logprobs)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if start == stop:
            return None
        return tuple(float(value) for value in values[start:stop])

    def prefix_value_preds_at(self, index: int) -> tuple[float, ...] | None:
        """Return behavior prefix values for one decision."""
        if not self.has_token_trace:
            return None
        offsets = cast(np.ndarray, self.token_offsets)
        values = cast(np.ndarray, self.prefix_value_preds)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if start == stop:
            return None
        return tuple(float(value) for value in values[start:stop])

    @property
    def has_engine_teacher_targets(self) -> bool:
        """Return whether this block carries explicit sparse teacher evidence."""
        fields = (
            self.engine_teacher_action_offsets,
            self.engine_teacher_action_indices,
            self.engine_teacher_confidences,
            self.engine_teacher_weights,
            self.engine_teacher_masks,
        )
        if all(field is None for field in fields):
            return False
        if any(field is None for field in fields):
            raise ValueError("trajectory block has incomplete engine teacher fields")
        return True

    def engine_teacher_target_at(self, index: int) -> EngineTeacherTarget | None:
        """Return one sparse engine target from compact trajectory arrays."""
        if not self.has_engine_teacher_targets:
            return None
        masks = cast(np.ndarray, self.engine_teacher_masks)
        if not bool(masks[index]):
            return None
        offsets = cast(np.ndarray, self.engine_teacher_action_offsets)
        indices = cast(np.ndarray, self.engine_teacher_action_indices)
        confidences = cast(np.ndarray, self.engine_teacher_confidences)
        weights = cast(np.ndarray, self.engine_teacher_weights)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        return EngineTeacherTarget(
            action=tuple(int(value) for value in indices[start:stop]),
            confidence=float(confidences[index]),
            weight=float(weights[index]),
            search_evidence=(
                None
                if self.engine_teacher_search is None
                else self.engine_teacher_search.evidence_at(index)
            ),
        )

    @property
    def has_engine_teacher_search_evidence(self) -> bool:
        """Return whether any row carries complete-action search candidates."""
        if self.engine_teacher_search is None:
            return False
        validate_search_evidence_array_block(
            self.engine_teacher_search,
            decision_count=self.decision_count,
        )
        return bool(self.engine_teacher_search.masks.any())

    @property
    def has_factual_targets(self) -> bool:
        """Return whether every decision carries dense factual evidence."""
        fields = (
            self.factual_effect_targets,
            self.factual_actor_relations,
            self.factual_next_contexts,
        )
        if all(field is None for field in fields):
            return False
        if any(field is None for field in fields):
            raise ValueError("trajectory block has incomplete factual target fields")
        return True

    def factual_target_at(self, index: int) -> FactualTransitionTarget | None:
        """Return one dense factual target, or ``None`` for a legacy block."""
        if not self.has_factual_targets:
            return None
        features = cast(np.ndarray, self.factual_effect_targets)
        actor_relations = cast(np.ndarray, self.factual_actor_relations)
        next_contexts = cast(np.ndarray, self.factual_next_contexts)
        return FactualTransitionTarget(
            effect_features=tuple(float(value) for value in features[index]),
            actor_relation=FactualActorRelation(int(actor_relations[index])),
            next_context=int(next_contexts[index]),
        )

    def planner_behavior_at(self, index: int) -> PlannerBehaviorEvidence | None:
        """Return the immutable schema-9 branch for one decision."""
        if self.planner_behavior is None:
            return None
        return self.planner_behavior.evidence_at(index)

    def macro_teacher_at(self, index: int) -> MacroTeacherEvidence | None:
        """Return native counterfactual macro evidence for one decision."""
        if self.macro_teacher is None:
            return None
        return self.macro_teacher.evidence_at(index)

    def executed_endpoint_value_row_at(self, index: int) -> Any | None:
        """Return the actual endpoint row referenced by one decision."""
        indices = self.executed_endpoint_value_indices
        table = self.executed_endpoint_values
        if indices is None and table is None:
            return None
        if indices is None or table is None:
            raise ValueError("trajectory block has incomplete endpoint value fields")
        row_index = int(indices[index])
        return None if row_index < 0 else table.row_at(row_index)


@dataclass(frozen=True, eq=False)
class TrajectoryDeckContext:
    """Immutable canonical deck identities stored once for both game seats."""

    seat_card_ids: np.ndarray
    seat_signatures: tuple[str, str]

    def __post_init__(self) -> None:
        """Validate compact cards and their human-auditable signatures."""
        if self.seat_card_ids.dtype != np.uint16:
            raise TypeError("trajectory deck cards must use uint16 dtype")
        if self.seat_card_ids.shape != (2, DECK_SIZE):
            raise ValueError(
                "trajectory deck cards must have shape "
                f"[2, {DECK_SIZE}], got {self.seat_card_ids.shape}"
            )
        if len(self.seat_signatures) != 2:
            raise ValueError("trajectory deck context requires two seat signatures")
        cards = np.array(
            self.seat_card_ids,
            dtype=np.uint16,
            order="C",
            copy=True,
        )
        for seat in (0, 1):
            deck = canonicalize_deck(cards[seat])
            if tuple(int(value) for value in cards[seat]) != deck.card_ids:
                raise ValueError(
                    "trajectory deck card rows must be canonical and sorted"
                )
            if self.seat_signatures[seat] != deck.signature:
                raise ValueError(
                    f"trajectory deck signature for seat {seat} does not match cards"
                )
        cards.setflags(write=False)
        object.__setattr__(self, "seat_card_ids", cards)

    def __eq__(self, other: object) -> bool:
        """Compare array content without NumPy's ambiguous truth semantics."""
        return (
            isinstance(other, TrajectoryDeckContext)
            and self.seat_signatures == other.seat_signatures
            and bool(np.array_equal(self.seat_card_ids, other.seat_card_ids))
        )

    @classmethod
    def from_deck_pair(cls, deck_pair: Any) -> TrajectoryDeckContext:
        """Canonicalize the immutable two-seat deck pair from the engine."""
        try:
            decks = tuple(canonicalize_deck(deck) for deck in deck_pair)
        except TypeError as exc:
            raise ValueError(
                "trajectory decision is missing a valid deck pair"
            ) from exc
        if len(decks) != 2:
            raise ValueError("trajectory deck pair must contain exactly two decks")
        cards = np.asarray([deck.card_ids for deck in decks], dtype=np.uint16)
        return cls(
            seat_card_ids=cards,
            seat_signatures=(decks[0].signature, decks[1].signature),
        )

    def deck_for_seat(self, seat: int) -> CanonicalDeck:
        """Return the canonical acting-seat deck without exposing its opponent."""
        if seat not in (0, 1):
            raise ValueError(f"invalid trajectory seat: {seat}")
        return canonicalize_deck(self.seat_card_ids[seat])


@dataclass(frozen=True)
class GameMetadata:
    """Small per-game metadata attached to a learner trajectory."""

    deck_signature: str = ""
    opponent_deck_signature: str = ""
    seat_0_signature: str = ""
    seat_1_signature: str = ""
    opponent_name: str = ""
    opponent_tier: int = -1
    policy_version: int = 0
    episode_length: int = 0
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GameTrajectory:
    """One completed game worth of trainable decisions."""

    game_id: str
    seats_reward: tuple[float, float]
    decisions: tuple[DecisionRecord, ...]
    metadata: GameMetadata = GameMetadata()
    deck_context: TrajectoryDeckContext | None = None
    archive: CompletedTrajectory | None = None
    array_block: TrajectoryArrayBlock | None = None
    policy_artifacts: (
        tuple[PolicyArtifactIdentity | None, PolicyArtifactIdentity | None] | None
    ) = None

    def __post_init__(self) -> None:
        """Bind sparse endpoint labels to this immutable game envelope."""
        validate_game_trajectory_schema12_contract(self)
        validate_game_trajectory_schema11_contract(self)
        validate_game_trajectory_schema10_contract(self)
        validate_game_trajectory_schema9_contract(self)
        validate_game_trajectory_endpoint_values(self)

    def reward_for_seat(self, seat: int) -> float:
        """Return the terminal reward from one seat's perspective."""
        if seat not in (0, 1):
            raise ValueError(f"invalid rollout seat: {seat}")
        return float(self.seats_reward[seat])

    @property
    def decision_count(self) -> int:
        """Return trainable decision count for object or array representation."""
        if self.decisions:
            return len(self.decisions)
        if self.array_block is not None:
            return self.array_block.decision_count
        return 0


@dataclass
class _TensorGameBuffer:
    decisions: list[DecisionRecord] = field(default_factory=list)
    metadata: GameMetadata = GameMetadata()
    deck_context: TrajectoryDeckContext | None = None
    policy_artifacts: list[PolicyArtifactIdentity | None] = field(
        default_factory=lambda: [None, None]
    )
    policy_artifact_presence: list[bool | None] = field(
        default_factory=lambda: [None, None]
    )


class TensorTrajectoryRecorder:
    """Rollout recorder that emits tensor-ready completed game trajectories."""

    def __init__(
        self,
        *,
        reanalysis_root_callback: (
            Callable[[ReanalysisRoot], bool | None] | None
        ) = None,
    ) -> None:
        """Initialize empty trajectory buffers."""
        self._buffers: dict[str, _TensorGameBuffer] = {}
        self._completed: list[GameTrajectory] = []
        self._counters: Counter[str] = Counter()
        self._reanalysis_root_callback = reanalysis_root_callback

    @property
    def counters(self) -> Counter[str]:
        """Return recorder counters."""
        return Counter(self._counters)

    @property
    def pending_game_count(self) -> int:
        """Return buffered games not yet finalized."""
        return len(self._buffers)

    @property
    def completed_game_count(self) -> int:
        """Return completed trajectories waiting to be consumed."""
        return len(self._completed)

    def record(self, decision: Any) -> None:
        """Record one encoded rollout decision."""
        encoded_state = getattr(decision, "encoded_state", None)
        raw_options = getattr(decision, "encoded_options", ())
        encoded_options: EncodedOptionInput
        if isinstance(raw_options, EncodedOptionArrayFeatures):
            encoded_options = raw_options
        else:
            encoded_options = tuple(raw_options)
        min_count = getattr(decision, "min_count", None)
        max_count = getattr(decision, "max_count", None)
        if encoded_state is None or len(encoded_options) == 0:
            raise ValueError("rollout decision is missing encoded learner inputs")
        if min_count is None or max_count is None:
            raise ValueError("rollout decision is missing select count bounds")
        raw_token_logprobs = getattr(decision, "token_logprobs", None)
        raw_prefix_values = getattr(decision, "prefix_value_preds", None)
        raw_stop_sampled = getattr(decision, "stop_sampled", None)

        seat = int(decision.seat)
        if seat not in (0, 1):
            raise ValueError(f"invalid rollout seat: {seat}")
        deck_context = TrajectoryDeckContext.from_deck_pair(
            getattr(decision, "deck_pair", None)
        )
        game_id = str(decision.game_id)
        buffer = self._buffers.get(game_id)
        if buffer is None:
            buffer = _TensorGameBuffer(
                metadata=_metadata_from_decision(decision, deck_context),
                deck_context=deck_context,
            )
            self._buffers[game_id] = buffer
        elif buffer.deck_context != deck_context:
            raise ValueError("trajectory deck pair changed during a game")
        raw_artifact = getattr(decision, "policy_artifact_identity", None)
        artifact = (
            None
            if raw_artifact is None
            else (
                raw_artifact
                if isinstance(raw_artifact, PolicyArtifactIdentity)
                else PolicyArtifactIdentity.model_validate(raw_artifact)
            )
        )
        artifact_present = artifact is not None
        prior_presence = buffer.policy_artifact_presence[seat]
        if prior_presence is not None and prior_presence != artifact_present:
            raise ValueError("trajectory changed artifact mode within a seat")
        buffer.policy_artifact_presence[seat] = artifact_present
        existing_artifact = buffer.policy_artifacts[seat]
        if existing_artifact is not None and artifact != existing_artifact:
            raise ValueError("trajectory policy artifact changed within a seat")
        if artifact is not None:
            buffer.policy_artifacts[seat] = artifact
        decision_index = _next_decision_index(buffer, seat)
        reanalysis_root = None
        reanalysis_observation = getattr(decision, "reanalysis_observation", None)
        reanalysis_context_snapshots = getattr(
            decision,
            "reanalysis_context_snapshots",
            None,
        )
        if reanalysis_observation is not None:
            if reanalysis_context_snapshots is None:
                raise ValueError("reanalysis root is missing context snapshots")
            from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
                freeze_reanalysis_root,
            )

            reanalysis_root = freeze_reanalysis_root(
                game_id=game_id,
                seat=seat,
                decision_index=decision_index,
                state=encoded_state,
                options=encoded_options,
                min_count=int(min_count),
                max_count=int(max_count),
                deck_pair=(
                    deck_context.deck_for_seat(0).card_ids,
                    deck_context.deck_for_seat(1).card_ids,
                ),
                observation=reanalysis_observation,
                context_snapshots=reanalysis_context_snapshots,
                behavior_kind=cast(
                    BehaviorKind,
                    str(getattr(decision, "behavior_kind", "policy_sample")),
                ),
                behavior_action=decision.action,
                behavior_logprob=float(decision.action_logprob),
                sampling_temperature=float(
                    getattr(decision, "sampling_temperature", 1.0)
                ),
                policy_version=int(getattr(decision, "policy_version", 0)),
            )
            if self._reanalysis_root_callback is not None:
                admitted = self._reanalysis_root_callback(reanalysis_root)
                # ``None`` preserves compatibility with fire-and-forget local
                # callbacks. A queue-backed callback returns False explicitly,
                # retaining the root for one terminal-boundary retry.
                if admitted is not False:
                    reanalysis_root = None
                    self._counters["reanalysis_roots_published_immediately"] += 1
        buffer.decisions.append(
            DecisionRecord(
                seat=seat,
                decision_index=decision_index,
                state=encoded_state.without_layout(),
                options=encoded_options,
                min_count=int(min_count),
                max_count=int(max_count),
                action=tuple(int(index) for index in decision.action),
                action_logprob=float(decision.action_logprob),
                value_pred=float(decision.value_pred),
                policy_version=int(getattr(decision, "policy_version", 0)),
                behavior_kind=cast(
                    BehaviorKind,
                    str(getattr(decision, "behavior_kind", "policy_sample")),
                ),
                sampling_temperature=float(
                    getattr(decision, "sampling_temperature", 1.0)
                ),
                token_logprobs=(
                    None
                    if raw_token_logprobs is None
                    else tuple(float(value) for value in raw_token_logprobs)
                ),
                prefix_value_preds=(
                    None
                    if raw_prefix_values is None
                    else tuple(float(value) for value in raw_prefix_values)
                ),
                stop_sampled=(
                    None if raw_stop_sampled is None else bool(raw_stop_sampled)
                ),
                engine_teacher_target=getattr(
                    decision,
                    "engine_teacher_target",
                    None,
                ),
                factual_target=getattr(decision, "factual_target", None),
                factual_transition_steps=getattr(
                    decision,
                    "factual_transition_steps",
                    None,
                ),
                macro_credit_enabled=bool(
                    getattr(decision, "macro_credit_enabled", False)
                ),
                planner_behavior=getattr(decision, "planner_behavior", None),
                executed_endpoint_value_leaf=getattr(
                    decision,
                    "executed_endpoint_value_leaf",
                    None,
                ),
                reanalysis_root=reanalysis_root,
                public_event_delta=getattr(decision, "public_event_delta", None),
            )
        )
        self._counters["recorded_decisions"] += 1

    def finalize(self, finished: Any) -> None:
        """Finalize one game and emit a ``GameTrajectory`` if it has decisions."""
        game_id = str(finished.game_id)
        buffer = self._buffers.pop(game_id, None)
        if buffer is None:
            self._counters["finished_without_decisions"] += 1
            return
        winner_index = int(finished.winner_index)
        seats_reward = (
            _reward_for_seat(0, winner_index),
            _reward_for_seat(1, winner_index),
        )
        trajectory = GameTrajectory(
            game_id=game_id,
            seats_reward=seats_reward,
            decisions=tuple(buffer.decisions),
            metadata=buffer.metadata,
            deck_context=buffer.deck_context,
            array_block=build_trajectory_array_block(
                buffer.decisions,
                game_id=game_id,
                seats_reward=seats_reward,
            ),
            policy_artifacts=(
                (buffer.policy_artifacts[0], buffer.policy_artifacts[1])
                if any(item is not None for item in buffer.policy_artifacts)
                else None
            ),
        )
        self._completed.append(trajectory)
        self._counters["finalized_games"] += 1
        self._counters["finalized_decisions"] += len(trajectory.decisions)

    def discard(self, game_id: str, *, reason: str = "discarded") -> None:
        """Drop a buffered game without emitting a trajectory."""
        if self._buffers.pop(game_id, None) is None:
            self._counters["discard_missing_games"] += 1
        else:
            self._counters["discarded_games"] += 1
            self._counters[f"discarded_{reason}"] += 1

    def pop_completed(self) -> tuple[GameTrajectory, ...]:
        """Return and clear completed trajectories."""
        completed = tuple(self._completed)
        self._completed.clear()
        return completed


def _metadata_from_decision(
    decision: Any,
    deck_context: TrajectoryDeckContext,
) -> GameMetadata:
    raw_extra = getattr(decision, "training_metadata", {})
    extra = (
        {str(key): str(value) for key, value in raw_extra.items()}
        if isinstance(raw_extra, Mapping)
        else {}
    )
    seat_0_signature, seat_1_signature = deck_context.seat_signatures
    return GameMetadata(
        deck_signature=seat_0_signature,
        opponent_deck_signature=seat_1_signature,
        seat_0_signature=seat_0_signature,
        seat_1_signature=seat_1_signature,
        opponent_name=str(getattr(decision, "opponent_name", "")),
        opponent_tier=int(getattr(decision, "opponent_tier", -1)),
        policy_version=int(getattr(decision, "policy_version", 0)),
        extra=extra or None,
    )


def _next_decision_index(buffer: _TensorGameBuffer, seat: int) -> int:
    return sum(1 for decision in buffer.decisions if decision.seat == seat)


def _reward_for_seat(seat: int, winner_index: int) -> float:
    if winner_index == 2:
        return 0.0
    return 1.0 if winner_index == seat else -1.0


def build_trajectory_array_block(
    decisions: tuple[DecisionRecord, ...] | list[DecisionRecord],
    *,
    game_id: str = "",
    seats_reward: tuple[float, float] | None = None,
) -> TrajectoryArrayBlock | None:
    """Build a compact padded array block from decision records."""
    if not decisions:
        return None
    state_batch = collate_state_tokens([decision.state for decision in decisions])
    option_batch = collate_encoded_options(
        [decision.options for decision in decisions],
        min_counts=[decision.min_count for decision in decisions],
        max_counts=[decision.max_count for decision in decisions],
    )
    action_offsets = [0]
    action_indices: list[int] = []
    token_offsets = [0]
    token_logprobs: list[float] = []
    prefix_value_preds: list[float] = []
    planner_behavior = build_planner_evidence_array_block(
        [decision.planner_behavior for decision in decisions]
    )
    schema9_behavior = planner_behavior is not None
    token_trace_presence = tuple(
        decision.token_logprobs is not None for decision in decisions
    )
    if (
        any(token_trace_presence)
        and not all(token_trace_presence)
        and not schema9_behavior
    ):
        raise ValueError("cannot mix decisions with and without token traces")
    has_token_trace = (
        any(token_trace_presence) if schema9_behavior else all(token_trace_presence)
    )
    engine_teacher_arrays = optional_target_arrays(
        [decision.engine_teacher_target for decision in decisions]
    )
    engine_teacher_search = build_search_evidence_array_block(
        [
            None
            if decision.engine_teacher_target is None
            else decision.engine_teacher_target.search_evidence
            for decision in decisions
        ]
    )
    macro_teacher = build_macro_teacher_array_block(
        [decision.engine_teacher_target for decision in decisions]
    )
    macro_enabled = any(decision.macro_credit_enabled for decision in decisions)
    if macro_enabled and seats_reward is None:
        raise ValueError("schema-10 trajectory requires terminal seat rewards")
    executed_macros = (
        None
        if not macro_enabled
        else build_executed_macro_array_block(
            decisions,
            seats_reward=cast(tuple[float, float], seats_reward),
        )
    )
    endpoint_value_indices, endpoint_value_rows = _build_executed_endpoint_values(
        decisions,
        game_id=game_id,
        seats_reward=seats_reward,
    )
    factual_presence = [decision.factual_target is not None for decision in decisions]
    if any(factual_presence) and not all(factual_presence):
        raise ValueError("cannot mix decisions with and without factual targets")
    has_factual_targets = all(factual_presence)
    event_presence = [decision.public_event_delta is not None for decision in decisions]
    if any(event_presence) and not all(event_presence):
        raise ValueError("cannot mix decisions with and without public event deltas")
    public_events = (
        build_public_event_array_block(
            [
                cast(PublicEventDelta, decision.public_event_delta)
                for decision in decisions
            ]
        )
        if all(event_presence)
        else None
    )
    for decision in decisions:
        action_indices.extend(int(index) for index in decision.action)
        action_offsets.append(len(action_indices))
        if has_token_trace and decision.token_logprobs is not None:
            decision_logprobs = decision.token_logprobs
            decision_values = cast(tuple[float, ...], decision.prefix_value_preds)
            token_logprobs.extend(decision_logprobs)
            prefix_value_preds.extend(decision_values)
        if has_token_trace:
            token_offsets.append(len(token_logprobs))
    int32_max = np.iinfo(np.int32).max
    if (
        len(decisions) > int32_max
        or len(action_indices) > int32_max
        or len(token_logprobs) > int32_max
    ):
        raise ValueError("trajectory ragged arrays exceed int32 wire capacity")
    block = TrajectoryArrayBlock(
        states=StateArrayBlock(
            card_ids=_to_numpy(state_batch.card_ids),
            areas=_to_numpy(state_batch.areas),
            owner_roles=_to_numpy(state_batch.owner_roles),
            token_kinds=_to_numpy(state_batch.token_kinds),
            scalars=_to_numpy(state_batch.scalars),
            last_attack_ids=_to_numpy(state_batch.last_attack_ids),
            padding_mask=_to_numpy(state_batch.padding_mask),
            attachment_card_ids=_optional_to_numpy(state_batch.attachment_card_ids),
            attachment_parent_indices=_optional_to_numpy(
                state_batch.attachment_parent_indices
            ),
            attachment_kinds=_optional_to_numpy(state_batch.attachment_kinds),
            entity_slots=_optional_to_numpy(state_batch.entity_slots),
        ),
        options=OptionArrayBlock(
            option_types=_to_numpy(option_batch.option_types),
            contexts=_to_numpy(option_batch.contexts),
            entity_slots=_to_numpy(option_batch.entity_slots),
            entity_slot_mask=_to_numpy(option_batch.entity_slot_mask),
            attack_ids=_to_numpy(option_batch.attack_ids),
            card_ids=_to_numpy(option_batch.card_ids),
            scalars=_to_numpy(option_batch.scalars),
            dynamic_effect_features=_to_numpy(option_batch.dynamic_effect_features),
            dynamic_effect_masks=_to_numpy(option_batch.dynamic_effect_masks),
            valid_options=_to_numpy(option_batch.valid_options),
            min_counts=_to_numpy(option_batch.min_counts),
            max_counts=_to_numpy(option_batch.max_counts),
        ),
        seats=np.asarray([decision.seat for decision in decisions], dtype=np.int8),
        decision_indices=np.asarray(
            [decision.decision_index for decision in decisions],
            dtype=np.int32,
        ),
        action_offsets=np.asarray(action_offsets, dtype=np.int32),
        action_indices=np.asarray(action_indices, dtype=np.int32),
        action_logprobs=np.asarray(
            [decision.action_logprob for decision in decisions],
            dtype=np.float32,
        ),
        value_preds=np.asarray(
            [decision.value_pred for decision in decisions],
            dtype=np.float32,
        ),
        policy_versions=np.asarray(
            [decision.policy_version for decision in decisions],
            dtype=np.int32,
        ),
        behavior_kinds=np.asarray(
            [
                0 if decision.behavior_kind == "policy_sample" else 1
                for decision in decisions
            ],
            dtype=np.uint8,
        ),
        sampling_temperatures=np.asarray(
            [decision.sampling_temperature for decision in decisions],
            dtype=np.float32,
        ),
        token_offsets=(
            np.asarray(token_offsets, dtype=np.int32) if has_token_trace else None
        ),
        token_logprobs=(
            np.asarray(token_logprobs, dtype=np.float32) if has_token_trace else None
        ),
        prefix_value_preds=(
            np.asarray(prefix_value_preds, dtype=np.float32)
            if has_token_trace
            else None
        ),
        stop_sampled=(
            np.asarray(
                [bool(decision.stop_sampled) for decision in decisions],
                dtype=np.bool_,
            )
            if has_token_trace
            else None
        ),
        engine_teacher_action_offsets=(
            None
            if engine_teacher_arrays is None
            else np.asarray(engine_teacher_arrays[0], dtype=np.int32)
        ),
        engine_teacher_action_indices=(
            None
            if engine_teacher_arrays is None
            else np.asarray(engine_teacher_arrays[1], dtype=np.int32)
        ),
        engine_teacher_confidences=(
            None
            if engine_teacher_arrays is None
            else np.asarray(engine_teacher_arrays[2], dtype=np.float32)
        ),
        engine_teacher_weights=(
            None
            if engine_teacher_arrays is None
            else np.asarray(engine_teacher_arrays[3], dtype=np.float32)
        ),
        engine_teacher_masks=(
            None
            if engine_teacher_arrays is None
            else np.asarray(engine_teacher_arrays[4], dtype=np.bool_)
        ),
        engine_teacher_search=engine_teacher_search,
        factual_effect_targets=(
            np.asarray(
                [
                    cast(
                        FactualTransitionTarget, decision.factual_target
                    ).effect_features
                    for decision in decisions
                ],
                dtype=np.float32,
            )
            if has_factual_targets
            else None
        ),
        factual_actor_relations=(
            np.asarray(
                [
                    int(
                        cast(
                            FactualTransitionTarget,
                            decision.factual_target,
                        ).actor_relation
                    )
                    for decision in decisions
                ],
                dtype=np.uint8,
            )
            if has_factual_targets
            else None
        ),
        factual_next_contexts=(
            np.asarray(
                [
                    cast(
                        FactualTransitionTarget,
                        decision.factual_target,
                    ).next_context
                    for decision in decisions
                ],
                dtype=np.uint8,
            )
            if has_factual_targets
            else None
        ),
        factual_transition_steps=(
            np.asarray(
                [
                    cast(int, decision.factual_transition_steps)
                    for decision in decisions
                ],
                dtype=np.int32,
            )
            if executed_macros is not None
            else None
        ),
        executed_macros=executed_macros,
        macro_teacher=macro_teacher,
        planner_behavior=planner_behavior,
        executed_endpoint_value_indices=endpoint_value_indices,
        executed_endpoint_values=endpoint_value_rows,
        public_events=public_events,
    )
    validate_trajectory_array_block(block)
    return block


def validate_trajectory_array_block(
    block: TrajectoryArrayBlock,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Validate one compact block while materializing planner rows once."""
    return _validate_trajectory_array_block(block, planner_evidence=None)


def _validate_trajectory_array_block(
    block: TrajectoryArrayBlock,
    *,
    planner_evidence: tuple[PlannerBehaviorEvidence, ...] | None,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Validate one block, optionally reusing rows validated in this module."""
    behavior_kinds = block.behavior_kinds
    if behavior_kinds is not None:
        if behavior_kinds.dtype != np.uint8 or behavior_kinds.shape != (
            block.decision_count,
        ):
            raise ValueError("behavior_kinds must be uint8 [decision_count]")
        if not bool(np.isin(behavior_kinds, (0, 1)).all()):
            raise ValueError("behavior_kinds contains an unsupported lane")
    planner_rows = _validated_planner_evidence_rows(
        block,
        planner_evidence=planner_evidence,
    )
    _validate_trajectory_array_token_trace(
        block,
        planner_evidence=planner_rows,
    )
    validate_trajectory_array_engine_teacher(block)
    validate_trajectory_array_factual_targets(block)
    validate_trajectory_array_macro_credit(block)
    _validate_trajectory_array_planner_behavior(
        block,
        planner_evidence=planner_rows,
    )
    validate_trajectory_array_endpoint_values(block)
    if block.public_events is not None:
        validate_public_event_array_block(block.public_events)
        if block.public_events.decision_count != block.decision_count:
            raise ValueError("public event rows differ from trajectory decisions")
    return planner_rows


def validate_trajectory_array_token_trace(
    block: TrajectoryArrayBlock,
) -> None:
    """Reject corrupt flattened token evidence in one compact block."""
    if not block.has_token_trace:
        return
    planner_rows = _validated_planner_evidence_rows(
        block,
        planner_evidence=None,
    )
    _validate_trajectory_array_token_trace(
        block,
        planner_evidence=planner_rows,
    )


def _validate_trajectory_array_token_trace(
    block: TrajectoryArrayBlock,
    *,
    planner_evidence: tuple[PlannerBehaviorEvidence, ...] | None,
) -> None:
    """Validate token traces against rows already checked in this module."""
    if not block.has_token_trace:
        return
    planner_rows = planner_evidence
    offsets = cast(np.ndarray, block.token_offsets)
    logprobs = cast(np.ndarray, block.token_logprobs)
    prefix_values = cast(np.ndarray, block.prefix_value_preds)
    stop_sampled = cast(np.ndarray, block.stop_sampled)
    decision_count = block.decision_count
    if offsets.ndim != 1 or offsets.shape[0] != decision_count + 1:
        raise ValueError("token_offsets must have shape [decision_count + 1]")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise TypeError("token_offsets must use an integer dtype")
    if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("token_offsets must start at zero and be monotonic")
    token_count = int(offsets[-1])
    if (
        logprobs.ndim != 1
        or prefix_values.ndim != 1
        or logprobs.shape[0] != token_count
        or prefix_values.shape[0] != token_count
    ):
        raise ValueError("flattened token trace arrays do not align with offsets")
    if stop_sampled.ndim != 1 or stop_sampled.shape[0] != decision_count:
        raise ValueError("stop_sampled must have shape [decision_count]")
    if not bool(np.isfinite(logprobs).all()) or not bool(
        np.isfinite(prefix_values).all()
    ):
        raise ValueError("flattened token trace values must be finite")
    for index in range(decision_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        planner_row = None if planner_rows is None else planner_rows[index]
        if (
            planner_row is not None
            and planner_row.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
        ):
            if stop != start or bool(stop_sampled[index]):
                raise ValueError(
                    "planner categorical rows cannot carry token behavior evidence"
                )
            continue
        if stop <= start:
            raise ValueError("autoregressive behavior rows need a token trace")
        action_length = int(block.action_offsets[index + 1]) - int(
            block.action_offsets[index]
        )
        expected_tokens = action_length + int(bool(stop_sampled[index]))
        if stop - start != expected_tokens:
            raise ValueError("token trace length does not match action termination")
        expected_stop_sampled = action_length < int(block.options.max_counts[index])
        if bool(stop_sampled[index]) != expected_stop_sampled:
            raise ValueError("stop_sampled does not match max_count termination")
        if not math.isclose(
            float(logprobs[start:stop].sum(dtype=np.float64)),
            float(block.action_logprobs[index]),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("token log-probabilities do not sum to action_logprob")
        if not math.isclose(
            float(prefix_values[start]),
            float(block.value_preds[index]),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("first prefix value does not match value_pred")


def validate_trajectory_array_engine_teacher(block: TrajectoryArrayBlock) -> None:
    """Reject corrupt sparse engine-teacher targets in a compact block."""
    if not block.has_engine_teacher_targets:
        if block.engine_teacher_search is not None:
            raise ValueError("search evidence requires engine teacher targets")
        return
    offsets = cast(np.ndarray, block.engine_teacher_action_offsets)
    indices = cast(np.ndarray, block.engine_teacher_action_indices)
    confidences = cast(np.ndarray, block.engine_teacher_confidences)
    weights = cast(np.ndarray, block.engine_teacher_weights)
    masks = cast(np.ndarray, block.engine_teacher_masks)
    decision_count = block.decision_count
    if offsets.ndim != 1 or offsets.shape[0] != decision_count + 1:
        raise ValueError(
            "engine_teacher_action_offsets must have shape [decision_count + 1]"
        )
    if not np.issubdtype(offsets.dtype, np.integer):
        raise TypeError("engine teacher offsets must use an integer dtype")
    if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("engine teacher offsets must start at zero and be monotonic")
    if indices.ndim != 1 or indices.shape[0] != int(offsets[-1]):
        raise ValueError("engine teacher indices do not align with offsets")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("engine teacher indices must use an integer dtype")
    if masks.dtype != np.bool_ or masks.shape != (decision_count,):
        raise ValueError("engine_teacher_masks must be bool [decision_count]")
    search = block.engine_teacher_search
    if search is not None:
        validate_search_evidence_array_block(
            search,
            decision_count=decision_count,
        )
        if bool(np.any(search.masks & ~masks)):
            raise ValueError("search evidence requires a valid engine teacher row")
    for name, values in (
        ("engine_teacher_confidences", confidences),
        ("engine_teacher_weights", weights),
    ):
        if values.ndim != 1 or values.shape != (decision_count,):
            raise ValueError(f"{name} must have shape [decision_count]")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"{name} must contain finite values")
    for row_index in range(decision_count):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if not bool(masks[row_index]):
            if stop != start:
                raise ValueError("masked engine teacher rows cannot carry an action")
            if confidences[row_index] != 0.0 or weights[row_index] != 0.0:
                raise ValueError("masked engine teacher rows must have zero weights")
            continue
        confidence = float(confidences[row_index])
        weight = float(weights[row_index])
        if not 0.0 < confidence <= 1.0 or weight <= 0.0:
            raise ValueError("valid engine teacher rows require positive evidence")
        action = tuple(int(value) for value in indices[start:stop])
        _validate_engine_teacher_action(
            action,
            min_count=int(block.options.min_counts[row_index]),
            max_count=int(block.options.max_counts[row_index]),
            valid_options=tuple(
                bool(value) for value in block.options.valid_options[row_index]
            ),
            context=int(block.options.contexts[row_index, 0]),
        )
        if search is not None:
            evidence = search.evidence_at(row_index)
            if evidence is not None:
                for candidate in evidence.actions:
                    _validate_engine_teacher_action(
                        candidate,
                        min_count=int(block.options.min_counts[row_index]),
                        max_count=int(block.options.max_counts[row_index]),
                        valid_options=tuple(
                            bool(value)
                            for value in block.options.valid_options[row_index]
                        ),
                        context=int(block.options.contexts[row_index, 0]),
                    )
                if action not in evidence.actions:
                    raise ValueError(
                        "engine teacher target must be present in search candidates"
                    )


def validate_trajectory_array_factual_targets(block: TrajectoryArrayBlock) -> None:
    """Reject incomplete or corrupt dense factual evidence."""
    if not block.has_factual_targets:
        return
    features = cast(np.ndarray, block.factual_effect_targets)
    actor_relations = cast(np.ndarray, block.factual_actor_relations)
    next_contexts = cast(np.ndarray, block.factual_next_contexts)
    expected_feature_shape = (block.decision_count, DYNAMIC_EFFECT_FEATURE_SIZE)
    if features.dtype != np.float32:
        raise TypeError("factual_effect_targets must use float32 dtype")
    if features.shape != expected_feature_shape:
        raise ValueError(
            f"factual_effect_targets must have shape {expected_feature_shape}"
        )
    if not bool(np.isfinite(features).all()):
        raise ValueError("factual_effect_targets must contain finite values")
    if actor_relations.dtype != np.uint8:
        raise TypeError("factual_actor_relations must use uint8 dtype")
    if actor_relations.shape != (block.decision_count,):
        raise ValueError("factual_actor_relations must have shape [decision_count]")
    valid_relations = np.asarray([int(value) for value in FactualActorRelation])
    if not bool(np.isin(actor_relations, valid_relations).all()):
        raise ValueError("factual_actor_relations contains an invalid relation")
    if next_contexts.dtype != np.uint8:
        raise TypeError("factual_next_contexts must use uint8 dtype")
    if next_contexts.shape != (block.decision_count,):
        raise ValueError("factual_next_contexts must have shape [decision_count]")
    if not bool((next_contexts < FACTUAL_NEXT_CONTEXT_COUNT).all()):
        raise ValueError("factual_next_contexts contains an invalid context")
    terminal = actor_relations == int(FactualActorRelation.TERMINAL)
    terminal_context = next_contexts == FACTUAL_NEXT_CONTEXT_COUNT - 1
    if not bool(np.array_equal(terminal, terminal_context)):
        raise ValueError("factual terminal relation/context labels disagree")


def validate_trajectory_array_macro_credit(block: TrajectoryArrayBlock) -> None:
    """Validate schema-10 chain, step-count, and native teacher evidence."""
    macros = block.executed_macros
    steps = block.factual_transition_steps
    teacher = block.macro_teacher
    if macros is None:
        if steps is not None or teacher is not None:
            raise ValueError("macro fields require an executed-macro contract")
        return
    if not block.has_factual_targets:
        raise ValueError("executed macro rows require dense factual targets")
    if (
        steps is None
        or steps.dtype != np.int32
        or steps.shape != (block.decision_count,)
    ):
        raise ValueError("macro factual step counts must be int32 [decision_count]")
    if bool((steps <= 0).any()):
        raise ValueError("macro factual step counts must be positive")
    validate_executed_macro_array_block(
        macros,
        decision_count=block.decision_count,
    )
    for row_index in range(macros.row_count):
        target = macros.row_at(row_index)
        positions = (
            target.root_decision_position,
            *target.continuation_decision_positions,
        )
        if sum(int(steps[position]) for position in positions) != target.engine_steps:
            raise ValueError("macro engine steps differ from referenced decisions")
        root_seat = int(block.seats[target.root_decision_position])
        if any(int(block.seats[position]) != root_seat for position in positions):
            raise ValueError("executed macro chain crosses trajectory seats")
        expected_indices = tuple(
            range(
                int(block.decision_indices[target.root_decision_position]),
                int(block.decision_indices[target.root_decision_position])
                + len(positions),
            )
        )
        actual_indices = tuple(
            int(block.decision_indices[position]) for position in positions
        )
        if actual_indices != expected_indices:
            raise ValueError("executed macro decision indices are discontinuous")
    if teacher is None:
        return
    validate_macro_teacher_array_block(teacher)
    if teacher.decision_count != block.decision_count:
        raise ValueError("macro teacher rows differ from trajectory decisions")
    if not block.has_engine_teacher_targets:
        raise ValueError("macro teacher evidence requires sparse teacher targets")
    teacher_masks = cast(np.ndarray, block.engine_teacher_masks)
    if bool(np.any(teacher.masks & ~teacher_masks)):
        raise ValueError("macro evidence requires a valid teacher action")
    for index in np.flatnonzero(teacher.masks):
        evidence = teacher.evidence_at(int(index))
        teacher_target = block.engine_teacher_target_at(int(index))
        if (
            evidence is None
            or teacher_target is None
            or teacher_target.action != evidence.target_action
        ):
            raise ValueError("macro teacher action differs from its evidence")


def validate_trajectory_array_planner_behavior(
    block: TrajectoryArrayBlock,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Validate schema-9 branch identity against actual behavior rows."""
    planner_evidence = _validated_planner_evidence_rows(
        block,
        planner_evidence=None,
    )
    return _validate_trajectory_array_planner_behavior(
        block,
        planner_evidence=planner_evidence,
    )


def _validate_trajectory_array_planner_behavior(
    block: TrajectoryArrayBlock,
    *,
    planner_evidence: tuple[PlannerBehaviorEvidence, ...] | None,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Validate branch behavior against rows checked in this module."""
    planner = block.planner_behavior
    if planner is None:
        if planner_evidence is not None:
            raise ValueError("planner evidence rows require a planner array block")
        return None
    evidence_rows = _validated_planner_evidence_rows(
        block,
        planner_evidence=planner_evidence,
    )
    if evidence_rows is None:
        raise AssertionError("planner block validation did not return evidence")
    for index, evidence in enumerate(evidence_rows):
        if evidence.policy_version != int(block.policy_versions[index]):
            raise ValueError("planner evidence policy version differs from behavior")
        if evidence.branch is PlannerBehaviorBranch.BASE_FALLBACK:
            if block.token_logprobs_at(index) is None:
                raise ValueError("schema-9 fallback rows require a token trace")
            continue
        if block.token_logprobs_at(index) is not None:
            raise ValueError("planner categorical behavior cannot carry token traces")
        selected_action = evidence.selected_action
        if selected_action != block.action_at(index):
            raise ValueError("planner selected candidate differs from behavior action")
        selected_logprob = evidence.selected_old_logprob
        if selected_logprob is None or not math.isclose(
            selected_logprob,
            float(block.action_logprobs[index]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError("planner old log-probability differs from behavior")
        if not math.isclose(
            evidence.planner_temperature,
            float(cast(np.ndarray, block.sampling_temperatures)[index]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError("planner temperature differs from behavior metadata")
        for candidate in evidence.candidates:
            _validate_engine_teacher_action(
                candidate.action,
                min_count=int(block.options.min_counts[index]),
                max_count=int(block.options.max_counts[index]),
                valid_options=tuple(
                    bool(value) for value in block.options.valid_options[index]
                ),
                context=int(block.options.contexts[index, 0]),
            )
    return evidence_rows


def _validated_planner_evidence_rows(
    block: TrajectoryArrayBlock,
    *,
    planner_evidence: tuple[PlannerBehaviorEvidence, ...] | None,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Return already-validated rows or reconstruct them from the CSR block."""
    planner = block.planner_behavior
    if planner is None:
        if planner_evidence is not None:
            raise ValueError("planner evidence rows require a planner array block")
        return None
    if planner_evidence is None:
        return validate_planner_evidence_array_block(
            planner,
            decision_count=block.decision_count,
        )
    if len(planner_evidence) != block.decision_count:
        raise ValueError("planner evidence row count differs from trajectory")
    return planner_evidence


def validate_trajectory_array_endpoint_values(block: TrajectoryArrayBlock) -> None:
    """Validate sparse decision indices into the deduplicated endpoint table."""
    indices = block.executed_endpoint_value_indices
    table = block.executed_endpoint_values
    if indices is None and table is None:
        return
    if indices is None or table is None:
        raise ValueError("trajectory block has incomplete endpoint value fields")
    if indices.dtype != np.int32 or indices.shape != (block.decision_count,):
        raise ValueError("executed endpoint indices must be int32 [decision_count]")
    validate_executed_endpoint_value_array_block(table)
    if bool(np.any(indices < -1)) or bool(np.any(indices >= table.row_count)):
        raise ValueError("executed endpoint index is outside the endpoint table")
    referenced = {int(value) for value in indices if int(value) >= 0}
    if referenced != set(range(table.row_count)):
        raise ValueError("executed endpoint table contains unreferenced rows")


def validate_game_trajectory_endpoint_values(trajectory: GameTrajectory) -> None:
    """Reject endpoint evidence that is not bound to its enclosing game."""
    block = trajectory.array_block
    if block is None:
        return
    validate_trajectory_array_endpoint_values(block)
    indices = block.executed_endpoint_value_indices
    table = block.executed_endpoint_values
    if indices is None and table is None:
        return
    if indices is None or table is None:
        raise ValueError("trajectory has incomplete endpoint value fields")
    if not trajectory.game_id:
        raise ValueError("executed endpoint rows require a non-empty game id")
    if len(trajectory.seats_reward) != 2:
        raise ValueError("executed endpoint rows require two seat outcomes")
    rewards = tuple(float(value) for value in trajectory.seats_reward)
    if any(
        not math.isfinite(value) or value not in (-1.0, 0.0, 1.0) for value in rewards
    ):
        raise ValueError("executed endpoint rows require root-perspective W/D/L")
    if not math.isclose(rewards[1], -rewards[0], abs_tol=0.0):
        raise ValueError("executed endpoint seat outcomes must be opposites")
    expected_game_fingerprint = _trajectory_game_fingerprint(trajectory.game_id)
    for row_index in range(table.row_count):
        row = table.row_at(row_index)
        if row.game_fingerprint != expected_game_fingerprint:
            raise ValueError(
                "executed endpoint game fingerprint differs from trajectory"
            )
        expected_outcome = int(rewards[row.root_player])
        if row.final_root_outcome != expected_outcome:
            raise ValueError("executed endpoint outcome differs from trajectory reward")
    for decision_index, row_index_raw in enumerate(indices):
        row_index = int(row_index_raw)
        if row_index < 0:
            continue
        if table.row_at(row_index).root_player != int(block.seats[decision_index]):
            raise ValueError("executed endpoint root player differs from decision seat")


def validate_game_trajectory_schema9_contract(
    trajectory: GameTrajectory,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Apply the same schema-9 semantic contract to local and wire games."""
    if trajectory.decisions:
        planner_presence = tuple(
            decision.planner_behavior is not None for decision in trajectory.decisions
        )
        if not any(planner_presence):
            return None
        if not all(planner_presence):
            raise ValueError("cannot mix schema-9 and legacy decisions in one game")
        if trajectory.deck_context is None:
            raise ValueError("schema 9 planner trajectories require deck context")
        if any(decision.factual_target is None for decision in trajectory.decisions):
            raise ValueError("schema 9 planner trajectories require factual targets")
        if any(
            decision.engine_teacher_target is not None
            for decision in trajectory.decisions
        ):
            raise ValueError("schema 9 cannot contain legacy engine teacher evidence")
        return tuple(
            cast(PlannerBehaviorEvidence, decision.planner_behavior)
            for decision in trajectory.decisions
        )
    block = trajectory.array_block
    if block is None:
        return None
    endpoint_fields_present = (
        block.executed_endpoint_value_indices is not None
        or block.executed_endpoint_values is not None
    )
    if block.planner_behavior is None:
        if endpoint_fields_present and block.executed_macros is None:
            raise ValueError(
                "executed endpoint rows require schema-9 behavior evidence"
            )
        return None
    if trajectory.deck_context is None or not block.has_factual_targets:
        raise ValueError(
            "schema 9 planner trajectories require deck and factual context"
        )
    if block.has_engine_teacher_targets or block.engine_teacher_search is not None:
        raise ValueError("schema 9 cannot contain legacy engine teacher evidence")
    return validate_trajectory_array_planner_behavior(block)


def validate_game_trajectory_schema10_contract(
    trajectory: GameTrajectory,
) -> None:
    """Bind macro rows to one factual, deck-aware, single-behavior game."""
    if trajectory.decisions:
        presence = tuple(
            bool(getattr(decision, "macro_credit_enabled", False))
            for decision in trajectory.decisions
        )
        if any(presence) and not all(presence):
            raise ValueError("cannot mix schema-10 and legacy decisions in one game")
        if not any(presence):
            return
        if trajectory.deck_context is None:
            raise ValueError("schema-10 trajectories require deck context")
        if any(
            decision.factual_target is None or decision.factual_transition_steps is None
            for decision in trajectory.decisions
        ):
            raise ValueError("schema-10 trajectories require factual step evidence")
        if any(
            decision.planner_behavior is not None for decision in trajectory.decisions
        ):
            raise ValueError("schema 10 cannot contain pre-action planner behavior")
        return
    block = trajectory.array_block
    if block is None or block.executed_macros is None:
        return
    if trajectory.deck_context is None:
        raise ValueError("schema-10 trajectories require deck context")
    if block.planner_behavior is not None:
        raise ValueError("schema 10 cannot contain pre-action planner behavior")
    validate_trajectory_array_macro_credit(block)


def validate_game_trajectory_schema11_contract(trajectory: GameTrajectory) -> None:
    """Bind public-event rows to one deck-aware PPO-only game sequence."""
    if trajectory.decisions:
        presence = tuple(
            getattr(decision, "public_event_delta", None) is not None
            for decision in trajectory.decisions
        )
        if any(presence) and not all(presence):
            raise ValueError("cannot mix schema-11 and legacy decisions in one game")
        object_has_events = all(presence)
        block = trajectory.array_block
        if block is not None:
            array_has_events = block.public_events is not None
            if object_has_events != array_has_events:
                raise ValueError(
                    "schema-11 object and array event representations disagree"
                )
            if object_has_events:
                event_block = cast(PublicEventArrayBlock, block.public_events)
                validate_public_event_array_block(event_block)
                if event_block.decision_count != len(trajectory.decisions):
                    raise ValueError(
                        "schema-11 object and array event row counts disagree"
                    )
                if any(
                    decision.public_event_delta != event_block.delta_at(index)
                    for index, decision in enumerate(trajectory.decisions)
                ):
                    raise ValueError(
                        "schema-11 object and array event contents disagree"
                    )
        if not any(presence):
            return
        if trajectory.deck_context is None:
            raise ValueError("schema-11 trajectories require deck context")
        if any(
            decision.token_logprobs is None
            or decision.prefix_value_preds is None
            or decision.stop_sampled is None
            for decision in trajectory.decisions
        ):
            raise ValueError("schema-11 trajectories require token behavior traces")
        if any(
            decision.behavior_kind != "policy_sample"
            or decision.engine_teacher_target is not None
            or decision.factual_target is not None
            or decision.macro_credit_enabled
            or decision.planner_behavior is not None
            or decision.reanalysis_root is not None
            for decision in trajectory.decisions
        ):
            raise ValueError("schema 11 is a PPO-only public-event trajectory")
        if trajectory.array_block is not None:
            _validate_schema11_array_contract(trajectory)
        return

    _validate_schema11_array_contract(trajectory)


def validate_game_trajectory_schema12_contract(trajectory: GameTrajectory) -> None:
    """Bind each recurrent game-seat sequence to one immutable artifact."""
    artifacts = trajectory.policy_artifacts
    if artifacts is None:
        return
    if len(artifacts) != 2:
        raise ValueError("schema-12 trajectory requires two seat artifact slots")
    if trajectory.deck_context is None:
        raise ValueError("schema-12 trajectories require exact deck context")
    if trajectory.decisions:
        rows = tuple(trajectory.decisions)
        if any(row.public_event_delta is None for row in rows):
            raise ValueError("schema-12 trajectories require public event rows")
        active_seats = {row.seat for row in rows}
    else:
        block = trajectory.array_block
        if block is None or not block.has_public_events:
            raise ValueError("schema-12 trajectories require public event arrays")
        active_seats = {int(seat) for seat in block.seats.tolist()}
    for seat in active_seats:
        if artifacts[seat] is None:
            raise ValueError(
                "schema-12 recurrent sequence is missing its policy artifact"
            )
    for seat, artifact in enumerate(artifacts):
        if artifact is not None and seat not in active_seats:
            raise ValueError("schema-12 artifact has no aligned decision sequence")


def _validate_schema11_array_contract(trajectory: GameTrajectory) -> None:
    """Validate the compact half of a schema-11 trajectory contract."""
    block = trajectory.array_block
    if block is None or not block.has_public_events:
        return
    if trajectory.deck_context is None or not block.has_token_trace:
        raise ValueError(
            "schema-11 trajectories require deck and token behavior context"
        )
    if (
        block.behavior_kinds is None
        or bool(np.any(block.behavior_kinds != 0))
        or block.has_engine_teacher_targets
        or block.engine_teacher_search is not None
        or block.has_factual_targets
        or block.factual_transition_steps is not None
        or block.executed_macros is not None
        or block.macro_teacher is not None
        or block.planner_behavior is not None
        or block.executed_endpoint_value_indices is not None
        or block.executed_endpoint_values is not None
    ):
        raise ValueError("schema 11 is a PPO-only public-event trajectory")


def validate_game_trajectory_array_block(
    trajectory: GameTrajectory,
) -> tuple[PlannerBehaviorEvidence, ...] | None:
    """Validate a transport-ready game and materialize planner rows once."""
    validate_game_trajectory_schema12_contract(trajectory)
    validate_game_trajectory_schema11_contract(trajectory)
    validate_game_trajectory_schema10_contract(trajectory)
    planner_evidence = validate_game_trajectory_schema9_contract(trajectory)
    validate_game_trajectory_endpoint_values(trajectory)
    block = trajectory.array_block
    if block is None:
        raise ValueError("transport-ready trajectories must be array-backed")
    return _validate_trajectory_array_block(
        block,
        planner_evidence=planner_evidence,
    )


def _validate_decision_token_trace(decision: DecisionRecord) -> None:
    fields = (
        decision.token_logprobs,
        decision.prefix_value_preds,
        decision.stop_sampled,
    )
    if all(field is None for field in fields):
        return
    if any(field is None for field in fields):
        raise ValueError("decision has an incomplete token trace")
    token_logprobs = cast(tuple[float, ...], decision.token_logprobs)
    prefix_values = cast(tuple[float, ...], decision.prefix_value_preds)
    if not token_logprobs:
        raise ValueError("decision token trace must contain an active token")
    if len(token_logprobs) != len(prefix_values):
        raise ValueError("token log-probabilities and prefix values must align")
    if not all(math.isfinite(value) for value in (*token_logprobs, *prefix_values)):
        raise ValueError("decision token trace values must be finite")
    if not math.isclose(
        sum(token_logprobs),
        decision.action_logprob,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("token log-probabilities do not sum to action_logprob")
    if not math.isclose(
        prefix_values[0],
        decision.value_pred,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("first prefix value does not match value_pred")
    expected_tokens = len(decision.action) + int(bool(decision.stop_sampled))
    if len(token_logprobs) != expected_tokens:
        raise ValueError("token trace length does not match action termination")
    expected_stop_sampled = len(decision.action) < decision.max_count
    if bool(decision.stop_sampled) != expected_stop_sampled:
        raise ValueError("stop_sampled does not match max_count termination")


def _validate_decision_engine_teacher_target(decision: DecisionRecord) -> None:
    target = decision.engine_teacher_target
    if target is None:
        return
    _validate_engine_teacher_action(
        target.action,
        min_count=decision.min_count,
        max_count=decision.max_count,
        valid_options=(True,) * len(decision.options),
        context=_decision_option_context(decision),
    )
    if target.search_evidence is not None:
        for candidate in target.search_evidence.actions:
            _validate_engine_teacher_action(
                candidate,
                min_count=decision.min_count,
                max_count=decision.max_count,
                valid_options=(True,) * len(decision.options),
                context=_decision_option_context(decision),
            )


def _validate_decision_macro_contract(decision: DecisionRecord) -> None:
    """Keep schema-10 object rows complete before compact assembly."""
    macro_evidence = (
        None
        if decision.engine_teacher_target is None
        else decision.engine_teacher_target.macro_evidence
    )
    if not decision.macro_credit_enabled:
        if decision.factual_transition_steps is not None or macro_evidence is not None:
            raise ValueError("macro-only decision fields require schema 10")
        return
    if decision.factual_target is None:
        raise ValueError("schema-10 decisions require factual targets")
    if (
        decision.factual_transition_steps is None
        or decision.factual_transition_steps <= 0
    ):
        raise ValueError("schema-10 decisions require positive factual step counts")


def _validate_decision_planner_behavior(decision: DecisionRecord) -> None:
    evidence = decision.planner_behavior
    if evidence is None:
        if (
            decision.executed_endpoint_value_leaf is not None
            and not decision.macro_credit_enabled
        ):
            raise ValueError("endpoint value rows require schema-9 behavior evidence")
        return
    if evidence.policy_version != decision.policy_version:
        raise ValueError("planner evidence policy version differs from behavior")
    if evidence.branch is PlannerBehaviorBranch.BASE_FALLBACK:
        if decision.token_logprobs is None:
            raise ValueError("schema-9 fallback behavior requires a token trace")
        return
    if decision.token_logprobs is not None:
        raise ValueError("planner categorical behavior cannot carry token log-probs")
    if evidence.selected_action != decision.action:
        raise ValueError("planner selected candidate differs from behavior action")
    selected_logprob = evidence.selected_old_logprob
    if selected_logprob is None or not math.isclose(
        selected_logprob,
        decision.action_logprob,
        rel_tol=1.0e-5,
        abs_tol=1.0e-5,
    ):
        raise ValueError("planner old log-probability differs from behavior")
    if not math.isclose(
        evidence.planner_temperature,
        decision.sampling_temperature,
        rel_tol=1.0e-5,
        abs_tol=1.0e-5,
    ):
        raise ValueError("planner temperature differs from behavior metadata")
    for candidate in evidence.candidates:
        _validate_engine_teacher_action(
            candidate.action,
            min_count=decision.min_count,
            max_count=decision.max_count,
            valid_options=(True,) * len(decision.options),
            context=_decision_option_context(decision),
        )


def _validate_engine_teacher_action(
    action: tuple[int, ...],
    *,
    min_count: int,
    max_count: int,
    valid_options: tuple[bool, ...],
    context: int,
) -> None:
    if len(action) < min_count or len(action) > max_count:
        raise ValueError("engine teacher action violates selection count bounds")
    if len(set(action)) != len(action):
        raise ValueError("engine teacher action contains duplicate option indices")
    if any(
        index < 0 or index >= len(valid_options) or not valid_options[index]
        for index in action
    ):
        raise ValueError("engine teacher action index is outside the option range")
    order_sensitive = context == int(SelectContext.SKILL_ORDER) or (
        max_count > 1
        and not is_unordered_set_selection(
            context=context,
            min_count=min_count,
            max_count=max_count,
        )
    )
    if not order_sensitive and any(left >= right for left, right in pairwise(action)):
        raise ValueError("unordered engine teacher actions must be strictly increasing")


def _decision_option_context(decision: DecisionRecord) -> int:
    raw_contexts = getattr(decision.options, "contexts", None)
    if raw_contexts is not None:
        return int(raw_contexts[0])
    options = cast(Sequence[Any], decision.options)
    return int(options[0].context)


def compact_game_trajectory(trajectory: GameTrajectory) -> GameTrajectory:
    """Return an array-backed trajectory with decision object rows removed."""
    array_block = trajectory.array_block
    if not trajectory.decisions and array_block is not None:
        return trajectory
    if array_block is None:
        array_block = build_trajectory_array_block(
            trajectory.decisions,
            game_id=trajectory.game_id,
            seats_reward=trajectory.seats_reward,
        )
    return GameTrajectory(
        game_id=trajectory.game_id,
        seats_reward=trajectory.seats_reward,
        decisions=(),
        metadata=trajectory.metadata,
        deck_context=trajectory.deck_context,
        archive=trajectory.archive,
        array_block=array_block,
        policy_artifacts=trajectory.policy_artifacts,
    )


def rebind_game_trajectory_id(
    trajectory: GameTrajectory,
    *,
    game_id: str,
) -> GameTrajectory:
    """Change an envelope game id while preserving endpoint-row integrity."""
    if not game_id:
        raise ValueError("trajectory game id must not be empty")
    if game_id == trajectory.game_id:
        return trajectory
    array_block = trajectory.array_block
    if array_block is not None and array_block.executed_endpoint_values is not None:
        endpoint_values = array_block.executed_endpoint_values
        fingerprint = np.frombuffer(
            bytes.fromhex(_trajectory_game_fingerprint(game_id)),
            dtype=np.uint8,
        )
        game_fingerprints = np.broadcast_to(
            fingerprint,
            (endpoint_values.row_count, fingerprint.shape[0]),
        ).copy()
        endpoint_values = replace(
            endpoint_values,
            game_fingerprints=game_fingerprints,
        )
        array_block = replace(
            array_block,
            executed_endpoint_values=endpoint_values,
        )
    return replace(trajectory, game_id=game_id, array_block=array_block)


def _build_executed_endpoint_values(
    decisions: Sequence[DecisionRecord],
    *,
    game_id: str,
    seats_reward: tuple[float, float] | None,
) -> tuple[np.ndarray | None, ExecutedEndpointValueArrayBlock | None]:
    leaves = tuple(decision.executed_endpoint_value_leaf for decision in decisions)
    if not any(leaf is not None for leaf in leaves):
        return (None, None)
    if not game_id or seats_reward is None:
        raise ValueError(
            "executed endpoint rows require the finalized game id and outcome"
        )
    if len(seats_reward) != 2 or any(
        reward not in (-1.0, 0.0, 1.0) for reward in seats_reward
    ):
        raise ValueError("executed endpoint rows require root-perspective W/D/L")
    if not math.isclose(seats_reward[1], -seats_reward[0], abs_tol=0.0):
        raise ValueError("executed endpoint seat outcomes must be opposites")
    game_fingerprint = _trajectory_game_fingerprint(game_id)
    builder = ExecutedEndpointValueTableBuilder()
    indices = np.full(len(decisions), -1, dtype=np.int32)
    for index, (decision, leaf) in enumerate(zip(decisions, leaves, strict=True)):
        if leaf is None:
            continue
        indices[index] = builder.register(
            game_fingerprint=game_fingerprint,
            root_player=decision.seat,
            leaf=leaf,
        )
    builder.finalize_game(
        game_fingerprint=game_fingerprint,
        root_player_zero_outcome=int(seats_reward[0]),
    )
    return (
        indices,
        build_executed_endpoint_value_array_block(builder.freeze()),
    )


def _trajectory_game_fingerprint(game_id: str) -> str:
    """Return the domain-separated immutable identity of one rollout game."""
    return hashlib.sha256(
        b"ptcg-rl/trajectory-game/v1\x00" + game_id.encode("utf-8")
    ).hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    return cast(np.ndarray, value.detach().cpu().numpy().copy())


def _optional_to_numpy(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return _to_numpy(value)
