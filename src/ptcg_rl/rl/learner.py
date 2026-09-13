"""Learner-side PPO minibatch assembly from completed trajectories."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.selection import (
    ENGINE_PROVEN_UNORDERED_SET_CONTEXTS,
    is_unordered_set_selection,
)
from ptcg_rl.agent.search.root_information import RootInformationLeaf
from ptcg_rl.agent.search.root_information_tensorizer import (
    ProductionRootInformationTensorizer,
    RootInformationTensorizerConfig,
)
from ptcg_rl.context import (
    PublicEventBatch,
    collate_public_event_deltas,
    select_public_event_batch_rows,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.model import (
    AgentPolicyValueNet,
    OptionBatch,
    StateBatch,
    collate_encoded_options,
    collate_state_tokens,
)
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS, teacher_forced_action_targets
from ptcg_rl.model.state_encoder import OWNER_UNKNOWN, TOKEN_KIND_OOV_INDEX
from ptcg_rl.profiling import StageTimer, time_stage
from ptcg_rl.rl.advantage import (
    AdvantageBatch,
    AdvantageEstimate,
    AdvantageStep,
    GaeConfig,
    compute_gae,
)
from ptcg_rl.rl.checkpoint_pair_io import (
    atomic_write_bytes,
    json_payload,
    publish_torch_file,
)
from ptcg_rl.rl.engine_teacher import EngineTeacherTarget
from ptcg_rl.rl.experience import (
    DecisionRecord,
    GameTrajectory,
    TrajectoryArrayBlock,
    compact_game_trajectory,
    validate_game_trajectory_endpoint_values,
    validate_game_trajectory_schema9_contract,
    validate_game_trajectory_schema10_contract,
    validate_game_trajectory_schema11_contract,
    validate_game_trajectory_schema12_contract,
)
from ptcg_rl.rl.factual import FactualTransitionTarget
from ptcg_rl.rl.learner_diagnostics import (
    LearnerDiagnosticRow,
    LearnerDiagnostics,
    summarize_learner_diagnostics,
)
from ptcg_rl.rl.macro_credit import (
    MACRO_CONTINUATION_SUMMARY_SIZE,
    MacroCreditConfig,
    continuation_summary_from_array_block,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.model_publication import (
    PreparedModelState,
    prepare_model_state,
    validate_model_fingerprint,
)
from ptcg_rl.rl.planner_evidence import (
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
)
from ptcg_rl.rl.planner_losses import (
    PlannerReplayBatch,
    move_planner_replay,
    pin_planner_replay,
    select_planner_replay,
)
from ptcg_rl.rl.ppo import (
    AnchorStepLogitsCache,
    PpoBatch,
    PpoConfig,
    PpoReferenceKl,
    PpoUpdateResult,
    ensure_finite_model,
    evaluate_reference_action_logprobs,
    ppo_accumulated_update_step,
    reference_kl_from_logprobs,
    validate_ppo_batch,
    with_reference_kl,
    with_training_batch_kl,
)
from ptcg_rl.rl.recurrent_runtime import PolicyArtifactIdentity
from ptcg_rl.rl.root_information_value_replay import (
    RootInformationValueReplayBatch,
    move_root_information_value_replay,
    pin_root_information_value_replay,
    select_root_information_value_replay,
)
from ptcg_rl.rl.token_credit import PromptTokenCredit, compute_prompt_token_credit


class Schema9LearnerBatchConfig(BaseModel):
    """Required schema-9 learner-only tensorization contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_information_tensorizer: RootInformationTensorizerConfig
    expected_constructor_fingerprint: str
    expected_scorer_fingerprint: str
    expected_controller_fingerprint: str
    expected_planner_fingerprint: str

    @field_validator(
        "expected_constructor_fingerprint",
        "expected_scorer_fingerprint",
        "expected_controller_fingerprint",
        "expected_planner_fingerprint",
    )
    @classmethod
    def valid_expected_fingerprint(cls, value: str) -> str:
        """Require canonical identities for the fixed formal-run contract."""
        if len(value) != 64 or value != value.lower():
            raise ValueError("schema-9 expected identities must be lowercase SHA-256")
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(
                "schema-9 expected identities must be lowercase SHA-256"
            ) from exc
        if len(raw) != 32:
            raise ValueError("schema-9 expected identities must be lowercase SHA-256")
        return value


class LearnerBatchConfig(BaseModel):
    """Config for turning completed trajectories into PPO minibatches."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    microbatch_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("microbatch_size", "minibatch_size"),
    )
    gradient_accumulation_steps: int = 1
    max_decisions: int | None = None
    max_staleness: int = 2
    staleness_scope: Literal["decision", "seat_trajectory"] = "decision"
    shuffle: bool = True
    shuffle_each_epoch: bool = False
    drop_last: bool = False
    pin_memory: bool = False
    non_blocking_transfer: bool = True
    copy_stream: bool = True
    shape_bucket_accumulation: bool = False
    route_bucket_minibatches: bool = False
    fuse_route_accumulation_batches: bool = True
    seed: int = 0
    legacy_sampling_temperature: float | None = None
    require_deck_context: bool = False
    schema9: Schema9LearnerBatchConfig | None = None
    macro_credit: MacroCreditConfig | None = None
    gae: GaeConfig = GaeConfig()

    @field_validator("microbatch_size", "gradient_accumulation_steps")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid minibatch sizes."""
        if value <= 0:
            raise ValueError("learner batch sizes must be positive")
        return value

    @field_validator("max_decisions")
    @classmethod
    def valid_optional_max_decisions(cls, value: int | None) -> int | None:
        """Reject invalid optional complete-trajectory window budgets."""
        if value is not None and value <= 0:
            raise ValueError("max_decisions must be positive when set")
        return value

    @property
    def minibatch_size(self) -> int:
        """Return the microbatch size under the legacy attribute name."""
        return self.microbatch_size

    @property
    def effective_batch_size(self) -> int:
        """Return the nominal number of rows per optimizer update."""
        return self.microbatch_size * self.gradient_accumulation_steps

    @field_validator("max_staleness")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject invalid staleness limits."""
        if value < 0:
            raise ValueError("max_staleness must be non-negative")
        return value

    @field_validator("legacy_sampling_temperature")
    @classmethod
    def valid_legacy_sampling_temperature(
        cls,
        value: float | None,
    ) -> float | None:
        """Validate an explicit fallback for schema-1 trajectory payloads."""
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("legacy_sampling_temperature must be finite and positive")
        return value


class WeightPublisherConfig(BaseModel):
    """Config for atomic policy weight publication."""

    model_config = ConfigDict(extra="forbid")

    keep_last: int = 2
    retain_every_versions: int | None = None
    filename_prefix: str = "policy"

    @field_validator("keep_last")
    @classmethod
    def valid_keep_last(cls, value: int) -> int:
        """Reject invalid retention counts."""
        if value <= 0:
            raise ValueError("keep_last must be positive")
        return value

    @field_validator("retain_every_versions")
    @classmethod
    def valid_retain_every_versions(cls, value: int | None) -> int | None:
        """Reject invalid optional milestone intervals."""
        if value is not None and value <= 0:
            raise ValueError("retain_every_versions must be positive when set")
        return value

    @field_validator("filename_prefix")
    @classmethod
    def valid_filename_prefix(cls, value: str) -> str:
        """Reject empty or path-like filename prefixes."""
        cleaned = value.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("filename_prefix must be a single filename segment")
        return cleaned


@dataclass(frozen=True)
class LearnerBatchStats:
    """Counters from one learner minibatch assembly pass."""

    trajectories: int
    total_decisions: int
    kept_decisions: int
    stale_decisions: int
    minibatches: int
    behavior_filtered_decisions: int = 0
    trimmed_decisions: int = 0
    budget_overshoot_decisions: int = 0
    legacy_temperature_inferred_decisions: int = 0
    intrinsically_stale_decisions: int = 0
    propagated_stale_decisions: int = 0
    stale_seat_trajectories: int = 0
    retained_seat_trajectories: int = 0
    active_tokens: int = 0
    voluntary_stop_tokens: int = 0
    forced_max_decisions: int = 0
    min_policy_version: int | None = None
    max_policy_version: int | None = None
    policy_version_span: int = 0
    mixed_policy_games: int = 0
    max_game_policy_version_span: int = 0
    policy_age_histogram: Mapping[str, int] = field(default_factory=dict)
    selection_count_histogram: Mapping[str, int] = field(default_factory=dict)
    token_count_histogram: Mapping[str, int] = field(default_factory=dict)
    prompt_context_histogram: Mapping[str, int] = field(default_factory=dict)
    curriculum_mass: Mapping[
        str,
        Mapping[str, Mapping[str, int]],
    ] = field(default_factory=dict)
    deck_mass: Mapping[str, Mapping[str, int | float]] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnerWindowStaleness:
    """Lightweight stale-row accounting before formal batch construction."""

    total_decisions: int
    stale_decisions: int
    intrinsically_stale_decisions: int
    propagated_stale_decisions: int
    stale_seat_trajectories: int

    @property
    def retained_budget_decisions(self) -> int:
        """Return raw window rows left after only the staleness exclusion."""
        return self.total_decisions - self.stale_decisions


@dataclass(frozen=True)
class LearnerBatchPool:
    """One collated learner window plus indexing metadata."""

    batch: PpoBatch
    sample_count: int


@dataclass(frozen=True, slots=True)
class _PlannedPpoBatches(Sequence[PpoBatch]):
    """Index plan that materializes only the minibatch being consumed."""

    full_batch: PpoBatch
    index_chunks: tuple[tuple[int, ...], ...]
    device: torch.device | str | None

    def __len__(self) -> int:
        """Return the number of planned minibatches without materializing them."""
        return len(self.index_chunks)

    @overload
    def __getitem__(self, index: int) -> PpoBatch: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PpoBatch, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> PpoBatch | tuple[PpoBatch, ...]:
        """Materialize one requested minibatch or an explicit sliced subset."""
        if isinstance(index, slice):
            return tuple(self._materialize(chunk) for chunk in self.index_chunks[index])
        return self._materialize(self.index_chunks[index])

    def __iter__(self) -> Iterator[PpoBatch]:
        """Yield ephemeral minibatches in the planned optimizer order."""
        return (self._materialize(chunk) for chunk in self.index_chunks)

    def _materialize(self, indices: tuple[int, ...]) -> PpoBatch:
        return _take_ppo_batch(
            self.full_batch,
            indices,
            device=self.device,
        )


@dataclass(frozen=True)
class FixedKlReference:
    """Canonical decision rows and pre-update outputs for KL early stopping."""

    batches: tuple[PpoBatch, ...]
    action_logprobs: tuple[torch.Tensor, ...]
    initial_kl: PpoReferenceKl


@dataclass(frozen=True)
class LearnerBatchResult:
    """PPO minibatches plus assembly diagnostics."""

    batches: Sequence[PpoBatch]
    stats: LearnerBatchStats
    pool: LearnerBatchPool | None = None


@dataclass(frozen=True)
class LearnerIterationResult:
    """Result from one synchronous PPO learner iteration."""

    batch_result: LearnerBatchResult
    updates: tuple[PpoUpdateResult, ...]
    published_weights: PublishedWeights | None


@dataclass(frozen=True)
class PolicyIterationTrajectoryBatch:
    """All real behavior rows retained for distributional off-policy learning."""

    states: StateBatch
    options: OptionBatch
    decks: DeckBatch
    actions: tuple[tuple[int, ...], ...]
    game_ids: tuple[str, ...]
    seats: tuple[int, ...]
    decision_indices: tuple[int, ...]
    route_ids: tuple[str, ...]
    behavior_probabilities: torch.Tensor
    terminal_wdl: torch.Tensor
    behavior_kinds: tuple[str, ...]

    @property
    def row_count(self) -> int:
        """Return the number of aligned real decisions."""
        return len(self.actions)


@dataclass(frozen=True)
class PublishedWeights:
    """Metadata for the latest atomically published policy checkpoint."""

    version: int
    path: Path
    latest_path: Path
    published_at: str
    metadata: Mapping[str, Any]
    model_fingerprint: str | None = None


@dataclass(frozen=True)
class _IndexedDecision:
    game_id: str
    record: DecisionRecord | None
    array_block: TrajectoryArrayBlock | None
    array_index: int
    seat: int
    decision_index: int
    value_pred: float
    policy_version: int
    terminal_reward: float
    action_logprob: float
    behavior_kind: str
    sampling_temperature: float | None
    training_metadata: Mapping[str, str]
    deck: CanonicalDeck | None
    planner_behavior: PlannerBehaviorEvidence | None
    policy_artifact: PolicyArtifactIdentity | None


@dataclass(frozen=True)
class _LearnerSample:
    indexed: _IndexedDecision
    estimate: AdvantageEstimate
    policy_age: int
    token_credit: PromptTokenCredit | None = None


@dataclass(frozen=True)
class _EndpointLearnerRow:
    """One referenced actual endpoint before learner-batch deduplication."""

    key: tuple[str, int, str]
    leaf: RootInformationLeaf
    final_root_outcome: int
    deck: CanonicalDeck


@dataclass(frozen=True)
class _ArrayBlockSelection:
    block: TrajectoryArrayBlock
    source_indices: np.ndarray
    target_start: int
    target_stop: int

    @property
    def target_slice(self) -> slice:
        return slice(self.target_start, self.target_stop)


@dataclass(frozen=True)
class _StalenessSelection:
    """Rows excluded by the configured behavior-policy age scope."""

    drop_mask: tuple[bool, ...]
    intrinsically_stale_decisions: int
    propagated_stale_decisions: int
    stale_seat_trajectories: int


def build_ppo_minibatches(
    trajectories: Sequence[GameTrajectory],
    *,
    current_policy_version: int,
    config: LearnerBatchConfig | None = None,
    device: torch.device | str | None = None,
    timer: StageTimer | None = None,
) -> LearnerBatchResult:
    """Build PPO minibatches from complete game trajectories."""
    if current_policy_version < 0:
        raise ValueError("current_policy_version must be non-negative")
    cfg = config or LearnerBatchConfig()
    with time_stage(timer, "learner_index_decisions"):
        indexed_decisions = tuple(_iter_indexed_decisions(trajectories))
    if not indexed_decisions:
        return LearnerBatchResult(
            batches=(),
            stats=LearnerBatchStats(
                trajectories=len(trajectories),
                total_decisions=0,
                kept_decisions=0,
                stale_decisions=0,
                minibatches=0,
            ),
        )
    _validate_schema9_indexed_contract(indexed_decisions, cfg.schema9)

    policy_decisions: list[_IndexedDecision] = []
    behavior_filtered_decisions = 0
    legacy_temperature_inferred_decisions = 0
    for indexed in indexed_decisions:
        if indexed.behavior_kind != "policy_sample":
            behavior_filtered_decisions += 1
            continue
        if not math.isfinite(indexed.action_logprob):
            raise ValueError(
                "policy_sample decisions require a finite behavior action_logprob"
            )
        sampling_temperature = indexed.sampling_temperature
        if sampling_temperature is None:
            sampling_temperature = cfg.legacy_sampling_temperature
            if sampling_temperature is None:
                raise ValueError(
                    "schema-1 policy_sample decisions require an explicit "
                    "legacy_sampling_temperature"
                )
            indexed = replace(
                indexed,
                sampling_temperature=sampling_temperature,
            )
            legacy_temperature_inferred_decisions += 1
        if not math.isfinite(sampling_temperature) or sampling_temperature <= 0.0:
            raise ValueError(
                "policy_sample decisions require a finite positive sampling_temperature"
            )
        policy_decisions.append(indexed)
    if cfg.require_deck_context and any(
        indexed.deck is None for indexed in policy_decisions
    ):
        raise ValueError("deck-conditioned learner requires trajectory deck context")
    recurrent_presence = tuple(
        indexed.policy_artifact is not None for indexed in policy_decisions
    )
    if any(recurrent_presence) and not all(recurrent_presence):
        raise ValueError("cannot mix recurrent and stateless learner decisions")
    recurrent = bool(recurrent_presence and all(recurrent_presence))
    if recurrent and cfg.staleness_scope != "seat_trajectory":
        raise ValueError("recurrent PPO requires seat-trajectory staleness")
    if not policy_decisions:
        return LearnerBatchResult(
            batches=(),
            stats=LearnerBatchStats(
                trajectories=len(trajectories),
                total_decisions=len(indexed_decisions),
                kept_decisions=0,
                stale_decisions=0,
                minibatches=0,
                behavior_filtered_decisions=behavior_filtered_decisions,
                legacy_temperature_inferred_decisions=(
                    legacy_temperature_inferred_decisions
                ),
            ),
        )

    with time_stage(timer, "learner_gae"):
        # GAE and critic targets must be computed on complete seat trajectories.
        # Advantage normalization is a learner-pool operation, however, and must
        # not include rows that are subsequently excluded for staleness.
        raw_gae_config = cfg.gae.model_copy(update={"normalize_advantages": False})
        advantage_batch = _compute_learner_gae(
            policy_decisions,
            raw_gae_config,
        )
    with time_stage(timer, "learner_filter_shuffle"):
        staleness = _select_stale_decisions(
            policy_decisions,
            current_policy_version=current_policy_version,
            max_staleness=cfg.max_staleness,
            scope=cfg.staleness_scope,
        )
        samples: list[_LearnerSample] = []
        for input_index, (indexed, estimate) in enumerate(
            zip(
                policy_decisions,
                advantage_batch.estimates,
                strict=True,
            )
        ):
            if staleness.drop_mask[input_index]:
                continue
            samples.append(
                _LearnerSample(
                    indexed=indexed,
                    estimate=estimate,
                    policy_age=current_policy_version - indexed.policy_version,
                )
            )
        if cfg.gae.credit_unit == "decode_token":
            samples = _attach_prompt_token_credit(samples, cfg.gae)
            samples = _normalize_retained_token_advantages(samples, cfg.gae)
        else:
            samples = _normalize_retained_advantages(samples, cfg.gae)
        if recurrent:
            samples = _order_recurrent_samples(samples)

        # ``trajectories`` contains completed games. Never enforce the learner
        # budget by slicing this flattened row list: doing so systematically
        # removes terminal decisions from the last game in a window. Training
        # callers partition their rolling buffers at trajectory boundaries;
        # direct callers that provide an oversized game get the whole game and
        # an explicit soft-budget overshoot diagnostic.
        budget_overshoot_decisions = (
            0 if cfg.max_decisions is None else max(0, len(samples) - cfg.max_decisions)
        )
        sample_indices = list(range(len(samples)))
        if cfg.shuffle:
            if recurrent:
                sample_indices = _shuffle_recurrent_sample_indices(
                    samples,
                    seed=cfg.seed,
                )
            else:
                random.Random(cfg.seed).shuffle(sample_indices)

    with time_stage(timer, "learner_collate"):
        batches, pool = _collate_ppo_minibatches(
            samples,
            sample_indices,
            cfg,
            device=device,
        )
    diagnostics = _learner_diagnostics(
        samples,
        current_policy_version=current_policy_version,
    )
    return LearnerBatchResult(
        batches=batches,
        stats=LearnerBatchStats(
            trajectories=len(trajectories),
            total_decisions=len(indexed_decisions),
            kept_decisions=len(samples),
            stale_decisions=sum(staleness.drop_mask),
            minibatches=len(batches),
            behavior_filtered_decisions=behavior_filtered_decisions,
            trimmed_decisions=0,
            budget_overshoot_decisions=budget_overshoot_decisions,
            legacy_temperature_inferred_decisions=(
                legacy_temperature_inferred_decisions
            ),
            intrinsically_stale_decisions=(staleness.intrinsically_stale_decisions),
            propagated_stale_decisions=staleness.propagated_stale_decisions,
            stale_seat_trajectories=staleness.stale_seat_trajectories,
            retained_seat_trajectories=diagnostics.retained_seat_trajectories,
            active_tokens=diagnostics.active_tokens,
            voluntary_stop_tokens=diagnostics.voluntary_stop_tokens,
            forced_max_decisions=diagnostics.forced_max_decisions,
            min_policy_version=diagnostics.min_policy_version,
            max_policy_version=diagnostics.max_policy_version,
            policy_version_span=(
                0
                if diagnostics.min_policy_version is None
                or diagnostics.max_policy_version is None
                else diagnostics.max_policy_version - diagnostics.min_policy_version
            ),
            mixed_policy_games=diagnostics.mixed_policy_games,
            max_game_policy_version_span=diagnostics.max_game_policy_version_span,
            policy_age_histogram=diagnostics.policy_age_histogram,
            selection_count_histogram=diagnostics.selection_count_histogram,
            token_count_histogram=diagnostics.token_count_histogram,
            prompt_context_histogram=diagnostics.prompt_context_histogram,
            curriculum_mass=diagnostics.curriculum_mass,
            deck_mass=diagnostics.deck_mass,
        ),
        pool=pool,
    )


def capture_fixed_kl_reference(
    model: AgentPolicyValueNet,
    batch_result: LearnerBatchResult,
    *,
    batch_config: LearnerBatchConfig,
    ppo_config: PpoConfig,
    device: torch.device | str | None,
    timer: StageTimer | None = None,
) -> FixedKlReference | None:
    """Capture pre-update outputs on canonical rows independent of shuffle order."""
    pool = batch_result.pool
    if pool is None or pool.sample_count <= 0:
        return None
    reference_count = min(ppo_config.kl_reference_decisions, pool.sample_count)
    reference_indices: Sequence[int]
    if pool.batch.sequence_offsets is None:
        reference_indices = range(reference_count)
    else:
        selected_groups: list[tuple[int, ...]] = []
        selected_count = 0
        for group in _recurrent_sequence_rows(pool.batch):
            selected_groups.append(group)
            selected_count += len(group)
            if selected_count >= reference_count:
                break
        reference_indices = tuple(index for group in selected_groups for index in group)
    reference_batch_size = (
        batch_config.microbatch_size
        if ppo_config.kl_reference_batch_size is None
        else ppo_config.kl_reference_batch_size
    )
    reference_config = batch_config.model_copy(
        update={
            "drop_last": False,
            "microbatch_size": reference_batch_size,
        }
    )
    batches = tuple(
        _take_ppo_minibatches(
            pool.batch,
            reference_indices,
            reference_config,
            device=None,
        )
    )
    action_logprobs = _fixed_reference_action_logprobs(
        model,
        batches,
        config=ppo_config,
        device=device,
        transfer_config=batch_config,
        timer=timer,
    )
    initial_current = _fixed_reference_action_logprobs(
        model,
        batches,
        config=ppo_config,
        device=device,
        transfer_config=batch_config,
        timer=timer,
    )
    initial_kl = reference_kl_from_logprobs(initial_current, action_logprobs)
    return FixedKlReference(
        batches=batches,
        action_logprobs=action_logprobs,
        initial_kl=initial_kl,
    )


def evaluate_fixed_kl_reference(
    model: AgentPolicyValueNet,
    reference: FixedKlReference,
    *,
    config: PpoConfig,
    device: torch.device | str | None,
    transfer_config: LearnerBatchConfig,
    timer: StageTimer | None = None,
) -> PpoReferenceKl:
    """Evaluate current policy output on exactly the captured reference rows."""
    current_action_logprobs = _fixed_reference_action_logprobs(
        model,
        reference.batches,
        config=config,
        device=device,
        transfer_config=transfer_config,
        timer=timer,
    )
    return reference_kl_from_logprobs(
        current_action_logprobs,
        reference.action_logprobs,
    )


def _fixed_reference_action_logprobs(
    model: AgentPolicyValueNet,
    batches: Sequence[PpoBatch],
    *,
    config: PpoConfig,
    device: torch.device | str | None,
    transfer_config: LearnerBatchConfig,
    timer: StageTimer | None,
) -> tuple[torch.Tensor, ...]:
    with time_stage(timer, "learner_kl_reference_forward"):
        return tuple(
            evaluate_reference_action_logprobs(model, batch, config)
            for batch in _iter_device_batches(
                batches,
                device=device,
                pin_memory=transfer_config.pin_memory,
                non_blocking=transfer_config.non_blocking_transfer,
                copy_stream=transfer_config.copy_stream,
            )
        )


def run_ppo_iteration(
    *,
    model: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[GameTrajectory],
    current_policy_version: int,
    batch_config: LearnerBatchConfig | None = None,
    ppo_config: PpoConfig | None = None,
    anchor_model: AgentPolicyValueNet | None = None,
    anchor_cache: AnchorStepLogitsCache | None = None,
    publisher: WeightPublisher | None = None,
    publish_version: int | None = None,
    first_update_index: int = 0,
    first_ppo_update_index: int | None = None,
    device: torch.device | str | None = None,
    timer: StageTimer | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batch_result: LearnerBatchResult | None = None,
    progress_callback: Callable[[int, PpoUpdateResult], None] | None = None,
    anchor_cache_key_offset: int = 0,
    kl_stop_baseline_k3: float | None = None,
    kl_reference: FixedKlReference | None = None,
    collect_diagnostics: bool = True,
) -> LearnerIterationResult:
    """Run one synchronous PPO iteration from trajectories to optional publish.

    ``anchor_cache`` may be shared across the PPO epochs of one iteration
    (same ``batch_result``): the frozen anchor's step logits per minibatch
    are computed on the first epoch and reused afterwards. Do not reuse a
    cache across different minibatch sets.
    """
    if first_update_index < 0:
        raise ValueError("first_update_index must be non-negative")
    if first_ppo_update_index is None:
        first_ppo_update_index = first_update_index
    if first_ppo_update_index < 0:
        raise ValueError("first_ppo_update_index must be non-negative")
    if anchor_cache_key_offset < 0:
        raise ValueError("anchor_cache_key_offset must be non-negative")
    if batch_result is None:
        batch_result = build_ppo_minibatches(
            trajectories,
            current_policy_version=current_policy_version,
            config=batch_config,
            device=None,
            timer=timer,
        )
    updates: list[PpoUpdateResult] = []
    cfg = ppo_config or PpoConfig()
    transfer_cfg = batch_config or LearnerBatchConfig()
    if (
        kl_reference is None
        and cfg.target_kl_early_stop is not None
        and cfg.kl_stop_mode == "fixed_reference"
    ):
        kl_reference = capture_fixed_kl_reference(
            model,
            batch_result,
            batch_config=transfer_cfg,
            ppo_config=cfg,
            device=device,
            timer=timer,
        )
    execution_batches = batch_result.batches
    nominal_microbatch_counts: tuple[int, ...] | None = None
    accumulation_steps = transfer_cfg.gradient_accumulation_steps
    if _can_fuse_route_accumulation(
        model,
        execution_batches,
        transfer_cfg,
    ):
        execution_batches, nominal_microbatch_counts = (
            _fuse_planned_accumulation_batches(
                cast(_PlannedPpoBatches, execution_batches),
                accumulation_steps=accumulation_steps,
            )
        )
        accumulation_steps = 1
    device_batches = _iter_device_batches(
        execution_batches,
        device=device,
        pin_memory=transfer_cfg.pin_memory,
        non_blocking=transfer_cfg.non_blocking_transfer,
        copy_stream=transfer_cfg.copy_stream,
    )
    consumed_microbatches = 0
    for offset, update_batches in enumerate(
        _iter_accumulation_groups(
            device_batches,
            accumulation_steps,
        )
    ):
        nominal_microbatch_count = (
            len(update_batches)
            if nominal_microbatch_counts is None
            else nominal_microbatch_counts[offset]
        )
        cache_keys = tuple(
            anchor_cache_key_offset + consumed_microbatches + microbatch_offset
            if anchor_cache is not None
            else None
            for microbatch_offset in range(len(update_batches))
        )
        update = ppo_accumulated_update_step(
            model,
            optimizer,
            update_batches,
            cfg,
            anchor_model=anchor_model,
            anchor_cache=anchor_cache,
            anchor_cache_keys=cache_keys,
            update_index=first_update_index + offset,
            distillation_update_index=first_ppo_update_index + offset,
            timer=timer,
            kl_stop_baseline_k3=kl_stop_baseline_k3,
            collect_diagnostics=collect_diagnostics,
        )
        if nominal_microbatch_counts is not None:
            update = replace(
                update,
                microbatch_count=nominal_microbatch_count,
            )
        consumed_microbatches += nominal_microbatch_count
        if lr_scheduler is not None:
            with time_stage(timer, "learner_lr_scheduler"):
                _advance_active_optimizer_group_clocks(optimizer)
                lr_scheduler.step()
        if kl_reference is not None:
            reference_kl = evaluate_fixed_kl_reference(
                model,
                kl_reference,
                config=cfg,
                device=device,
                transfer_config=transfer_cfg,
                timer=timer,
            )
            update = with_reference_kl(
                update,
                reference_kl,
                initial_reference_kl_k3=kl_reference.initial_kl.approx_kl_k3,
                config=cfg,
            )
        elif cfg.kl_stop_mode == "training_batch_delta":
            update = with_training_batch_kl(
                update,
                baseline_k3=kl_stop_baseline_k3,
                config=cfg,
            )
            kl_stop_baseline_k3 = update.kl_stop_baseline_k3
        if cfg.transition_distillation.active(first_ppo_update_index + offset):
            update = replace(
                update,
                should_stop=False,
                kl_stop_threshold=None,
            )
        updates.append(update)
        if progress_callback is not None:
            progress_callback(first_update_index + offset, update)
        if update.should_stop:
            break

    published_weights: PublishedWeights | None = None
    if publisher is not None and updates:
        ensure_finite_model(model)
        version = (
            publish_version
            if publish_version is not None
            else current_policy_version + 1
        )
        published_weights = publisher.publish(
            model.state_dict(),
            version=version,
            metadata={
                "updates": len(updates),
                "effective_updates": len(updates),
                "microbatches": sum(update.microbatch_count for update in updates),
                "updated_objective_units": sum(
                    update.objective_unit_count for update in updates
                ),
                "kept_decisions": batch_result.stats.kept_decisions,
                "stale_decisions": batch_result.stats.stale_decisions,
                "behavior_filtered_decisions": (
                    batch_result.stats.behavior_filtered_decisions
                ),
                "trimmed_decisions": batch_result.stats.trimmed_decisions,
                "budget_overshoot_decisions": (
                    batch_result.stats.budget_overshoot_decisions
                ),
                "legacy_temperature_inferred_decisions": (
                    batch_result.stats.legacy_temperature_inferred_decisions
                ),
                "retained_seat_trajectories": (
                    batch_result.stats.retained_seat_trajectories
                ),
                "active_tokens": batch_result.stats.active_tokens,
                "voluntary_stop_tokens": (batch_result.stats.voluntary_stop_tokens),
                "forced_max_decisions": batch_result.stats.forced_max_decisions,
                "min_policy_version": batch_result.stats.min_policy_version,
                "max_policy_version": batch_result.stats.max_policy_version,
                "policy_version_span": batch_result.stats.policy_version_span,
                "mixed_policy_games": batch_result.stats.mixed_policy_games,
                "max_game_policy_version_span": (
                    batch_result.stats.max_game_policy_version_span
                ),
                "policy_age_histogram": dict(batch_result.stats.policy_age_histogram),
                "selection_count_histogram": dict(
                    batch_result.stats.selection_count_histogram
                ),
                "token_count_histogram": dict(batch_result.stats.token_count_histogram),
                "prompt_context_histogram": dict(
                    batch_result.stats.prompt_context_histogram
                ),
                "curriculum_mass": {
                    dimension: {label: dict(counts) for label, counts in labels.items()}
                    for dimension, labels in (
                        batch_result.stats.curriculum_mass.items()
                    )
                },
            },
        )
    return LearnerIterationResult(
        batch_result=batch_result,
        updates=tuple(updates),
        published_weights=published_weights,
    )


def _advance_active_optimizer_group_clocks(
    optimizer: torch.optim.Optimizer,
) -> None:
    """Advance LR clocks only for groups that participated in this update."""
    for group in optimizer.param_groups:
        if "scheduler_active_updates" not in group:
            continue
        if any(parameter.grad is not None for parameter in group["params"]):
            group["scheduler_active_updates"] = (
                int(group["scheduler_active_updates"]) + 1
            )


def _iter_accumulation_groups(
    batches: Iterator[PpoBatch],
    accumulation_steps: int,
) -> Iterator[tuple[PpoBatch, ...]]:
    """Yield consecutive microbatch groups, retaining a short final group."""
    while True:
        group: list[PpoBatch] = []
        for _ in range(accumulation_steps):
            try:
                group.append(next(batches))
            except StopIteration:
                break
        if not group:
            return
        yield tuple(group)


def _can_fuse_route_accumulation(
    model: AgentPolicyValueNet,
    batches: Sequence[PpoBatch],
    config: LearnerBatchConfig,
) -> bool:
    """Return whether one update can use a single mixed-route forward safely."""
    if (
        not config.fuse_route_accumulation_batches
        or not config.route_bucket_minibatches
        or config.gradient_accumulation_steps <= 1
        or not isinstance(batches, _PlannedPpoBatches)
    ):
        return False
    for module in model.modules():
        if not module.training:
            continue
        if isinstance(module, torch.nn.Dropout) and module.p > 0.0:
            return False
        if isinstance(module, torch.nn.MultiheadAttention) and module.dropout > 0.0:
            return False
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return False
    return True


def _fuse_planned_accumulation_batches(
    batches: _PlannedPpoBatches,
    *,
    accumulation_steps: int,
) -> tuple[_PlannedPpoBatches, tuple[int, ...]]:
    """Combine route-local index chunks lazily at effective-update boundaries."""
    if accumulation_steps <= 1:
        raise ValueError("fused accumulation requires at least two microbatches")
    fused_chunks: list[tuple[int, ...]] = []
    nominal_counts: list[int] = []
    for start in range(0, len(batches.index_chunks), accumulation_steps):
        group = batches.index_chunks[start : start + accumulation_steps]
        fused_chunks.append(tuple(index for chunk in group for index in chunk))
        nominal_counts.append(len(group))
    return (
        _PlannedPpoBatches(
            full_batch=batches.full_batch,
            index_chunks=tuple(fused_chunks),
            device=batches.device,
        ),
        tuple(nominal_counts),
    )


def reshuffle_ppo_minibatches(
    batch_result: LearnerBatchResult,
    *,
    config: LearnerBatchConfig,
    seed: int,
    device: torch.device | str | None = None,
) -> LearnerBatchResult:
    """Return ``batch_result`` with minibatches re-indexed from its pool."""
    if batch_result.pool is None or batch_result.pool.sample_count == 0:
        return batch_result
    pool_batch = batch_result.pool.batch
    sample_indices = list(range(batch_result.pool.sample_count))
    if config.shuffle:
        if pool_batch.sequence_offsets is None:
            random.Random(seed).shuffle(sample_indices)
        else:
            groups = list(_recurrent_sequence_rows(pool_batch))
            random.Random(seed).shuffle(groups)
            sample_indices = [index for group in groups for index in group]
    batches = _take_ppo_minibatches(
        pool_batch,
        sample_indices,
        config,
        device=device,
    )
    stats = replace(batch_result.stats, minibatches=len(batches))
    return LearnerBatchResult(
        batches=batches,
        stats=stats,
        pool=batch_result.pool,
    )


def collate_policy_iteration_trajectories(
    trajectories: Sequence[GameTrajectory],
    *,
    max_decisions: int,
    device: torch.device | str,
) -> PolicyIterationTrajectoryBatch | None:
    """Collate complete acting-seat sequences without filtering behavior kind."""
    if max_decisions <= 0:
        raise ValueError("policy-iteration max_decisions must be positive")
    compact = tuple(compact_game_trajectory(trajectory) for trajectory in trajectories)
    indexed = _iter_indexed_decisions(compact)
    if not indexed:
        return None

    grouped: dict[tuple[str, int, str], list[_IndexedDecision]] = {}
    for row in indexed:
        if row.deck is None:
            raise ValueError("policy-iteration rows require exact acting decks")
        key = (row.game_id, row.seat, row.deck.signature)
        grouped.setdefault(key, []).append(row)
    retained: list[_IndexedDecision] = []
    for group in grouped.values():
        group.sort(key=lambda row: row.decision_index)
        if retained and len(retained) + len(group) > max_decisions:
            continue
        retained.extend(group)
        if len(retained) >= max_decisions:
            break
    if not retained:
        retained.extend(next(iter(grouped.values())))

    selections = _array_block_selections(retained)
    decks = tuple(cast(CanonicalDeck, row.deck) for row in retained)
    behavior_probabilities = torch.tensor(
        [max(math.exp(row.action_logprob), 1.0e-30) for row in retained],
        dtype=torch.float32,
        device=device,
    )
    terminal_wdl = torch.tensor(
        [_terminal_wdl(row.terminal_reward) for row in retained],
        dtype=torch.float32,
        device=device,
    )
    _validate_policy_iteration_actions(retained)
    return PolicyIterationTrajectoryBatch(
        states=_collate_array_states(selections, device=device),
        options=_collate_array_options(selections, device=device),
        decks=DeckBatch.from_decks(decks, device=device),
        actions=tuple(_array_action(row) for row in retained),
        game_ids=tuple(row.game_id for row in retained),
        seats=tuple(row.seat for row in retained),
        decision_indices=tuple(row.decision_index for row in retained),
        route_ids=tuple(deck.signature for deck in decks),
        behavior_probabilities=behavior_probabilities,
        terminal_wdl=terminal_wdl,
        behavior_kinds=tuple(row.behavior_kind for row in retained),
    )


def _validate_policy_iteration_actions(rows: Sequence[_IndexedDecision]) -> None:
    """Validate count-first actions once before repeated trusted GPU forwards."""
    for row in rows:
        block = _require_array_block(row)
        index = row.array_index
        action = block.action_at(index)
        minimum = int(block.options.min_counts[index])
        maximum = int(block.options.max_counts[index])
        valid = block.options.valid_options[index]
        if len(action) < minimum or len(action) > maximum:
            raise ValueError("policy-iteration action violates selection count bounds")
        if len(set(action)) != len(action) or any(
            option < 0 or option >= int(valid.shape[0]) or not bool(valid[option])
            for option in action
        ):
            raise ValueError("policy-iteration action contains an invalid option")
        context = (
            -1
            if not bool(valid.any()) or int(block.options.contexts.shape[1]) == 0
            else int(block.options.contexts[index, 0])
        )
        if is_unordered_set_selection(
            context=context,
            min_count=minimum,
            max_count=maximum,
        ) and any(left >= right for left, right in pairwise(action)):
            raise ValueError(
                "policy-iteration unordered actions must be strictly increasing"
            )


def _terminal_wdl(reward: float) -> tuple[float, float, float]:
    if reward == -1.0:
        return (1.0, 0.0, 0.0)
    if reward == 0.0:
        return (0.0, 1.0, 0.0)
    if reward == 1.0:
        return (0.0, 0.0, 1.0)
    raise ValueError("policy-iteration terminal reward must be categorical W/D/L")


def _iter_indexed_decisions(
    trajectories: Sequence[GameTrajectory],
) -> list[_IndexedDecision]:
    indexed: list[_IndexedDecision] = []
    for trajectory in trajectories:
        validate_game_trajectory_schema12_contract(trajectory)
        validate_game_trajectory_schema11_contract(trajectory)
        validate_game_trajectory_schema10_contract(trajectory)
        planner_rows = validate_game_trajectory_schema9_contract(trajectory)
        validate_game_trajectory_endpoint_values(trajectory)
        schema10_array = (
            trajectory.array_block is not None
            and trajectory.array_block.executed_macros is not None
        )
        if trajectory.decisions and not schema10_array:
            indexed.extend(
                _iter_record_decisions(
                    trajectory,
                    planner_rows=planner_rows,
                )
            )
            continue
        if trajectory.array_block is not None:
            indexed.extend(
                _iter_array_decisions(
                    trajectory,
                    planner_rows=planner_rows,
                )
            )
    return indexed


def _iter_record_decisions(
    trajectory: GameTrajectory,
    *,
    planner_rows: tuple[PlannerBehaviorEvidence, ...] | None,
) -> list[_IndexedDecision]:
    indexed: list[_IndexedDecision] = []
    seat_decks = _trajectory_seat_decks(trajectory)
    if planner_rows is not None and len(planner_rows) != len(trajectory.decisions):
        raise ValueError("planner evidence rows differ from trajectory decisions")
    for row_index, record in enumerate(trajectory.decisions):
        indexed.append(
            _IndexedDecision(
                game_id=trajectory.game_id,
                record=record,
                array_block=None,
                array_index=-1,
                seat=record.seat,
                decision_index=record.decision_index,
                value_pred=record.value_pred,
                policy_version=record.policy_version,
                terminal_reward=trajectory.reward_for_seat(record.seat),
                action_logprob=record.action_logprob,
                behavior_kind=_record_behavior_kind(record),
                sampling_temperature=float(
                    getattr(record, "sampling_temperature", 1.0)
                ),
                training_metadata=_trajectory_training_metadata(trajectory),
                deck=None if seat_decks is None else seat_decks[record.seat],
                planner_behavior=(
                    record.planner_behavior
                    if planner_rows is None
                    else planner_rows[row_index]
                ),
                policy_artifact=_trajectory_policy_artifact(
                    trajectory,
                    record.seat,
                ),
            )
        )
    return indexed


def _iter_array_decisions(
    trajectory: GameTrajectory,
    *,
    planner_rows: tuple[PlannerBehaviorEvidence, ...] | None,
) -> list[_IndexedDecision]:
    block = trajectory.array_block
    if block is None:
        return []
    sampling_temperatures = block.sampling_temperatures
    if sampling_temperatures is not None and (
        sampling_temperatures.ndim != 1
        or sampling_temperatures.shape[0] != block.decision_count
    ):
        raise ValueError("sampling_temperatures must have shape [decision_count]")
    indexed: list[_IndexedDecision] = []
    if planner_rows is not None and len(planner_rows) != block.decision_count:
        raise ValueError("planner evidence rows differ from trajectory decisions")
    seat_decks = _trajectory_seat_decks(trajectory)
    for index in range(block.decision_count):
        seat = int(block.seats[index])
        indexed.append(
            _IndexedDecision(
                game_id=trajectory.game_id,
                record=None,
                array_block=block,
                array_index=index,
                seat=seat,
                decision_index=int(block.decision_indices[index]),
                value_pred=float(block.value_preds[index]),
                policy_version=int(block.policy_versions[index]),
                terminal_reward=trajectory.reward_for_seat(seat),
                action_logprob=float(block.action_logprobs[index]),
                behavior_kind=_array_behavior_kind(block, index),
                sampling_temperature=(
                    None
                    if sampling_temperatures is None
                    else float(sampling_temperatures[index])
                ),
                training_metadata=_trajectory_training_metadata(trajectory),
                deck=None if seat_decks is None else seat_decks[seat],
                planner_behavior=(
                    None if planner_rows is None else planner_rows[index]
                ),
                policy_artifact=_trajectory_policy_artifact(trajectory, seat),
            )
        )
    return indexed


def _trajectory_seat_decks(
    trajectory: GameTrajectory,
) -> tuple[CanonicalDeck, CanonicalDeck] | None:
    context = trajectory.deck_context
    if context is None:
        return None
    return (context.deck_for_seat(0), context.deck_for_seat(1))


def _trajectory_policy_artifact(
    trajectory: GameTrajectory,
    seat: int,
) -> PolicyArtifactIdentity | None:
    artifacts = trajectory.policy_artifacts
    return None if artifacts is None else artifacts[seat]


def _record_behavior_kind(record: DecisionRecord) -> str:
    raw_kind = getattr(record, "behavior_kind", "policy_sample")
    return str(raw_kind)


def _array_behavior_kind(block: TrajectoryArrayBlock, index: int) -> str:
    raw_kinds = getattr(block, "behavior_kinds", None)
    if raw_kinds is None:
        return "policy_sample"
    raw_kind = raw_kinds[index]
    if isinstance(raw_kind, np.integer):
        code = int(raw_kind)
        if code == 0:
            return "policy_sample"
        if code == 1:
            return "improvement"
        raise ValueError("trajectory contains an unsupported behavior kind code")
    if isinstance(raw_kind, bytes):
        return raw_kind.decode("utf-8")
    return str(raw_kind)


def _trajectory_training_metadata(
    trajectory: GameTrajectory,
) -> Mapping[str, str]:
    extra = trajectory.metadata.extra
    if not isinstance(extra, Mapping):
        return {}
    allowed_keys = (
        "candidate_lane",
        "candidate_deck_label",
        "frozen_sampling_lane",
        "opponent_deck_label",
        "opponent_kind",
        "opponent_id",
    )
    return {
        key: str(extra[key]) for key in allowed_keys if key in extra and str(extra[key])
    }


class WeightPublisher:
    """Publish policy state_dict files with an atomic latest pointer."""

    def __init__(
        self,
        directory: Path,
        config: WeightPublisherConfig | None = None,
    ) -> None:
        """Initialize the publisher root directory and retention policy."""
        self.directory = Path(directory)
        self.config = config or WeightPublisherConfig()

    @property
    def latest_path(self) -> Path:
        """Return the latest-pointer JSON path."""
        return self.directory / "latest.json"

    def publish(
        self,
        state_dict: Mapping[str, Any] | PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_fields: Mapping[str, Any] | None = None,
    ) -> PublishedWeights:
        """Write a policy checkpoint and atomically update ``latest.json``."""
        if version < 0:
            raise ValueError("version must be non-negative")
        self.directory.mkdir(parents=True, exist_ok=True)
        prepared = prepare_model_state(state_dict)
        final_path = self.path_for_version(version)
        publish_torch_file(
            final_path,
            _checkpoint_payload(prepared, checkpoint_fields=checkpoint_fields),
        )
        published = self._publish_latest_pointer(
            final_path,
            version=version,
            metadata=metadata,
            model_fingerprint=prepared.model_fingerprint,
        )
        self._prune_old_weights()
        return published

    def adopt_existing(
        self,
        path: Path,
        *,
        version: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> PublishedWeights:
        """Publish a pointer to an immutable checkpoint without rewriting it."""
        final_path = self.path_for_version(version)
        if Path(path).resolve() != final_path.resolve():
            raise ValueError(
                "adopted checkpoint must use the publisher's numbered path: "
                f"{path} != {final_path}"
            )
        if not final_path.is_file():
            raise FileNotFoundError(f"adopted checkpoint is missing: {final_path}")
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(final_path, map_location="cpu")
        checkpoint_state = _state_dict_from_published_checkpoint(checkpoint)
        model_fingerprint = canonical_model_state_fingerprint(checkpoint_state)
        embedded_fingerprint = (
            checkpoint.get("model_fingerprint")
            if isinstance(checkpoint, Mapping)
            else None
        )
        if embedded_fingerprint is not None:
            validate_model_fingerprint(
                str(embedded_fingerprint),
                name="checkpoint model_fingerprint",
            )
            if embedded_fingerprint != model_fingerprint:
                raise ValueError(
                    "adopted checkpoint model fingerprint does not match its state"
                )
        return self._publish_latest_pointer(
            final_path,
            version=version,
            metadata=metadata,
            model_fingerprint=model_fingerprint,
        )

    def commit_prepared(
        self,
        path: Path,
        *,
        version: int,
        metadata: Mapping[str, Any] | None,
        model_fingerprint: str,
        prune: bool = True,
    ) -> PublishedWeights:
        """Publish a pointer for an already durable prepared checkpoint."""
        final_path = self.path_for_version(version)
        if Path(path).resolve() != final_path.resolve():
            raise ValueError("prepared checkpoint path is not canonical")
        if not final_path.is_file():
            raise FileNotFoundError(f"prepared checkpoint is missing: {final_path}")
        published = self._publish_latest_pointer(
            final_path,
            version=version,
            metadata=metadata,
            model_fingerprint=model_fingerprint,
        )
        if prune:
            self._prune_old_weights()
        return published

    def prune(self) -> None:
        """Apply the configured retention policy after a pair commit."""
        self._prune_old_weights()

    def read_latest(self) -> PublishedWeights | None:
        """Read the latest-pointer JSON, if it exists."""
        return read_latest_published_weights(self.directory)

    def path_for_version(self, version: int) -> Path:
        """Return the canonical numbered checkpoint path for one version."""
        if version < 0:
            raise ValueError("version must be non-negative")
        return self.directory / self._weight_filename(version)

    def _weight_filename(self, version: int) -> str:
        return f"{self.config.filename_prefix}_v{version}.pt"

    def _publish_latest_pointer(
        self,
        path: Path,
        *,
        version: int,
        metadata: Mapping[str, Any] | None,
        model_fingerprint: str,
    ) -> PublishedWeights:
        validate_model_fingerprint(model_fingerprint)
        published_at = datetime.now(UTC).isoformat()
        latest_record = {
            "version": version,
            "path": str(path),
            "published_at": published_at,
            "model_fingerprint": model_fingerprint,
            "metadata": dict(metadata or {}),
        }
        atomic_write_bytes(
            self.latest_path,
            json_payload(latest_record),
            overwrite=True,
        )
        return PublishedWeights(
            version=version,
            path=path,
            latest_path=self.latest_path,
            published_at=published_at,
            metadata=metadata or {},
            model_fingerprint=model_fingerprint,
        )

    def _prune_old_weights(self) -> None:
        versioned = [
            (version, path)
            for path in self.directory.glob(f"{self.config.filename_prefix}_v*.pt")
            if (version := _published_weight_version(path, self.config.filename_prefix))
            is not None
        ]
        ordered = sorted(versioned, reverse=True)
        protected = {version for version, _path in ordered[: self.config.keep_last]}
        interval = self.config.retain_every_versions
        if interval is not None:
            protected.update(
                version for version, _path in ordered if version % interval == 0
            )
        for version, path in ordered:
            if version in protected:
                continue
            path.unlink(missing_ok=True)


def read_latest_published_weights(directory: Path) -> PublishedWeights | None:
    """Return the latest published weight metadata from a publisher directory."""
    latest_path = Path(directory) / "latest.json"
    if not latest_path.exists():
        return None
    record = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"latest weight record must be a JSON object: {latest_path}")
    version = int(record["version"])
    raw_path = record["path"]
    raw_published_at = record["published_at"]
    raw_model_fingerprint = record.get("model_fingerprint")
    raw_metadata = record.get("metadata", {})
    if not isinstance(raw_path, str):
        raise ValueError("latest weight record path must be a string")
    if not isinstance(raw_published_at, str):
        raise ValueError("latest weight record published_at must be a string")
    if not isinstance(raw_metadata, dict):
        raise ValueError("latest weight record metadata must be an object")
    if raw_model_fingerprint is not None:
        if not isinstance(raw_model_fingerprint, str):
            raise ValueError("latest weight model_fingerprint must be a string")
        validate_model_fingerprint(raw_model_fingerprint)
    return PublishedWeights(
        version=version,
        path=Path(raw_path),
        latest_path=latest_path,
        published_at=raw_published_at,
        metadata=raw_metadata,
        model_fingerprint=raw_model_fingerprint,
    )


def _checkpoint_payload(
    prepared: PreparedModelState,
    *,
    checkpoint_fields: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    cpu_state_dict = dict(prepared.state_dict)
    if checkpoint_fields is None:
        return cpu_state_dict
    fields = dict(checkpoint_fields)
    embedded_fingerprint = fields.get("model_fingerprint")
    if (
        embedded_fingerprint is not None
        and embedded_fingerprint != prepared.model_fingerprint
    ):
        raise ValueError("checkpoint model_fingerprint conflicts with model state")
    return {
        "model_state_dict": cpu_state_dict,
        **fields,
        "model_fingerprint": prepared.model_fingerprint,
    }


def prepared_checkpoint_payload(
    prepared: PreparedModelState,
    *,
    checkpoint_fields: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return the serializable policy payload for a frozen model snapshot."""
    return _checkpoint_payload(prepared, checkpoint_fields=checkpoint_fields)


def _state_dict_from_published_checkpoint(
    checkpoint: Any,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("published checkpoint must be a mapping")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return cast(Mapping[str, torch.Tensor], value)
    return cast(Mapping[str, torch.Tensor], checkpoint)


def _published_weight_version(path: Path, prefix: str) -> int | None:
    name = path.name
    stem = name.removesuffix(".pt")
    expected_prefix = f"{prefix}_v"
    if not stem.startswith(expected_prefix):
        return None
    raw_version = stem.removeprefix(expected_prefix)
    if not raw_version.isdigit():
        return None
    return int(raw_version)


def _advantage_step(indexed: _IndexedDecision) -> AdvantageStep:
    return AdvantageStep(
        game_id=indexed.game_id,
        seat=indexed.seat,
        decision_index=indexed.decision_index,
        terminal_reward=indexed.terminal_reward,
        value_pred=indexed.value_pred,
    )


def _compute_learner_gae(
    indexed_decisions: Sequence[_IndexedDecision],
    config: GaeConfig,
) -> AdvantageBatch:
    if (
        indexed_decisions
        and config.prize_diff_shaping_beta == 0.0
        and all(indexed.array_block is not None for indexed in indexed_decisions)
    ):
        return _compute_array_gae(indexed_decisions, config)
    return compute_gae(
        tuple(_advantage_step(indexed) for indexed in indexed_decisions),
        config,
    )


def _compute_array_gae(
    indexed_decisions: Sequence[_IndexedDecision],
    config: GaeConfig,
) -> AdvantageBatch:
    grouped: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for input_index, indexed in enumerate(indexed_decisions):
        _validate_array_gae_indexed(indexed)
        grouped[(indexed.game_id, indexed.seat)].append(input_index)

    advantages = np.zeros(len(indexed_decisions), dtype=np.float64)
    returns = np.zeros(len(indexed_decisions), dtype=np.float64)
    discount = config.gamma * config.gae_lambda
    for group_indices in grouped.values():
        ordered = sorted(
            group_indices,
            key=lambda index: indexed_decisions[index].decision_index,
        )
        values = np.asarray(
            [indexed_decisions[index].value_pred for index in ordered],
            dtype=np.float64,
        )
        rewards = np.zeros(len(ordered), dtype=np.float64)
        rewards[-1] = indexed_decisions[ordered[-1]].terminal_reward
        next_values = np.zeros_like(values)
        if len(values) > 1:
            next_values[:-1] = values[1:]
        group_advantages = _discounted_cumsum_numpy(
            rewards + config.gamma * next_values - values,
            discount,
        )
        if config.value_target_mode == "monte_carlo":
            group_returns = _discounted_cumsum_numpy(rewards, config.gamma)
        else:
            group_returns = group_advantages + values
        advantages[np.asarray(ordered, dtype=np.intp)] = group_advantages
        returns[np.asarray(ordered, dtype=np.intp)] = group_returns

    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())
    if config.normalize_advantages:
        scale = advantage_std if advantage_std > config.normalize_epsilon else math.inf
        normalized = (advantages - advantage_mean) / scale
    else:
        normalized = advantages

    estimates = tuple(
        AdvantageEstimate(
            input_index=input_index,
            game_id=indexed.game_id,
            seat=indexed.seat,
            decision_index=indexed.decision_index,
            advantage=float(advantages[input_index]),
            normalized_advantage=float(normalized[input_index]),
            return_value=float(returns[input_index]),
        )
        for input_index, indexed in enumerate(indexed_decisions)
    )
    return AdvantageBatch(
        estimates=estimates,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )


def _validate_array_gae_indexed(indexed: _IndexedDecision) -> None:
    if indexed.decision_index < 0:
        raise ValueError("decision_index must be non-negative")
    for name, value in (
        ("terminal_reward", indexed.terminal_reward),
        ("value_pred", indexed.value_pred),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _discounted_cumsum_numpy(values: np.ndarray, discount: float) -> np.ndarray:
    if discount == 0.0:
        return values.copy()
    if discount == 1.0:
        return np.cumsum(values[::-1], dtype=np.float64)[::-1]
    powers = np.power(discount, np.arange(len(values), dtype=np.float64))
    if powers[-1] == 0.0:
        return _discounted_cumsum_scan_numpy(values, discount)
    reversed_result = np.cumsum(values[::-1] / powers, dtype=np.float64) * powers
    if not np.isfinite(reversed_result).all():
        return _discounted_cumsum_scan_numpy(values, discount)
    return reversed_result[::-1]


def _discounted_cumsum_scan_numpy(values: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    running = 0.0
    for position in range(len(values) - 1, -1, -1):
        running = float(values[position]) + discount * running
        result[position] = running
    return result


def _is_stale(
    indexed: _IndexedDecision,
    *,
    current_policy_version: int,
    max_staleness: int,
) -> bool:
    policy_age = current_policy_version - indexed.policy_version
    if policy_age < 0:
        raise ValueError(
            "behavior policy_version cannot be newer than the learner policy"
        )
    return policy_age > max_staleness


def _select_stale_decisions(
    indexed_decisions: Sequence[_IndexedDecision],
    *,
    current_policy_version: int,
    max_staleness: int,
    scope: Literal["decision", "seat_trajectory"],
) -> _StalenessSelection:
    """Return stale-row exclusions without truncating a configured GAE chain."""
    intrinsic_mask = tuple(
        _is_stale(
            indexed,
            current_policy_version=current_policy_version,
            max_staleness=max_staleness,
        )
        for indexed in indexed_decisions
    )
    stale_chain_keys = {
        (indexed.game_id, indexed.seat)
        for indexed, is_stale in zip(
            indexed_decisions,
            intrinsic_mask,
            strict=True,
        )
        if is_stale
    }
    if scope == "decision":
        drop_mask = intrinsic_mask
    else:
        drop_mask = tuple(
            (indexed.game_id, indexed.seat) in stale_chain_keys
            for indexed in indexed_decisions
        )
    intrinsic_count = sum(intrinsic_mask)
    dropped_count = sum(drop_mask)
    return _StalenessSelection(
        drop_mask=drop_mask,
        intrinsically_stale_decisions=intrinsic_count,
        propagated_stale_decisions=dropped_count - intrinsic_count,
        stale_seat_trajectories=len(stale_chain_keys),
    )


def summarize_learner_window_staleness(
    trajectories: Sequence[GameTrajectory],
    *,
    current_policy_version: int,
    max_staleness: int,
    staleness_scope: Literal["decision", "seat_trajectory"],
) -> LearnerWindowStaleness:
    """Count exactly the stale rows that formal learner assembly will drop.

    The learner window budget historically counts all decoded rows before the
    behavior-kind filter. Refill therefore subtracts only stale policy rows;
    teacher or other non-policy rows retain their existing budget semantics.
    """
    if current_policy_version < 0:
        raise ValueError("current_policy_version must be non-negative")
    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")
    indexed_decisions = tuple(_iter_indexed_decisions(trajectories))
    policy_decisions = tuple(
        indexed
        for indexed in indexed_decisions
        if indexed.behavior_kind == "policy_sample"
    )
    staleness = _select_stale_decisions(
        policy_decisions,
        current_policy_version=current_policy_version,
        max_staleness=max_staleness,
        scope=staleness_scope,
    )
    return LearnerWindowStaleness(
        total_decisions=len(indexed_decisions),
        stale_decisions=sum(staleness.drop_mask),
        intrinsically_stale_decisions=staleness.intrinsically_stale_decisions,
        propagated_stale_decisions=staleness.propagated_stale_decisions,
        stale_seat_trajectories=staleness.stale_seat_trajectories,
    )


def _normalize_retained_advantages(
    samples: Sequence[_LearnerSample],
    config: GaeConfig,
) -> list[_LearnerSample]:
    """Normalize raw advantages over exactly the retained learner rows."""
    if not samples or not config.normalize_advantages:
        return list(samples)
    advantages = np.asarray(
        [sample.estimate.advantage for sample in samples],
        dtype=np.float64,
    )
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())
    scale = advantage_std if advantage_std > config.normalize_epsilon else math.inf
    normalized = (advantages - advantage_mean) / scale
    return [
        replace(
            sample,
            estimate=replace(
                sample.estimate,
                normalized_advantage=float(normalized[index]),
            ),
        )
        for index, sample in enumerate(samples)
    ]


def _attach_prompt_token_credit(
    samples: Sequence[_LearnerSample],
    config: GaeConfig,
) -> list[_LearnerSample]:
    """Compute prompt-local credit from recorded behavior prefix values."""
    credited: list[_LearnerSample] = []
    for sample in samples:
        if _is_planner_sample(sample):
            credited.append(sample)
            continue
        token_logprobs, prefix_values, _stop_sampled = _indexed_token_trace(
            sample.indexed
        )
        if len(token_logprobs) != len(prefix_values):
            raise ValueError("token behavior evidence must align by active token")
        credited.append(
            replace(
                sample,
                token_credit=compute_prompt_token_credit(
                    prefix_values,
                    downstream_return=sample.estimate.return_value,
                    gamma=config.intra_prompt_gamma,
                    gae_lambda=config.intra_prompt_gae_lambda,
                ),
            )
        )
    return credited


def _normalize_retained_token_advantages(
    samples: Sequence[_LearnerSample],
    config: GaeConfig,
) -> list[_LearnerSample]:
    """Normalize only active token advantages from retained seat chains."""
    if not samples:
        return []
    if not config.normalize_advantages:
        return list(samples)
    advantages = np.asarray(
        [
            advantage
            for sample in samples
            for advantage in (
                (sample.estimate.advantage,)
                if sample.token_credit is None
                else sample.token_credit.advantages
            )
        ],
        dtype=np.float64,
    )
    if advantages.size == 0:
        raise ValueError("token-credit learner samples require active tokens")
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())
    scale = advantage_std if advantage_std > config.normalize_epsilon else math.inf
    normalized = (advantages - advantage_mean) / scale
    output: list[_LearnerSample] = []
    offset = 0
    for sample in samples:
        credit = sample.token_credit
        length = 1 if credit is None else len(credit.advantages)
        stop = offset + length
        if credit is None:
            if not _is_planner_sample(sample):
                raise ValueError(
                    "token-credit batches may omit token targets only for planner rows"
                )
            output.append(
                replace(
                    sample,
                    estimate=replace(
                        sample.estimate,
                        normalized_advantage=float(normalized[offset]),
                    ),
                )
            )
        else:
            output.append(
                replace(
                    sample,
                    token_credit=replace(
                        credit,
                        advantages=tuple(
                            float(value) for value in normalized[offset:stop]
                        ),
                    ),
                )
            )
        offset = stop
    return output


def _indexed_token_trace(
    indexed: _IndexedDecision,
) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    """Return complete active-token evidence for one indexed decision."""
    if indexed.record is not None:
        record = indexed.record
        if (
            record.token_logprobs is None
            or record.prefix_value_preds is None
            or record.stop_sampled is None
        ):
            raise ValueError(
                "decode_token credit requires schema-3 token behavior evidence"
            )
        return (
            record.token_logprobs,
            record.prefix_value_preds,
            record.stop_sampled,
        )
    block = _require_array_block(indexed)
    token_logprobs = block.token_logprobs_at(indexed.array_index)
    prefix_values = block.prefix_value_preds_at(indexed.array_index)
    if token_logprobs is None or prefix_values is None or block.stop_sampled is None:
        raise ValueError(
            "decode_token credit requires schema-3 token behavior evidence"
        )
    return (
        token_logprobs,
        prefix_values,
        bool(block.stop_sampled[indexed.array_index]),
    )


def _optional_indexed_token_trace(
    indexed: _IndexedDecision,
) -> tuple[tuple[float, ...], tuple[float, ...], bool] | None:
    if indexed.record is not None and indexed.record.token_logprobs is None:
        return None
    if indexed.array_block is not None:
        block = indexed.array_block
        if not block.has_token_trace:
            return None
        if block.token_logprobs_at(indexed.array_index) is None:
            return None
    return _indexed_token_trace(indexed)


def _indexed_planner_behavior(
    indexed: _IndexedDecision,
) -> PlannerBehaviorEvidence | None:
    """Return the explicit schema-9 branch evidence for one learner row."""
    return indexed.planner_behavior


def _is_planner_sample(sample: _LearnerSample) -> bool:
    evidence = _indexed_planner_behavior(sample.indexed)
    return (
        evidence is not None
        and evidence.branch is PlannerBehaviorBranch.PLANNER_CONDITIONED
    )


def _learner_diagnostics(
    samples: Sequence[_LearnerSample],
    *,
    current_policy_version: int,
) -> LearnerDiagnostics:
    """Summarize retained prompt shape and behavior-version mass."""
    rows = tuple(_learner_diagnostic_row(sample) for sample in samples)
    return summarize_learner_diagnostics(
        rows,
        current_policy_version=current_policy_version,
    )


def _learner_diagnostic_row(sample: _LearnerSample) -> LearnerDiagnosticRow:
    indexed = sample.indexed
    trace = _optional_indexed_token_trace(indexed)
    training_advantages = (
        (sample.estimate.normalized_advantage,)
        if sample.token_credit is None
        else sample.token_credit.advantages
    )
    return LearnerDiagnosticRow(
        game_id=indexed.game_id,
        seat=indexed.seat,
        policy_version=indexed.policy_version,
        action_count=len(_indexed_action(indexed)),
        max_count=_indexed_max_count(indexed),
        prompt_context=_indexed_prompt_context(indexed),
        token_count=0 if trace is None else len(trace[0]),
        stop_sampled=False if trace is None else trace[2],
        deck_signature=("legacy" if indexed.deck is None else indexed.deck.signature),
        training_advantage_abs=sum(abs(value) for value in training_advantages),
        training_metadata=indexed.training_metadata,
    )


def _indexed_action(indexed: _IndexedDecision) -> tuple[int, ...]:
    if indexed.record is not None:
        return indexed.record.action
    return _array_action(indexed)


def _indexed_max_count(indexed: _IndexedDecision) -> int:
    if indexed.record is not None:
        return indexed.record.max_count
    block = _require_array_block(indexed)
    return int(block.options.max_counts[indexed.array_index])


def _indexed_prompt_context(indexed: _IndexedDecision) -> int:
    if indexed.record is not None:
        raw_contexts = getattr(indexed.record.options, "contexts", None)
        if raw_contexts is not None:
            return int(raw_contexts[0])
        tuple_options = cast(Sequence[Any], indexed.record.options)
        return int(tuple_options[0].context)
    block = _require_array_block(indexed)
    return int(block.options.contexts[indexed.array_index, 0])


def _require_token_credit(sample: _LearnerSample) -> PromptTokenCredit:
    credit = sample.token_credit
    if credit is None:
        raise ValueError("token-credit learner sample is missing prompt targets")
    return credit


def _order_recurrent_samples(
    samples: Sequence[_LearnerSample],
) -> list[_LearnerSample]:
    """Return complete game-seat histories as contiguous ordered sequences."""
    grouped: dict[tuple[str, int], list[_LearnerSample]] = {}
    for sample in samples:
        indexed = sample.indexed
        if indexed.policy_artifact is None:
            raise ValueError("recurrent learner row is missing its policy artifact")
        grouped.setdefault((indexed.game_id, indexed.seat), []).append(sample)
    ordered: list[_LearnerSample] = []
    for group in grouped.values():
        group.sort(key=lambda sample: sample.indexed.decision_index)
        indices = tuple(sample.indexed.decision_index for sample in group)
        if indices != tuple(range(len(group))):
            raise ValueError("recurrent decision sequence is not continuous from zero")
        versions = {sample.indexed.policy_version for sample in group}
        if len(versions) != 1:
            raise ValueError("recurrent sequence crossed a behavior policy version")
        artifacts = {sample.indexed.policy_artifact for sample in group}
        if len(artifacts) != 1:
            raise ValueError("recurrent sequence crossed a policy artifact")
        decks = {sample.indexed.deck for sample in group}
        if None in decks or len(decks) != 1:
            raise ValueError("recurrent sequence crossed an exact deck identity")
        ordered.extend(group)
    return ordered


def _recurrent_sample_groups(
    samples: Sequence[_LearnerSample],
) -> tuple[tuple[int, ...], ...]:
    """Return contiguous row groups from already ordered recurrent samples."""
    groups: list[list[int]] = []
    previous_key: tuple[str, int] | None = None
    closed: set[tuple[str, int]] = set()
    for index, sample in enumerate(samples):
        key = (sample.indexed.game_id, sample.indexed.seat)
        if key != previous_key:
            if key in closed:
                raise ValueError("recurrent sequence rows are not contiguous")
            if previous_key is not None:
                closed.add(previous_key)
            groups.append([])
            previous_key = key
        groups[-1].append(index)
    return tuple(tuple(group) for group in groups)


def _shuffle_recurrent_sample_indices(
    samples: Sequence[_LearnerSample],
    *,
    seed: int,
) -> list[int]:
    """Shuffle whole recurrent sequences without changing timestep order."""
    groups = list(_recurrent_sample_groups(samples))
    random.Random(seed).shuffle(groups)
    return [index for group in groups for index in group]


def _collate_ppo_minibatches(
    samples: Sequence[_LearnerSample],
    sample_indices: Sequence[int],
    config: LearnerBatchConfig,
    *,
    device: torch.device | str | None,
) -> tuple[Sequence[PpoBatch], LearnerBatchPool | None]:
    if len(samples) != len(sample_indices):
        raise ValueError("samples and sample_indices must align")
    if not samples:
        return ((), None)
    full_batch = _collate_ppo_batch(
        samples,
        device=None,
        schema9_config=config.schema9,
        macro_credit_config=config.macro_credit,
    )
    full_batch = replace(
        full_batch,
        sample_indices=torch.arange(len(samples), dtype=torch.long),
    )
    validate_ppo_batch(full_batch)
    full_batch = replace(full_batch, integrity_validated=True)
    pool = LearnerBatchPool(batch=full_batch, sample_count=len(samples))
    batches = _take_ppo_minibatches(
        full_batch,
        sample_indices,
        config,
        device=device,
    )
    return (batches, pool)


def _take_ppo_minibatches(
    full_batch: PpoBatch,
    sample_indices: Sequence[int],
    config: LearnerBatchConfig,
    *,
    device: torch.device | str | None,
) -> Sequence[PpoBatch]:
    ordered_indices = list(sample_indices)
    if full_batch.sequence_offsets is not None:
        if config.route_bucket_minibatches or config.shape_bucket_accumulation:
            raise ValueError(
                "recurrent PPO does not support row-level route or shape bucketing"
            )
        sequence_groups = _ordered_recurrent_sequence_rows(
            full_batch,
            ordered_indices,
        )
        index_chunks: list[list[int]] = []
        chunk: list[int] = []
        for group in sequence_groups:
            if chunk and len(chunk) + len(group) > config.microbatch_size:
                index_chunks.append(chunk)
                chunk = []
            chunk.extend(group)
        if chunk:
            index_chunks.append(chunk)
    elif config.route_bucket_minibatches:
        index_chunks = _route_bucket_minibatches(
            full_batch,
            ordered_indices,
            microbatch_size=config.microbatch_size,
            shape_bucket=config.shape_bucket_accumulation,
        )
    else:
        if config.shape_bucket_accumulation and config.gradient_accumulation_steps > 1:
            ordered_indices = _shape_bucket_accumulation_groups(
                full_batch,
                ordered_indices,
                effective_batch_size=config.effective_batch_size,
            )
        index_chunks = [
            ordered_indices[start : start + config.microbatch_size]
            for start in range(0, len(ordered_indices), config.microbatch_size)
        ]
    retained_chunks = tuple(
        tuple(chunk_indices)
        for chunk_indices in index_chunks
        if len(chunk_indices) >= config.microbatch_size or not config.drop_last
    )
    return _PlannedPpoBatches(
        full_batch=full_batch,
        index_chunks=retained_chunks,
        device=device,
    )


def _recurrent_sequence_rows(batch: PpoBatch) -> tuple[tuple[int, ...], ...]:
    """Materialize row ranges from validated sequence offsets."""
    offsets = batch.sequence_offsets
    if offsets is None:
        raise ValueError("PPO batch has no recurrent sequence offsets")
    values = tuple(
        int(value)
        for value in offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    )
    return tuple(tuple(range(start, stop)) for start, stop in pairwise(values))


def _ordered_recurrent_sequence_rows(
    batch: PpoBatch,
    ordered_indices: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Require an ordering made exclusively of complete sequence ranges."""
    groups = _recurrent_sequence_rows(batch)
    by_first = {group[0]: group for group in groups}
    sequence_index_by_first = {
        group[0]: sequence_index for sequence_index, group in enumerate(groups)
    }
    selected: list[tuple[int, ...]] = []
    seen: set[int] = set()
    cursor = 0
    values = tuple(int(index) for index in ordered_indices)
    while cursor < len(values):
        group = by_first.get(values[cursor])
        if group is None or values[cursor : cursor + len(group)] != group:
            raise ValueError("recurrent minibatches must select complete sequences")
        sequence_index = sequence_index_by_first[group[0]]
        if sequence_index in seen:
            raise ValueError("recurrent minibatch ordering repeats a sequence")
        seen.add(sequence_index)
        selected.append(group)
        cursor += len(group)
    return tuple(selected)


def _route_bucket_minibatches(
    batch: PpoBatch,
    sample_indices: Sequence[int],
    *,
    microbatch_size: int,
    shape_bucket: bool,
) -> list[list[int]]:
    """Build route-local chunks and interleave them across private experts.

    Dense-private experts still receive full route-local GEMMs inside each
    microbatch. Interleaving those chunks ensures consecutive gradient-
    accumulation slots cover distinct experts instead of applying a long run
    of optimizer pressure to one route. Short per-route tails are packed only
    after all full chunks so no rows are dropped.
    """
    if microbatch_size <= 0:
        raise ValueError("route microbatch size must be positive")
    if batch.decks is None:
        return [
            list(sample_indices[start : start + microbatch_size])
            for start in range(0, len(sample_indices), microbatch_size)
        ]
    signatures = batch.decks.signatures
    if len(signatures) != len(batch.actions):
        raise ValueError("deck signatures must align with PPO rows")
    route_rows: dict[str, list[int]] = {}
    for index in sample_indices:
        signature = signatures[index]
        route_rows.setdefault(signature, []).append(index)
    if shape_bucket:
        state_lengths = (~batch.states.padding_mask).sum(dim=1).tolist()
        option_lengths = batch.options.valid_options.sum(dim=1).tolist()
        token_lengths = (
            [0] * len(state_lengths)
            if batch.token_mask is None
            else batch.token_mask.sum(dim=1).tolist()
        )
        for rows in route_rows.values():
            rows.sort(
                key=lambda index: (
                    int(state_lengths[index]),
                    int(option_lengths[index]),
                    int(token_lengths[index]),
                )
            )

    full_chunks_by_route: list[list[list[int]]] = []
    tail_rows: list[int] = []
    for rows in route_rows.values():
        full_row_count = len(rows) - len(rows) % microbatch_size
        full_chunks_by_route.append(
            [
                rows[start : start + microbatch_size]
                for start in range(0, full_row_count, microbatch_size)
            ]
        )
        tail_rows.extend(rows[full_row_count:])

    chunks: list[list[int]] = []
    max_route_chunks = max(
        (len(route_chunks) for route_chunks in full_chunks_by_route),
        default=0,
    )
    for chunk_index in range(max_route_chunks):
        chunks.extend(
            route_chunks[chunk_index]
            for route_chunks in full_chunks_by_route
            if chunk_index < len(route_chunks)
        )
    chunks.extend(
        tail_rows[start : start + microbatch_size]
        for start in range(0, len(tail_rows), microbatch_size)
    )
    return chunks


def _shape_bucket_accumulation_groups(
    batch: PpoBatch,
    sample_indices: Sequence[int],
    *,
    effective_batch_size: int,
) -> list[int]:
    """Sort only within each optimizer update to reduce microbatch padding."""
    state_lengths = (~batch.states.padding_mask).sum(dim=1).tolist()
    option_lengths = batch.options.valid_options.sum(dim=1).tolist()
    token_lengths = (
        [0] * len(state_lengths)
        if batch.token_mask is None
        else batch.token_mask.sum(dim=1).tolist()
    )
    ordered: list[int] = []
    for start in range(0, len(sample_indices), effective_batch_size):
        group = list(sample_indices[start : start + effective_batch_size])
        group.sort(
            key=lambda index: (
                int(state_lengths[index]),
                int(option_lengths[index]),
                int(token_lengths[index]),
            )
        )
        ordered.extend(group)
    return ordered


def _collate_ppo_batch(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
    schema9_config: Schema9LearnerBatchConfig | None,
    macro_credit_config: MacroCreditConfig | None,
) -> PpoBatch:
    if not samples:
        raise ValueError("samples must be non-empty")
    _validate_schema9_learner_contract(samples, schema9_config)
    _validate_schema10_learner_contract(samples, macro_credit_config)
    if all(sample.indexed.record is not None for sample in samples):
        return _collate_record_ppo_batch(
            samples,
            device=device,
            schema9_config=schema9_config,
            macro_credit_config=macro_credit_config,
        )
    if all(sample.indexed.array_block is not None for sample in samples):
        return _collate_array_ppo_batch(
            samples,
            device=device,
            schema9_config=schema9_config,
            macro_credit_config=macro_credit_config,
        )
    raise ValueError("cannot mix record and array-backed learner samples")


def _collate_record_ppo_batch(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
    schema9_config: Schema9LearnerBatchConfig | None,
    macro_credit_config: MacroCreditConfig | None,
) -> PpoBatch:
    records = [_require_record(sample) for sample in samples]
    options = collate_encoded_options(
        [record.options for record in records],
        min_counts=[record.min_count for record in records],
        max_counts=[record.max_count for record in records],
        device=device,
    )
    actions = tuple(record.action for record in records)
    token_tensors = _collate_token_credit_tensors(samples, device=device)
    engine_teacher = _collate_engine_teacher_tensors(
        samples,
        behavior_actions=actions,
        options=options,
        device=device,
    )
    factual = _collate_factual_tensors(samples, device=device)
    macro = _collate_macro_tensors(samples, device=device)
    recurrent = _collate_recurrent_context(samples, device=device)
    return PpoBatch(
        states=collate_state_tokens(
            [record.state for record in records],
            device=device,
        ),
        options=options,
        actions=actions,
        old_action_logprobs=torch.tensor(
            [record.action_logprob for record in records],
            dtype=torch.float32,
            device=device,
        ),
        sampling_temperatures=torch.tensor(
            [_required_sampling_temperature(sample.indexed) for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        old_values=torch.tensor(
            [record.value_pred for record in records],
            dtype=torch.float32,
            device=device,
        ),
        returns=torch.tensor(
            [sample.estimate.return_value for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        advantages=torch.tensor(
            [sample.estimate.normalized_advantage for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        decks=_collate_sample_decks(samples, device=device),
        action_targets=_ppo_action_targets(actions, options, device=device),
        old_token_logprobs=token_tensors[0],
        old_prefix_values=token_tensors[1],
        token_returns=token_tensors[2],
        token_advantages=token_tensors[3],
        token_mask=token_tensors[4],
        engine_teacher_actions=engine_teacher[0],
        engine_teacher_action_targets=engine_teacher[1],
        engine_teacher_confidences=engine_teacher[2],
        engine_teacher_weights=engine_teacher[3],
        engine_teacher_mask=engine_teacher[4],
        engine_teacher_candidate_actions=engine_teacher[5],
        engine_teacher_candidate_features=engine_teacher[6],
        factual_effect_targets=factual[0],
        factual_actor_relations=factual[1],
        factual_next_contexts=factual[2],
        macro_effect_targets=macro[0],
        macro_endpoints=macro[1],
        macro_continuation_summaries=macro[2],
        macro_mask=macro[3],
        planner_replay=_collate_planner_replay(samples, device=device),
        root_information_value_replay=_collate_root_information_value_replay(
            samples,
            schema9_config=schema9_config,
            macro_credit_config=macro_credit_config,
            device=device,
        ),
        public_events=recurrent[0],
        sequence_offsets=recurrent[1],
        sequence_artifacts=recurrent[2],
    )


def _collate_array_ppo_batch(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
    schema9_config: Schema9LearnerBatchConfig | None,
    macro_credit_config: MacroCreditConfig | None,
) -> PpoBatch:
    indexed = [sample.indexed for sample in samples]
    selections = _array_block_selections(indexed)
    options = _collate_array_options(selections, device=device)
    actions = tuple(_array_action(item) for item in indexed)
    token_tensors = _collate_token_credit_tensors(samples, device=device)
    engine_teacher = _collate_engine_teacher_tensors(
        samples,
        behavior_actions=actions,
        options=options,
        device=device,
    )
    factual = _collate_factual_tensors(samples, device=device)
    macro = _collate_macro_tensors(samples, device=device)
    recurrent = _collate_recurrent_context(samples, device=device)
    return PpoBatch(
        states=_collate_array_states(selections, device=device),
        options=options,
        actions=actions,
        old_action_logprobs=torch.as_tensor(
            _concat_selected_1d(
                selections,
                lambda block: block.action_logprobs,
            ),
            dtype=torch.float32,
            device=device,
        ),
        sampling_temperatures=torch.as_tensor(
            [_required_sampling_temperature(sample.indexed) for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        old_values=torch.as_tensor(
            _concat_selected_1d(
                selections,
                lambda block: block.value_preds,
            ),
            dtype=torch.float32,
            device=device,
        ),
        returns=torch.as_tensor(
            [sample.estimate.return_value for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        advantages=torch.as_tensor(
            [sample.estimate.normalized_advantage for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        decks=_collate_sample_decks(samples, device=device),
        action_targets=_ppo_action_targets(actions, options, device=device),
        old_token_logprobs=token_tensors[0],
        old_prefix_values=token_tensors[1],
        token_returns=token_tensors[2],
        token_advantages=token_tensors[3],
        token_mask=token_tensors[4],
        engine_teacher_actions=engine_teacher[0],
        engine_teacher_action_targets=engine_teacher[1],
        engine_teacher_confidences=engine_teacher[2],
        engine_teacher_weights=engine_teacher[3],
        engine_teacher_mask=engine_teacher[4],
        engine_teacher_candidate_actions=engine_teacher[5],
        engine_teacher_candidate_features=engine_teacher[6],
        factual_effect_targets=factual[0],
        factual_actor_relations=factual[1],
        factual_next_contexts=factual[2],
        macro_effect_targets=macro[0],
        macro_endpoints=macro[1],
        macro_continuation_summaries=macro[2],
        macro_mask=macro[3],
        planner_replay=_collate_planner_replay(samples, device=device),
        root_information_value_replay=_collate_root_information_value_replay(
            samples,
            schema9_config=schema9_config,
            macro_credit_config=macro_credit_config,
            device=device,
        ),
        public_events=recurrent[0],
        sequence_offsets=recurrent[1],
        sequence_artifacts=recurrent[2],
    )


def _collate_recurrent_context(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> tuple[
    PublicEventBatch | None,
    torch.Tensor | None,
    tuple[PolicyArtifactIdentity, ...] | None,
]:
    """Collate compact event rows and explicit full-sequence boundaries."""
    presence = tuple(sample.indexed.policy_artifact is not None for sample in samples)
    if not any(presence):
        return (None, None, None)
    if not all(presence):
        raise ValueError("cannot mix recurrent and stateless PPO rows")
    groups = _recurrent_sample_groups(samples)
    artifacts: list[PolicyArtifactIdentity] = []
    offsets = [0]
    for group in groups:
        first = samples[group[0]].indexed
        artifact = first.policy_artifact
        if artifact is None:
            raise AssertionError("recurrent sequence artifact disappeared")
        if any(samples[index].indexed.policy_artifact != artifact for index in group):
            raise ValueError("recurrent sequence crossed a policy artifact")
        artifacts.append(artifact)
        offsets.append(offsets[-1] + len(group))
    deltas = tuple(_indexed_public_event_delta(sample.indexed) for sample in samples)
    return (
        collate_public_event_deltas(deltas, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        tuple(artifacts),
    )


def _indexed_public_event_delta(indexed: _IndexedDecision) -> Any:
    """Return one aligned public event delta from object or compact storage."""
    if indexed.record is not None:
        delta = indexed.record.public_event_delta
    else:
        block = _require_array_block(indexed)
        delta = (
            None
            if block.public_events is None
            else block.public_events.delta_at(indexed.array_index)
        )
    if delta is None:
        raise ValueError("recurrent learner row is missing its public event delta")
    return delta


def _collate_sample_decks(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> DeckBatch | None:
    decks = tuple(sample.indexed.deck for sample in samples)
    if not any(deck is not None for deck in decks):
        return None
    if not all(deck is not None for deck in decks):
        raise ValueError("cannot mix learner rows with and without deck context")
    return DeckBatch.from_decks(
        tuple(cast(CanonicalDeck, deck) for deck in decks),
        device=device,
    )


def _collate_token_credit_tensors(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Pad active token evidence and targets for one learner batch."""
    credit_presence = tuple(sample.token_credit is not None for sample in samples)
    if not any(credit_presence):
        return (None, None, None, None, None)
    if any(
        not present and not _is_planner_sample(sample)
        for sample, present in zip(samples, credit_presence, strict=True)
    ):
        raise ValueError("token credit may be absent only for planner behavior rows")
    traces = tuple(
        None if sample.token_credit is None else _indexed_token_trace(sample.indexed)
        for sample in samples
    )
    width = max(len(trace[0]) for trace in traces if trace is not None)
    old_logprobs = np.zeros((len(samples), width), dtype=np.float32)
    old_prefix_values = np.zeros((len(samples), width), dtype=np.float32)
    token_returns = np.zeros((len(samples), width), dtype=np.float32)
    token_advantages = np.zeros((len(samples), width), dtype=np.float32)
    token_mask = np.zeros((len(samples), width), dtype=np.bool_)
    for index, (trace, sample) in enumerate(zip(traces, samples, strict=True)):
        if trace is None:
            continue
        credit = _require_token_credit(sample)
        logprobs, prefix_values, _stop_sampled = trace
        length = len(logprobs)
        if (
            len(prefix_values) != length
            or len(credit.value_targets) != length
            or len(credit.advantages) != length
        ):
            raise ValueError("token evidence and prompt targets must align")
        old_logprobs[index, :length] = logprobs
        old_prefix_values[index, :length] = prefix_values
        token_returns[index, :length] = credit.value_targets
        token_advantages[index, :length] = credit.advantages
        token_mask[index, :length] = True
    return (
        torch.as_tensor(old_logprobs, device=device),
        torch.as_tensor(old_prefix_values, device=device),
        torch.as_tensor(token_returns, device=device),
        torch.as_tensor(token_advantages, device=device),
        torch.as_tensor(token_mask, device=device),
    )


def _collate_planner_replay(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> PlannerReplayBatch | None:
    """Flatten planner-conditioned schema-9 rows for exact PPO replay."""
    evidence_rows = tuple(
        _indexed_planner_behavior(sample.indexed) for sample in samples
    )
    presence = tuple(evidence is not None for evidence in evidence_rows)
    if not any(presence):
        return None
    if not all(presence):
        raise ValueError("cannot mix schema-9 and legacy rows in one learner batch")
    planner_rows = tuple(
        (index, cast(PlannerBehaviorEvidence, evidence), samples[index])
        for index, evidence in enumerate(evidence_rows)
        if cast(PlannerBehaviorEvidence, evidence).branch
        is PlannerBehaviorBranch.PLANNER_CONDITIONED
    )
    if not planner_rows:
        return None
    candidate_offsets = [0]
    target_probabilities: list[float] = []
    old_behavior_probabilities: list[float] = []
    score_priors: list[float] = []
    rules_exact: list[bool] = []
    candidate_actions: list[tuple[tuple[int, ...], ...]] = []
    candidate_features: list[torch.Tensor] = []
    for _index, evidence, _sample in planner_rows:
        candidates = evidence.candidates
        target_probabilities.extend(
            candidate.target_probability for candidate in candidates
        )
        old_behavior_probabilities.extend(
            candidate.behavior_probability for candidate in candidates
        )
        score_priors.extend(candidate.score_prior for candidate in candidates)
        rules_exact.extend(candidate.rules_exact for candidate in candidates)
        candidate_actions.append(tuple(candidate.action for candidate in candidates))
        candidate_features.append(
            torch.tensor(
                [candidate.aggregate_features for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
        )
        candidate_offsets.append(candidate_offsets[-1] + len(candidates))
    replay = PlannerReplayBatch(
        decision_indices=torch.tensor(
            [index for index, _evidence, _sample in planner_rows],
            dtype=torch.long,
            device=device,
        ),
        candidate_offsets=torch.tensor(
            candidate_offsets,
            dtype=torch.long,
            device=device,
        ),
        selected_candidate_indices=torch.tensor(
            [
                evidence.selected_candidate_index
                for _index, evidence, _sample in planner_rows
            ],
            dtype=torch.long,
            device=device,
        ),
        target_probabilities=torch.tensor(
            target_probabilities,
            dtype=torch.float32,
            device=device,
        ),
        old_behavior_probabilities=torch.tensor(
            old_behavior_probabilities,
            dtype=torch.float32,
            device=device,
        ),
        immutable_score_priors=torch.tensor(
            score_priors,
            dtype=torch.float32,
            device=device,
        ),
        rules_exact=torch.tensor(rules_exact, dtype=torch.bool, device=device),
        scenario_grid_complete=torch.tensor(
            [
                evidence.scenario_grid_complete
                for _index, evidence, _sample in planner_rows
            ],
            dtype=torch.bool,
            device=device,
        ),
        identity_valid=torch.ones(
            len(planner_rows),
            dtype=torch.bool,
            device=device,
        ),
        policy_ages=torch.tensor(
            [sample.policy_age for _index, _evidence, sample in planner_rows],
            dtype=torch.long,
            device=device,
        ),
        support_exhaustive=torch.tensor(
            [evidence.support_exhaustive for _index, evidence, _sample in planner_rows],
            dtype=torch.bool,
            device=device,
        ),
        support_censored=torch.tensor(
            [evidence.support_censored for _index, evidence, _sample in planner_rows],
            dtype=torch.bool,
            device=device,
        ),
        planner_temperatures=torch.tensor(
            [
                evidence.planner_temperature
                for _index, evidence, _sample in planner_rows
            ],
            dtype=torch.float32,
            device=device,
        ),
        candidate_actions=tuple(candidate_actions),
        candidate_features=tuple(candidate_features),
    )
    replay.validate(batch_size=len(samples))
    return replay


def _validate_schema9_learner_contract(
    samples: Sequence[_LearnerSample],
    config: Schema9LearnerBatchConfig | None,
) -> None:
    """Revalidate a selected schema-9 batch before tensor collation."""
    _validate_schema9_indexed_contract(
        tuple(sample.indexed for sample in samples),
        config,
    )


def _validate_schema10_learner_contract(
    samples: Sequence[_LearnerSample],
    config: MacroCreditConfig | None,
) -> None:
    """Bind selected macro rows to the configured immutable featurizers."""
    presence = tuple(
        sample.indexed.array_block is not None
        and sample.indexed.array_block.executed_macros is not None
        for sample in samples
    )
    if any(presence) and not all(presence):
        raise ValueError("cannot mix schema-10 and legacy learner rows")
    if any(presence) and (config is None or not config.enabled):
        raise ValueError("schema-10 learner rows require macro credit config")
    if config is not None and config.enabled and not all(presence):
        raise ValueError("macro credit learner requires schema-10 trajectories")
    if not any(presence):
        return
    resolved_config = cast(MacroCreditConfig, config)
    validated_blocks: set[int] = set()
    adapter_by_policy_version: dict[int, str] = {}
    policy_version_by_adapter: dict[str, int] = {}
    for sample in samples:
        block = sample.indexed.array_block
        if block is None or block.executed_macros is None:
            raise AssertionError("schema-10 learner row lost its macro block")
        block_identity = id(block)
        if block_identity in validated_blocks:
            continue
        validated_blocks.add(block_identity)
        if block.executed_macros.aggregation_fingerprint != (
            resolved_config.aggregation_fingerprint
        ):
            raise ValueError("macro aggregation identity differs from learner")
        if block.executed_macros.continuation_summary_fingerprint != (
            resolved_config.continuation_summary_fingerprint
        ):
            raise ValueError("macro continuation identity differs from learner")
        identity = resolved_config.native_teacher_identity
        teacher = block.macro_teacher
        if teacher is None:
            continue
        if identity is None:
            raise ValueError("macro teacher evidence has no learner identity contract")
        expected = (
            identity.constructor_fingerprint,
            identity.scorer_fingerprint,
            identity.controller_fingerprint,
            identity.planner_fingerprint,
        )
        for decision_index in np.flatnonzero(teacher.masks):
            evidence = teacher.evidence_at(int(decision_index))
            if evidence is None:
                raise AssertionError("active macro teacher row is unavailable")
            policy_version = int(block.policy_versions[int(decision_index)])
            if evidence.policy_version != policy_version:
                raise ValueError(
                    "macro teacher adapter differs from the behavior policy version"
                )
            actual = (
                evidence.constructor_fingerprint,
                evidence.scorer_fingerprint,
                evidence.controller_fingerprint,
                evidence.producer_fingerprint,
            )
            if actual != expected:
                raise ValueError("macro teacher identity differs from learner contract")
            existing_adapter = adapter_by_policy_version.setdefault(
                policy_version,
                evidence.adapter_fingerprint,
            )
            if existing_adapter != evidence.adapter_fingerprint:
                raise ValueError("one policy version has multiple macro adapters")
            existing_version = policy_version_by_adapter.setdefault(
                evidence.adapter_fingerprint,
                policy_version,
            )
            if existing_version != policy_version:
                raise ValueError("one macro adapter has multiple policy versions")


def _validate_schema9_indexed_contract(
    indexed_rows: Sequence[_IndexedDecision],
    config: Schema9LearnerBatchConfig | None,
) -> None:
    """Hard-reject wire rows incompatible with the formal learner contract."""
    evidence_rows = tuple(_indexed_planner_behavior(row) for row in indexed_rows)
    presence = tuple(evidence is not None for evidence in evidence_rows)
    if any(presence) and not all(presence):
        raise ValueError("cannot mix schema-9 and legacy rows in one learner batch")
    if any(presence) and config is None:
        raise ValueError("schema-9 learner rows require schema9 batch configuration")
    if config is None:
        return
    model_by_policy_version: dict[int, str] = {}
    policy_version_by_model: dict[str, int] = {}
    for indexed, evidence in zip(indexed_rows, evidence_rows, strict=True):
        if evidence is None:
            continue
        if evidence.policy_version != indexed.policy_version:
            raise ValueError("schema-9 model lease differs from behavior version")
        existing_model = model_by_policy_version.setdefault(
            evidence.policy_version,
            evidence.model_fingerprint,
        )
        if existing_model != evidence.model_fingerprint:
            raise ValueError("one policy version has multiple model fingerprints")
        existing_version = policy_version_by_model.setdefault(
            evidence.model_fingerprint,
            evidence.policy_version,
        )
        if existing_version != evidence.policy_version:
            raise ValueError("one model fingerprint has multiple policy versions")
    expected_identities = (
        ("constructor", config.expected_constructor_fingerprint),
        ("scorer", config.expected_scorer_fingerprint),
        ("controller", config.expected_controller_fingerprint),
        ("planner", config.expected_planner_fingerprint),
    )
    for evidence in evidence_rows:
        if evidence is None or evidence.branch is PlannerBehaviorBranch.BASE_FALLBACK:
            continue
        for name, expected in expected_identities:
            if getattr(evidence, f"{name}_fingerprint") != expected:
                raise ValueError(
                    f"schema-9 planner {name} identity differs from learner contract"
                )


def _collate_root_information_value_replay(
    samples: Sequence[_LearnerSample],
    *,
    schema9_config: Schema9LearnerBatchConfig | None,
    macro_credit_config: MacroCreditConfig | None,
    device: torch.device | str | None,
) -> RootInformationValueReplayBatch | None:
    """Deduplicate actual endpoints, tensorize once, and align WDL targets."""
    rows: list[_EndpointLearnerRow] = []
    canonical_decisions: list[int] = []
    row_by_key: dict[tuple[str, int, str], int] = {}
    for decision_index, sample in enumerate(samples):
        row = _indexed_endpoint_learner_row(sample.indexed)
        if row is None:
            continue
        existing = row_by_key.get(row.key)
        if existing is not None:
            if rows[existing] != row:
                raise ValueError("one actual endpoint has conflicting learner inputs")
            continue
        row_by_key[row.key] = len(rows)
        rows.append(row)
        canonical_decisions.append(decision_index)
    if not rows:
        return None
    if schema9_config is not None and macro_credit_config is not None:
        raise ValueError("endpoint replay cannot mix schema-9 and schema-10 configs")
    tensorizer_config = (
        schema9_config.root_information_tensorizer
        if schema9_config is not None
        else None
        if macro_credit_config is None
        else macro_credit_config.root_information_tensorizer
    )
    if tensorizer_config is None:
        raise ValueError("endpoint value rows require a tensorizer configuration")
    unique_rows: list[_EndpointLearnerRow] = []
    input_by_fingerprint: dict[str, int] = {}
    value_input_indices: list[int] = []
    for row in rows:
        fingerprint = row.leaf.information_history_fingerprint
        input_index = input_by_fingerprint.get(fingerprint)
        if input_index is None:
            input_index = len(unique_rows)
            input_by_fingerprint[fingerprint] = input_index
            unique_rows.append(row)
        elif (
            unique_rows[input_index].leaf != row.leaf
            or unique_rows[input_index].deck != row.deck
        ):
            raise ValueError("one root-information input has conflicting deck data")
        value_input_indices.append(input_index)
    model_inputs = ProductionRootInformationTensorizer(tensorizer_config).tensorize_all(
        tuple(row.leaf for row in unique_rows)
    )
    replay = RootInformationValueReplayBatch(
        decision_indices=torch.tensor(canonical_decisions, dtype=torch.long),
        value_input_indices=torch.tensor(value_input_indices, dtype=torch.long),
        model_inputs=model_inputs,
        decks=DeckBatch.from_decks(tuple(row.deck for row in unique_rows)),
        final_root_outcomes=torch.tensor(
            [row.final_root_outcome for row in rows],
            dtype=torch.float32,
        ),
    )
    replay.validate(batch_size=len(samples))
    return (
        move_root_information_value_replay(replay, device=device)
        if device is not None
        else replay
    )


def _indexed_endpoint_learner_row(
    indexed: _IndexedDecision,
) -> _EndpointLearnerRow | None:
    """Reconstruct one referenced endpoint without scanning unreferenced rows."""
    deck = indexed.deck
    if indexed.record is not None:
        leaf = indexed.record.executed_endpoint_value_leaf
        if leaf is None:
            return None
        if deck is None:
            raise ValueError("root-information endpoint rows require deck context")
        outcome = int(indexed.terminal_reward)
        if float(outcome) != indexed.terminal_reward or outcome not in (-1, 0, 1):
            raise ValueError("root-information endpoint outcome must be W/D/L")
        return _EndpointLearnerRow(
            key=(indexed.game_id, indexed.seat, leaf.information_history_fingerprint),
            leaf=leaf,
            final_root_outcome=outcome,
            deck=deck,
        )
    block = _require_array_block(indexed)
    row = block.executed_endpoint_value_row_at(indexed.array_index)
    if row is None:
        return None
    if deck is None:
        raise ValueError("root-information endpoint rows require deck context")
    if row.root_player != indexed.seat:
        raise ValueError("endpoint root player differs from its learner decision")
    return _EndpointLearnerRow(
        key=(
            row.game_fingerprint,
            row.root_player,
            row.semantic_endpoint_fingerprint,
        ),
        leaf=row.leaf,
        final_root_outcome=row.final_root_outcome,
        deck=deck,
    )


def _collate_engine_teacher_tensors(
    samples: Sequence[_LearnerSample],
    *,
    behavior_actions: Sequence[Sequence[int]],
    options: OptionBatch,
    device: torch.device | str | None,
) -> tuple[
    tuple[tuple[int, ...], ...] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[tuple[tuple[int, ...], ...], ...] | None,
    tuple[torch.Tensor, ...] | None,
]:
    """Collate sparse engine targets without touching behavior evidence."""
    targets = tuple(
        _indexed_engine_teacher_target(sample.indexed) for sample in samples
    )
    if not any(target is not None for target in targets):
        return (None, None, None, None, None, None, None)
    teacher_actions = tuple(
        tuple(int(index) for index in behavior_action)
        if target is None
        else target.action
        for target, behavior_action in zip(targets, behavior_actions, strict=True)
    )
    confidences = torch.tensor(
        [0.0 if target is None else target.confidence for target in targets],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        [0.0 if target is None else target.weight for target in targets],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.tensor(
        [target is not None for target in targets],
        dtype=torch.bool,
        device=device,
    )
    evidence_present = any(
        target is not None and target.search_evidence is not None for target in targets
    )
    candidate_actions = None
    candidate_features = None
    if evidence_present:
        candidate_actions = tuple(
            ()
            if target is None or target.search_evidence is None
            else target.search_evidence.actions
            for target in targets
        )
        candidate_features = tuple(
            torch.empty(
                (0, SEARCH_EVIDENCE_FEATURE_SIZE),
                dtype=torch.float32,
                device=device,
            )
            if target is None or target.search_evidence is None
            else torch.tensor(
                target.search_evidence.feature_rows,
                dtype=torch.float32,
                device=device,
            )
            for target in targets
        )
    return (
        teacher_actions,
        teacher_forced_action_targets(
            teacher_actions,
            max_options=int(options.valid_options.shape[1]),
            device=device,
        ),
        confidences,
        weights,
        mask,
        candidate_actions,
        candidate_features,
    )


def _indexed_engine_teacher_target(
    indexed: _IndexedDecision,
) -> EngineTeacherTarget | None:
    if indexed.record is not None:
        return indexed.record.engine_teacher_target
    block = _require_array_block(indexed)
    return block.engine_teacher_target_at(indexed.array_index)


def _collate_factual_tensors(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Collate dense executed-transition targets for one learner batch."""
    targets = tuple(_indexed_factual_target(sample.indexed) for sample in samples)
    target_presence = tuple(target is not None for target in targets)
    if not any(target_presence):
        return (None, None, None)
    if not all(target_presence):
        raise ValueError("cannot mix learner rows with and without factual targets")
    dense_targets = tuple(cast(FactualTransitionTarget, target) for target in targets)
    return (
        torch.tensor(
            [target.effect_features for target in dense_targets],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [int(target.actor_relation) for target in dense_targets],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [target.next_context for target in dense_targets],
            dtype=torch.long,
            device=device,
        ),
    )


def _indexed_factual_target(
    indexed: _IndexedDecision,
) -> FactualTransitionTarget | None:
    if indexed.record is not None:
        return indexed.record.factual_target
    block = _require_array_block(indexed)
    return block.factual_target_at(indexed.array_index)


def _collate_macro_tensors(
    samples: Sequence[_LearnerSample],
    *,
    device: torch.device | str | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Materialize root targets while deriving summaries from CSR references."""
    presence = tuple(
        sample.indexed.array_block is not None
        and sample.indexed.array_block.executed_macros is not None
        for sample in samples
    )
    if not any(presence):
        return (None, None, None, None)
    if not all(presence):
        raise ValueError("cannot mix schema-10 and legacy learner rows")
    effects = np.zeros(
        (len(samples), DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    endpoints = np.full(len(samples), -1, dtype=np.int64)
    summaries = np.zeros(
        (len(samples), MACRO_CONTINUATION_SUMMARY_SIZE),
        dtype=np.float32,
    )
    masks = np.zeros(len(samples), dtype=np.bool_)
    for batch_index, sample in enumerate(samples):
        indexed = sample.indexed
        block = _require_array_block(indexed)
        macros = block.executed_macros
        if macros is None:
            raise AssertionError("schema-10 row has no executed macro table")
        row = macros.row_for_decision(indexed.array_index)
        if row is None:
            continue
        effects[batch_index] = row.aggregate_effect_features
        endpoints[batch_index] = int(row.endpoint)
        summaries[batch_index] = continuation_summary_from_array_block(
            block,
            row.continuation_decision_positions,
            engine_steps=row.engine_steps,
        )
        masks[batch_index] = True
    return (
        torch.as_tensor(effects, dtype=torch.float32, device=device),
        torch.as_tensor(endpoints, dtype=torch.long, device=device),
        torch.as_tensor(summaries, dtype=torch.float32, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
    )


def _ppo_action_targets(
    actions: Sequence[Sequence[int]],
    options: OptionBatch,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    return teacher_forced_action_targets(
        actions,
        max_options=int(options.valid_options.shape[1]),
        device=device,
    )


def _required_sampling_temperature(indexed: _IndexedDecision) -> float:
    sampling_temperature = indexed.sampling_temperature
    if sampling_temperature is None:
        raise ValueError("learner sample is missing sampling_temperature")
    return sampling_temperature


def _take_action_targets(
    action_targets: torch.Tensor | None,
    actions: Sequence[Sequence[int]],
    *,
    max_options: int,
) -> torch.Tensor | None:
    if action_targets is None:
        return None
    return teacher_forced_action_targets(
        actions,
        max_options=max_options,
        max_steps=_action_target_width(actions),
        device=action_targets.device,
    )


def _action_target_width(actions: Sequence[Sequence[int]]) -> int:
    return max((len(action) for action in actions), default=0) + 1


def _collate_array_states(
    selections: Sequence[_ArrayBlockSelection],
    *,
    device: torch.device | str | None,
) -> StateBatch:
    max_tokens = max(
        int(selection.block.states.card_ids.shape[1]) for selection in selections
    )
    max_attachments = max(
        int(_state_attachment_card_ids(selection.block).shape[1])
        for selection in selections
    )
    return StateBatch(
        card_ids=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.card_ids,
                width=max_tokens,
                fill_value=0,
            ),
            device=device,
        ),
        areas=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.areas,
                width=max_tokens,
                fill_value=0,
            ),
            device=device,
        ),
        owner_roles=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.owner_roles,
                width=max_tokens,
                fill_value=OWNER_UNKNOWN,
            ),
            device=device,
        ),
        token_kinds=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.token_kinds,
                width=max_tokens,
                fill_value=TOKEN_KIND_OOV_INDEX,
            ),
            device=device,
        ),
        scalars=torch.as_tensor(
            _collate_selected_3d(
                selections,
                lambda block: block.states.scalars,
                width=max_tokens,
                fill_value=0.0,
            ),
            device=device,
        ),
        last_attack_ids=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.last_attack_ids,
                width=max_tokens,
                fill_value=0,
            ),
            device=device,
        ),
        padding_mask=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.states.padding_mask,
                width=max_tokens,
                fill_value=True,
            ),
            device=device,
        ),
        attachment_card_ids=torch.as_tensor(
            _collate_selected_2d(
                selections,
                _state_attachment_card_ids,
                width=max_attachments,
                fill_value=0,
            ),
            device=device,
        ),
        attachment_parent_indices=torch.as_tensor(
            _collate_selected_2d(
                selections,
                _state_attachment_parent_indices,
                width=max_attachments,
                fill_value=0,
            ),
            device=device,
        ),
        attachment_kinds=torch.as_tensor(
            _collate_selected_2d(
                selections,
                _state_attachment_kinds,
                width=max_attachments,
                fill_value=0,
            ),
            device=device,
        ),
        entity_slots=torch.as_tensor(
            _collate_selected_2d(
                selections,
                _state_entity_slots,
                width=max_tokens,
                fill_value=0,
            ),
            device=device,
        ),
    )


def _state_attachment_card_ids(block: TrajectoryArrayBlock) -> np.ndarray:
    values = block.states.attachment_card_ids
    if values is not None:
        return values
    return np.zeros((block.decision_count, 1), dtype=np.uint16)


def _state_attachment_parent_indices(block: TrajectoryArrayBlock) -> np.ndarray:
    values = block.states.attachment_parent_indices
    if values is not None:
        return values
    return np.zeros((block.decision_count, 1), dtype=np.uint16)


def _state_attachment_kinds(block: TrajectoryArrayBlock) -> np.ndarray:
    values = block.states.attachment_kinds
    if values is not None:
        return values
    return np.zeros((block.decision_count, 1), dtype=np.uint8)


def _state_entity_slots(block: TrajectoryArrayBlock) -> np.ndarray:
    values = block.states.entity_slots
    if values is not None:
        return values
    return np.zeros_like(block.states.card_ids, dtype=np.uint8)


def _collate_array_options(
    selections: Sequence[_ArrayBlockSelection],
    *,
    device: torch.device | str | None,
) -> OptionBatch:
    max_options = max(
        int(selection.block.options.valid_options.shape[1]) for selection in selections
    )
    return OptionBatch(
        option_types=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.option_types,
                width=max_options,
                fill_value=0,
            ),
            device=device,
        ),
        contexts=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.contexts,
                width=max_options,
                fill_value=0,
            ),
            device=device,
        ),
        entity_slots=torch.as_tensor(
            _collate_selected_3d(
                selections,
                lambda block: block.options.entity_slots,
                width=max_options,
                fill_value=0,
                depth=MAX_ENTITY_SLOTS,
            ),
            device=device,
        ),
        entity_slot_mask=torch.as_tensor(
            _collate_selected_3d(
                selections,
                lambda block: block.options.entity_slot_mask,
                width=max_options,
                fill_value=False,
                depth=MAX_ENTITY_SLOTS,
            ),
            device=device,
        ),
        attack_ids=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.attack_ids,
                width=max_options,
                fill_value=0,
            ),
            device=device,
        ),
        card_ids=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.card_ids,
                width=max_options,
                fill_value=0,
            ),
            device=device,
        ),
        scalars=torch.as_tensor(
            _collate_selected_3d(
                selections,
                lambda block: block.options.scalars,
                width=max_options,
                fill_value=0.0,
            ),
            device=device,
        ),
        dynamic_effect_features=torch.as_tensor(
            _collate_selected_3d(
                selections,
                lambda block: block.options.dynamic_effect_features,
                width=max_options,
                fill_value=0.0,
            ),
            device=device,
        ),
        dynamic_effect_masks=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.dynamic_effect_masks,
                width=max_options,
                fill_value=False,
            ),
            device=device,
        ),
        valid_options=torch.as_tensor(
            _collate_selected_2d(
                selections,
                lambda block: block.options.valid_options,
                width=max_options,
                fill_value=False,
            ),
            device=device,
        ),
        min_counts=torch.as_tensor(
            _concat_selected_1d(
                selections,
                lambda block: block.options.min_counts,
            ),
            dtype=torch.long,
            device=device,
        ),
        max_counts=torch.as_tensor(
            _concat_selected_1d(
                selections,
                lambda block: block.options.max_counts,
            ),
            dtype=torch.long,
            device=device,
        ),
    )


def _array_block_selections(
    indexed: Sequence[_IndexedDecision],
) -> tuple[_ArrayBlockSelection, ...]:
    selections: list[_ArrayBlockSelection] = []
    start = 0
    while start < len(indexed):
        block = _require_array_block(indexed[start])
        source_indices: list[int] = []
        stop = start
        while stop < len(indexed) and _require_array_block(indexed[stop]) is block:
            source_indices.append(indexed[stop].array_index)
            stop += 1
        selections.append(
            _ArrayBlockSelection(
                block=block,
                source_indices=np.asarray(source_indices, dtype=np.intp),
                target_start=start,
                target_stop=stop,
            )
        )
        start = stop
    return tuple(selections)


def _concat_selected_1d(
    selections: Sequence[_ArrayBlockSelection],
    array_getter: Callable[[TrajectoryArrayBlock], np.ndarray],
) -> np.ndarray:
    if not selections:
        raise ValueError("selections must be non-empty")
    return np.concatenate(
        tuple(
            array_getter(selection.block)[selection.source_indices]
            for selection in selections
        ),
        axis=0,
    )


def _collate_selected_2d(
    selections: Sequence[_ArrayBlockSelection],
    array_getter: Callable[[TrajectoryArrayBlock], np.ndarray],
    *,
    width: int,
    fill_value: int | bool | float,
) -> np.ndarray:
    if not selections:
        raise ValueError("selections must be non-empty")
    prototype = array_getter(selections[0].block)
    result = np.full(
        (_selection_count(selections), width),
        fill_value,
        dtype=prototype.dtype,
    )
    for selection in selections:
        selected = array_getter(selection.block)[selection.source_indices]
        result[selection.target_slice, : int(selected.shape[1])] = selected
    return result


def _collate_selected_3d(
    selections: Sequence[_ArrayBlockSelection],
    array_getter: Callable[[TrajectoryArrayBlock], np.ndarray],
    *,
    width: int,
    fill_value: int | bool | float,
    depth: int | None = None,
) -> np.ndarray:
    if not selections:
        raise ValueError("selections must be non-empty")
    prototype = array_getter(selections[0].block)
    active_depth = int(depth if depth is not None else prototype.shape[2])
    result = np.full(
        (_selection_count(selections), width, active_depth),
        fill_value,
        dtype=prototype.dtype,
    )
    for selection in selections:
        selected = array_getter(selection.block)[selection.source_indices]
        result[
            selection.target_slice,
            : int(selected.shape[1]),
            : int(selected.shape[2]),
        ] = selected
    return result


def _selection_count(selections: Sequence[_ArrayBlockSelection]) -> int:
    return sum(
        selection.target_stop - selection.target_start for selection in selections
    )


def _require_record(sample: _LearnerSample) -> DecisionRecord:
    record = sample.indexed.record
    if record is None:
        raise ValueError("expected record-backed learner sample")
    return record


def _require_array_block(indexed: _IndexedDecision) -> TrajectoryArrayBlock:
    block = indexed.array_block
    if block is None:
        raise ValueError("expected array-backed learner sample")
    return block


def _array_action(indexed: _IndexedDecision) -> tuple[int, ...]:
    return _require_array_block(indexed).action_at(indexed.array_index)


def _pad_1d_rows(
    rows: Sequence[np.ndarray],
    *,
    width: int,
    fill_value: int | bool | float,
) -> np.ndarray:
    if not rows:
        raise ValueError("rows must be non-empty")
    result = np.full((len(rows), width), fill_value, dtype=rows[0].dtype)
    for row_index, row in enumerate(rows):
        length = int(row.shape[0])
        result[row_index, :length] = row
    return result


def _pad_2d_rows(
    rows: Sequence[np.ndarray],
    *,
    width: int,
    fill_value: int | bool | float,
    depth: int | None = None,
) -> np.ndarray:
    if not rows:
        raise ValueError("rows must be non-empty")
    active_depth = int(depth if depth is not None else rows[0].shape[1])
    result = np.full(
        (len(rows), width, active_depth),
        fill_value,
        dtype=rows[0].dtype,
    )
    for row_index, row in enumerate(rows):
        length = int(row.shape[0])
        result[row_index, :length, : int(row.shape[1])] = row
    return result


def _take_ppo_batch(
    batch: PpoBatch,
    indices: Sequence[int],
    *,
    device: torch.device | str | None,
    non_blocking: bool = False,
) -> PpoBatch:
    if not indices:
        raise ValueError("indices must be non-empty")
    index_tensor = torch.as_tensor(
        tuple(int(index) for index in indices),
        dtype=torch.long,
        device=batch.old_values.device,
    )
    recurrent = _take_recurrent_context(batch, indices, index_tensor=index_tensor)
    actions = tuple(batch.actions[index] for index in indices)
    engine_teacher_actions = (
        None
        if batch.engine_teacher_actions is None
        else tuple(batch.engine_teacher_actions[index] for index in indices)
    )
    engine_teacher_candidate_actions = (
        None
        if batch.engine_teacher_candidate_actions is None
        else tuple(batch.engine_teacher_candidate_actions[index] for index in indices)
    )
    engine_teacher_candidate_features = (
        None
        if batch.engine_teacher_candidate_features is None
        else tuple(batch.engine_teacher_candidate_features[index] for index in indices)
    )
    token_mask = _take_optional_token_tensor(batch.token_mask, index_tensor)
    token_counts = (
        None
        if token_mask is None
        else tuple(int(value) for value in token_mask.sum(dim=1).tolist())
    )
    token_width = (
        None if token_mask is None else int(token_mask.sum(dim=1).max().item())
    )
    if token_mask is not None and token_width is not None:
        token_mask = token_mask[:, :token_width]
    planner_replay = select_planner_replay(batch.planner_replay, indices)
    root_value_replay = select_root_information_value_replay(
        batch.root_information_value_replay,
        indices,
    )
    engine_teacher_mask = _take_optional_row_tensor(
        batch.engine_teacher_mask,
        index_tensor,
    )
    engine_teacher_indices = (
        None
        if engine_teacher_mask is None
        else tuple(
            int(value)
            for value in torch.nonzero(
                engine_teacher_mask,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
    )
    states = _take_state_batch(batch.states, index_tensor)
    options = _take_option_batch(batch.options, index_tensor)
    return _move_ppo_batch(
        PpoBatch(
            states=states,
            options=options,
            actions=actions,
            decks=(None if batch.decks is None else batch.decks.select(index_tensor)),
            old_action_logprobs=batch.old_action_logprobs.index_select(0, index_tensor),
            sampling_temperatures=batch.sampling_temperatures.index_select(
                0,
                index_tensor,
            ),
            old_values=batch.old_values.index_select(0, index_tensor),
            returns=batch.returns.index_select(0, index_tensor),
            advantages=batch.advantages.index_select(0, index_tensor),
            action_targets=_take_action_targets(
                batch.action_targets,
                actions,
                max_options=int(options.valid_options.shape[1]),
            ),
            sample_indices=(
                None
                if batch.sample_indices is None
                else batch.sample_indices.index_select(
                    0,
                    index_tensor.to(device=batch.sample_indices.device),
                )
            ),
            old_token_logprobs=_take_optional_token_tensor(
                batch.old_token_logprobs,
                index_tensor,
                width=token_width,
            ),
            old_prefix_values=_take_optional_token_tensor(
                batch.old_prefix_values,
                index_tensor,
                width=token_width,
            ),
            token_returns=_take_optional_token_tensor(
                batch.token_returns,
                index_tensor,
                width=token_width,
            ),
            token_advantages=_take_optional_token_tensor(
                batch.token_advantages,
                index_tensor,
                width=token_width,
            ),
            token_mask=token_mask,
            engine_teacher_actions=engine_teacher_actions,
            engine_teacher_action_targets=(
                None
                if engine_teacher_actions is None
                else _take_action_targets(
                    batch.engine_teacher_action_targets,
                    engine_teacher_actions,
                    max_options=int(options.valid_options.shape[1]),
                )
            ),
            engine_teacher_confidences=_take_optional_row_tensor(
                batch.engine_teacher_confidences,
                index_tensor,
            ),
            engine_teacher_weights=_take_optional_row_tensor(
                batch.engine_teacher_weights,
                index_tensor,
            ),
            engine_teacher_mask=engine_teacher_mask,
            engine_teacher_candidate_actions=engine_teacher_candidate_actions,
            engine_teacher_candidate_features=(engine_teacher_candidate_features),
            factual_effect_targets=_take_optional_row_tensor(
                batch.factual_effect_targets,
                index_tensor,
            ),
            factual_actor_relations=_take_optional_row_tensor(
                batch.factual_actor_relations,
                index_tensor,
            ),
            factual_next_contexts=_take_optional_row_tensor(
                batch.factual_next_contexts,
                index_tensor,
            ),
            macro_effect_targets=_take_optional_row_tensor(
                batch.macro_effect_targets,
                index_tensor,
            ),
            macro_endpoints=_take_optional_row_tensor(
                batch.macro_endpoints,
                index_tensor,
            ),
            macro_continuation_summaries=_take_optional_row_tensor(
                batch.macro_continuation_summaries,
                index_tensor,
            ),
            macro_mask=_take_optional_row_tensor(
                batch.macro_mask,
                index_tensor,
            ),
            integrity_validated=batch.integrity_validated,
            count_first_rows_present=_count_first_rows_present(options),
            objective_unit_count=(
                len(actions)
                if token_counts is None
                else sum(token_counts)
                + (0 if planner_replay is None else planner_replay.group_count)
            ),
            engine_teacher_indices=engine_teacher_indices,
            token_counts=token_counts,
            planner_replay=planner_replay,
            root_information_value_replay=root_value_replay,
            public_events=recurrent[0],
            sequence_offsets=recurrent[1],
            sequence_artifacts=recurrent[2],
        ),
        device=device,
        non_blocking=non_blocking,
    )


def _take_recurrent_context(
    batch: PpoBatch,
    indices: Sequence[int],
    *,
    index_tensor: torch.Tensor,
) -> tuple[
    PublicEventBatch | None,
    torch.Tensor | None,
    tuple[PolicyArtifactIdentity, ...] | None,
]:
    """Select complete recurrent sequences and rebuild local offsets."""
    if batch.sequence_offsets is None:
        return (None, None, None)
    events = batch.public_events
    artifacts = batch.sequence_artifacts
    if events is None or artifacts is None:
        raise ValueError("recurrent PPO batch is missing aligned context")
    groups = _recurrent_sequence_rows(batch)
    ordered = _ordered_recurrent_sequence_rows(batch, indices)
    artifact_by_first = {
        group[0]: artifact for group, artifact in zip(groups, artifacts, strict=True)
    }
    selected_artifacts = tuple(artifact_by_first[group[0]] for group in ordered)
    offsets = [0]
    for group in ordered:
        offsets.append(offsets[-1] + len(group))
    return (
        select_public_event_batch_rows(
            events,
            index_tensor.to(device=events.event_types.device),
        ),
        torch.tensor(
            offsets,
            dtype=torch.long,
            device=batch.sequence_offsets.device,
        ),
        selected_artifacts,
    )


def _take_optional_token_tensor(
    tensor: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    width: int | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    selected = tensor.index_select(0, indices.to(device=tensor.device))
    if width is None:
        return selected
    return selected[:, :width]


def _take_optional_row_tensor(
    tensor: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.index_select(0, indices.to(device=tensor.device))


def _take_state_batch(states: StateBatch, indices: torch.Tensor) -> StateBatch:
    padding_mask = states.padding_mask.index_select(0, indices)
    token_width = _active_prefix_width(~padding_mask)
    return StateBatch(
        card_ids=states.card_ids.index_select(0, indices)[:, :token_width],
        areas=states.areas.index_select(0, indices)[:, :token_width],
        owner_roles=states.owner_roles.index_select(0, indices)[:, :token_width],
        token_kinds=states.token_kinds.index_select(0, indices)[:, :token_width],
        scalars=states.scalars.index_select(0, indices)[:, :token_width],
        last_attack_ids=states.last_attack_ids.index_select(0, indices)[
            :, :token_width
        ],
        padding_mask=padding_mask[:, :token_width],
        attachment_card_ids=_optional_index_select(
            states.attachment_card_ids,
            indices,
        ),
        attachment_parent_indices=_optional_index_select(
            states.attachment_parent_indices,
            indices,
        ),
        attachment_kinds=_optional_index_select(states.attachment_kinds, indices),
        entity_slots=_optional_index_select_width(
            states.entity_slots,
            indices,
            width=token_width,
        ),
    )


def _optional_index_select(
    tensor: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.index_select(0, indices)


def _optional_index_select_width(
    tensor: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor | None:
    """Select rows and trim a token- or option-aligned optional tensor."""
    if tensor is None:
        return None
    return tensor.index_select(0, indices)[:, :width]


def _take_option_batch(options: OptionBatch, indices: torch.Tensor) -> OptionBatch:
    valid_options = options.valid_options.index_select(0, indices)
    option_width = _active_prefix_width(valid_options)
    return OptionBatch(
        option_types=options.option_types.index_select(0, indices)[:, :option_width],
        contexts=options.contexts.index_select(0, indices)[:, :option_width],
        entity_slots=options.entity_slots.index_select(0, indices)[:, :option_width],
        entity_slot_mask=options.entity_slot_mask.index_select(0, indices)[
            :, :option_width
        ],
        attack_ids=options.attack_ids.index_select(0, indices)[:, :option_width],
        card_ids=options.card_ids.index_select(0, indices)[:, :option_width],
        scalars=options.scalars.index_select(0, indices)[:, :option_width],
        dynamic_effect_features=options.dynamic_effect_features.index_select(
            0,
            indices,
        )[:, :option_width],
        dynamic_effect_masks=options.dynamic_effect_masks.index_select(0, indices)[
            :, :option_width
        ],
        valid_options=valid_options[:, :option_width],
        min_counts=options.min_counts.index_select(0, indices),
        max_counts=options.max_counts.index_select(0, indices),
    )


def _active_prefix_width(valid: torch.Tensor) -> int:
    """Return the last batch column containing a non-padding value."""
    if valid.ndim != 2:
        raise ValueError("prefix validity mask must be two-dimensional")
    active_columns = torch.nonzero(valid.any(dim=0), as_tuple=False)
    if active_columns.numel() == 0:
        return 0
    return int(active_columns[-1, 0].item()) + 1


def _count_first_rows_present(options: OptionBatch) -> bool:
    """Resolve the static teacher-forced branch before the batch reaches CUDA."""
    valid_options = options.valid_options.detach().to(device="cpu")
    contexts = options.contexts.detach().to(device="cpu")
    rows = torch.zeros_like(valid_options.any(dim=1))
    for context in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS:
        rows |= (valid_options & contexts.eq(int(context))).any(dim=1)
    variable_count = (
        options.min_counts.detach()
        .to(device="cpu")
        .lt(options.max_counts.detach().to(device="cpu"))
    )
    return bool((rows & variable_count).any())


def _pin_ppo_batch(batch: PpoBatch) -> PpoBatch:
    if not torch.cuda.is_available():
        return batch
    return PpoBatch(
        states=_pin_state_batch(batch.states),
        options=_pin_option_batch(batch.options),
        actions=batch.actions,
        decks=None if batch.decks is None else batch.decks.pin_memory(),
        old_action_logprobs=_pin_tensor(batch.old_action_logprobs),
        sampling_temperatures=_pin_tensor(batch.sampling_temperatures),
        old_values=_pin_tensor(batch.old_values),
        returns=_pin_tensor(batch.returns),
        advantages=_pin_tensor(batch.advantages),
        action_targets=(
            None if batch.action_targets is None else _pin_tensor(batch.action_targets)
        ),
        sample_indices=batch.sample_indices,
        old_token_logprobs=_optional_pin_tensor(batch.old_token_logprobs),
        old_prefix_values=_optional_pin_tensor(batch.old_prefix_values),
        token_returns=_optional_pin_tensor(batch.token_returns),
        token_advantages=_optional_pin_tensor(batch.token_advantages),
        token_mask=_optional_pin_tensor(batch.token_mask),
        engine_teacher_actions=batch.engine_teacher_actions,
        engine_teacher_action_targets=_optional_pin_tensor(
            batch.engine_teacher_action_targets
        ),
        engine_teacher_confidences=_optional_pin_tensor(
            batch.engine_teacher_confidences
        ),
        engine_teacher_weights=_optional_pin_tensor(batch.engine_teacher_weights),
        engine_teacher_mask=_optional_pin_tensor(batch.engine_teacher_mask),
        engine_teacher_candidate_actions=batch.engine_teacher_candidate_actions,
        engine_teacher_candidate_features=_optional_pin_tensor_sequence(
            batch.engine_teacher_candidate_features
        ),
        factual_effect_targets=_optional_pin_tensor(batch.factual_effect_targets),
        factual_actor_relations=_optional_pin_tensor(batch.factual_actor_relations),
        factual_next_contexts=_optional_pin_tensor(batch.factual_next_contexts),
        macro_effect_targets=_optional_pin_tensor(batch.macro_effect_targets),
        macro_endpoints=_optional_pin_tensor(batch.macro_endpoints),
        macro_continuation_summaries=_optional_pin_tensor(
            batch.macro_continuation_summaries
        ),
        macro_mask=_optional_pin_tensor(batch.macro_mask),
        integrity_validated=batch.integrity_validated,
        count_first_rows_present=batch.count_first_rows_present,
        objective_unit_count=batch.objective_unit_count,
        engine_teacher_indices=batch.engine_teacher_indices,
        token_counts=batch.token_counts,
        planner_replay=pin_planner_replay(batch.planner_replay),
        root_information_value_replay=pin_root_information_value_replay(
            batch.root_information_value_replay
        ),
        public_events=_pin_public_event_batch(batch.public_events),
        sequence_offsets=_optional_pin_tensor(batch.sequence_offsets),
        sequence_artifacts=batch.sequence_artifacts,
    )


def _pin_state_batch(states: StateBatch) -> StateBatch:
    return StateBatch(
        card_ids=_pin_tensor(states.card_ids),
        areas=_pin_tensor(states.areas),
        owner_roles=_pin_tensor(states.owner_roles),
        token_kinds=_pin_tensor(states.token_kinds),
        scalars=_pin_tensor(states.scalars),
        last_attack_ids=_pin_tensor(states.last_attack_ids),
        padding_mask=_pin_tensor(states.padding_mask),
        attachment_card_ids=_optional_pin_tensor(states.attachment_card_ids),
        attachment_parent_indices=_optional_pin_tensor(
            states.attachment_parent_indices
        ),
        attachment_kinds=_optional_pin_tensor(states.attachment_kinds),
        entity_slots=_optional_pin_tensor(states.entity_slots),
    )


def _optional_pin_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return _pin_tensor(tensor)


def _pin_public_event_batch(
    batch: PublicEventBatch | None,
) -> PublicEventBatch | None:
    if batch is None:
        return None
    return _map_public_event_batch(batch, _pin_tensor)


def _optional_pin_tensor_sequence(
    tensors: Sequence[torch.Tensor] | None,
) -> tuple[torch.Tensor, ...] | None:
    """Pin every tensor in an optional ragged row-aligned sequence."""
    if tensors is None:
        return None
    return tuple(_pin_tensor(tensor) for tensor in tensors)


def _pin_option_batch(options: OptionBatch) -> OptionBatch:
    return OptionBatch(
        option_types=_pin_tensor(options.option_types),
        contexts=_pin_tensor(options.contexts),
        entity_slots=_pin_tensor(options.entity_slots),
        entity_slot_mask=_pin_tensor(options.entity_slot_mask),
        attack_ids=_pin_tensor(options.attack_ids),
        card_ids=_pin_tensor(options.card_ids),
        scalars=_pin_tensor(options.scalars),
        dynamic_effect_features=_pin_tensor(options.dynamic_effect_features),
        dynamic_effect_masks=_pin_tensor(options.dynamic_effect_masks),
        valid_options=_pin_tensor(options.valid_options),
        min_counts=_pin_tensor(options.min_counts),
        max_counts=_pin_tensor(options.max_counts),
    )


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


def _iter_device_batches(
    batches: Sequence[PpoBatch],
    *,
    device: torch.device | str | None,
    pin_memory: bool,
    non_blocking: bool,
    copy_stream: bool,
) -> Iterator[PpoBatch]:
    if device is None:
        yield from batches
        return
    device_obj = torch.device(device)
    if not _should_prefetch_batches(device_obj, non_blocking, copy_stream):
        for batch in batches:
            transfer_batch = _stage_ppo_batch_for_transfer(
                batch,
                device=device_obj,
                pin_memory=pin_memory,
            )
            yield _move_ppo_batch(
                transfer_batch,
                device=device_obj,
                non_blocking=non_blocking,
            )
        return
    yield from _iter_prefetched_device_batches(
        batches,
        device=device_obj,
        pin_memory=pin_memory,
    )


def _should_prefetch_batches(
    device: torch.device,
    non_blocking: bool,
    copy_stream: bool,
) -> bool:
    return bool(
        non_blocking
        and copy_stream
        and device.type == "cuda"
        and torch.cuda.is_available()
    )


def _iter_prefetched_device_batches(
    batches: Sequence[PpoBatch],
    *,
    device: torch.device,
    pin_memory: bool,
) -> Iterator[PpoBatch]:
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return
    stream = cast(Any, torch.cuda.Stream)(device=device)
    first = _stage_ppo_batch_for_transfer(
        first,
        device=device,
        pin_memory=pin_memory,
    )
    with torch.cuda.stream(stream):
        next_batch = _move_ppo_batch(first, device=device, non_blocking=True)
    for cpu_batch in iterator:
        torch.cuda.current_stream(device).wait_stream(stream)
        current_batch = next_batch
        cpu_batch = _stage_ppo_batch_for_transfer(
            cpu_batch,
            device=device,
            pin_memory=pin_memory,
        )
        with torch.cuda.stream(stream):
            next_batch = _move_ppo_batch(cpu_batch, device=device, non_blocking=True)
        yield current_batch
    torch.cuda.current_stream(device).wait_stream(stream)
    yield next_batch


def _stage_ppo_batch_for_transfer(
    batch: PpoBatch,
    *,
    device: torch.device,
    pin_memory: bool,
) -> PpoBatch:
    """Pin only the bounded transfer frontier instead of a complete epoch."""
    if not pin_memory or device.type != "cuda":
        return batch
    return _pin_ppo_batch(batch)


def _move_ppo_batch(
    batch: PpoBatch,
    *,
    device: torch.device | str | None,
    non_blocking: bool = False,
) -> PpoBatch:
    if device is None:
        return batch
    return PpoBatch(
        states=_move_state_batch(
            batch.states,
            device=device,
            non_blocking=non_blocking,
        ),
        options=_move_option_batch(
            batch.options,
            device=device,
            non_blocking=non_blocking,
        ),
        actions=batch.actions,
        decks=(
            None
            if batch.decks is None
            else batch.decks.to(device, non_blocking=non_blocking)
        ),
        old_action_logprobs=batch.old_action_logprobs.to(
            device=device,
            non_blocking=non_blocking,
        ),
        sampling_temperatures=batch.sampling_temperatures.to(
            device=device,
            non_blocking=non_blocking,
        ),
        old_values=batch.old_values.to(device=device, non_blocking=non_blocking),
        returns=batch.returns.to(device=device, non_blocking=non_blocking),
        advantages=batch.advantages.to(device=device, non_blocking=non_blocking),
        action_targets=(
            None
            if batch.action_targets is None
            else batch.action_targets.to(device=device, non_blocking=non_blocking)
        ),
        sample_indices=batch.sample_indices,
        old_token_logprobs=_optional_move_tensor(
            batch.old_token_logprobs,
            device=device,
            non_blocking=non_blocking,
        ),
        old_prefix_values=_optional_move_tensor(
            batch.old_prefix_values,
            device=device,
            non_blocking=non_blocking,
        ),
        token_returns=_optional_move_tensor(
            batch.token_returns,
            device=device,
            non_blocking=non_blocking,
        ),
        token_advantages=_optional_move_tensor(
            batch.token_advantages,
            device=device,
            non_blocking=non_blocking,
        ),
        token_mask=_optional_move_tensor(
            batch.token_mask,
            device=device,
            non_blocking=non_blocking,
        ),
        engine_teacher_actions=batch.engine_teacher_actions,
        engine_teacher_action_targets=_optional_move_tensor(
            batch.engine_teacher_action_targets,
            device=device,
            non_blocking=non_blocking,
        ),
        engine_teacher_confidences=_optional_move_tensor(
            batch.engine_teacher_confidences,
            device=device,
            non_blocking=non_blocking,
        ),
        engine_teacher_weights=_optional_move_tensor(
            batch.engine_teacher_weights,
            device=device,
            non_blocking=non_blocking,
        ),
        engine_teacher_mask=_optional_move_tensor(
            batch.engine_teacher_mask,
            device=device,
            non_blocking=non_blocking,
        ),
        engine_teacher_candidate_actions=batch.engine_teacher_candidate_actions,
        engine_teacher_candidate_features=_optional_move_tensor_sequence(
            batch.engine_teacher_candidate_features,
            device=device,
            non_blocking=non_blocking,
        ),
        factual_effect_targets=_optional_move_tensor(
            batch.factual_effect_targets,
            device=device,
            non_blocking=non_blocking,
        ),
        factual_actor_relations=_optional_move_tensor(
            batch.factual_actor_relations,
            device=device,
            non_blocking=non_blocking,
        ),
        factual_next_contexts=_optional_move_tensor(
            batch.factual_next_contexts,
            device=device,
            non_blocking=non_blocking,
        ),
        macro_effect_targets=_optional_move_tensor(
            batch.macro_effect_targets,
            device=device,
            non_blocking=non_blocking,
        ),
        macro_endpoints=_optional_move_tensor(
            batch.macro_endpoints,
            device=device,
            non_blocking=non_blocking,
        ),
        macro_continuation_summaries=_optional_move_tensor(
            batch.macro_continuation_summaries,
            device=device,
            non_blocking=non_blocking,
        ),
        macro_mask=_optional_move_tensor(
            batch.macro_mask,
            device=device,
            non_blocking=non_blocking,
        ),
        integrity_validated=batch.integrity_validated,
        count_first_rows_present=batch.count_first_rows_present,
        objective_unit_count=batch.objective_unit_count,
        engine_teacher_indices=batch.engine_teacher_indices,
        token_counts=batch.token_counts,
        planner_replay=move_planner_replay(
            batch.planner_replay,
            device=device,
            non_blocking=non_blocking,
        ),
        root_information_value_replay=move_root_information_value_replay(
            batch.root_information_value_replay,
            device=device,
            non_blocking=non_blocking,
        ),
        public_events=_move_public_event_batch(
            batch.public_events,
            device=device,
            non_blocking=non_blocking,
        ),
        sequence_offsets=_optional_move_tensor(
            batch.sequence_offsets,
            device=device,
            non_blocking=non_blocking,
        ),
        sequence_artifacts=batch.sequence_artifacts,
    )


def _move_public_event_batch(
    batch: PublicEventBatch | None,
    *,
    device: torch.device | str,
    non_blocking: bool,
) -> PublicEventBatch | None:
    if batch is None:
        return None
    return _map_public_event_batch(
        batch,
        lambda tensor: tensor.to(device=device, non_blocking=non_blocking),
    )


def _map_public_event_batch(
    batch: PublicEventBatch,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> PublicEventBatch:
    """Apply one storage/device transform to every event tensor."""
    return PublicEventBatch(
        event_types=transform(batch.event_types),
        actor_roles=transform(batch.actor_roles),
        from_areas=transform(batch.from_areas),
        to_areas=transform(batch.to_areas),
        card_ids=transform(batch.card_ids),
        serials=transform(batch.serials),
        entity_mask=transform(batch.entity_mask),
        attack_ids=transform(batch.attack_ids),
        attack_id_mask=transform(batch.attack_id_mask),
        values=transform(batch.values),
        value_mask=transform(batch.value_mask),
        categorical_values=transform(batch.categorical_values),
        padding_mask=transform(batch.padding_mask),
        overflow_type_actor_counts=transform(batch.overflow_type_actor_counts),
    )


def _optional_move_tensor_sequence(
    tensors: Sequence[torch.Tensor] | None,
    *,
    device: torch.device | str,
    non_blocking: bool,
) -> tuple[torch.Tensor, ...] | None:
    """Move every tensor in an optional ragged row-aligned sequence."""
    if tensors is None:
        return None
    return tuple(
        tensor.to(device=device, non_blocking=non_blocking) for tensor in tensors
    )


def _move_state_batch(
    states: StateBatch,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
) -> StateBatch:
    return StateBatch(
        card_ids=states.card_ids.to(device=device, non_blocking=non_blocking),
        areas=states.areas.to(device=device, non_blocking=non_blocking),
        owner_roles=states.owner_roles.to(device=device, non_blocking=non_blocking),
        token_kinds=states.token_kinds.to(device=device, non_blocking=non_blocking),
        scalars=states.scalars.to(device=device, non_blocking=non_blocking),
        last_attack_ids=states.last_attack_ids.to(
            device=device,
            non_blocking=non_blocking,
        ),
        padding_mask=states.padding_mask.to(device=device, non_blocking=non_blocking),
        attachment_card_ids=_optional_move_tensor(
            states.attachment_card_ids,
            device=device,
            non_blocking=non_blocking,
        ),
        attachment_parent_indices=_optional_move_tensor(
            states.attachment_parent_indices,
            device=device,
            non_blocking=non_blocking,
        ),
        attachment_kinds=_optional_move_tensor(
            states.attachment_kinds,
            device=device,
            non_blocking=non_blocking,
        ),
        entity_slots=_optional_move_tensor(
            states.entity_slots,
            device=device,
            non_blocking=non_blocking,
        ),
    )


def _optional_move_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device | str,
    non_blocking: bool,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device, non_blocking=non_blocking)


def _move_option_batch(
    options: OptionBatch,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
) -> OptionBatch:
    return OptionBatch(
        option_types=options.option_types.to(device=device, non_blocking=non_blocking),
        contexts=options.contexts.to(device=device, non_blocking=non_blocking),
        entity_slots=options.entity_slots.to(device=device, non_blocking=non_blocking),
        entity_slot_mask=options.entity_slot_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        attack_ids=options.attack_ids.to(device=device, non_blocking=non_blocking),
        card_ids=options.card_ids.to(device=device, non_blocking=non_blocking),
        scalars=options.scalars.to(device=device, non_blocking=non_blocking),
        dynamic_effect_features=options.dynamic_effect_features.to(
            device=device,
            non_blocking=non_blocking,
        ),
        dynamic_effect_masks=options.dynamic_effect_masks.to(
            device=device,
            non_blocking=non_blocking,
        ),
        valid_options=options.valid_options.to(
            device=device, non_blocking=non_blocking
        ),
        min_counts=options.min_counts.to(device=device, non_blocking=non_blocking),
        max_counts=options.max_counts.to(device=device, non_blocking=non_blocking),
    )
