"""Configured rollout trajectory collection."""

from __future__ import annotations

import hashlib
import math
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from ptcg_rl.agent.search.proposal_generation import (
    PLANNER_PROPOSAL_ARCHITECTURE_VERSION,
    PlannerProposalBatchResult,
    PlannerProposalSearchLimits,
)
from ptcg_rl.context import PublicEventBatch
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.engine.vector_battle import DeckPair, DeckPairSampler, VectorBattlePool
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    AgentPolicyValueNet,
    ConditionedStateOutput,
    OptionBatch,
    PlannerCandidateEvaluation,
    PolicyEvaluationContext,
    RecurrentPolicyState,
    StateBatch,
    build_agent_policy_value_net,
)
from ptcg_rl.model.network import SampleDecodeTensorTrace, SampleDecodeTrace
from ptcg_rl.opponents import (
    OpponentSpec,
    build_opponent,
    opponent_registry,
)
from ptcg_rl.rl.graph_decode import CudaGraphDecodeRunner
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.planner_context_cache import (
    PlannerContextCapacityError,
    PlannerPolicyContextCache,
)
from ptcg_rl.rl.recurrent_runtime import PolicyArtifactIdentity
from ptcg_rl.rl.rollout import (
    RolloutActors,
    RolloutMode,
    RolloutPolicy,
    RolloutRecorder,
    RolloutStepper,
    RolloutStepStats,
    VectorPoolLike,
)
from ptcg_rl.runtime.bounded_store import BoundedStoreStats
from ptcg_rl.training.run_config import TrainingRunConfig

RolloutPolicyKind = Literal["model", "min_count"]
AutocastMode = Literal["bf16", "off"]


class RolloutOpponentConfig(BaseModel):
    """Hydra-backed opponent selection for trajectory rollout."""

    model_config = ConfigDict(extra="forbid")

    mode: RolloutMode = "self_play"
    frozen_checkpoint_path: Path | None = None
    scripted_name: str | None = None

    @model_validator(mode="after")
    def valid_mode_fields(self) -> RolloutOpponentConfig:
        """Reject missing mode-specific opponent settings."""
        if self.mode == "frozen" and self.frozen_checkpoint_path is None:
            raise ValueError("frozen rollout mode requires frozen_checkpoint_path")
        if self.mode == "scripted" and not self.scripted_name:
            raise ValueError("scripted rollout mode requires scripted_name")
        return self


class RolloutDeckConfig(BaseModel):
    """Deck paths used by local vector battle rollout."""

    model_config = ConfigDict(extra="forbid")

    candidate: Path = Path("data/sample_submission/deck.csv")
    opponent: Path | None = None


class RolloutConfig(BaseModel):
    """Hydra-backed config for writing RL rollout trajectory shards."""

    model_config = ConfigDict(extra="forbid")

    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    checkpoint_path: Path | None = None
    policy_kind: RolloutPolicyKind = "model"
    policy_version: str | None = None
    device: str = "auto"
    autocast: AutocastMode = "bf16"
    num_concurrent_games: int = 256
    total_games: int | None = 2_000
    total_steps: int | None = None
    max_iterations: int = 100_000
    sampling_temperature: float = 1.0
    opponent: RolloutOpponentConfig = Field(default_factory=RolloutOpponentConfig)
    decks: RolloutDeckConfig = Field(default_factory=RolloutDeckConfig)
    record_forced_actions: bool = False
    rows_per_shard: int = 65_536
    compression: str = "zstd"
    seed: int = 0
    parallel_workers: int = 1
    parallel_worker_output_subdir: str = "workers"
    model: AgentNetworkConfig = Field(default_factory=AgentNetworkConfig)

    @field_validator(
        "num_concurrent_games",
        "max_iterations",
        "rows_per_shard",
        "parallel_workers",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive rollout limits."""
        if value <= 0:
            raise ValueError("rollout limits must be positive")
        return value

    @field_validator("total_games", "total_steps")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional rollout budgets."""
        if value is not None and value <= 0:
            raise ValueError("rollout budgets must be positive when set")
        return value

    @field_validator("sampling_temperature")
    @classmethod
    def valid_non_negative_float(cls, value: float) -> float:
        """Reject invalid sampling temperatures."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("sampling_temperature must be finite and non-negative")
        return value

    @field_validator("device", "compression")
    @classmethod
    def valid_non_empty_string(cls, value: str) -> str:
        """Reject empty string settings."""
        if not value.strip():
            raise ValueError("string rollout settings must be non-empty")
        return value

    @field_validator("parallel_worker_output_subdir")
    @classmethod
    def valid_parallel_worker_subdir(cls, value: str) -> str:
        """Reject empty or path-like parallel rollout labels."""
        cleaned = value.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("parallel_worker_output_subdir must be a path segment")
        return cleaned

    @model_validator(mode="after")
    def valid_budget(self) -> RolloutConfig:
        """Require at least one rollout stopping condition."""
        if self.total_games is None and self.total_steps is None:
            raise ValueError("total_games or total_steps must be set")
        return self


RolloutPoolFactory = Callable[
    [int, DeckPairSampler],
    AbstractContextManager[VectorPoolLike],
]
RolloutPolicyFactory = Callable[
    [RolloutPolicyKind, Path | None, torch.device, AgentNetworkConfig],
    RolloutPolicy,
]


class TrajectoryRecorderLike(RolloutRecorder, Protocol):
    """Recorder interface needed by the rollout runner."""

    @property
    def counters(self) -> Counter[str]:
        """Return recorder counters."""

    def pop_completed(self) -> Sequence[Any]:
        """Return completed trajectories ready for writing."""


class TrajectoryWriterLike(Protocol):
    """Writer interface needed by the rollout runner."""

    def add_completed(self, trajectories: Any) -> None:
        """Write completed trajectories."""

    def close(self) -> Any:
        """Flush output and return a write result."""


class MinCountRolloutPolicy:
    """Policy that selects the first ``minCount`` options for rollout smoke."""

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Return legal deterministic actions, zero log-probs, and zero values."""
        trace = self.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> SampleDecodeTrace:
        """Return deterministic actions with an exact zero-probability trace."""
        del temperature
        _validate_policy_decks(states, options, decks)
        actions = tuple(
            tuple(range(int(options.min_counts[row].item())))
            for row in range(int(options.valid_options.shape[0]))
        )
        batch_size = len(actions)
        device = options.valid_options.device
        values = torch.zeros(batch_size, dtype=torch.float32, device=device)
        stop_sampled = torch.tensor(
            [
                len(action) < int(options.max_counts[index].item())
                for index, action in enumerate(actions)
            ],
            dtype=torch.bool,
            device=device,
        )
        active_lengths = [
            len(action) + int(stop_sampled[index].item())
            for index, action in enumerate(actions)
        ]
        token_width = max(active_lengths, default=0)
        token_mask = torch.zeros(
            (batch_size, token_width),
            dtype=torch.bool,
            device=device,
        )
        for index, active_length in enumerate(active_lengths):
            token_mask[index, :active_length] = True
        return SampleDecodeTrace(
            actions=actions,
            action_logprobs=torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=device,
            ),
            values=values,
            token_logprobs=torch.zeros(
                (batch_size, token_width),
                dtype=torch.float32,
                device=device,
            ),
            prefix_values=values.unsqueeze(1).expand(-1, token_width).clone(),
            token_mask=token_mask,
            stop_sampled=stop_sampled,
        )


class ModelRolloutPolicy:
    """Thin rollout policy wrapper around ``AgentPolicyValueNet``."""

    def __init__(
        self,
        model: AgentPolicyValueNet,
        *,
        policy_version: int = 0,
        autocast: AutocastMode = "off",
        graph_decode: bool = False,
        graph_warmup_steps: int = 2,
        graph_max_captures: int = 0,
        graph_capture_idle_replays: int = 64,
        graph_capture_allowed: Callable[[], bool] | None = None,
        graph_replay_allowed: Callable[[], bool] | None = None,
        planner_context_capacity: int = 0,
        model_fingerprint: str | None = None,
        verified_model_fingerprint: str | None = None,
        proposal_version: int = PLANNER_PROPOSAL_ARCHITECTURE_VERSION,
        generator: torch.Generator | None = None,
    ) -> None:
        """Store an eval-mode model."""
        self._model = model
        self.behavior_kind = "policy_sample"
        self.policy_version = int(policy_version)
        if model_fingerprint is not None and verified_model_fingerprint is not None:
            raise ValueError("model fingerprint trust paths are mutually exclusive")
        if verified_model_fingerprint is not None:
            if len(verified_model_fingerprint) != 64 or any(
                character not in "0123456789abcdef"
                for character in verified_model_fingerprint
            ):
                raise ValueError("verified model fingerprint must be SHA-256")
            # Trust boundary: the publication registry already canonical-hashed
            # the complete CPU state and strictly loaded these exact tensors.
            # Rehashing the fresh accelerator copy would force another full
            # device-to-host transfer for every published snapshot.
            self.model_fingerprint = verified_model_fingerprint
        else:
            computed_fingerprint = canonical_model_state_fingerprint(model)
            if (
                model_fingerprint is not None
                and model_fingerprint != computed_fingerprint
            ):
                raise ValueError(
                    "supplied model fingerprint differs from full model state"
                )
            self.model_fingerprint = computed_fingerprint
        self._policy_artifact_identity = PolicyArtifactIdentity.from_model_config(
            model.config,
            model_fingerprint=self.model_fingerprint,
        )
        self.proposal_version = int(proposal_version)
        if self.proposal_version < 0:
            raise ValueError("planner proposal version must be non-negative")
        self._autocast = autocast
        self._generator = generator
        self._graph_decode = (
            CudaGraphDecodeRunner(
                model,
                warmup_steps=graph_warmup_steps,
                max_captures=graph_max_captures,
                capture_idle_replays=graph_capture_idle_replays,
                capture_allowed=graph_capture_allowed,
                replay_allowed=graph_replay_allowed,
                generator=generator,
            )
            if graph_decode
            else None
        )
        if planner_context_capacity < 0:
            raise ValueError("planner_context_capacity must be non-negative")
        self._planner_context_cache = (
            None
            if planner_context_capacity == 0
            else PlannerPolicyContextCache(planner_context_capacity)
        )
        self.planner_inference_device_type = next(model.parameters()).device.type
        if self.planner_inference_device_type not in ("cpu", "cuda"):
            raise ValueError("planner inference requires a CPU or CUDA model")

    @property
    def captured_graph_buckets(self) -> int:
        """Return the number of captured graph-decode buckets."""
        if self._graph_decode is None:
            return 0
        return self._graph_decode.captured_buckets

    @property
    def recurrent_enabled(self) -> bool:
        """Return whether this model requires explicit actor-side state."""
        return self._model.config.recurrent is not None

    @property
    def policy_artifact_fingerprint(self) -> str:
        """Return the immutable weights-and-input-contract identity."""
        return self._policy_artifact_identity.fingerprint

    @property
    def policy_artifact_identity(self) -> PolicyArtifactIdentity:
        """Return the full immutable recurrent policy contract."""
        return self._policy_artifact_identity

    def initial_recurrent_state(self, batch_size: int) -> RecurrentPolicyState:
        """Create an explicit zero state at a game-seat registration boundary."""
        if not self.recurrent_enabled:
            raise RuntimeError("rollout policy recurrent state is disabled")
        device = _module_device(self._model)
        dtype = _module_dtype(self._model)
        return self._model.initial_recurrent_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    @property
    def graph_decode_enabled(self) -> bool:
        """Return whether this policy can consume padded static graph batches."""
        return self._graph_decode is not None

    def graph_decode_stats(self) -> Mapping[str, int]:
        """Expose graph capture/replay activity without reaching into internals."""
        if self._graph_decode is None:
            return {}
        return self._graph_decode.stats().as_dict()

    @property
    def planner_context_enabled(self) -> bool:
        """Return whether decode must retain root contexts for planning."""
        return self._planner_context_cache is not None

    def predict_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
    ) -> Tensor:
        """Predict root values without running the autoregressive decoder."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            return self._model.predict_values(states, decks)

    def predict_root_information_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
        *,
        actor_relations: Tensor,
        endpoints: Tensor,
        belief_summaries: Tensor,
    ) -> Tensor:
        """Batch semantic endpoint values in the immutable root perspective."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            conditioned = self._model.encode_conditioned_state(states, decks)
            return self._model.root_information_values_from_conditioned(
                conditioned,
                actor_relations=actor_relations,
                endpoints=endpoints,
                belief_summaries=belief_summaries,
            )

    def evaluate_planner_candidates(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        ordered_rows: Tensor | None = None,
        decks: DeckBatch,
        planner_context_handles: Sequence[str] | None = None,
        model_version_lease: int | None = None,
        tensor_schema_fingerprint: str = "",
    ) -> PlannerCandidateEvaluation:
        """Evaluate schema-9 candidates through the isolated planner head."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            if planner_context_handles is None:
                return self._model.evaluate_planner_candidates(
                    states,
                    options,
                    candidate_actions,
                    candidate_features,
                    ordered_rows=ordered_rows,
                    decks=decks,
                )
            context = self._acquire_planner_context(
                planner_context_handles,
                states,
                options,
                decks,
                model_version_lease=model_version_lease,
                tensor_schema_fingerprint=tensor_schema_fingerprint,
            )
            return self._model.evaluate_planner_candidates_from_context(
                context,
                options,
                candidate_actions,
                candidate_features,
                ordered_rows=ordered_rows,
                decks=decks,
            )

    def generate_planner_proposals(
        self,
        states: StateBatch,
        options: OptionBatch,
        *,
        ordered_rows: Tensor,
        limits: PlannerProposalSearchLimits,
        decks: DeckBatch,
        planner_context_handles: Sequence[str] | None = None,
        model_version_lease: int | None = None,
        tensor_schema_fingerprint: str = "",
        deadline_monotonic: float | None = None,
    ) -> PlannerProposalBatchResult:
        """Generate bounded learned proposals through one leased model pass."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            if planner_context_handles is None:
                return self._model.generate_planner_proposals(
                    states,
                    options,
                    ordered_rows=ordered_rows,
                    limits=limits,
                    decks=decks,
                    deadline_monotonic=deadline_monotonic,
                )
            context = self._acquire_planner_context(
                planner_context_handles,
                states,
                options,
                decks,
                model_version_lease=model_version_lease,
                tensor_schema_fingerprint=tensor_schema_fingerprint,
            )
            return self._model.generate_planner_proposals_from_context(
                context,
                options,
                ordered_rows=ordered_rows,
                limits=limits,
                decks=decks,
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
        """Bind newly issued decode handles to the server request schema."""
        del deadline_monotonic
        cache = self._required_planner_context_cache()
        cache.bind_schema(
            handles,
            policy_version=policy_version,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
        )

    def release_planner_context_handles(self, handles: Sequence[str]) -> int:
        """Release root contexts after planner completion or fast fallback."""
        return self._required_planner_context_cache().release(handles)

    def reset_planner_contexts(
        self,
        *,
        policy_version: int,
        model_fingerprint: str | None = None,
    ) -> None:
        """Publish identity only after every prior planner context drained."""
        cache = self._planner_context_cache
        if cache is not None:
            if cache.stats().size:
                raise RuntimeError(
                    "cannot publish model identity with live planner contexts"
                )
            cache.clear()
        fingerprint = (
            canonical_model_state_fingerprint(self._model)
            if model_fingerprint is None
            else model_fingerprint
        )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("published model fingerprint must be SHA-256")
        self.model_fingerprint = fingerprint
        self.policy_version = int(policy_version)
        self._policy_artifact_identity = PolicyArtifactIdentity.from_model_config(
            self._model.config,
            model_fingerprint=fingerprint,
        )

    def planner_context_cache_stats(self) -> BoundedStoreStats | None:
        """Expose bounded context reuse telemetry to runtime profiling."""
        cache = self._planner_context_cache
        return None if cache is None else cache.stats()

    def _acquire_planner_context(
        self,
        handles: Sequence[str],
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        model_version_lease: int | None,
        tensor_schema_fingerprint: str,
    ) -> PolicyEvaluationContext:
        if model_version_lease is None or model_version_lease != self.policy_version:
            raise RuntimeError("planner context model-version lease is invalid")
        return self._required_planner_context_cache().acquire_batch(
            handles,
            states,
            options,
            decks,
            policy_version=model_version_lease,
            tensor_schema_fingerprint=tensor_schema_fingerprint,
            deck_conditioning=self._model.config.deck_conditioning,
        )

    def _required_planner_context_cache(self) -> PlannerPolicyContextCache:
        cache = self._planner_context_cache
        if cache is None:
            raise RuntimeError("planner policy context caching is disabled")
        return cache

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample actions from the wrapped model."""
        trace = self.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> SampleDecodeTrace:
        """Sample actions without retaining a planner root context."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            return self._model.sample_decode_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                generator=self._generator,
            )

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
        """Prepare memory once and reuse it throughout one complete action."""
        if not self.recurrent_enabled:
            raise RuntimeError("rollout policy recurrent state is disabled")
        if self._graph_decode is not None:
            raise RuntimeError(
                "recurrent rollout does not yet support CUDA graph decode"
            )
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            conditioned = self._model.encode_conditioned_state(states, decks)
            prepared, proposed_state = self._model.recurrent_step_from_conditioned(
                conditioned,
                public_events,
                previous_state=previous_state,
            )
            trace = self._sample_decode_from_conditioned_for_request(
                prepared,
                states,
                options,
                decks,
                temperature=temperature,
                retain_planner_context=retain_planner_context,
            )
        return (trace, proposed_state.detach())

    def sample_decode_with_trace_for_request(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
        model_version_lease: int | None,
        retain_planner_context: bool,
    ) -> SampleDecodeTrace:
        """Decode against one serving lease with explicit context ownership."""
        if model_version_lease is not None and (
            int(model_version_lease) != self.policy_version
        ):
            raise RuntimeError("planner decode model-version lease is unavailable")
        if retain_planner_context and model_version_lease is not None:
            raise ValueError("planner root context acquisition must be unbound")
        if retain_planner_context and self._planner_context_cache is None:
            raise RuntimeError("planner policy context caching is disabled")
        if not retain_planner_context:
            trace = self.sample_decode_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
            )
            return replace(
                trace,
                served_policy_version=self.policy_version,
                served_model_fingerprint=self.model_fingerprint,
                served_proposal_version=self.proposal_version,
            )
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            conditioned = self._model.encode_conditioned_state(states, decks)
            return self._sample_decode_from_conditioned_for_request(
                conditioned,
                states,
                options,
                decks,
                temperature=temperature,
                retain_planner_context=True,
            )

    def _sample_decode_from_conditioned_for_request(
        self,
        conditioned: ConditionedStateOutput,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float,
        retain_planner_context: bool,
    ) -> SampleDecodeTrace:
        """Decode from one immutable prepared root and optionally retain it."""
        context = self._model.policy_context_from_conditioned(
            conditioned,
            options,
        )
        trace = self._model.sample_decode_with_trace_from_context(
            conditioned,
            context,
            options,
            temperature=temperature,
            generator=self._generator,
        )
        if not retain_planner_context:
            return replace(
                trace,
                served_policy_version=self.policy_version,
                served_model_fingerprint=self.model_fingerprint,
                served_proposal_version=self.proposal_version,
            )
        cache = self._required_planner_context_cache()
        try:
            handles = cache.store_batch(
                context,
                states,
                options,
                decks,
                policy_version=self.policy_version,
            )
        except PlannerContextCapacityError:
            return replace(
                trace,
                served_policy_version=self.policy_version,
                served_model_fingerprint=self.model_fingerprint,
                served_proposal_version=self.proposal_version,
                planner_fallback_reason="model_lease_capacity",
            )
        return replace(
            trace,
            planner_context_handles=handles,
            served_policy_version=self.policy_version,
            served_model_fingerprint=self.model_fingerprint,
            served_proposal_version=self.proposal_version,
        )

    def sample_decode_static(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample using a fixed decode cap for bucketed inference serving."""
        trace = self.sample_decode_static_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_static_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
    ) -> SampleDecodeTrace:
        """Sample a fixed-cap decode and retain token-level evidence."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            if self._graph_decode is not None and temperature > 0.0:
                _validate_policy_decks(states, options, decks)
                return self._graph_decode.sample_decode_static_with_trace(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                    gumbel_noise=gumbel_noise,
                )
            return self._model.sample_decode_static_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
                generator=self._generator,
            )

    def sample_decode_tensors(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample actions and keep decode traces on device for serving."""
        trace = self.sample_decode_tensors_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
        )
        return (
            trace.choice_indices,
            trace.append_masks,
            trace.action_logprobs,
            trace.values,
        )

    def sample_decode_tensors_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
        max_select_steps: int,
    ) -> SampleDecodeTensorTrace:
        """Sample tensor actions and retain token-level serving evidence."""
        with (
            torch.inference_mode(),
            _rollout_autocast_context(
                self._model,
                self._autocast,
            ),
        ):
            if self._graph_decode is not None and temperature > 0.0:
                _validate_policy_decks(states, options, decks)
                return self._graph_decode.sample_decode_tensors_with_trace(
                    states,
                    options,
                    decks,
                    temperature=temperature,
                    max_select_steps=max_select_steps,
                )
            return self._model.sample_decode_tensors_with_trace(
                states,
                options,
                decks,
                temperature=temperature,
                max_select_steps=max_select_steps,
                generator=self._generator,
            )


def _validate_policy_decks(
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
) -> None:
    batch_size = int(options.valid_options.shape[0])
    if int(states.card_ids.shape[0]) != batch_size or len(decks) != batch_size:
        raise ValueError("states, options, and decks must have matching batch rows")


def run_rollout(
    config: RolloutConfig,
    *,
    pool_factory: RolloutPoolFactory | None = None,
    policy_factory: RolloutPolicyFactory | None = None,
) -> dict[str, Any]:
    """Run vectorized battles and write BC-compatible rollout trajectory shards."""
    if config.parallel_workers > 1:
        warnings.warn(
            "RolloutConfig.parallel_workers is deprecated for training throughput; "
            "use rl.train execution.actors with the centralized inference service "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if pool_factory is not None or policy_factory is not None:
            raise ValueError("parallel rollout does not support injected factories")
        from ptcg_rl.rl.parallel_rollout import run_parallel_rollout

        return run_parallel_rollout(config)
    return _run_single_rollout(
        config,
        pool_factory=pool_factory,
        policy_factory=policy_factory,
    )


def _run_single_rollout(
    config: RolloutConfig,
    *,
    pool_factory: RolloutPoolFactory | None = None,
    policy_factory: RolloutPolicyFactory | None = None,
) -> dict[str, Any]:
    """Run one process of vectorized battles and write rollout shards."""
    if config.record_forced_actions:
        raise NotImplementedError("record_forced_actions=True is not implemented")
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    candidate_deck = _read_deck(config.decks.candidate)
    opponent_deck = _read_deck(config.decks.opponent or config.decks.candidate)
    deck_pair: DeckPair = (candidate_deck, opponent_deck)

    def deck_pair_sampler() -> DeckPair:
        return deck_pair

    output_dir = deck_records.repo_path(resolve_rollout_output_dir(config))
    candidate_policy = _make_policy(
        config,
        checkpoint_path=config.checkpoint_path,
        device=device,
        policy_factory=policy_factory,
    )
    opponent_spec = _scripted_opponent_spec(config)
    actors = _rollout_actors(
        config,
        candidate_policy=candidate_policy,
        device=device,
        policy_factory=policy_factory,
        opponent_spec=opponent_spec,
    )
    policy_version = config.policy_version or _policy_version(
        config.checkpoint_path,
        policy_kind=config.policy_kind,
    )
    recorder, writer = _trajectory_io(
        config,
        output_dir=output_dir,
        policy_version=policy_version,
        opponent_spec=opponent_spec,
    )
    factory = pool_factory or _default_pool_factory

    start = time.perf_counter()
    run_summary = _MutableRunSummary()
    with factory(config.num_concurrent_games, deck_pair_sampler) as pool:
        stepper = RolloutStepper(
            pool=pool,
            actors=actors,
            recorder=recorder,
            temperature=config.sampling_temperature,
            device=device,
        )
        _run_step_loop(config, stepper, recorder, writer, run_summary)
        live_games = len(getattr(pool, "live_games", ()))
    write_result = writer.close()
    elapsed_seconds = time.perf_counter() - start
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": _rollout_config_dump(config, output_dir=output_dir),
        "device": str(device),
        "output_dir": deck_records.display_path(output_dir),
        "manifest_path": deck_records.display_path(write_result.manifest_path),
        "games_path": deck_records.display_path(write_result.games_path),
        "elapsed_seconds": elapsed_seconds,
        "run": run_summary.as_dict(),
        "rates": _rollout_rate_summary(run_summary, elapsed_seconds),
        "recorder": dict(sorted(recorder.counters.items())),
        "writer": write_result.manifest.get("summary", {}),
        "live_games_after_run": live_games,
    }


def resolve_rollout_output_dir(config: RolloutConfig) -> Path:
    """Return the explicit or version-derived rollout artifact directory."""
    if config.output_dir is not None:
        return config.output_dir
    return config.run.output_root / "rl" / "rollouts" / config.run.version


@dataclass
class _MutableRunSummary:
    iterations: int = 0
    forced_actions: int = 0
    scripted_actions: int = 0
    policy_actions: int = 0
    recorded_decisions: int = 0
    finished_games: int = 0

    def update(self, stats: RolloutStepStats) -> None:
        """Accumulate one step worth of counters."""
        self.iterations += 1
        self.forced_actions += stats.forced_actions
        self.scripted_actions += stats.scripted_actions
        self.policy_actions += stats.policy_actions
        self.recorded_decisions += stats.recorded_decisions
        self.finished_games += stats.finished_games

    def as_dict(self) -> dict[str, int]:
        """Return JSON-friendly run counters."""
        return {
            "iterations": self.iterations,
            "forced_actions": self.forced_actions,
            "scripted_actions": self.scripted_actions,
            "policy_actions": self.policy_actions,
            "recorded_decisions": self.recorded_decisions,
            "finished_games": self.finished_games,
            "engine_submissions": (
                self.forced_actions + self.scripted_actions + self.policy_actions
            ),
        }


def _run_step_loop(
    config: RolloutConfig,
    stepper: RolloutStepper,
    recorder: TrajectoryRecorderLike,
    writer: TrajectoryWriterLike,
    summary: _MutableRunSummary,
) -> None:
    while summary.iterations < config.max_iterations:
        if (
            config.total_games is not None
            and summary.finished_games >= config.total_games
        ):
            break
        if (
            config.total_steps is not None
            and summary.recorded_decisions >= config.total_steps
        ):
            break
        stats = stepper.step()
        summary.update(stats)
        completed = recorder.pop_completed()
        if completed:
            writer.add_completed(completed)
        if stats.pending_games == 0 and stats.finished_games == 0:
            break
    completed = recorder.pop_completed()
    if completed:
        writer.add_completed(completed)


def _trajectory_io(
    config: RolloutConfig,
    *,
    output_dir: Path,
    policy_version: str,
    opponent_spec: OpponentSpec | None,
) -> tuple[TrajectoryRecorderLike, TrajectoryWriterLike]:
    from ptcg_rl.rl.trajectory import TrajectoryRecorder, TrajectoryShardWriter

    opponent_name = _opponent_name(config, opponent_spec, policy_version=policy_version)
    opponent_tier = -1 if opponent_spec is None else int(opponent_spec.tier)
    recorder = TrajectoryRecorder(
        policy_name="policy",
        policy_version=policy_version,
        opponent_name=opponent_name,
        opponent_tier=opponent_tier,
    )
    writer = TrajectoryShardWriter(
        output_dir=output_dir,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
        config=_rollout_config_dump(config, output_dir=output_dir),
        metadata={
            "policy_version": policy_version,
            "opponent_name": opponent_name,
            "opponent_tier": opponent_tier,
        },
    )
    return recorder, writer


def _rollout_actors(
    config: RolloutConfig,
    *,
    candidate_policy: RolloutPolicy,
    device: torch.device,
    policy_factory: RolloutPolicyFactory | None,
    opponent_spec: OpponentSpec | None,
) -> RolloutActors:
    if config.opponent.mode == "self_play":
        return RolloutActors(mode="self_play", candidate_policy=candidate_policy)
    if config.opponent.mode == "frozen":
        frozen_policy = _make_policy(
            config,
            checkpoint_path=config.opponent.frozen_checkpoint_path,
            device=device,
            policy_factory=policy_factory,
        )
        return RolloutActors(
            mode="frozen",
            candidate_policy=candidate_policy,
            frozen_policy=frozen_policy,
        )
    if opponent_spec is None:
        raise RuntimeError("scripted opponent spec was not resolved")
    scripted_agent = build_opponent(opponent_spec, seed=config.seed + 17)
    return RolloutActors(
        mode="scripted",
        candidate_policy=candidate_policy,
        scripted_agent=scripted_agent,
    )


def _make_policy(
    config: RolloutConfig,
    *,
    checkpoint_path: Path | None,
    device: torch.device,
    policy_factory: RolloutPolicyFactory | None,
) -> RolloutPolicy:
    if policy_factory is not None:
        return policy_factory(config.policy_kind, checkpoint_path, device, config.model)
    return build_rollout_policy(
        policy_kind=config.policy_kind,
        checkpoint_path=checkpoint_path,
        model_config=config.model,
        device=device,
        autocast=config.autocast,
    )


def build_rollout_policy(
    *,
    policy_kind: RolloutPolicyKind,
    checkpoint_path: Path | None,
    model_config: AgentNetworkConfig,
    device: torch.device,
    autocast: AutocastMode = "off",
) -> RolloutPolicy:
    """Build a rollout policy from config and optional checkpoint."""
    if policy_kind == "min_count":
        return MinCountRolloutPolicy()
    model = build_agent_policy_value_net(
        _checkpoint_model_config(checkpoint_path) or model_config
    ).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            deck_records.repo_path(checkpoint_path), map_location="cpu"
        )
        incompatible = model.load_state_dict(
            _checkpoint_state_dict(checkpoint),
            strict=False,
        )
        allowed_missing = {
            "opponent_hand_head.weight",
            "opponent_hand_head.bias",
        } | LEGACY_STATE_ENCODER_MISSING_KEYS
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                "checkpoint state dict is incompatible with rollout model"
            )
    model.eval()
    return ModelRolloutPolicy(model, autocast=autocast)


def _rollout_autocast_context(
    model: torch.nn.Module,
    autocast: AutocastMode,
) -> AbstractContextManager[None]:
    if autocast == "off":
        return nullcontext()
    device = _module_device(model)
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _module_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _scripted_opponent_spec(config: RolloutConfig) -> OpponentSpec | None:
    if config.opponent.mode != "scripted":
        return None
    scripted_name = config.opponent.scripted_name
    if scripted_name is None:
        raise ValueError("scripted rollout mode requires scripted_name")
    registry = opponent_registry()
    spec = registry[scripted_name]
    if not spec.vector_safe:
        raise ValueError(f"scripted opponent is not vector_safe: {scripted_name}")
    return spec


def _opponent_name(
    config: RolloutConfig,
    opponent_spec: OpponentSpec | None,
    *,
    policy_version: str,
) -> str:
    if config.opponent.mode == "self_play":
        return "self_play"
    if config.opponent.mode == "scripted":
        if opponent_spec is None:
            raise RuntimeError("scripted opponent spec was not resolved")
        return opponent_spec.name
    frozen_path = config.opponent.frozen_checkpoint_path
    version = _policy_version(frozen_path, policy_kind=config.policy_kind)
    return f"frozen@{version or policy_version}"


def _default_pool_factory(
    num_games: int,
    deck_pair_sampler: DeckPairSampler,
) -> AbstractContextManager[VectorPoolLike]:
    return VectorBattlePool(num_games, deck_pair_sampler, include_search_input=True)


def _read_deck(path: Path) -> tuple[int, ...]:
    return tuple(deck_records.read_deck(deck_records.repo_path(path)))


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _checkpoint_model_config(checkpoint_path: Path | None) -> AgentNetworkConfig | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(deck_records.repo_path(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _policy_version(
    checkpoint_path: Path | None,
    *,
    policy_kind: RolloutPolicyKind,
) -> str:
    if policy_kind == "min_count":
        return "min_count"
    if checkpoint_path is None:
        return "untrained"
    resolved = deck_records.repo_path(checkpoint_path)
    if not resolved.exists():
        return "missing"
    digest = hashlib.sha1()
    with resolved.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _rollout_config_dump(
    config: RolloutConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data["output_dir"] = deck_records.display_path(output_dir)
    return data


def _rollout_rate_summary(
    summary: _MutableRunSummary,
    elapsed_seconds: float,
) -> dict[str, float]:
    return {
        "recorded_decisions_per_second": _rate(
            summary.recorded_decisions,
            elapsed_seconds,
        ),
        "finished_games_per_second": _rate(summary.finished_games, elapsed_seconds),
        "engine_submissions_per_second": _rate(
            summary.forced_actions + summary.scripted_actions + summary.policy_actions,
            elapsed_seconds,
        ),
    }


def _rate(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return float(count) / elapsed_seconds
