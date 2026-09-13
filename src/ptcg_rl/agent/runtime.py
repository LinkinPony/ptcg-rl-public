"""Kaggle agent runtime with policy inference and legal fallbacks."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.actions.selection import (
    forced_action,
    is_legal_action,
    normalize_action_order,
    random_legal_action,
)
from ptcg_rl.agent.probe import (
    ActTimeSearchConfig,
    RuntimeProbeResult,
    all_worlds_verified_lethal,
    all_worlds_verified_self_loss,
    enumerate_select_actions,
    is_core_action,
    observation_with_probe_features,
    run_runtime_probe_features,
)
from ptcg_rl.agent.search.budget import (
    QuotaClass,
    SearchBudgetManager,
    SearchBudgetPlan,
)
from ptcg_rl.agent.search.context import observation_with_context
from ptcg_rl.agent.search.evidence import search_evidence_from_macro_result
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.policy_inference import (
    EncodedObservationCache,
    batched_observation_values,
    batched_root_information_values,
)
from ptcg_rl.agent.search.policy_inputs import (
    CanonicalPolicyInput,
    build_canonical_policy_input,
)
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.agent.search.reranker import (
    MacroSearchPolicy,
    MacroSearchResult,
    PairedMacroSearcher,
)
from ptcg_rl.agent.search.root_information import RootActorRelation
from ptcg_rl.agent.search.telemetry import SearchActTelemetry
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContext,
    GameContextFeatures,
    GameContextSnapshot,
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    PublicEventDecisionToken,
    PublicEventDelta,
    collate_public_event_deltas,
)
from ptcg_rl.decks import CanonicalDeck, DeckBatch, canonicalize_deck
from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.engine.search_evidence import SearchEvidence


class SelectPolicy(Protocol):
    """Callable policy used by the act-time runtime."""

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return one complete select action for the observation prompt."""


class ActTimeConfig(BaseModel):
    """Configuration for safe act-time policy execution."""

    model_config = ConfigDict(extra="forbid")

    deck_path: Path | None = None
    checkpoint_path: Path | None = None
    # Validated lazily as PackagedPlannerConfig. Importing the RL planner stack
    # while this lightweight Kaggle boundary initializes creates a package
    # cycle through ptcg_rl.rl.__init__.
    planner: Any | None = None
    seed: int = 0
    policy_temperature: float = 0.0
    default_remaining_overage_time: float = 600.0
    min_budget_seconds: float = 0.2
    max_budget_seconds: float = 10.0
    low_overage_seconds: float = 60.0
    critical_overage_seconds: float = 10.0
    min_remaining_decisions: int = 20
    decision_history_size: int = 64
    prewarm_on_startup: bool = True
    prewarm_engine: bool = False
    prewarm_checkpoint: bool = True
    prewarm_policy_forward: bool = False
    belief: OpponentBeliefFeatureConfig = OpponentBeliefFeatureConfig()
    search: ActTimeSearchConfig = ActTimeSearchConfig()

    @field_validator(
        "default_remaining_overage_time",
        "min_budget_seconds",
        "max_budget_seconds",
        "low_overage_seconds",
        "critical_overage_seconds",
    )
    @classmethod
    def valid_non_negative_float(cls, value: float) -> float:
        """Reject invalid time budgets."""
        if value < 0.0:
            raise ValueError("time fields must be non-negative")
        return value

    @field_validator("policy_temperature")
    @classmethod
    def valid_policy_temperature(cls, value: float) -> float:
        """Require a finite deployment temperature; zero means greedy."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy_temperature must be finite and non-negative")
        return value

    @field_validator("min_remaining_decisions", "decision_history_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid positive counters."""
        if value <= 0:
            raise ValueError("counter fields must be positive")
        return value

    @model_validator(mode="after")
    def valid_budget_order(self) -> ActTimeConfig:
        """Reject inconsistent budget clamps."""
        if self.max_budget_seconds < self.min_budget_seconds:
            raise ValueError("max_budget_seconds must be >= min_budget_seconds")
        if self.low_overage_seconds < self.critical_overage_seconds:
            raise ValueError("low_overage_seconds must be >= critical_overage_seconds")
        if self.planner is not None:
            if self.checkpoint_path is None:
                raise ValueError("packaged planner requires a checkpoint")
            if self.search.enabled or self.search.conservative_override_enabled:
                raise ValueError("packaged planner requires the legacy probe disabled")
            if self.search.macro.mode != "disabled":
                raise ValueError(
                    "packaged planner requires legacy macro search disabled"
                )
            if self.belief != self.planner.belief_producer:
                raise ValueError("ActTime belief differs from packaged planner")
            if self.search.sampler != self.planner.belief_sampler:
                raise ValueError("ActTime sampler differs from packaged planner")
        return self

    @classmethod
    def from_env(cls) -> ActTimeConfig:
        """Build runtime config from packaged defaults and optional env overrides."""
        planner_path = _env_planner_runtime_path()
        packaged_planner = (
            None
            if planner_path is None
            else _load_packaged_planner_config(planner_path)
        )
        planner = None
        if packaged_planner is not None and _profile_planner_enabled(
            packaged_planner.planner_enabled_by_default
        ):
            planner = packaged_planner
        search = ActTimeSearchConfig()
        belief = OpponentBeliefFeatureConfig()
        if packaged_planner is not None:
            search = search.model_copy(
                update={
                    "enabled": False,
                    "conservative_override_enabled": False,
                    "worlds": packaged_planner.planner.scenario.belief_world_count,
                    "manual_coin": False,
                    "sampler": packaged_planner.belief_sampler,
                    "macro": search.macro.model_copy(update={"mode": "disabled"}),
                }
            )
            belief = packaged_planner.belief_producer
        config = cls(
            deck_path=_optional_env_path("PTCG_RL_DECK_PATH"),
            checkpoint_path=_env_checkpoint_path(),
            planner=planner,
            seed=_optional_env_int("PTCG_RL_AGENT_SEED", 0),
            policy_temperature=_deployment_policy_temperature(),
            belief=belief,
            search=search,
        )
        belief_summary_path = _env_belief_summary_path()
        if belief_summary_path is not None and planner is None:
            config = config.model_copy(deep=True)
            config.belief.deck_signature_summary_path = belief_summary_path
            config.search.sampler.prior_deck_signature_summary_path = (
                belief_summary_path
            )
        return config


@dataclass(frozen=True)
class ActTimeBudget:
    """Decision-time budget derived from the Kaggle overage pool."""

    remaining_overage_time: float
    estimated_remaining_decisions: int
    budget_seconds: float
    allow_model_load: bool
    allow_search: bool


class TimeBudgetManager:
    """Compute conservative per-decision budgets from observation metadata."""

    def __init__(self, config: ActTimeConfig) -> None:
        """Initialize budget history."""
        self._config = config
        self._elapsed_seconds: deque[float] = deque(
            maxlen=config.decision_history_size,
        )

    def budget_for(self, observation: Any) -> ActTimeBudget:
        """Return a safe time budget for one decision."""
        remaining = _float_field(
            observation,
            "remainingOverageTime",
            self._config.default_remaining_overage_time,
        )
        estimated_decisions = _estimate_remaining_decisions(
            observation,
            self._config.min_remaining_decisions,
        )
        raw_budget = remaining / max(estimated_decisions, 1)
        budget = min(
            self._config.max_budget_seconds,
            max(self._config.min_budget_seconds, raw_budget),
        )
        return ActTimeBudget(
            remaining_overage_time=remaining,
            estimated_remaining_decisions=estimated_decisions,
            budget_seconds=budget,
            allow_model_load=remaining > self._config.critical_overage_seconds,
            allow_search=remaining >= self._config.low_overage_seconds,
        )

    def record_elapsed(self, elapsed_seconds: float) -> None:
        """Record one completed decision duration."""
        if elapsed_seconds >= 0.0:
            self._elapsed_seconds.append(elapsed_seconds)

    @property
    def average_elapsed_seconds(self) -> float:
        """Return the moving average decision time, or zero before decisions."""
        if not self._elapsed_seconds:
            return 0.0
        return sum(self._elapsed_seconds) / len(self._elapsed_seconds)


def _resolve_policy_device(torch_module: Any, device: str) -> Any:
    """Resolve a user-facing policy device string into a torch device."""
    normalized = device.strip().lower()
    if not normalized:
        raise ValueError("policy device must be non-empty")
    if normalized == "auto":
        normalized = "cuda" if torch_module.cuda.is_available() else "cpu"
    elif normalized == "gpu":
        normalized = "cuda"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(f"requested CUDA policy device is unavailable: {device}")
    return torch_module.device(normalized)


@dataclass(frozen=True)
class _PendingRecurrentRuntimeDecision:
    """One pure recurrent proposal awaiting the Kaggle return boundary."""

    observation_id: int
    event_generation: int
    conditioned: Any
    options: Any
    proposed_state: Any


class CheckpointPolicy:
    """Temperature-controlled inference loaded from a training checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Load a policy/value checkpoint for inference."""
        import torch

        from ptcg_rl.checkpoint_storage import (
            direct_recurrent_policy_missing_keys,
        )
        from ptcg_rl.model import (
            LEGACY_STATE_ENCODER_MISSING_KEYS,
            AgentNetworkConfig,
            build_agent_policy_value_net,
        )

        self._torch = torch
        self._device = _resolve_policy_device(torch, device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        from ptcg_rl.model.weights_only_migration_classification import (
            checkpoint_publish_version,
        )

        config = _checkpoint_model_config(checkpoint) or AgentNetworkConfig()
        self._checkpoint_path = checkpoint_path
        self._model_version = _file_sha256(checkpoint_path)
        published_version = checkpoint_publish_version(checkpoint)
        self._policy_version = 0 if published_version is None else published_version
        self._model = build_agent_policy_value_net(config).to(self._device)
        checkpoint_state = _checkpoint_state_dict(checkpoint)
        direct_policy_missing = direct_recurrent_policy_missing_keys(checkpoint)
        if direct_policy_missing & {str(key) for key in checkpoint_state}:
            raise RuntimeError(
                "direct policy-only checkpoint retained declared omitted state"
            )
        incompatible = self._model.load_state_dict(
            checkpoint_state,
            strict=False,
        )
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        legacy_aux_head_keys = {
            "opponent_hand_head.weight",
            "opponent_hand_head.bias",
        }
        allowed_missing = (
            legacy_aux_head_keys
            | LEGACY_STATE_ENCODER_MISSING_KEYS
            | direct_policy_missing
        )
        conditioning = config.deck_conditioning
        if (
            conditioning is not None
            and conditioning.enabled
            and conditioning.deck_context_mode == "folded"
        ):
            alias_missing = {
                name
                for name in missing
                if name.startswith("state_encoder.card_encoder.")
                and name.removeprefix("state_encoder.") in checkpoint_state
            }
            allowed_missing.update(alias_missing)
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                "checkpoint state dict is incompatible with AgentPolicyValueNet"
            )
        self._has_opponent_hand_head = not bool(missing & legacy_aux_head_keys)
        self._direct_policy_only = bool(direct_policy_missing)
        self._model.eval()
        self._recurrent_enabled = config.recurrent is not None
        self._recurrent_state: Any | None = None
        self._pending_recurrent: _PendingRecurrentRuntimeDecision | None = None
        self._runtime_event_delta: PublicEventDelta | None = None
        self._runtime_event_generation: int | None = None
        self._prewarmed = False
        self._policy_temperature = 0.0
        self._sampling_seed = 0
        self._sampling_generator = torch.Generator(device=self._device).manual_seed(0)
        self._own_deck: CanonicalDeck | None = None
        self._inference_cache = EncodedObservationCache(
            model=self._model,
            device=self._device,
        )
        self._inference_cache.bind_context(
            decks=None,
            deck_signature=None,
            model_version=self._model_version,
        )
        if own_deck is not None:
            self.bind_own_deck(own_deck)

    def configure_deployment_sampling(self, *, temperature: float, seed: int) -> None:
        """Bind reproducible deployment decoding without global RNG state."""
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("deployment policy temperature must be non-negative")
        self._policy_temperature = float(temperature)
        self._sampling_seed = int(seed)
        self._sampling_generator = self._torch.Generator(
            device=self._device
        ).manual_seed(self._sampling_seed)

    def configure_inference_cache(self, *, enabled: bool) -> None:
        """Enable per-callback encoded-state reuse for repeated observations."""
        self._inference_cache.configure(enabled=enabled)

    def clear_inference_cache(self) -> None:
        """Release cached observations and device tensors at an act boundary."""
        self._inference_cache.clear()

    def bind_own_deck(self, card_ids: Sequence[int]) -> None:
        """Bind one canonical own deck and invalidate every conditioned cache."""
        deck = canonicalize_deck(card_ids)
        conditioning = self._model.config.deck_conditioning
        if (
            conditioning is not None
            and conditioning.enabled
            and (
                (
                    conditioning.lora is not None
                    and conditioning.lora.export_mode == "merged"
                )
                or (
                    conditioning.dense_private is not None
                    and conditioning.dense_private.export_mode == "fixed"
                )
                or (
                    conditioning.compositional is not None
                    and conditioning.compositional.export_mode == "fixed"
                )
            )
            and deck.signature not in conditioning.profile_by_signature
        ):
            raise ValueError(
                "fixed deck-specialized checkpoint cannot bind a different deck"
            )
        if self._own_deck == deck:
            return
        self.reset_runtime_episode()
        self._own_deck = deck
        self._prewarmed = False
        self._inference_cache.bind_context(
            decks=DeckBatch.from_decks((deck,), device=self._device),
            deck_signature=deck.signature,
            model_version=self._model_version,
        )

    @property
    def recurrent_enabled(self) -> bool:
        """Return whether this checkpoint requires deployment state commits."""
        return self._recurrent_enabled

    @property
    def direct_policy_only(self) -> bool:
        """Return whether deployment-only auxiliary heads were omitted."""
        return self._direct_policy_only

    @property
    def own_deck_signature(self) -> str | None:
        """Return the currently bound canonical own-deck signature."""
        return None if self._own_deck is None else self._own_deck.signature

    @property
    def selected_private_profile_module_key(self) -> str | None:
        """Return the exact private module selected by the bound deck, if any."""
        if self._own_deck is None:
            return None
        conditioning = self._model.config.deck_conditioning
        if conditioning is None or not conditioning.enabled:
            return None
        profile = conditioning.profile_by_signature.get(self._own_deck.signature)
        return None if profile is None else profile.module_key

    @property
    def checkpoint_registry_sha256(self) -> str | None:
        """Return the resolved conditioning registry carried by the checkpoint."""
        conditioning = self._model.config.deck_conditioning
        if conditioning is None or not conditioning.enabled:
            return None
        return conditioning.resolved_registry_sha256

    @property
    def packaged_private_profile_count(self) -> int:
        """Return private profiles physically represented by this model config."""
        conditioning = self._model.config.deck_conditioning
        if conditioning is None or not conditioning.enabled:
            return 0
        return len(conditioning.active_routes)

    @property
    def deck_conditioning_enabled(self) -> bool:
        """Return whether this checkpoint requires a bound deck context."""
        conditioning = self._model.config.deck_conditioning
        return conditioning is not None and conditioning.enabled

    @property
    def device(self) -> str:
        """Return the resolved inference device for audit manifests."""
        return str(self._device)

    @property
    def planner_model(self) -> Any:
        """Expose the loaded eval model to the production planner adapter."""
        return self._model

    @property
    def planner_device(self) -> Any:
        """Expose the exact torch device used by the loaded model."""
        return self._device

    @property
    def checkpoint_sha256(self) -> str:
        """Return the immutable source checkpoint fingerprint."""
        return self._model_version

    @property
    def policy_version(self) -> int:
        """Return the checkpoint publication version, or zero when legacy."""
        return self._policy_version

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Decode a legal index sequence at the configured temperature."""
        select = _select_from_observation(observation)
        if select is None:
            return ()
        with self._torch.inference_mode():
            prepared = self._conditioned_observation(
                observation,
                require_options=True,
            )
            if prepared is None or prepared[1] is None:
                return ()
            conditioned, options = prepared[0], cast(Any, prepared[1])
            conditioned = self._prepare_recurrent_decision(
                observation,
                conditioned,
                options,
            )
            if self._policy_temperature == 0.0:
                (decoded_action,) = self._model.greedy_decode_from_conditioned(
                    conditioned,
                    options,
                )
            else:
                context = self._model.policy_context_from_conditioned(
                    conditioned,
                    options,
                )
                trace = self._model.sample_decode_with_trace_from_context(
                    conditioned,
                    context,
                    options,
                    temperature=self._policy_temperature,
                    generator=self._sampling_generator,
                )
                (decoded_action,) = trace.actions
        return normalize_action_order(select, decoded_action)

    def select_actions(
        self,
        observations: Sequence[Any],
    ) -> tuple[tuple[int, ...], ...]:
        """Decode a stateless observation batch in one model forward."""
        if self._recurrent_enabled:
            raise ValueError("recurrent checkpoint actions require per-game state")
        if not observations:
            return ()
        from ptcg_rl.model import collate_encoded_options, collate_state_tokens

        policy_inputs = tuple(
            build_canonical_policy_input(observation) for observation in observations
        )
        if any(policy_input is None for policy_input in policy_inputs):
            raise ValueError("batched policy observation could not be tensorized")
        inputs = cast(tuple[CanonicalPolicyInput, ...], policy_inputs)
        states = collate_state_tokens(
            [policy_input.state for policy_input in inputs],
            device=self._device,
        )
        options = collate_encoded_options(
            [policy_input.options for policy_input in inputs],
            min_counts=[policy_input.min_count for policy_input in inputs],
            max_counts=[policy_input.max_count for policy_input in inputs],
            device=self._device,
        )
        with self._torch.inference_mode():
            decks = self._deck_batch(len(inputs))
            if self._policy_temperature == 0.0:
                decoded = self._model.greedy_decode(states, options, decks)
            else:
                decoded, _logprobs, _values = self._model.sample_decode(
                    states,
                    options,
                    decks,
                    temperature=self._policy_temperature,
                    generator=self._sampling_generator,
                )
        return tuple(
            normalize_action_order(
                _select_from_observation(observation),
                action,
            )
            for observation, action in zip(
                observations,
                decoded,
                strict=True,
            )
        )

    def select_preencoded_actions(
        self,
        states: Any,
        options: Any,
        decks: DeckBatch | None,
        *,
        max_select_steps: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Decode validated numeric inputs without observations."""
        if self._recurrent_enabled:
            raise ValueError("recurrent checkpoint actions require per-game state")
        state_rows = int(states.card_ids.shape[0])
        option_rows = int(options.valid_options.shape[0])
        if state_rows <= 0 or option_rows != state_rows:
            raise ValueError("preencoded policy state and option rows must align")
        if states.card_ids.device.type != self._device.type or (
            self._device.index is not None
            and states.card_ids.device.index != self._device.index
        ):
            raise ValueError("preencoded policy state is on the wrong device")
        if options.valid_options.device.type != self._device.type or (
            self._device.index is not None
            and options.valid_options.device.index != self._device.index
        ):
            raise ValueError("preencoded policy options are on the wrong device")
        if decks is not None and (
            decks.card_ids.device.type != self._device.type
            or (
                self._device.index is not None
                and decks.card_ids.device.index != self._device.index
            )
        ):
            raise ValueError("preencoded policy decks are on the wrong device")
        if self.deck_conditioning_enabled and decks is None:
            raise ValueError("deck-conditioned preencoded policy requires decks")
        with self._torch.inference_mode():
            if self._policy_temperature == 0.0:
                return self._model.greedy_decode(
                    states,
                    options,
                    decks,
                    max_select_steps=max_select_steps,
                )
            decoded, _logprobs, _values = self._model.sample_decode(
                states,
                options,
                decks,
                temperature=self._policy_temperature,
                generator=self._sampling_generator,
            )
            return decoded

    def rank_actions(
        self, observation: Any, *, top_k: int
    ) -> tuple[tuple[int, ...], ...]:
        """Return top complete actions by policy probability."""
        select = _select_from_observation(observation)
        if select is None or top_k <= 0:
            return ()
        actions = enumerate_select_actions(select, max_actions=64)
        priors = self.action_priors(observation, actions)
        return tuple(
            action
            for action, _ in sorted(
                priors.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
        )

    def action_priors(
        self,
        observation: Any,
        actions: Sequence[tuple[int, ...]],
    ) -> Mapping[tuple[int, ...], float]:
        """Return normalized policy priors for complete candidate actions."""
        if not actions:
            return {}
        with self._torch.inference_mode():
            if self._recurrent_enabled:
                pending = self._required_pending_recurrent(observation)
                conditioned, options = pending.conditioned, pending.options
            else:
                prepared = self._conditioned_observation(
                    observation,
                    require_options=True,
                )
                if prepared is None or prepared[1] is None:
                    return {}
                conditioned, options = prepared[0], cast(Any, prepared[1])
            logprobs = self._model.action_logprobs_from_conditioned(
                conditioned,
                options,
                actions,
            )
            probabilities = self._torch.exp(logprobs).detach().cpu().tolist()
            raw_priors = {
                tuple(int(index) for index in action): float(probability)
                for action, probability in zip(actions, probabilities, strict=True)
            }
        total = sum(raw_priors.values())
        if total <= 0.0:
            return raw_priors
        return {action: prior / total for action, prior in raw_priors.items()}

    def select_search_conditioned_action(
        self,
        observation: Any,
        evidence: SearchEvidence,
    ) -> tuple[int, ...]:
        """Greedily select from complete candidates using search evidence."""
        select = _select_from_observation(observation)
        if select is None:
            return ()
        with self._torch.inference_mode():
            prepared = self._conditioned_observation(
                observation,
                require_options=True,
            )
            if prepared is None or prepared[1] is None:
                return ()
            conditioned, options = prepared[0], cast(Any, prepared[1])
            features = self._torch.tensor(
                evidence.feature_rows,
                dtype=conditioned.encoded_state.global_embedding.dtype,
                device=self._device,
            )
            action = self._model.select_search_action_from_conditioned(
                conditioned,
                options,
                evidence.actions,
                features,
            )
        return normalize_action_order(select, action)

    def value(self, observation: Any, root_player_index: int) -> float:
        """Return model value from the root player's perspective."""
        terminal = _terminal_value(observation, root_player_index)
        if terminal is not None:
            return terminal
        current_player = _int_field(
            _field(observation, "current"),
            "yourIndex",
            root_player_index,
        )
        self._validate_conditioned_search_perspective(
            current_player=current_player,
            root_player_index=root_player_index,
        )
        with self._torch.inference_mode():
            conditioned = self._value_conditioned_state(observation)
            value = float(
                self._model.root_values_from_conditioned(conditioned)
                .detach()
                .cpu()
                .item()
            )
        return value if current_player == root_player_index else -value

    def values(
        self,
        observations: Sequence[Any],
        root_player_index: int,
    ) -> tuple[float, ...]:
        """Evaluate multiple leaf values in one state-encoder forward pass."""
        for observation in observations:
            if _terminal_value(observation, root_player_index) is not None:
                continue
            self._validate_conditioned_search_perspective(
                current_player=_int_field(
                    _field(observation, "current"),
                    "yourIndex",
                    root_player_index,
                ),
                root_player_index=root_player_index,
            )
        nonterminal_count = sum(
            _terminal_value(observation, root_player_index) is None
            for observation in observations
        )
        return batched_observation_values(
            observations,
            root_player_index=root_player_index,
            model=self._model,
            device=self._device,
            torch_module=self._torch,
            terminal_value=_terminal_value,
            current_player_index=lambda observation, fallback: _int_field(
                _field(observation, "current"),
                "yourIndex",
                fallback,
            ),
            decks=self._deck_batch(nonterminal_count),
        )

    def root_information_values(
        self,
        observations: Sequence[Any],
        root_player_index: int,
        endpoints: Sequence[MacroEndpoint],
    ) -> tuple[float, ...]:
        """Evaluate same-seat and handoff leaves in one root-adapter forward."""
        if len(observations) != len(endpoints):
            raise ValueError("semantic endpoints must align with observations")
        relations: list[int] = []
        semantic_endpoints: list[int] = []
        for endpoint in endpoints:
            if endpoint is MacroEndpoint.SAME_SEAT_MAIN:
                relations.append(int(RootActorRelation.SAME_SEAT))
                semantic_endpoints.append(int(SemanticEndpoint.SAME_SEAT_MAIN))
            elif endpoint is MacroEndpoint.TURN_HANDOFF:
                relations.append(int(RootActorRelation.OTHER_SEAT))
                semantic_endpoints.append(int(SemanticEndpoint.TURN_HANDOFF))
            else:
                raise ValueError("root adapter received a non-value endpoint")
        belief_dim = self._model.config.root_perspective_value.belief_summary_dim
        return batched_root_information_values(
            observations,
            actor_relations=relations,
            semantic_endpoints=semantic_endpoints,
            root_player_index=root_player_index,
            belief_summary_dim=belief_dim,
            model=self._model,
            device=self._device,
            torch_module=self._torch,
            decks=self._deck_batch(len(observations)),
            current_player_index=lambda observation, fallback: _int_field(
                _field(observation, "current"),
                "yourIndex",
                fallback,
            ),
        )

    def belief_distributions(
        self,
        observation: Any,
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Return card and hand head distributions for belief sampling."""
        if not self._has_opponent_hand_head:
            return None
        with self._torch.inference_mode():
            conditioned = self._value_conditioned_state(observation)
            card_logits, hand_logits = self._model.opponent_logits_from_conditioned(
                conditioned
            )
            card_probs = self._torch.softmax(
                card_logits,
                dim=-1,
            )[0]
            hand_probs = self._torch.softmax(
                hand_logits,
                dim=-1,
            )[0]
        return (
            tuple(float(value) for value in card_probs.detach().cpu().tolist()),
            tuple(float(value) for value in hand_probs.detach().cpu().tolist()),
        )

    def prewarm(self) -> None:
        """Run one dummy forward pass to initialize torch/card-encoder paths."""
        if self._prewarmed:
            return
        if self._recurrent_enabled:
            self._prewarm_recurrent()
        else:
            self.select_action(_dummy_policy_observation())
        self._prewarmed = True

    def bind_runtime_context(
        self,
        *,
        context_snapshot: GameContextSnapshot,
        context_features: GameContextFeatures,
        deadline_monotonic: float,
    ) -> None:
        """Bind the exact public-event delta for one Kaggle callback."""
        del deadline_monotonic
        if not self._recurrent_enabled:
            return
        if self._pending_recurrent is not None:
            raise RuntimeError("a recurrent runtime decision is already pending")
        if context_features.public_event_delta != PublicEventDelta(
            events=context_snapshot.pending_public_events,
            overflow=context_snapshot.public_event_overflow,
        ):
            raise ValueError("runtime context event delta differs from its snapshot")
        self._runtime_event_delta = context_features.public_event_delta
        self._runtime_event_generation = context_snapshot.public_event_generation

    def commit_runtime_decision(self, event_generation: int) -> None:
        """Commit one locally validated Kaggle action exactly once."""
        if not self._recurrent_enabled:
            return
        pending = self._pending_recurrent
        if pending is None:
            raise RuntimeError("recurrent runtime has no prepared decision to commit")
        if pending.event_generation != event_generation:
            raise RuntimeError("recurrent runtime event generation changed")
        self._recurrent_state = pending.proposed_state.detach()
        self._pending_recurrent = None
        self._runtime_event_delta = None
        self._runtime_event_generation = None

    def abort_runtime_decision(self) -> None:
        """Discard an unserved recurrent proposal without advancing memory."""
        self._pending_recurrent = None
        self._runtime_event_delta = None
        self._runtime_event_generation = None

    def reset_runtime_episode(self) -> None:
        """Release every recurrent tensor at a registration/game boundary."""
        self._recurrent_state = None
        self.abort_runtime_decision()

    def close(self) -> None:
        """Release deployment recurrent state without retaining CUDA tensors."""
        self.reset_runtime_episode()

    def canonical_input(
        self,
        observation: Any,
        *,
        require_options: bool = True,
    ) -> CanonicalPolicyInput | None:
        """Expose the shared pre-collation features for parity audits."""
        return build_canonical_policy_input(
            observation,
            require_options=require_options,
        )

    def first_step_logits(self, observation: Any) -> tuple[float, ...]:
        """Return valid option plus STOP logits for numerical parity audits."""
        with self._torch.inference_mode():
            if self._recurrent_enabled:
                pending = self._required_pending_recurrent(observation)
                conditioned, options = pending.conditioned, pending.options
            else:
                prepared = self._conditioned_observation(
                    observation,
                    require_options=True,
                )
                if prepared is None or prepared[1] is None:
                    return ()
                conditioned, options = prepared[0], cast(Any, prepared[1])
            logits = self._model.first_step_logits_from_conditioned(
                conditioned,
                options,
            )[0]
        return tuple(float(value) for value in logits.detach().cpu().tolist())

    def conditioned_diagnostics(
        self,
        observation: Any,
        action: Sequence[int],
    ) -> Mapping[str, Any]:
        """Materialize conditioned outputs used by checkpoint asset audits."""
        with self._torch.inference_mode():
            if self._recurrent_enabled:
                pending = self._required_pending_recurrent(observation)
                conditioned, options = pending.conditioned, pending.options
            else:
                prepared = self._conditioned_observation(
                    observation,
                    require_options=True,
                )
                if prepared is None or prepared[1] is None:
                    raise ValueError("diagnostic observation could not be tensorized")
                conditioned, options = prepared[0], cast(Any, prepared[1])
            evaluation = self._model.evaluate_actions_from_conditioned(
                conditioned,
                options,
                (tuple(int(index) for index in action),),
            )
            auxiliary = self._model.auxiliary_outputs_from_conditioned(
                conditioned,
                options,
            )
        return {
            "action_logprobs": _flat_tensor(evaluation.action_logprobs),
            "root_values": _flat_tensor(evaluation.values),
            "prefix_values": _flat_tensor(evaluation.prefix_values),
            "teacher_forced_step_logits": tuple(
                _flat_tensor(logits) for logits in evaluation.step_logits
            ),
            "prize_diff": _flat_tensor(auxiliary[0]),
            "opponent_card_logits": _flat_tensor(auxiliary[1]),
            "opponent_hand_logits": _flat_tensor(auxiliary[2]),
            "effect_predictions": _flat_tensor(auxiliary[3]),
        }

    def _policy_inputs(self, observation: Any) -> tuple[Any, Any] | None:
        from ptcg_rl.model import collate_encoded_options, collate_state_tokens

        policy_input = build_canonical_policy_input(observation)
        if policy_input is None:
            return None

        states = collate_state_tokens([policy_input.state], device=self._device)
        option_batch = collate_encoded_options(
            [policy_input.options],
            min_counts=[policy_input.min_count],
            max_counts=[policy_input.max_count],
            device=self._device,
        )
        return states, option_batch

    def _value_conditioned_state(self, observation: Any) -> Any:
        if self._recurrent_enabled:
            return self._required_pending_recurrent(observation).conditioned
        from ptcg_rl.model import collate_state_tokens

        if self._inference_cache.enabled:
            cached = self._encoded_observation(
                observation,
                require_options=False,
            )
            if cached is None:
                raise ValueError("value observation could not be tensorized")
            return cached.conditioned_state
        policy_input = build_canonical_policy_input(
            observation,
            require_options=False,
        )
        if policy_input is None:
            raise ValueError("value observation could not be tensorized")
        states = collate_state_tokens([policy_input.state], device=self._device)
        return self._model.encode_conditioned_state(
            states,
            self._deck_batch(1),
        )

    def _conditioned_observation(
        self,
        observation: Any,
        *,
        require_options: bool,
    ) -> tuple[Any, Any | None] | None:
        if self._inference_cache.enabled:
            cached = self._encoded_observation(
                observation,
                require_options=require_options,
            )
            if cached is None:
                return None
            return (cached.conditioned_state, cached.options)
        policy_inputs = self._policy_inputs(observation)
        if policy_inputs is None:
            return None
        states, options = policy_inputs
        return (
            self._model.encode_conditioned_state(states, self._deck_batch(1)),
            options,
        )

    def _prepare_recurrent_decision(
        self,
        observation: Any,
        conditioned: Any,
        options: Any,
    ) -> Any:
        if not self._recurrent_enabled:
            return conditioned
        if self._pending_recurrent is not None:
            raise RuntimeError("a recurrent runtime decision is already pending")
        delta = self._runtime_event_delta
        generation = self._runtime_event_generation
        if delta is None or generation is None:
            raise RuntimeError("recurrent runtime context was not bound")
        snapshot = conditioned.encoded_state.global_embedding
        previous = self._recurrent_state
        if previous is None:
            previous = self._model.initial_recurrent_state(
                1,
                device=snapshot.device,
                dtype=snapshot.dtype,
            )
        prepared, proposed = self._model.recurrent_step_from_conditioned(
            conditioned,
            collate_public_event_deltas((delta,), device=self._device),
            previous_state=previous,
        )
        self._pending_recurrent = _PendingRecurrentRuntimeDecision(
            observation_id=id(observation),
            event_generation=generation,
            conditioned=prepared,
            options=options,
            proposed_state=proposed,
        )
        return prepared

    def _required_pending_recurrent(
        self,
        observation: Any,
    ) -> _PendingRecurrentRuntimeDecision:
        pending = self._pending_recurrent
        if pending is None or pending.observation_id != id(observation):
            raise RuntimeError(
                "recurrent auxiliary inference requires the prepared decision root"
            )
        return pending

    def _prewarm_recurrent(self) -> None:
        prepared = self._conditioned_observation(
            _dummy_policy_observation(),
            require_options=True,
        )
        if prepared is None or prepared[1] is None:
            raise RuntimeError("recurrent prewarm observation could not be tensorized")
        conditioned, options = prepared[0], cast(Any, prepared[1])
        snapshot = conditioned.encoded_state.global_embedding
        initial = self._model.initial_recurrent_state(
            1,
            device=snapshot.device,
            dtype=snapshot.dtype,
        )
        conditioned, _proposed = self._model.recurrent_step_from_conditioned(
            conditioned,
            collate_public_event_deltas((PublicEventDelta(),), device=self._device),
            previous_state=initial,
        )
        self._model.greedy_decode_from_conditioned(conditioned, options)

    def _encoded_observation(
        self,
        observation: Any,
        *,
        require_options: bool,
    ) -> Any:
        if self._own_deck is None and self.deck_conditioning_enabled:
            raise ValueError(
                "enabled CheckpointPolicy requires bind_own_deck before inference"
            )
        return self._inference_cache.encode(
            observation,
            require_options=require_options,
        )

    def _deck_batch(self, batch_size: int) -> DeckBatch | None:
        if batch_size < 0:
            raise ValueError("deck batch size cannot be negative")
        if batch_size == 0:
            return None
        if self._own_deck is None:
            if self.deck_conditioning_enabled:
                raise ValueError(
                    "enabled CheckpointPolicy requires bind_own_deck before inference"
                )
            return None
        return DeckBatch.from_decks(
            (self._own_deck,) * batch_size,
            device=self._device,
        )

    def _validate_conditioned_search_perspective(
        self,
        *,
        current_player: int,
        root_player_index: int,
    ) -> None:
        if self.deck_conditioning_enabled and current_player != root_player_index:
            raise ValueError(
                "deck-conditioned leaf evaluation requires root-player perspective"
            )


class PolicyRuntimeAgent:
    """Act-time Kaggle agent with optional checkpoint policy and hard fallback."""

    def __init__(
        self,
        *,
        config: ActTimeConfig | None = None,
        deck_path: Path | None = None,
        checkpoint_path: Path | None = None,
        seed: int | None = None,
        policy: SelectPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
        strict_runtime_errors: bool = False,
    ) -> None:
        """Initialize runtime state without loading optional model weights."""
        base_config = config or ActTimeConfig()
        updates: dict[str, Any] = {}
        if deck_path is not None:
            updates["deck_path"] = deck_path
        if checkpoint_path is not None:
            updates["checkpoint_path"] = checkpoint_path
        if seed is not None:
            updates["seed"] = seed
        self.config = base_config.model_copy(update=updates)
        self._clock = clock
        self._strict_runtime_errors = strict_runtime_errors
        self._rng = random.Random(self.config.seed)
        self._search_rng = random.Random(self.config.seed + 1_000_003)
        self._deck: list[int] | None = None
        self._context = GameContext()
        self._belief_producer: OpponentBeliefFeatureProducer | None = None
        self._belief_sampler: BeliefSampler | None = None
        self._policy = policy
        self._configure_deployment_policy()
        self._time_manager = TimeBudgetManager(self.config)
        self._search_budget_manager = SearchBudgetManager(
            self.config.search.macro.budget,
            clock=self._clock,
        )
        self.last_budget: ActTimeBudget | None = None
        self.last_policy_error: Exception | None = None
        self.last_belief_error: Exception | None = None
        self.last_probe_error: Exception | None = None
        self.last_probe_seconds: float = 0.0
        self.last_base_policy_seconds: float = 0.0
        self.last_callback_startup_seconds: float = 0.0
        self.last_override_reason: str | None = None
        self.last_base_action: tuple[int, ...] | None = None
        self.last_search_error: Exception | None = None
        self.last_search_telemetry = SearchActTelemetry(
            bank_left_seconds=self._search_budget_manager.bank_left_seconds,
        )
        self.last_prewarm_error: Exception | None = None
        self.last_engine_prewarm_error: Exception | None = None
        self.last_prewarm_seconds: float = 0.0
        self._startup_prewarmed = False
        self._warned_random_fallback = False
        self._search_high_value_turn: int | None = None
        self._recurrent_ppo_only_runtime = False
        self._simple_stateless_runtime = bool(
            getattr(
                policy,
                "simple_stateless_runtime",
                getattr(policy, "simple_stateless_fixed_deck", False),
            )
        )
        self._closed = False

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Reset local-battle episode state without reloading runtime assets."""
        if self._closed:
            raise RuntimeError("cannot begin a game on a closed runtime agent")
        if own_deck is not None:
            canonical = canonicalize_deck(own_deck)
            self._deck = list(canonical.card_ids)
            bind_deck = getattr(self._policy, "bind_own_deck", None)
            if callable(bind_deck):
                bind_deck(canonical.card_ids)
        self._context.reset(
            player_index=player_index,
            own_deck=self._deck if self._deck is not None else None,
        )
        self._reset_policy_episode()
        self._configure_deployment_policy()
        self._time_manager = TimeBudgetManager(self.config)
        self._search_budget_manager.reset()
        self._search_rng = random.Random(self.config.seed + 1_000_003)
        self.last_budget = None
        self.last_probe_seconds = 0.0
        self.last_base_policy_seconds = 0.0
        self.last_callback_startup_seconds = 0.0
        self.last_probe_error = None
        self.last_override_reason = None
        self.last_base_action = None
        self.last_policy_error = None
        self.last_search_error = None
        self.last_search_telemetry = SearchActTelemetry(
            bank_left_seconds=self._search_budget_manager.bank_left_seconds,
        )
        self._search_high_value_turn = None

    def bind_independent_root_context(
        self,
        snapshot: GameContextSnapshot,
    ) -> None:
        """Bind an exact independent replay root without leaking episode state."""
        self._context = GameContext.from_snapshot(snapshot)

    @classmethod
    def from_env(cls) -> PolicyRuntimeAgent:
        """Build the default Kaggle runtime from environment variables."""
        return cls(config=ActTimeConfig.from_env())

    def act(self, observation: Any, configuration: Any | None = None) -> list[int]:
        """Return a legal deck registration or select action."""
        if self._closed:
            raise RuntimeError("cannot act with a closed runtime agent")
        reset_planner_telemetry = getattr(
            self._policy,
            "reset_decision_telemetry",
            None,
        )
        if callable(reset_planner_telemetry):
            reset_planner_telemetry()
        del configuration
        start_time = self._clock()
        self.last_probe_seconds = 0.0
        self.last_base_policy_seconds = 0.0
        self.last_callback_startup_seconds = 0.0
        self._configure_policy_inference_cache()
        select = _select_from_observation(observation)
        fallback = _first_legal_action(select) if select is not None else ()
        self.last_base_action = fallback if select is not None else None
        macro_enabled = self.config.search.macro.mode != "disabled"
        self.last_search_telemetry = SearchActTelemetry(
            enabled=macro_enabled,
            shadow_only=self.config.search.macro.mode == "shadow",
            remaining_overage_time=_float_field(
                observation,
                "remainingOverageTime",
                self.config.default_remaining_overage_time,
            ),
            bank_spent_seconds=self._search_budget_manager.spent_seconds,
            bank_left_seconds=self._search_budget_manager.bank_left_seconds,
            stop_reason="not_eligible" if macro_enabled else "disabled",
            fallback_available=select is None or is_legal_action(select, fallback),
        )
        try:
            return self._act_inner(observation, act_started_at=start_time)
        except Exception as exc:  # Defensive Kaggle boundary: never crash agent().
            self.last_policy_error = exc
            if self._strict_runtime_errors:
                raise
            return [int(index) for index in fallback]
        finally:
            whole_act_seconds = max(0.0, self._clock() - start_time)
            self._time_manager.record_elapsed(whole_act_seconds)
            self.last_search_telemetry = replace(
                self.last_search_telemetry,
                whole_act_seconds=whole_act_seconds,
                startup_seconds=self.last_callback_startup_seconds,
                probe_seconds=self.last_probe_seconds,
                base_policy_seconds=self.last_base_policy_seconds,
                bank_spent_seconds=self._search_budget_manager.spent_seconds,
                bank_left_seconds=self._search_budget_manager.bank_left_seconds,
            )
            self._clear_policy_inference_cache()

    def _configure_policy_inference_cache(self) -> None:
        policy = self._policy
        configure = getattr(policy, "configure_inference_cache", None)
        if callable(configure):
            configure(
                enabled=self.config.search.macro.root_inference_cache_enabled,
            )

    def _clear_policy_inference_cache(self) -> None:
        clear = getattr(self._policy, "clear_inference_cache", None)
        if callable(clear):
            clear()

    def _act_inner(self, observation: Any, *, act_started_at: float) -> list[int]:
        self._prewarm_startup(observation)
        select = _select_from_observation(observation)
        if select is None:
            deck = self._load_deck()
            self._context.reset(own_deck=deck)
            self._reset_policy_episode()
            return deck

        context_features = self._context.update(observation)
        if not self._recurrent_ppo_only_runtime and not self._simple_stateless_runtime:
            context_features = self._augment_belief_features(
                observation,
                context_features,
            )
        policy_observation = observation_with_context(observation, context_features)
        observe_public = getattr(
            self._policy,
            "observe_public_observation",
            None,
        )
        if callable(observe_public):
            observe_public(policy_observation)

        action = forced_action(select)
        if action is not None:
            self.last_base_action = action
            return [int(index) for index in action]

        decision_token = self._context.prepare_decision()
        try:
            budget = self._time_manager.budget_for(observation)
            self.last_budget = budget
            bind_runtime_context = getattr(
                self._policy,
                "bind_runtime_context",
                None,
            )
            if callable(bind_runtime_context):
                bind_runtime_context(
                    context_snapshot=self._context.snapshot(),
                    context_features=context_features,
                    deadline_monotonic=act_started_at + budget.budget_seconds,
                )
            probe_result: RuntimeProbeResult | None = None
            if (
                not self._recurrent_ppo_only_runtime
                and not self._simple_stateless_runtime
                and budget.allow_search
                and self.config.search.enabled
            ):
                policy_observation, probe_result = self._with_runtime_probe_features(
                    policy_observation,
                    context_features=context_features,
                )
            served_action: tuple[int, ...] | None = None
            if self._should_try_policy(budget):
                policy_action = self._safe_policy_action(
                    policy_observation,
                    select,
                    probe_result=probe_result,
                )
                if policy_action is not None:
                    self.last_base_action = policy_action
                    served_action = self._run_macro_search(
                        policy_observation,
                        greedy_action=policy_action,
                        context_features=context_features,
                        budget=budget,
                        act_started_at=act_started_at,
                        probe_result=probe_result,
                    )
            if served_action is None:
                if self._strict_runtime_errors:
                    raise RuntimeError(
                        "strict runtime could not produce a legal policy action"
                    )
                self._warn_random_fallback()
                served_action = random_legal_action(select, rng=self._rng)
                self.last_base_action = served_action
            if not is_legal_action(select, served_action):
                raise RuntimeError("deployment adapter selected an illegal action")
            self._commit_runtime_decision(decision_token)
            return [int(index) for index in served_action]
        except Exception:
            self._abort_runtime_decision(decision_token)
            raise

    def _run_macro_search(
        self,
        observation: Any,
        *,
        greedy_action: tuple[int, ...],
        context_features: GameContextFeatures,
        budget: ActTimeBudget,
        act_started_at: float,
        probe_result: RuntimeProbeResult | None,
    ) -> tuple[int, ...]:
        """Run paired search and apply the configured complete-action policy."""
        macro = self.config.search.macro
        select = _select_from_observation(observation)
        if macro.mode == "disabled":
            return greedy_action
        if macro.mode == "conditioned":
            if not _search_conditioning_eligible(select, probe_result):
                self.last_search_telemetry = replace(
                    self.last_search_telemetry,
                    stop_reason="not_conditioning_eligible",
                )
                return greedy_action
        elif _int_field(select, "context", -1) != int(SelectContext.MAIN):
            return greedy_action
        policy = self._policy
        if not _supports_macro_search(policy):
            self.last_search_telemetry = replace(
                self.last_search_telemetry,
                stop_reason="policy_surface_unavailable",
            )
            return greedy_action
        plan = self._search_budget_manager.plan(
            remaining_overage_time=budget.remaining_overage_time,
            act_started_at=act_started_at,
            quota_class=self._search_quota_class(observation),
            now=self._clock(),
        )
        if not plan.can_start(
            self._clock(),
            call_guard_seconds=macro.budget.uninterruptible_guard_seconds,
        ):
            self.last_search_telemetry = replace(
                self.last_search_telemetry,
                planned_quota_seconds=plan.quota_seconds,
                stop_reason=plan.stop_reason or "call_guard",
            )
            return greedy_action

        result = None
        charge = self._search_budget_manager.track_search()
        try:
            with charge:
                opponent_card_probs, opponent_hand_weights = (
                    self._belief_head_distributions(observation)
                )
                searcher = PairedMacroSearcher(
                    policy=cast(MacroSearchPolicy, policy),
                    sampler=cast(BeliefSampler, self._load_belief_sampler()),
                    your_deck=self._load_deck(),
                    context_snapshot=self._context.snapshot(),
                    root_context_features=context_features,
                    belief_producer=self._load_belief_producer(),
                    config=macro,
                    rng=self._search_rng,
                    opponent_card_probs=opponent_card_probs,
                    opponent_hand_weights=opponent_hand_weights,
                    clock=self._clock,
                )
                result = searcher.run(
                    observation,
                    greedy_action=greedy_action,
                    deadline=plan.search_soft_deadline,
                )
            self.last_search_error = None
        except Exception as exc:
            self.last_search_error = exc
            if self._strict_runtime_errors:
                raise

        actual_search_seconds = charge.elapsed_seconds
        deadline_overshoot = max(0.0, self._clock() - plan.search_soft_deadline)
        selected_score = result.decision.selected_score if result is not None else None
        recommended_action_changed = bool(
            result is not None and result.decision.action_changed
        )
        if macro.mode == "conditioned":
            served_action, override_gate_reason = self._conditioned_search_action(
                observation,
                select=select,
                greedy_action=greedy_action,
                result=result,
                plan=plan,
                deadline_overshoot=deadline_overshoot,
            )
            recommended_action_changed = served_action != greedy_action
        else:
            override_gate_reason = _macro_override_gate_reason(
                mode=macro.mode,
                result=result,
                select=select,
                plan=plan,
                now=self._clock(),
                deadline_overshoot=deadline_overshoot,
                bank_spent=self._search_budget_manager.spent_seconds,
                return_guard_seconds=macro.budget.return_guard_seconds,
            )
            can_apply_override = override_gate_reason == "applied"
            served_action = (
                normalize_action_order(select, result.decision.selected_action)
                if can_apply_override and result is not None
                else greedy_action
            )
            if served_action != greedy_action:
                self.last_override_reason = "paired_macro"
        self.last_search_telemetry = SearchActTelemetry(
            enabled=True,
            shadow_only=macro.mode == "shadow",
            planned_quota_seconds=plan.quota_seconds,
            actual_search_seconds=actual_search_seconds,
            bank_spent_seconds=self._search_budget_manager.spent_seconds,
            bank_left_seconds=self._search_budget_manager.bank_left_seconds,
            deadline_overshoot_seconds=deadline_overshoot,
            stop_reason=(
                result.stop_reason
                if result is not None
                else f"search_error:{type(self.last_search_error).__name__}"
            ),
            candidates=len(result.candidates.actions) if result is not None else 0,
            worlds_requested=result.worlds_requested if result is not None else 0,
            worlds_completed=result.worlds_completed if result is not None else 0,
            transitions=result.transitions if result is not None else 0,
            engine_sessions=result.engine_sessions if result is not None else 0,
            state_pool_peak=result.state_pool_peak if result is not None else 0,
            state_leaks=result.state_leaks if result is not None else 0,
            same_seat_value_rows=(
                result.same_seat_value_rows if result is not None else 0
            ),
            handoff_value_rows=(result.handoff_value_rows if result is not None else 0),
            selection_reason=(
                "search_conditioned_model"
                if macro.mode == "conditioned"
                and override_gate_reason == "conditioned_applied"
                else result.decision.reason
                if result is not None
                else "search_error"
            ),
            override_gate_reason=override_gate_reason,
            recommended_action_changed=recommended_action_changed,
            selected_mean_delta=(
                selected_score.mean_delta if selected_score is not None else None
            ),
            selected_robust_delta=(
                selected_score.robust_delta if selected_score is not None else None
            ),
            selected_downside_cvar=(
                selected_score.downside_cvar if selected_score is not None else None
            ),
            remaining_overage_time=budget.remaining_overage_time,
            search_start_remaining_overage_time=budget.remaining_overage_time,
            fallback_available=True,
            action_changed=served_action != greedy_action,
        )
        return served_action

    def _conditioned_search_action(
        self,
        observation: Any,
        *,
        select: Any,
        greedy_action: tuple[int, ...],
        result: MacroSearchResult | None,
        plan: SearchBudgetPlan,
        deadline_overshoot: float,
    ) -> tuple[tuple[int, ...], str]:
        """Apply model conditioning only to complete, in-budget evidence."""
        gate_reason = _conditioned_search_gate_reason(
            result=result,
            plan=plan,
            now=self._clock(),
            deadline_overshoot=deadline_overshoot,
            bank_spent=self._search_budget_manager.spent_seconds,
            return_guard_seconds=self.config.search.macro.budget.return_guard_seconds,
        )
        if gate_reason is not None or result is None:
            return (greedy_action, gate_reason or "search_error")
        select_space = describe_prompt_action_space(select)
        evidence = search_evidence_from_macro_result(
            result,
            legal_action_count=select_space.legal_action_count,
            config=self.config.search.macro.rerank,
        )
        if evidence is None:
            return (greedy_action, "invalid_search_evidence")
        conditioned_select = getattr(
            self._policy,
            "select_search_conditioned_action",
            None,
        )
        if not callable(conditioned_select):
            return (greedy_action, "conditioned_policy_surface_unavailable")
        try:
            selected = normalize_action_order(
                select,
                conditioned_select(observation, evidence),
            )
        except Exception as exc:
            self.last_search_error = exc
            return (greedy_action, f"conditioned_policy_error:{type(exc).__name__}")
        if selected not in evidence.actions:
            return (greedy_action, "conditioned_action_not_in_candidates")
        if not is_legal_action(select, selected):
            return (greedy_action, "illegal_conditioned_action")
        if selected != greedy_action:
            self.last_override_reason = "search_conditioned_model"
        return (selected, "conditioned_applied")

    def _search_quota_class(self, observation: Any) -> QuotaClass:
        """Classify first/later high-value MAINs without reimplementing rules."""
        select = _select_from_observation(observation)
        option_types = {
            _int_field(option, "type", -1)
            for option in _sequence(_field(select, "option", ()))
        }
        high_value_types = {
            int(OptionType.PLAY),
            int(OptionType.ATTACH),
            int(OptionType.EVOLVE),
            int(OptionType.ABILITY),
            int(OptionType.RETREAT),
            int(OptionType.ATTACK),
        }
        if not option_types & high_value_types:
            return "ordinary_main"
        turn = _int_field(_field(observation, "current"), "turn", -1)
        if self._search_high_value_turn != turn:
            self._search_high_value_turn = turn
            return "first_high_value_main"
        return "later_high_value_main"

    def _warn_random_fallback(self) -> None:
        """Emit one stderr warning the first time a non-forced prompt falls back."""
        if self._warned_random_fallback:
            return
        self._warned_random_fallback = True
        print(
            "ptcg_rl runtime warning: legal policy action unavailable, using random "
            f"fallback (checkpoint_path={self.config.checkpoint_path}, "
            f"policy_error={self.last_policy_error!r})",
            file=sys.stderr,
            flush=True,
        )

    def runtime_status(self) -> dict[str, Any]:
        """Return policy readiness details for submission validation."""
        belief_producer = (
            None if self._recurrent_ppo_only_runtime else self._load_belief_producer()
        )
        belief_prior_decks = (
            len(belief_producer.prior.decks)
            if belief_producer is not None and belief_producer.prior is not None
            else 0
        )
        policy = self._policy
        return {
            "checkpoint_path": (
                str(self.config.checkpoint_path)
                if self.config.checkpoint_path is not None
                else None
            ),
            "belief_summary_path": (
                str(self.config.belief.deck_signature_summary_path)
                if self.config.belief.deck_signature_summary_path is not None
                else None
            ),
            "belief_prior_decks": belief_prior_decks,
            "belief_error": (
                repr(self.last_belief_error)
                if self.last_belief_error is not None
                else None
            ),
            "policy_loaded": self._policy is not None,
            "recurrent_ppo_only_runtime": self._recurrent_ppo_only_runtime,
            "simple_stateless_runtime": self._simple_stateless_runtime,
            "fixed_deck_digest": getattr(policy, "fixed_deck_digest", None),
            "public_deck_catalog_fingerprint": getattr(
                policy,
                "public_deck_catalog_fingerprint",
                None,
            ),
            "planner_configured": self.config.planner is not None,
            "planner_enabled": bool(getattr(policy, "planner_enabled", False)),
            "planner_fingerprint": (
                None
                if self.config.planner is None
                else self.config.planner.expected_planner_fingerprint
            ),
            "planner_runtime_fingerprint": (
                None
                if self.config.planner is None
                else self.config.planner.expected_runtime_fingerprint
            ),
            "deck_conditioning_enabled": bool(
                getattr(policy, "deck_conditioning_enabled", False)
            ),
            "bound_deck_signature": getattr(
                policy,
                "own_deck_signature",
                None,
            ),
            "selected_private_profile_module_key": getattr(
                policy,
                "selected_private_profile_module_key",
                None,
            ),
            "checkpoint_registry_sha256": getattr(
                policy,
                "checkpoint_registry_sha256",
                None,
            ),
            "packaged_private_profile_count": int(
                getattr(policy, "packaged_private_profile_count", 0)
            ),
            "used_random_fallback": self._warned_random_fallback,
            "strict_runtime_errors": self._strict_runtime_errors,
            "prewarm_on_startup": self.config.prewarm_on_startup,
            "prewarm_engine": self.config.prewarm_engine,
            "prewarm_checkpoint": self.config.prewarm_checkpoint,
            "prewarm_policy_forward": self.config.prewarm_policy_forward,
            "prewarm_error": repr(self.last_prewarm_error)
            if self.last_prewarm_error is not None
            else None,
            "engine_prewarm_error": repr(self.last_engine_prewarm_error)
            if self.last_engine_prewarm_error is not None
            else None,
            "policy_error": repr(self.last_policy_error)
            if self.last_policy_error is not None
            else None,
            "search_error": repr(self.last_search_error)
            if self.last_search_error is not None
            else None,
            "search_telemetry": self.last_search_telemetry.as_dict(),
        }

    def last_act_telemetry(self) -> dict[str, Any]:
        """Return the latest compact callback telemetry for arena collection."""
        result = self.last_search_telemetry.as_dict()
        planner_decision = getattr(self._policy, "last_decision", None)
        result.update(
            {
                "planner_configured": self.config.planner is not None,
                "planner_used": bool(
                    planner_decision is not None
                    and not planner_decision.used_base_trace
                ),
                "planner_fallback_reason": (
                    None
                    if planner_decision is None
                    else planner_decision.planner_behavior.fallback_reason
                ),
            }
        )
        batch_telemetry = getattr(
            self._policy,
            "last_inference_batch_telemetry",
            None,
        )
        if callable(batch_telemetry):
            result.update(batch_telemetry())
        return result

    def close(self) -> None:
        """Drain optional packaged planner resources exactly once."""
        if self._closed:
            return
        self._closed = True
        self._reset_policy_episode()
        close_policy = getattr(self._policy, "close", None)
        if callable(close_policy):
            close_policy()

    def _commit_runtime_decision(self, token: PublicEventDecisionToken) -> None:
        """Commit a recurrent proposal immediately before returning to Kaggle."""
        policy = self._policy
        if not bool(getattr(policy, "recurrent_enabled", False)):
            return
        # Validate the context token before changing either side of the lease.
        self._context.abort_decision(token)
        commit = getattr(policy, "commit_runtime_decision", None)
        if not callable(commit):
            raise RuntimeError("recurrent policy has no deployment commit adapter")
        commit(token.generation)
        self._context.commit_decision(token)

    def _abort_runtime_decision(self, token: PublicEventDecisionToken) -> None:
        """Retain public events and discard any unserved recurrent proposal."""
        abort = getattr(self._policy, "abort_runtime_decision", None)
        if callable(abort):
            abort()
        self._context.abort_decision(token)

    def _reset_policy_episode(self) -> None:
        """Clear recurrent state at registration, explicit reset, and shutdown."""
        reset = getattr(self._policy, "reset_runtime_episode", None)
        if callable(reset):
            reset()

    def _should_try_policy(self, budget: ActTimeBudget) -> bool:
        if self._policy is not None:
            return True
        if self.config.checkpoint_path is None:
            return False
        return budget.allow_model_load

    def _safe_policy_action(
        self,
        observation: Any,
        select: Any,
        *,
        probe_result: RuntimeProbeResult | None,
    ) -> tuple[int, ...] | None:
        start_time = self._clock()
        try:
            action = self._policy_action(observation)
        except Exception as exc:
            self.last_base_policy_seconds = self._clock() - start_time
            self.last_policy_error = exc
            if self._strict_runtime_errors:
                raise
            return None
        self.last_base_policy_seconds = self._clock() - start_time
        if not is_legal_action(select, action):
            if self._strict_runtime_errors:
                raise ValueError("policy returned an illegal action")
            return None
        action = normalize_action_order(select, action)
        return self._maybe_apply_engine_override(
            observation,
            select,
            action,
            probe_result=probe_result,
        )

    def _augment_belief_features(
        self,
        observation: Any,
        context_features: GameContextFeatures,
    ) -> GameContextFeatures:
        producer = self._load_belief_producer()
        if producer is None:
            return context_features
        return producer.augment(observation, context_features)

    def _with_runtime_probe_features(
        self,
        observation: Any,
        *,
        context_features: GameContextFeatures,
    ) -> tuple[Any, RuntimeProbeResult | None]:
        start_time = self._clock()
        try:
            result = self._runtime_probe_features(observation, context_features)
        except Exception as exc:
            self.last_probe_error = exc
            self.last_probe_seconds = self._clock() - start_time
            return observation, None
        if result is None:
            self.last_probe_error = None
            self.last_probe_seconds = self._clock() - start_time
            return observation, None
        self.last_probe_error = None
        self.last_probe_seconds = self._clock() - start_time
        return observation_with_probe_features(observation, result), result

    def _runtime_probe_features(
        self,
        observation: Any,
        context_features: GameContextFeatures,
    ) -> RuntimeProbeResult | None:
        sampler = self._load_belief_sampler()
        if sampler is None:
            return None
        opponent_card_probs, opponent_hand_weights = self._belief_head_distributions(
            observation
        )
        return run_runtime_probe_features(
            observation,
            context_features,
            your_deck=self._load_deck(),
            sampler=sampler,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=self._rng,
            config=self.config.search,
        )

    def _belief_head_distributions(
        self,
        observation: Any,
    ) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        if self.config.search.sampler.mode != "model":
            return None, None
        policy = self._policy
        if policy is None and self.config.checkpoint_path is not None:
            policy = self._load_checkpoint_policy()
            self._policy = policy
        belief_distributions = getattr(policy, "belief_distributions", None)
        if not callable(belief_distributions):
            return None, None
        distributions = belief_distributions(observation)
        if distributions is None:
            return None, None
        return cast(
            tuple[tuple[float, ...] | None, tuple[float, ...] | None],
            distributions,
        )

    def _maybe_apply_engine_override(
        self,
        observation: Any,
        select: Any,
        greedy_action: tuple[int, ...],
        *,
        probe_result: RuntimeProbeResult | None,
    ) -> tuple[int, ...]:
        self.last_override_reason = None
        if probe_result is None or not self.config.search.conservative_override_enabled:
            return greedy_action
        top_actions = self._rank_policy_actions(observation)
        if greedy_action not in top_actions:
            top_actions = (greedy_action, *top_actions)
        for action in top_actions:
            if (
                is_legal_action(select, action)
                and is_core_action(select, action)
                and all_worlds_verified_lethal(action, probe_result)
            ):
                self.last_override_reason = "verified_lethal"
                return normalize_action_order(select, action)
        if is_core_action(select, greedy_action) and all_worlds_verified_self_loss(
            greedy_action, probe_result
        ):
            for action in top_actions:
                if action == greedy_action or not is_legal_action(select, action):
                    continue
                if not all_worlds_verified_self_loss(action, probe_result):
                    self.last_override_reason = "avoid_verified_self_loss"
                    return normalize_action_order(select, action)
        return greedy_action

    def _rank_policy_actions(self, observation: Any) -> tuple[tuple[int, ...], ...]:
        rank_actions = getattr(self._policy, "rank_actions", None)
        if not callable(rank_actions):
            return ()
        return tuple(
            tuple(int(index) for index in action)
            for action in rank_actions(
                observation,
                top_k=self.config.search.top_k,
            )
        )

    def _policy_action(self, observation: Any) -> tuple[int, ...]:
        policy = self._policy
        if policy is None:
            policy = self._load_checkpoint_policy()
            if policy is None:
                return ()
            self._policy = policy
        return policy.select_action(observation)

    def _load_belief_producer(self) -> OpponentBeliefFeatureProducer | None:
        if not self.config.belief.enabled:
            return None
        if self._belief_producer is None:
            belief_config = self.config.belief
            if belief_config.deck_signature_summary_path is not None:
                belief_config = belief_config.model_copy(
                    update={
                        "deck_signature_summary_path": _resolve_packaged_path(
                            belief_config.deck_signature_summary_path
                        ),
                    }
                )
            try:
                self._belief_producer = OpponentBeliefFeatureProducer.from_config(
                    belief_config
                )
                self.last_belief_error = None
            except Exception as exc:
                # A broken/missing prior must never break act(); belief tokens
                # degrade to empty and validation surfaces the error.
                self.last_belief_error = exc
                if self._strict_runtime_errors:
                    raise
                self._belief_producer = OpponentBeliefFeatureProducer(
                    prior=None,
                    top_k=belief_config.top_k,
                    enabled=belief_config.enabled,
                )
        return self._belief_producer

    def _load_belief_sampler(self) -> BeliefSampler | None:
        if self._belief_sampler is None:
            sampler_config = self.config.search.sampler
            if sampler_config.prior_deck_signature_summary_path is not None:
                sampler_config = sampler_config.model_copy(
                    update={
                        "prior_deck_signature_summary_path": _resolve_packaged_path(
                            sampler_config.prior_deck_signature_summary_path
                        ),
                    }
                )
            self._belief_sampler = BeliefSampler(config=sampler_config)
        return self._belief_sampler

    def _prewarm_startup(self, observation: Any) -> None:
        if self._startup_prewarmed or not self.config.prewarm_on_startup:
            return
        start_time = self._clock()
        # Engine and checkpoint prewarm are isolated so a missing/broken engine
        # module can never block loading the checkpoint policy.
        if self.config.prewarm_engine:
            try:
                _prewarm_engine()
            except Exception as exc:
                self.last_engine_prewarm_error = exc
                if self._strict_runtime_errors:
                    raise
        try:
            if self.config.prewarm_checkpoint:
                policy = self._policy
                if policy is None and _can_prewarm_model(observation, self.config):
                    policy = self._load_checkpoint_policy()
                if policy is not None:
                    self._policy = policy
                    if self.config.prewarm_policy_forward:
                        _prewarm_policy(policy)
        except Exception as exc:
            self.last_prewarm_error = exc
            if self._strict_runtime_errors:
                raise
        finally:
            self.last_prewarm_seconds = self._clock() - start_time
            self.last_callback_startup_seconds = self.last_prewarm_seconds
            self._startup_prewarmed = True

    def _load_checkpoint_policy(self) -> SelectPolicy | None:
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path is None:
            return None
        resolved_checkpoint = _resolve_packaged_path(checkpoint_path)
        stateless_catalog_path = _env_simple_stateless_catalog_path()
        if stateless_catalog_path is not None:
            from ptcg_rl.agent.simple_stateless_runtime import (
                FixedDeckStatelessPolicy,
            )

            fixed_policy = FixedDeckStatelessPolicy(
                resolved_checkpoint,
                public_catalog_manifest_path=stateless_catalog_path,
                own_deck=self._load_deck(),
            )
            self._configure_deployment_policy(fixed_policy)
            self._simple_stateless_runtime = True
            return fixed_policy
        legacy_policy = CheckpointPolicy(
            resolved_checkpoint,
            own_deck=self._load_deck(),
        )
        legacy_policy.configure_inference_cache(
            enabled=self.config.search.macro.root_inference_cache_enabled,
        )
        self._configure_deployment_policy(legacy_policy)
        planner = self.config.planner
        if bool(getattr(legacy_policy, "recurrent_enabled", False)):
            if planner is not None:
                raise ValueError(
                    "recurrent PPO checkpoint cannot use a runtime planner"
                )
            self._recurrent_ppo_only_runtime = True
        if planner is None:
            return legacy_policy
        wrapped, _runtime = _build_packaged_planner_policy(legacy_policy, planner)
        wrapped.bind_own_deck(self._load_deck())
        return cast(SelectPolicy, wrapped)

    def _configure_deployment_policy(self, policy: SelectPolicy | None = None) -> None:
        """Apply runtime-owned sampling semantics to a loaded policy."""
        target = self._policy if policy is None else policy
        configure = getattr(target, "configure_deployment_sampling", None)
        if callable(configure):
            configure(
                temperature=self.config.policy_temperature,
                seed=self.config.seed,
            )

    def _load_deck(self) -> list[int]:
        if self._deck is None:
            path = self.config.deck_path or _default_deck_path()
            self._deck = _read_deck(path)
            bind_deck = getattr(self._policy, "bind_own_deck", None)
            if callable(bind_deck):
                bind_deck(self._deck)
        return list(self._deck)


class RandomFallbackAgent:
    """A defensive baseline agent that never intentionally returns illegal selects."""

    def __init__(
        self,
        *,
        deck_path: Path | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize with a fixed deck and deterministic RNG."""
        self._runtime = PolicyRuntimeAgent(deck_path=deck_path, seed=seed)

    def act(self, observation: Any, configuration: Any | None = None) -> list[int]:
        """Return a legal deck registration or select action."""
        return self._runtime.act(observation, configuration)


def _select_from_observation(observation: Any) -> Any:
    if isinstance(observation, Mapping):
        return observation.get("select")
    return getattr(observation, "select", None)


def _first_legal_action(select: Any) -> tuple[int, ...]:
    """Return a deterministic legal fallback without invoking model or engine."""
    options = _sequence(_field(select, "option", ()))
    min_count = min(len(options), max(0, _int_field(select, "minCount", 0)))
    return normalize_action_order(select, tuple(range(min_count)))


def _supports_macro_search(policy: Any) -> bool:
    """Return whether a policy exposes proposal, continuation, and value calls."""
    return policy is not None and all(
        callable(getattr(policy, name, None))
        for name in ("select_action", "rank_actions", "value")
    )


def _search_conditioning_eligible(
    select: Any,
    probe_result: RuntimeProbeResult | None,
) -> bool:
    """Select generic prompts where complete-action evidence adds information."""
    if select is None:
        return False
    if _int_field(select, "context", -1) == int(SelectContext.MAIN):
        return True
    if probe_result is not None and probe_result.unresolved_options > 0:
        return True
    space = describe_prompt_action_space(select)
    return bool(
        space.ordered or space.max_count > 1 or space.min_count != space.max_count
    )


def _conditioned_search_gate_reason(
    *,
    result: MacroSearchResult | None,
    plan: SearchBudgetPlan,
    now: float,
    deadline_overshoot: float,
    bank_spent: float,
    return_guard_seconds: float,
) -> str | None:
    """Reject only incomplete, unsafe, or out-of-budget serving evidence."""
    if result is None:
        return "search_error"
    if result.stop_reason != "complete" or not result.complete_coverage:
        return result.stop_reason
    if result.worlds_completed != result.worlds_requested:
        return "unpaired_world_coverage"
    if result.state_leaks != 0:
        return "state_leak"
    if bank_spent > plan.bank_before_seconds + plan.bank_left_seconds:
        return "search_bank_exceeded"
    if deadline_overshoot > 0.0:
        return "deadline_overshoot"
    if now + return_guard_seconds >= plan.whole_act_deadline:
        return "whole_act_deadline"
    return None


def _macro_override_gate_reason(
    *,
    mode: str,
    result: MacroSearchResult | None,
    select: Any,
    plan: SearchBudgetPlan,
    now: float,
    deadline_overshoot: float,
    bank_spent: float,
    return_guard_seconds: float,
) -> str:
    """Classify every runtime safety check before an action switch."""
    if mode != "override":
        return "shadow"
    if result is None:
        return "search_error"
    if result.stop_reason != "complete":
        return result.stop_reason
    if not result.complete_coverage:
        return "incomplete_coverage"
    if result.worlds_completed != result.worlds_requested:
        return "unpaired_world_coverage"
    if result.state_leaks != 0:
        return "state_leak"
    if result.state_pool_peak > 128:
        return "state_pool_limit"
    if bank_spent > plan.bank_before_seconds + plan.bank_left_seconds:
        return "search_bank_exceeded"
    if deadline_overshoot > 0.0:
        return "deadline_overshoot"
    if now + return_guard_seconds >= plan.whole_act_deadline:
        return "whole_act_deadline"
    if not result.decision.action_changed:
        return "no_recommendation"
    if not is_legal_action(select, result.decision.selected_action):
        return "illegal_recommendation"
    return "applied"


def _default_deck_path() -> Path:
    packaged_path = Path("deck.csv")
    if packaged_path.exists():
        return packaged_path
    package_path = _package_root() / "deck.csv"
    if package_path.exists():
        return package_path
    return (
        Path(__file__).resolve().parents[3] / "data" / "sample_submission" / "deck.csv"
    )


def _read_deck(path: Path) -> list[int]:
    deck = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 card IDs: {path}")
    return deck


def _file_sha256(path: Path) -> str:
    """Hash one immutable checkpoint without materializing it again."""
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _flat_tensor(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    from ptcg_rl.checkpoint_storage import unpack_checkpoint_state_dict

    return _strip_lightning_model_prefix(unpack_checkpoint_state_dict(checkpoint))


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_model_config(checkpoint: Any) -> Any | None:
    from ptcg_rl.model import AgentNetworkConfig

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


def _prewarm_engine() -> None:
    from ptcg_rl.engine.runtime import load_cg_api, load_cg_game

    load_cg_api()
    load_cg_game()


def _prewarm_policy(policy: SelectPolicy) -> None:
    prewarm = getattr(policy, "prewarm", None)
    if callable(prewarm):
        prewarm()


def _terminal_value(observation: Any, root_player_index: int) -> float | None:
    result = _int_field(_field(observation, "current"), "result", -1)
    if result < 0:
        return None
    if result == int(root_player_index):
        return 1.0
    if result == 2:
        return 0.0
    return -1.0


def _can_prewarm_model(observation: Any, config: ActTimeConfig) -> bool:
    if config.checkpoint_path is None:
        return False
    remaining = _float_field(
        observation,
        "remainingOverageTime",
        config.default_remaining_overage_time,
    )
    return remaining > config.critical_overage_seconds


def _dummy_policy_observation() -> dict[str, Any]:
    return {
        "remainingOverageTime": 600.0,
        "current": {
            "turn": 0,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": [],
            "players": [
                _dummy_player_state(),
                _dummy_player_state(),
            ],
        },
        "logs": [],
        "gameContext": GameContextFeatures().as_observation_dict(),
        "search_begin_input": None,
        "select": {
            "type": 9,
            "context": 41,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "contextCard": None,
            "effect": None,
            "deck": [],
            "option": [{"type": 1}, {"type": 2}],
        },
        "step": 0,
    }


def _dummy_player_state() -> dict[str, Any]:
    return {
        "active": [],
        "bench": [],
        "benchMax": 8,
        "deckCount": 53,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 1,
        "hand": [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _estimate_remaining_decisions(observation: Any, floor: int) -> int:
    current = _field(observation, "current")
    players = _sequence(_field(current, "players", ()))
    prize_count = 0
    for player in players:
        prizes = _sequence(_field(player, "prize", ()))
        prize_count += len(prizes) if prizes else 6
    if prize_count <= 0:
        prize_count = 12
    return max(floor, min(150, floor + prize_count * 8))


def _resolve_packaged_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    for root in (Path.cwd(), _package_root(), _repo_root()):
        packaged_path = root / path
        if packaged_path.exists():
            return packaged_path
    return Path(__file__).resolve().parents[3] / path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_checkpoint_path() -> Path | None:
    env_path = _optional_env_path("PTCG_RL_CHECKPOINT_PATH")
    if env_path is not None:
        return env_path
    # Kaggle imports agents with cwd outside the extracted archive, so packaged
    # assets must also be probed relative to the package/repo roots.
    for filename in ("agent_checkpoint.pt", "model.pt", "checkpoint.pt"):
        candidate = _resolve_packaged_path(Path(filename))
        if candidate.exists():
            return candidate
    return None


def _env_belief_summary_path() -> Path | None:
    env_path = _optional_env_path("PTCG_RL_BELIEF_SUMMARY_PATH")
    if env_path is not None:
        return env_path
    candidate = _resolve_packaged_path(Path("belief_prior.csv"))
    if candidate.exists():
        return candidate
    return None


def _env_simple_stateless_catalog_path() -> Path | None:
    env_path = _optional_env_path("PTCG_RL_PUBLIC_DECK_CATALOG_PATH")
    if env_path is not None:
        resolved = _resolve_packaged_path(env_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"public deck catalog manifest not found: {resolved}"
            )
        return resolved
    candidate = _resolve_packaged_path(Path("public_catalog/manifest.json"))
    return candidate if candidate.is_file() else None


def _deployment_policy_temperature() -> float:
    """Resolve an explicit env override or the immutable packaged setting."""
    raw = os.environ.get("PTCG_RL_POLICY_TEMPERATURE")
    if raw is not None and raw.strip():
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError("PTCG_RL_POLICY_TEMPERATURE must be a float") from error
    path = _resolve_packaged_path(Path("deployment_runtime.json"))
    if not path.is_file():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("packaged deployment runtime is invalid") from error
    if not isinstance(payload, Mapping) or payload.get("format") != (
        "ptcg_rl_deployment_runtime_v1"
    ):
        raise ValueError("packaged deployment runtime has an invalid format")
    value = payload.get("policy_temperature")
    if not isinstance(value, int | float):
        raise ValueError("packaged policy_temperature must be numeric")
    return float(value)


def _env_planner_runtime_path() -> Path | None:
    env_path = _optional_env_path("PTCG_RL_PLANNER_RUNTIME_PATH")
    if env_path is not None:
        resolved = _resolve_packaged_path(env_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"packaged planner config not found: {resolved}")
        return resolved
    candidate = _resolve_packaged_path(Path("planner_runtime.json"))
    return candidate if candidate.is_file() else None


def _load_packaged_planner_config(path: Path) -> Any:
    from ptcg_rl.agent.packaged_planner import PackagedPlannerConfig

    expected = os.environ.get("PTCG_RL_PLANNER_RUNTIME_SHA256")
    if expected is not None and expected.strip():
        actual = _file_sha256(path)
        if actual != expected.strip():
            raise ValueError("packaged planner config fingerprint differs from env")
    return PackagedPlannerConfig.from_file(path)


def _build_packaged_planner_policy(policy: Any, config: Any) -> tuple[Any, Any]:
    from ptcg_rl.agent.packaged_planner import build_packaged_planner_policy

    return build_packaged_planner_policy(policy, config)


def _profile_planner_enabled(planner_enabled_by_default: bool) -> bool:
    """Apply only a safe planner-disable override to an immutable package."""
    raw = os.environ.get("PTCG_RL_PLANNER_PROFILE_ENABLED")
    if raw is None or not raw.strip():
        return planner_enabled_by_default
    canonical = raw.strip()
    if canonical not in {"0", "1"}:
        raise ValueError("PTCG_RL_PLANNER_PROFILE_ENABLED must be 0 or 1")
    requested = canonical == "1"
    if requested and not planner_enabled_by_default:
        raise ValueError(
            "PTCG_RL_PLANNER_PROFILE_ENABLED cannot enable an immutable "
            "planner-off package"
        )
    return requested


def _optional_env_path(name: str) -> Path | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return Path(raw_value)


def _optional_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default


def _float_field(value: Any, name: str, default: float) -> float:
    field_value = _field(value, name, default)
    return float(field_value) if field_value is not None else default


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()
