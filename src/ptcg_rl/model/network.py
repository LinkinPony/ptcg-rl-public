"""Policy/value network wiring for the Kaggle select agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

from ptcg_rl.agent.search.proposal_generation import (
    PlannerProposalBatchResult,
    PlannerProposalPrefixQuery,
    PlannerProposalProblem,
    PlannerProposalSearchLimits,
    ensure_planner_proposal_deadline,
    generate_batched_planner_proposals,
)
from ptcg_rl.cards.card_encoder import (
    CardEncoder,
    CardEncoderConfig,
    build_card_encoder,
)
from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS
from ptcg_rl.context import PublicEventBatch, select_public_event_batch_rows
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.model.action_value import (
    ActionValueHeadConfig,
    ActionValuePrediction,
    CompleteActionValueHead,
    wdl_expected_score,
)
from ptcg_rl.model.compositional_capsule import (
    ExactDeckCapsule,
    FixedExactDeckCapsule,
    exact_capsule,
)
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
    DeckConditioningConfig,
    DeckEncoder,
    DeckRoutePlan,
    PrivateScalarResidual,
    apply_private_residual,
    encode_deck_compositions,
    private_residual_modules,
    private_scalar_modules,
    resolve_deck_route_plan,
)
from ptcg_rl.model.macro_outcome import (
    MacroOutcomeHeadConfig,
    MacroOutcomeHeads,
    MacroOutcomePrediction,
)
from ptcg_rl.model.policy import (
    OptionBatch,
    PointerPolicyConfig,
    PointerPolicyHead,
    TeacherForcedActionScores,
    TeacherForcedEvaluation,
    actions_from_decode_tensors,
)
from ptcg_rl.model.private_strategy import (
    DensePrivatePrefixValueHead,
    DensePrivateRootValueHead,
)
from ptcg_rl.model.recurrent import (
    RecurrentPolicyConfig,
    RecurrentPolicyCore,
    RecurrentPolicyState,
)
from ptcg_rl.model.root_perspective_value import (
    RootPerspectiveValueAdapter,
    RootPerspectiveValueAdapterConfig,
)
from ptcg_rl.model.search_reranker import (
    SEARCH_RERANKER_ARCHITECTURE_VERSION,
    SearchCandidateReranker,
)
from ptcg_rl.model.state_encoder import (
    StateBatch,
    StateEncoder,
    StateEncoderConfig,
    StateEncoderOutput,
    compositional_transformer_shapes,
)

_PREFIX_VALUE_DELTA_PARAMETER_NAMES = (
    "prefix_value_delta_head.0.weight",
    "prefix_value_delta_head.0.bias",
    "prefix_value_delta_head.2.weight",
    "prefix_value_delta_head.2.bias",
)
_SEARCH_RERANKER_STATE_PREFIX = "search_reranker."
_PLANNER_RERANKER_STATE_PREFIX = "planner_reranker."
_ROOT_PERSPECTIVE_VALUE_STATE_PREFIX = "root_perspective_value_adapter."
_MACRO_OUTCOME_STATE_PREFIX = "macro_outcome_heads."
_ACTION_VALUE_STATE_PREFIX = "action_value_head."
SCHEMA9_WARM_START_INITIALIZATION_SEED = 2026071706
SCHEMA10_WARM_START_INITIALIZATION_SEED = 2026071801
ACTION_VALUE_WARM_START_INITIALIZATION_SEED = 2026072001
DECK_CONDITIONING_STATE_PREFIXES = (
    "deck_encoder.",
    "deck_input_projection.",
    "state_encoder.private_adapters.",
    "state_encoder.private_lora.",
    "state_encoder.private_strategy_stacks.",
    "policy_head.private_lora.",
    "policy_head.private_strategies.",
    "private_policy_adapters.",
    "private_root_value_heads.",
    "private_prefix_value_heads.",
    "dense_private_root_value_heads.",
    "dense_private_prefix_value_heads.",
    "state_encoder.compositional_projections.",
    "state_encoder.compositional_film.",
    "state_encoder.compositional_shared_adapters.",
    "policy_head.compositional_linears.",
    "policy_head.compositional_option_set.",
    "policy_head.compositional_option_film.",
    "policy_head.compositional_count_set_projection.",
    "exact_capsules.",
)


class AgentNetworkConfig(BaseModel):
    """Config for the state encoder, pointer policy, and value heads."""

    model_config = ConfigDict(extra="forbid")

    state_encoder: StateEncoderConfig = StateEncoderConfig()
    policy: PointerPolicyConfig = PointerPolicyConfig()
    card_encoder: CardEncoderConfig | None = None
    value_hidden_dim: int = 64
    opponent_card_classes: int = DEFAULT_NUM_CARD_IDS
    effect_feature_size: int = DYNAMIC_EFFECT_FEATURE_SIZE
    search_reranker_architecture_version: int = SEARCH_RERANKER_ARCHITECTURE_VERSION
    planner_reranker_architecture_version: int = SEARCH_RERANKER_ARCHITECTURE_VERSION
    root_perspective_value: RootPerspectiveValueAdapterConfig = (
        RootPerspectiveValueAdapterConfig()
    )
    macro_outcome: MacroOutcomeHeadConfig = MacroOutcomeHeadConfig()
    action_value: ActionValueHeadConfig = ActionValueHeadConfig()
    deck_conditioning: DeckConditioningConfig | None = None
    recurrent: RecurrentPolicyConfig | None = None

    @field_validator("value_hidden_dim", "opponent_card_classes", "effect_feature_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive dimensions."""
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator(
        "search_reranker_architecture_version",
        "planner_reranker_architecture_version",
    )
    @classmethod
    def valid_search_reranker_version(cls, value: int) -> int:
        """Require the complete-action evidence architecture implemented here."""
        if value != SEARCH_RERANKER_ARCHITECTURE_VERSION:
            raise ValueError("unsupported search reranker architecture version")
        return value


@dataclass(frozen=True)
class AgentNetworkOutput:
    """Outputs used by BC training and act-time inference."""

    policy_logits: Tensor
    value: Tensor
    prize_diff: Tensor
    opponent_card_logits: Tensor
    opponent_hand_logits: Tensor
    effect_predictions: Tensor


@dataclass(frozen=True)
class PolicyEvaluationContext:
    """Action-independent policy tensors reusable by auxiliary objectives."""

    policy_global: Tensor
    option_embeddings: Tensor
    projected_options: Tensor
    route_plan: DeckRoutePlan | None


@dataclass(frozen=True)
class ActionEvaluation:
    """Differentiable policy/value scores for selected action sequences."""

    action_logprobs: Tensor
    entropies: Tensor
    values: Tensor
    step_logits: tuple[Tensor, ...]
    first_logits: Tensor | None = None
    token_logprobs: Tensor | None = None
    token_entropies: Tensor | None = None
    token_mask: Tensor | None = None
    prefix_values: Tensor | None = None
    stop_sampled: Tensor | None = None
    completed_action_latents: Tensor | None = None
    factual_presence_logits: Tensor | None = None
    factual_magnitude_predictions: Tensor | None = None
    factual_actor_relation_logits: Tensor | None = None
    factual_next_context_logits: Tensor | None = None
    policy_context: PolicyEvaluationContext | None = None


@dataclass(frozen=True)
class SampleDecodeTrace:
    """Materialized sampled actions plus behavior-time token evidence."""

    actions: tuple[tuple[int, ...], ...]
    action_logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor
    prefix_values: Tensor
    token_mask: Tensor
    stop_sampled: Tensor
    planner_context_handles: tuple[str, ...] = ()
    served_policy_version: int | None = None
    served_model_fingerprint: str = ""
    served_proposal_version: int | None = None
    planner_fallback_reason: str = ""


@dataclass(frozen=True)
class SampleDecodeTensorTrace:
    """Tensor-only sampled actions plus behavior-time token evidence."""

    choice_indices: Tensor
    append_masks: Tensor
    action_logprobs: Tensor
    values: Tensor
    token_logprobs: Tensor
    prefix_values: Tensor
    token_mask: Tensor
    stop_sampled: Tensor


@dataclass(frozen=True)
class SampledActionCandidates:
    """Ragged repeated policy samples without redundant value traces."""

    actions: tuple[tuple[tuple[int, ...], ...], ...]
    action_logprobs: Tensor
    candidate_counts: tuple[int, ...]


@dataclass(frozen=True)
class ConditionedStateOutput:
    """One shared state encoding plus model-relative deck routing context."""

    encoded_state: StateEncoderOutput
    deck_embedding: Tensor | None
    route_plan: DeckRoutePlan | None
    decision_embedding: Tensor | None = None


@dataclass(frozen=True)
class SearchCandidateEvaluation:
    """Flattened search-conditioned logits with decision group boundaries."""

    logits: Tensor
    base_action_logprobs: Tensor
    proposal_action_logprobs: Tensor
    reranker_residuals: Tensor
    candidate_counts: tuple[int, ...]


@dataclass(frozen=True)
class PlannerCandidateEvaluation:
    """Schema-9 candidate probabilities from an isolated planner head."""

    base_action_logprobs: Tensor
    proposal_action_logprobs: Tensor
    reranker_residuals: Tensor
    candidate_counts: tuple[int, ...]


class AgentPolicyValueNet(nn.Module):
    """Entity-set encoder with pointer policy and value-style heads."""

    def __init__(
        self,
        config: AgentNetworkConfig | None = None,
        *,
        card_encoder: CardEncoder | None = None,
    ) -> None:
        """Initialize the policy/value network."""
        super().__init__()
        self.config = config or AgentNetworkConfig()
        if self.config.state_encoder.d_model != self.config.policy.d_model:
            raise ValueError("state_encoder.d_model must match policy.d_model")

        d_model = self.config.state_encoder.d_model
        self.card_encoder = card_encoder or CardEncoder(d_model=d_model)
        conditioning = self.config.deck_conditioning
        if conditioning is not None:
            conditioning.validate_for_model(
                num_layers=self.config.state_encoder.num_layers,
                max_card_id=self.card_encoder.num_card_ids,
            )
        self.state_encoder = StateEncoder(
            config=self.config.state_encoder,
            card_encoder=self.card_encoder,
            deck_conditioning=conditioning,
        )
        self.policy_head = PointerPolicyHead(
            self.config.policy,
            deck_conditioning=conditioning,
        )
        self.exact_capsules = nn.ModuleDict()
        if (
            conditioning is not None
            and conditioning.enabled
            and conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        ):
            compositional = conditioning.compositional
            if compositional is None:
                raise RuntimeError("architecture v4 requires compositional config")
            transformer_shapes = compositional_transformer_shapes(
                self.config.state_encoder,
                layer_indices=compositional.transformer_layer_indices,
            )
            policy_shapes = self.policy_head.compositional_policy_shapes()
            with torch.random.fork_rng(devices=[]):
                self.exact_capsules = nn.ModuleDict(
                    {
                        route.module_key: (
                            FixedExactDeckCapsule(
                                d_model=d_model,
                                option_adapter_bottleneck_dim=(
                                    compositional.option_adapter_bottleneck_dim
                                ),
                                policy_global_bottleneck_dim=(
                                    conditioning.policy_bottleneck_dim
                                ),
                                value_bottleneck_dim=(
                                    compositional.value_bottleneck_dim
                                ),
                            )
                            if compositional.export_mode == "fixed"
                            else ExactDeckCapsule(
                                d_model=d_model,
                                basis_count=compositional.shared_basis_count,
                                transformer_shapes=transformer_shapes,
                                policy_shapes=policy_shapes,
                                transformer_exact_rank=(
                                    compositional.transformer_exact_rank
                                ),
                                policy_exact_rank=compositional.policy_exact_rank,
                                transformer_layer_keys=tuple(
                                    f"layer_{index:02d}"
                                    for index in (
                                        compositional.transformer_layer_indices
                                    )
                                ),
                                option_set_width=compositional.option_set_width,
                                option_adapter_bottleneck_dim=(
                                    compositional.option_adapter_bottleneck_dim
                                ),
                                policy_global_bottleneck_dim=(
                                    conditioning.policy_bottleneck_dim
                                ),
                                count_hidden_dim=(
                                    self.config.policy.hidden_dim or d_model
                                ),
                                value_bottleneck_dim=(
                                    compositional.value_bottleneck_dim
                                ),
                            )
                        )
                        for route in conditioning.active_routes
                    }
                )
        self.search_reranker = SearchCandidateReranker(d_model)
        self.root_perspective_value_adapter = RootPerspectiveValueAdapter(
            d_model,
            self.config.root_perspective_value,
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, self.config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.value_hidden_dim, 1),
            nn.Tanh(),
        )
        self.prefix_value_delta_head = nn.Sequential(
            nn.Linear(d_model, self.config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.value_hidden_dim, 1),
        )
        prefix_value_output = cast(nn.Linear, self.prefix_value_delta_head[-1])
        nn.init.zeros_(prefix_value_output.weight)
        nn.init.zeros_(prefix_value_output.bias)
        self.prize_diff_head = nn.Sequential(
            nn.Linear(d_model, self.config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.value_hidden_dim, 1),
        )
        self.opponent_card_head = nn.Linear(d_model, self.config.opponent_card_classes)
        self.opponent_hand_head = nn.Linear(d_model, self.config.opponent_card_classes)
        self.effect_head = nn.Sequential(
            nn.Linear(d_model, self.config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.value_hidden_dim, self.config.effect_feature_size),
        )
        self.deck_encoder: DeckEncoder | None = None
        self.deck_input_projection: nn.Linear | None = None
        self.private_policy_adapters = nn.ModuleDict()
        self.private_root_value_heads = nn.ModuleDict()
        self.private_prefix_value_heads = nn.ModuleDict()
        self.dense_private_root_value_heads = nn.ModuleDict()
        self.dense_private_prefix_value_heads = nn.ModuleDict()
        if conditioning is not None and conditioning.enabled:
            if conditioning.deck_context_mode == "encoded":
                self.deck_encoder = DeckEncoder(
                    d_model,
                    conditioning.encoder_hidden_dim,
                )
                self.deck_input_projection = nn.Linear(d_model, d_model)
                _zero_linear(self.deck_input_projection)
            if (
                conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            ):
                dense_config = conditioning.dense_private
                if dense_config is None:
                    raise RuntimeError(
                        "architecture v3 requires dense-private configuration"
                    )
                with torch.random.fork_rng(devices=[]):
                    self.dense_private_root_value_heads = nn.ModuleDict(
                        {
                            route.module_key: (
                                DensePrivateRootValueHead.from_sequential(
                                    self.value_head,
                                    PrivateScalarResidual(
                                        d_model,
                                        conditioning.value_bottleneck_dim,
                                    ),
                                    dense_hidden_dim=dense_config.value_hidden_dim,
                                    dense_bottleneck_dim=(
                                        dense_config.value_bottleneck_dim
                                    ),
                                )
                            )
                            for route in conditioning.active_routes
                        }
                    )
                    self.dense_private_prefix_value_heads = nn.ModuleDict(
                        {
                            route.module_key: (
                                DensePrivatePrefixValueHead.from_sequential(
                                    self.prefix_value_delta_head,
                                    PrivateScalarResidual(
                                        d_model,
                                        conditioning.value_bottleneck_dim,
                                    ),
                                    dense_hidden_dim=dense_config.value_hidden_dim,
                                    dense_bottleneck_dim=(
                                        dense_config.value_bottleneck_dim
                                    ),
                                )
                            )
                            for route in conditioning.active_routes
                        }
                    )
                if dense_config.export_mode == "fixed":
                    # The selected private value heads contain cloned base
                    # paths, so retaining generic copies would only inflate a
                    # fixed-deck deployment checkpoint.
                    self.value_head = nn.Sequential()
                    self.prefix_value_delta_head = nn.Sequential()
            elif (
                conditioning.architecture_version
                != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            ):
                self.private_policy_adapters = private_residual_modules(
                    conditioning.active_routes,
                    d_model=d_model,
                    bottleneck_dim=conditioning.policy_bottleneck_dim,
                    dropout=conditioning.adapter_dropout,
                )
                self.private_root_value_heads = private_scalar_modules(
                    conditioning.active_routes,
                    d_model=d_model,
                    bottleneck_dim=conditioning.value_bottleneck_dim,
                )
                self.private_prefix_value_heads = private_scalar_modules(
                    conditioning.active_routes,
                    d_model=d_model,
                    bottleneck_dim=conditioning.value_bottleneck_dim,
                )
        # Schema 8 and schema 9 intentionally use different residual modules.
        # Keep this new module after every legacy initializer so adding it does
        # not perturb deterministic initialization of existing checkpoint keys.
        # Its zero output preserves collection/replay parity at migration.
        self.planner_reranker = SearchCandidateReranker(d_model)
        self._reset_schema9_migration_parameters()
        self.macro_outcome_heads = MacroOutcomeHeads(
            d_model,
            self.config.macro_outcome,
        )
        self._reset_schema10_migration_parameters()
        self.action_value_head: CompleteActionValueHead | None = None
        if self.config.action_value.enabled:
            self.action_value_head = CompleteActionValueHead(
                d_model,
                self.config.action_value,
            )
            self._reset_action_value_migration_parameters()
        # Keep recurrent initialization after every historical module so the
        # new architecture cannot perturb any existing checkpoint tensor.
        self.recurrent_policy: RecurrentPolicyCore | None = None
        if self.config.recurrent is not None:
            self.recurrent_policy = RecurrentPolicyCore(
                d_model,
                self.config.recurrent,
            )

    def _reset_action_value_migration_parameters(self) -> None:
        """Initialize the new persistent critic independently of process RNG."""
        head = self.action_value_head
        if head is None:
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ACTION_VALUE_WARM_START_INITIALIZATION_SEED)
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            # A fresh critic must not invent an action preference before it has
            # engine or real-trajectory evidence.  Keep the state W/D/L branch
            # trainable while making every initial sibling advantage exactly
            # zero.  Gradients still reach both advantage layers on the first
            # supervised update.
            advantage_output = cast(nn.Linear, head.advantage_branch[-1])
            nn.init.zeros_(advantage_output.weight)
            nn.init.zeros_(advantage_output.bias)

    def _reset_schema10_migration_parameters(self) -> None:
        """Make missing schema-10 tensors deterministic and output-inert."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEMA10_WARM_START_INITIALIZATION_SEED)
            for module in self.macro_outcome_heads.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            for branch in (
                self.macro_outcome_heads.expected,
                self.macro_outcome_heads.conditional,
            ):
                branch.reset_output_parameters()

    def _reset_schema9_migration_parameters(self) -> None:
        """Make missing schema-9 checkpoint tensors process-seed invariant."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEMA9_WARM_START_INITIALIZATION_SEED)
            for parent in (
                self.root_perspective_value_adapter,
                self.planner_reranker,
            ):
                for module in parent.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
            root_output = cast(
                nn.Linear,
                self.root_perspective_value_adapter.residual[-1],
            )
            planner_output = cast(
                nn.Linear,
                self.planner_reranker.residual_head[-1],
            )
            for output in (root_output, planner_output):
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
            nn.init.zeros_(self.policy_head.proposal_query_projection.weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Migrate checkpoints predating inert auxiliary policy modules."""
        qualified_names = tuple(
            f"{prefix}{name}" for name in _PREFIX_VALUE_DELTA_PARAMETER_NAMES
        )
        present = tuple(name in state_dict for name in qualified_names)
        conditioning = self.config.deck_conditioning
        fixed_dense_private = (
            conditioning is not None
            and conditioning.dense_private is not None
            and conditioning.dense_private.export_mode == "fixed"
        )
        if not any(present) and not fixed_dense_private:
            for name, qualified_name in zip(
                _PREFIX_VALUE_DELTA_PARAMETER_NAMES,
                qualified_names,
                strict=True,
            ):
                state_dict[qualified_name] = self.get_parameter(name).detach().clone()
        reranker_names = tuple(
            f"{_SEARCH_RERANKER_STATE_PREFIX}{name}"
            for name in self.search_reranker.state_dict()
        )
        reranker_keys = tuple(f"{prefix}{name}" for name in reranker_names)
        if not any(key in state_dict for key in reranker_keys):
            for name, key in zip(reranker_names, reranker_keys, strict=True):
                state_dict[key] = self.get_parameter(name).detach().clone()
        planner_reranker_names = tuple(
            f"{_PLANNER_RERANKER_STATE_PREFIX}{name}"
            for name in self.planner_reranker.state_dict()
        )
        planner_reranker_keys = tuple(
            f"{prefix}{name}" for name in planner_reranker_names
        )
        if not any(key in state_dict for key in planner_reranker_keys):
            for name, key in zip(
                planner_reranker_names,
                planner_reranker_keys,
                strict=True,
            ):
                state_dict[key] = self.get_parameter(name).detach().clone()
        adapter_names = tuple(
            f"{_ROOT_PERSPECTIVE_VALUE_STATE_PREFIX}{name}"
            for name in self.root_perspective_value_adapter.state_dict()
        )
        adapter_keys = tuple(f"{prefix}{name}" for name in adapter_names)
        if not any(key in state_dict for key in adapter_keys):
            for name, key in zip(adapter_names, adapter_keys, strict=True):
                state_dict[key] = self.get_parameter(name).detach().clone()
        macro_names = tuple(
            f"{_MACRO_OUTCOME_STATE_PREFIX}{name}"
            for name in self.macro_outcome_heads.state_dict()
        )
        macro_keys = tuple(f"{prefix}{name}" for name in macro_names)
        if not any(key in state_dict for key in macro_keys):
            for name, key in zip(macro_names, macro_keys, strict=True):
                state_dict[key] = self.get_parameter(name).detach().clone()
        if self.action_value_head is not None:
            action_value_names = tuple(
                f"{_ACTION_VALUE_STATE_PREFIX}{name}"
                for name in self.action_value_head.state_dict()
            )
            action_value_keys = tuple(f"{prefix}{name}" for name in action_value_names)
            if not any(key in state_dict for key in action_value_keys):
                for name, key in zip(
                    action_value_names,
                    action_value_keys,
                    strict=True,
                ):
                    state_dict[key] = self.get_parameter(name).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
    ) -> AgentNetworkOutput:
        """Return policy logits, value, and auxiliary predictions."""
        conditioned = self._encode_conditioned_state(states, decks)
        return self.forward_from_conditioned(conditioned, options)

    def forward_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
    ) -> AgentNetworkOutput:
        """Evaluate every head from one immutable decision context."""
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        policy_logits = self.policy_head(
            self._policy_global(conditioned),
            option_embeddings,
            options,
            route_plan=conditioned.route_plan,
        )
        value = self._root_values(conditioned)
        prize_diff = cast(
            Tensor,
            self.prize_diff_head(encoded_state.global_embedding),
        ).squeeze(-1)
        opponent_card_logits = cast(
            Tensor,
            self.opponent_card_head(encoded_state.global_embedding),
        )
        opponent_hand_logits = cast(
            Tensor,
            self.opponent_hand_head(encoded_state.global_embedding),
        )
        effect_predictions = cast(Tensor, self.effect_head(option_embeddings))
        return AgentNetworkOutput(
            policy_logits=policy_logits,
            value=value,
            prize_diff=prize_diff,
            opponent_card_logits=opponent_card_logits,
            opponent_hand_logits=opponent_hand_logits,
            effect_predictions=effect_predictions,
        )

    def greedy_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        max_select_steps: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return greedy autoregressive option-index actions."""
        with torch.no_grad():
            conditioned = self._encode_conditioned_state(states, decks)
            return self.greedy_decode_from_conditioned(
                conditioned,
                options,
                max_select_steps=max_select_steps,
            )

    def encode_conditioned_state(
        self,
        states: StateBatch,
        decks: DeckBatch | None = None,
    ) -> ConditionedStateOutput:
        """Expose one snapshot encoding before any recurrent state transition."""
        return self._encode_conditioned_state(states, decks)

    def recurrent_step_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        events: PublicEventBatch,
        *,
        previous_state: RecurrentPolicyState,
    ) -> tuple[ConditionedStateOutput, RecurrentPolicyState]:
        """Prepare one decision context and candidate next recurrent state."""
        if conditioned.decision_embedding is not None:
            raise ValueError("conditioned state already consumed a recurrent step")
        core = self._required_recurrent_policy()
        decision_embedding, proposed_state = core.step(
            conditioned.encoded_state.global_embedding,
            events,
            card_encoder=self.card_encoder,
            previous_state=previous_state,
        )
        return (
            replace(conditioned, decision_embedding=decision_embedding),
            proposed_state,
        )

    def recurrent_unroll_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        events: PublicEventBatch,
        sequence_padding_mask: Tensor,
        *,
        initial_state: RecurrentPolicyState,
    ) -> tuple[ConditionedStateOutput, RecurrentPolicyState]:
        """Replay flattened snapshot rows as padded full decision sequences."""
        if conditioned.decision_embedding is not None:
            raise ValueError("conditioned state already consumed a recurrent step")
        if sequence_padding_mask.ndim != 2:
            raise ValueError("sequence padding mask must have shape [B, S]")
        batch_size, sequence_length = sequence_padding_mask.shape
        snapshot_global = conditioned.encoded_state.global_embedding
        if int(snapshot_global.shape[0]) != batch_size * sequence_length:
            raise ValueError("conditioned rows do not match flattened sequences")
        core = self._required_recurrent_policy()
        decisions, final_state = core.unroll(
            snapshot_global.reshape(batch_size, sequence_length, -1),
            events,
            sequence_padding_mask,
            card_encoder=self.card_encoder,
            initial_state=initial_state,
        )
        return (
            replace(
                conditioned,
                decision_embedding=decisions.reshape(
                    batch_size * sequence_length,
                    -1,
                ),
            ),
            final_state,
        )

    def recurrent_replay_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        events: PublicEventBatch,
        sequence_offsets: Tensor,
    ) -> ConditionedStateOutput:
        """Replay packed complete sequences from exact zero-state boundaries."""
        if conditioned.decision_embedding is not None:
            raise ValueError("conditioned state already consumed a recurrent step")
        snapshot_global = conditioned.encoded_state.global_embedding
        row_count = int(snapshot_global.shape[0])
        if events.batch_size != row_count:
            raise ValueError("recurrent events must align with packed decision rows")
        if sequence_offsets.ndim != 1 or sequence_offsets.numel() < 2:
            raise ValueError("sequence offsets must contain at least one sequence")
        cpu_offsets = sequence_offsets.detach().to(device="cpu", dtype=torch.long)
        offsets = tuple(int(value) for value in cpu_offsets.tolist())
        if offsets[0] != 0 or offsets[-1] != row_count:
            raise ValueError("sequence offsets must span every decision row")
        lengths = tuple(right - left for left, right in pairwise(offsets))
        if any(length <= 0 for length in lengths):
            raise ValueError("recurrent sequences must be non-empty")
        sequence_count = len(lengths)
        sequence_length = max(lengths)
        source_indices = torch.empty(
            sequence_count * sequence_length,
            dtype=torch.long,
            device=snapshot_global.device,
        )
        padding_mask = torch.ones(
            (sequence_count, sequence_length),
            dtype=torch.bool,
            device=snapshot_global.device,
        )
        for sequence_index, (start, length) in enumerate(
            zip(offsets, lengths, strict=False)
        ):
            target_start = sequence_index * sequence_length
            source_indices[target_start : target_start + length] = torch.arange(
                start,
                start + length,
                dtype=torch.long,
                device=snapshot_global.device,
            )
            if length < sequence_length:
                source_indices[
                    target_start + length : target_start + sequence_length
                ] = start
            padding_mask[sequence_index, :length] = False
        padded_events = select_public_event_batch_rows(events, source_indices)
        core = self._required_recurrent_policy()
        initial_state = core.initial_state(
            sequence_count,
            device=snapshot_global.device,
            dtype=snapshot_global.dtype,
        )
        decisions, _final_state = core.unroll(
            snapshot_global.index_select(0, source_indices).reshape(
                sequence_count,
                sequence_length,
                -1,
            ),
            padded_events,
            padding_mask,
            card_encoder=self.card_encoder,
            initial_state=initial_state,
        )
        decision_embedding = decisions.reshape(
            sequence_count * sequence_length,
            -1,
        )[~padding_mask.reshape(-1)]
        if int(decision_embedding.shape[0]) != row_count:
            raise RuntimeError("recurrent replay changed the packed decision count")
        return replace(conditioned, decision_embedding=decision_embedding)

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentPolicyState:
        """Create an explicit reset state at a validated sequence boundary."""
        return self._required_recurrent_policy().initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def freeze_conditioned_encoder_prefix(self, prefix_layers: int) -> tuple[str, ...]:
        """Freeze every parameter consumed by a shared encoder prefix.

        The compositional upper Transformer, shared DCCR projections, exact
        capsules, and all policy/value/Q heads remain trainable.  Optimizers may
        retain the frozen parameters so exact-resume state layouts stay stable;
        parameters without gradients are skipped by AdamW.
        """
        self.state_encoder.validate_frozen_prefix_layers(prefix_layers)
        if self.deck_encoder is None or self.deck_input_projection is None:
            raise ValueError("conditioned prefix freezing requires encoded decks")
        modules: tuple[nn.Module, ...] = (
            self.card_encoder,
            self.state_encoder.area_embedding,
            self.state_encoder.owner_embedding,
            self.state_encoder.kind_embedding,
            self.state_encoder.entity_slot_embedding,
            self.state_encoder.scalar_projection,
            self.state_encoder.global_context_projection,
            self.state_encoder.public_state_projection,
            self.policy_head.attack_embedding,
            self.policy_head.attachment_identity_projection,
            self.deck_encoder,
            self.deck_input_projection,
            *tuple(self.state_encoder.transformer.layers[:prefix_layers]),
        )
        frozen_ids = {
            id(parameter) for module in modules for parameter in module.parameters()
        }
        frozen_ids.add(id(self.state_encoder.attachment_kind_gates))
        names = tuple(
            name
            for name, parameter in self.named_parameters()
            if id(parameter) in frozen_ids
        )
        for _name, parameter in self.named_parameters():
            if id(parameter) in frozen_ids:
                parameter.requires_grad_(False)
        return names

    def predict_values(
        self,
        states: StateBatch,
        decks: DeckBatch | None = None,
    ) -> Tensor:
        """Return no-grad root-critic predictions without decoding actions."""
        with torch.no_grad():
            conditioned = self._encode_conditioned_state(states, decks)
            return self._root_values(conditioned)

    def greedy_decode_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        *,
        max_select_steps: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Decode from a cache produced by :meth:`encode_conditioned_state`."""
        encoded_state = conditioned.encoded_state
        return self.policy_head.greedy_decode(
            self._policy_global(conditioned),
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
            max_select_steps=max_select_steps,
        )

    def action_logprobs_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
    ) -> Tensor:
        """Score complete actions without bypassing private policy residuals."""
        if int(conditioned.encoded_state.global_embedding.shape[0]) != 1:
            raise ValueError("runtime action scoring requires one conditioned row")
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        policy_global = self._policy_global(conditioned)
        projected_options = self.policy_head.project_option_embeddings(
            option_embeddings,
            route_plan=conditioned.route_plan,
        )
        return torch.stack(
            [
                self.policy_head.teacher_forced_action_scores(
                    policy_global,
                    option_embeddings,
                    options,
                    (action,),
                    route_plan=conditioned.route_plan,
                    projected_options=projected_options,
                ).action_logprobs[0]
                for action in actions
            ]
        )

    def search_candidate_logits_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[int]],
        candidate_features: Tensor,
    ) -> Tensor:
        """Score complete candidates from one cached runtime state."""
        return self.search_candidate_evaluation_from_conditioned(
            conditioned,
            options,
            candidate_actions,
            candidate_features,
        ).logits

    def search_candidate_evaluation_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[int]],
        candidate_features: Tensor,
    ) -> SearchCandidateEvaluation:
        """Evaluate all planner branches from one cached runtime state."""
        candidate_count = len(candidate_actions)
        if int(conditioned.encoded_state.global_embedding.shape[0]) != 1:
            raise ValueError("runtime search reranking requires one conditioned row")
        if int(options.valid_options.shape[0]) != 1:
            raise ValueError("runtime search options require one row")
        if candidate_count <= 0:
            raise ValueError("search reranking requires at least one candidate")
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        policy_global = self._policy_global(conditioned)
        projected_options = self.policy_head.project_option_embeddings(
            option_embeddings,
            route_plan=conditioned.route_plan,
        )
        evaluations = tuple(
            self.policy_head.teacher_forced_action_scores(
                policy_global,
                option_embeddings,
                options,
                (action,),
                route_plan=conditioned.route_plan,
                projected_options=projected_options,
                include_completion=True,
                include_proposal=True,
            )
            for action in candidate_actions
        )
        base_logprobs = torch.stack(
            tuple(evaluation.action_logprobs[0] for evaluation in evaluations)
        )
        action_latents = torch.cat(
            tuple(
                cast(Tensor, evaluation.completed_action_latents)
                for evaluation in evaluations
            ),
            dim=0,
        )
        proposal_logprobs = torch.stack(
            tuple(
                _required_proposal_action_logprobs(evaluation)[0]
                for evaluation in evaluations
            )
        )
        features = candidate_features.to(
            device=base_logprobs.device,
            dtype=action_latents.dtype,
        )
        state_latents = policy_global.expand(candidate_count, -1)
        residuals = self.search_reranker.residual_logits(
            base_logprobs,
            state_latents,
            action_latents,
            features,
            candidate_counts=(candidate_count,),
        )
        logits = self.search_reranker(
            base_logprobs,
            state_latents,
            action_latents,
            features,
            candidate_counts=(candidate_count,),
        )
        return SearchCandidateEvaluation(
            logits=logits,
            base_action_logprobs=base_logprobs,
            proposal_action_logprobs=proposal_logprobs,
            reranker_residuals=residuals,
            candidate_counts=(candidate_count,),
        )

    def planner_candidate_evaluation_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[int]],
        candidate_features: Tensor,
    ) -> PlannerCandidateEvaluation:
        """Evaluate one schema-9 support without consuming schema-8 weights."""
        candidate_count = len(candidate_actions)
        if int(conditioned.encoded_state.global_embedding.shape[0]) != 1:
            raise ValueError("runtime planner reranking requires one conditioned row")
        if int(options.valid_options.shape[0]) != 1:
            raise ValueError("runtime planner options require one row")
        if candidate_count <= 0:
            raise ValueError("runtime planner reranking requires a candidate")
        if candidate_features.shape != (
            candidate_count,
            SEARCH_EVIDENCE_FEATURE_SIZE,
        ):
            raise ValueError("runtime planner candidate features are misaligned")
        policy_global = self._policy_global(conditioned)
        option_embeddings = self.policy_head.option_embeddings(
            conditioned.encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=conditioned.encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        context = PolicyEvaluationContext(
            policy_global=policy_global,
            option_embeddings=option_embeddings,
            projected_options=self.policy_head.project_option_embeddings(
                option_embeddings,
                route_plan=conditioned.route_plan,
            ),
            route_plan=conditioned.route_plan,
        )
        evaluations = tuple(
            self._teacher_forced_action_scores_from_context(
                context,
                options,
                (action,),
                include_completion=True,
                include_proposal=True,
            )
            for action in candidate_actions
        )
        action_latents = torch.cat(
            tuple(
                _required_completed_action_latents(evaluation)
                for evaluation in evaluations
            ),
            dim=0,
        )
        base_logprobs = torch.stack(
            tuple(evaluation.action_logprobs[0] for evaluation in evaluations)
        )
        proposal_logprobs = torch.stack(
            tuple(
                _required_proposal_action_logprobs(evaluation)[0]
                for evaluation in evaluations
            )
        )
        features = candidate_features.to(
            device=base_logprobs.device,
            dtype=action_latents.dtype,
        )
        residuals = self.planner_reranker.residual_logits(
            base_logprobs,
            policy_global.expand(candidate_count, -1),
            action_latents,
            features,
            candidate_counts=(candidate_count,),
        )
        return PlannerCandidateEvaluation(
            base_action_logprobs=base_logprobs,
            proposal_action_logprobs=proposal_logprobs,
            reranker_residuals=residuals,
            candidate_counts=(candidate_count,),
        )

    def select_search_action_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[int]],
        candidate_features: Tensor,
    ) -> tuple[int, ...]:
        """Return the highest-logit complete candidate for one runtime state."""
        with torch.no_grad():
            logits = self.search_candidate_logits_from_conditioned(
                conditioned,
                options,
                candidate_actions,
                candidate_features,
            )
            selected_index = int(logits.argmax().item())
        return tuple(int(index) for index in candidate_actions[selected_index])

    def evaluate_search_candidates(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        decks: DeckBatch | None = None,
    ) -> SearchCandidateEvaluation:
        """Evaluate ragged complete-action groups for auxiliary supervision."""
        conditioned = self._encode_conditioned_state(states, decks)
        context = self._policy_context_from_conditioned(conditioned, options)
        return self.evaluate_search_candidates_from_context(
            context,
            options,
            candidate_actions,
            candidate_features,
            decks=decks,
        )

    def evaluate_planner_candidates(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        ordered_rows: Tensor | None = None,
        decks: DeckBatch | None = None,
    ) -> PlannerCandidateEvaluation:
        """Evaluate ragged schema-9 supports with the isolated planner head."""
        conditioned = self._encode_conditioned_state(states, decks)
        context = self._policy_context_from_conditioned(conditioned, options)
        return self.evaluate_planner_candidates_from_context(
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
        decks: DeckBatch | None = None,
        deadline_monotonic: float | None = None,
    ) -> PlannerProposalBatchResult:
        """Generate learned complete-action proposals from one state encoding."""
        ensure_planner_proposal_deadline(deadline_monotonic)
        conditioned = self._encode_conditioned_state(states, decks)
        context = self._policy_context_from_conditioned(conditioned, options)
        return self.generate_planner_proposals_from_context(
            context,
            options,
            ordered_rows=ordered_rows,
            limits=limits,
            decks=decks,
            deadline_monotonic=deadline_monotonic,
        )

    def generate_planner_proposals_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        *,
        ordered_rows: Tensor,
        limits: PlannerProposalSearchLimits,
        decks: DeckBatch | None = None,
        deadline_monotonic: float | None = None,
    ) -> PlannerProposalBatchResult:
        """Run bounded top-K search while reusing one pointer-policy context."""
        ensure_planner_proposal_deadline(deadline_monotonic)
        batch_size = int(context.policy_global.shape[0])
        if int(options.valid_options.shape[0]) != batch_size:
            raise ValueError("proposal options must align with policy context")
        if ordered_rows.shape != (batch_size,) or ordered_rows.dtype != torch.bool:
            raise ValueError("ordered_rows must be a bool tensor with shape [batch]")
        if context.route_plan is not None and decks is None:
            raise ValueError("deck-conditioned proposal context requires decks")
        if decks is not None and len(decks) != batch_size:
            raise ValueError("proposal decks must align with policy context")
        valid = options.valid_options.to(dtype=torch.long).sum(dim=1)
        option_counts = tuple(int(value) for value in valid.detach().cpu().tolist())
        min_counts = tuple(
            int(value) for value in options.min_counts.detach().cpu().tolist()
        )
        max_counts = tuple(
            int(value) for value in options.max_counts.detach().cpu().tolist()
        )
        ordered = tuple(bool(value) for value in ordered_rows.detach().cpu().tolist())
        problems = tuple(
            PlannerProposalProblem(
                root_index=index,
                option_count=option_count,
                min_count=min_count,
                max_count=max_count,
                ordered=is_ordered,
            )
            for index, (option_count, min_count, max_count, is_ordered) in enumerate(
                zip(
                    option_counts,
                    min_counts,
                    max_counts,
                    ordered,
                    strict=True,
                )
            )
        )
        scorer = _PolicyContextProposalScorer(
            model=self,
            context=context,
            options=options,
            ordered_rows=ordered_rows,
            decks=decks,
        )
        proposals = generate_batched_planner_proposals(
            problems,
            limits=limits,
            scorer=scorer,
            deadline_monotonic=deadline_monotonic,
        )
        ensure_planner_proposal_deadline(deadline_monotonic)
        base_greedy = self.greedy_decode_from_policy_context(
            context,
            options,
            ordered_rows=ordered_rows,
        )
        return replace(proposals, base_greedy_actions=base_greedy)

    def greedy_decode_from_policy_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        *,
        ordered_rows: Tensor | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Decode the true base-policy greedy anchors from a retained context."""
        output = self.policy_head.sample_decode_tensors_from_embeddings(
            context.policy_global,
            context.option_embeddings,
            options,
            temperature=0.0,
            route_plan=context.route_plan,
            ordered_rows=ordered_rows,
            projected_options=context.projected_options,
        )
        return actions_from_decode_tensors(
            output.choice_indices,
            output.append_masks,
        )

    def _planner_proposal_prefix_logprobs_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        queries: tuple[PlannerProposalPrefixQuery, ...],
        *,
        ordered_rows: Tensor,
        decks: DeckBatch | None,
    ) -> tuple[tuple[float, ...], ...]:
        """Score a ragged prefix wave through one proposal residual batch."""
        if not queries:
            return ()
        source_indices = torch.tensor(
            [query.root_index for query in queries],
            dtype=torch.long,
            device=context.policy_global.device,
        )
        expanded_options = _select_option_rows(
            options,
            source_indices.to(device=options.valid_options.device),
        )
        expanded_decks = (
            None
            if decks is None
            else decks.select(source_indices.to(device=decks.card_ids.device))
        )
        expanded_context = self.select_policy_context(
            context,
            source_indices,
            decks=expanded_decks,
        )
        expanded_ordered = ordered_rows.index_select(
            0,
            source_indices.to(device=ordered_rows.device),
        ).to(device=expanded_options.valid_options.device)
        selected_mask = torch.zeros_like(expanded_options.valid_options)
        for row_index, query in enumerate(queries):
            if query.prefix.selected:
                selected_mask[
                    row_index,
                    torch.tensor(
                        query.prefix.selected,
                        dtype=torch.long,
                        device=selected_mask.device,
                    ),
                ] = True
        selected_counts = torch.tensor(
            [query.prefix.selected_count for query in queries],
            dtype=expanded_options.min_counts.dtype,
            device=expanded_options.min_counts.device,
        )
        ordered_history = self.policy_head.initial_ordered_history(
            expanded_context.option_embeddings
        )
        max_prefix_length = max(query.prefix.selected_count for query in queries)
        for step in range(max_prefix_length):
            choice_indices = torch.zeros(
                len(queries),
                dtype=torch.long,
                device=expanded_context.option_embeddings.device,
            )
            append_mask = torch.zeros(
                len(queries),
                dtype=torch.bool,
                device=expanded_context.option_embeddings.device,
            )
            for row_index, query in enumerate(queries):
                if step < query.prefix.selected_count:
                    choice_indices[row_index] = query.prefix.selected[step]
                    append_mask[row_index] = True
            ordered_history = self.policy_head.advance_ordered_history(
                ordered_history,
                expanded_context.option_embeddings,
                choice_indices,
                append_mask,
            )
        logits = self.policy_head.base_and_proposal_step_logits(
            expanded_context.policy_global,
            expanded_context.option_embeddings,
            expanded_options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            ordered_rows=expanded_ordered,
            route_plan=expanded_context.route_plan,
            projected_options=expanded_context.projected_options,
        ).proposal
        logprobs = torch.log_softmax(logits.float(), dim=1).detach().cpu()
        stop_index = int(expanded_options.valid_options.shape[1])
        return tuple(
            tuple(
                float(value)
                for value in logprobs[row_index, : query.prefix.option_count]
            )
            + (float(logprobs[row_index, stop_index]),)
            for row_index, query in enumerate(queries)
        )

    def evaluate_search_candidates_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        decks: DeckBatch | None = None,
    ) -> SearchCandidateEvaluation:
        """Evaluate candidate groups from an existing policy context."""
        batch_size = int(context.policy_global.shape[0])
        if (
            len(candidate_actions) != batch_size
            or len(candidate_features) != batch_size
        ):
            raise ValueError("search candidate groups must align with state rows")
        candidate_counts = tuple(len(group) for group in candidate_actions)
        if any(count <= 0 for count in candidate_counts):
            raise ValueError("each search evidence row must contain candidates")
        for actions, features in zip(
            candidate_actions,
            candidate_features,
            strict=True,
        ):
            if features.ndim != 2 or int(features.shape[0]) != len(actions):
                raise ValueError("search feature rows must align with candidates")

        source_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=context.policy_global.device),
            torch.tensor(candidate_counts, device=context.policy_global.device),
        )
        expanded_options = _select_option_rows(
            options,
            source_indices.to(device=options.valid_options.device),
        )
        expanded_decks = (
            None
            if decks is None
            else decks.select(source_indices.to(device=decks.card_ids.device))
        )
        flattened_actions = tuple(
            action for group in candidate_actions for action in group
        )
        expanded_context = self.select_policy_context(
            context,
            source_indices,
            decks=expanded_decks,
        )
        policy_evaluation = self._teacher_forced_action_scores_from_context(
            expanded_context,
            expanded_options,
            flattened_actions,
            include_completion=True,
            include_proposal=True,
        )
        action_latents = policy_evaluation.completed_action_latents
        if action_latents is None:
            raise RuntimeError("model did not return completed-action latents")
        feature_tensor = torch.cat(
            tuple(
                features.to(device=action_latents.device, dtype=action_latents.dtype)
                for features in candidate_features
            ),
            dim=0,
        )
        logits = self.search_reranker(
            policy_evaluation.action_logprobs,
            expanded_context.policy_global,
            action_latents,
            feature_tensor,
            candidate_counts=candidate_counts,
        )
        residuals = self.search_reranker.residual_logits(
            policy_evaluation.action_logprobs,
            expanded_context.policy_global,
            action_latents,
            feature_tensor,
            candidate_counts=candidate_counts,
        )
        proposal_action_logprobs = policy_evaluation.proposal_action_logprobs
        if proposal_action_logprobs is None:
            raise RuntimeError("model did not return proposal action log-probabilities")
        return SearchCandidateEvaluation(
            logits=logits,
            base_action_logprobs=policy_evaluation.action_logprobs,
            proposal_action_logprobs=proposal_action_logprobs,
            reranker_residuals=residuals,
            candidate_counts=candidate_counts,
        )

    def evaluate_planner_candidates_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        candidate_features: Sequence[Tensor],
        *,
        ordered_rows: Tensor | None = None,
        decks: DeckBatch | None = None,
    ) -> PlannerCandidateEvaluation:
        """Replay schema-9 candidates from one shared policy context."""
        batch_size = int(context.policy_global.shape[0])
        if (
            len(candidate_actions) != batch_size
            or len(candidate_features) != batch_size
        ):
            raise ValueError("planner candidate groups must align with state rows")
        if ordered_rows is not None and (
            ordered_rows.shape != (batch_size,) or ordered_rows.dtype != torch.bool
        ):
            raise ValueError("ordered_rows must be a bool tensor with shape [batch]")
        candidate_counts = tuple(len(group) for group in candidate_actions)
        if any(count <= 0 for count in candidate_counts):
            raise ValueError("each planner evidence row must contain candidates")
        for actions, features in zip(
            candidate_actions,
            candidate_features,
            strict=True,
        ):
            if features.ndim != 2 or int(features.shape[0]) != len(actions):
                raise ValueError("planner feature rows must align with candidates")

        source_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=context.policy_global.device),
            torch.tensor(candidate_counts, device=context.policy_global.device),
        )
        expanded_options = _select_option_rows(
            options,
            source_indices.to(device=options.valid_options.device),
        )
        expanded_decks = (
            None
            if decks is None
            else decks.select(source_indices.to(device=decks.card_ids.device))
        )
        flattened_actions = tuple(
            action for group in candidate_actions for action in group
        )
        expanded_context = self.select_policy_context(
            context,
            source_indices,
            decks=expanded_decks,
        )
        expanded_ordered = (
            None
            if ordered_rows is None
            else ordered_rows.index_select(
                0,
                source_indices.to(device=ordered_rows.device),
            ).to(device=expanded_options.valid_options.device)
        )
        policy_evaluation = self._teacher_forced_action_scores_from_context(
            expanded_context,
            expanded_options,
            flattened_actions,
            include_completion=True,
            include_proposal=True,
            ordered_rows=expanded_ordered,
        )
        action_latents = _required_completed_action_latents(policy_evaluation)
        feature_tensor = torch.cat(
            tuple(
                features.to(device=action_latents.device, dtype=action_latents.dtype)
                for features in candidate_features
            ),
            dim=0,
        )
        residuals = self.planner_reranker.residual_logits(
            policy_evaluation.action_logprobs,
            expanded_context.policy_global,
            action_latents,
            feature_tensor,
            candidate_counts=candidate_counts,
        )
        proposal_action_logprobs = policy_evaluation.proposal_action_logprobs
        if proposal_action_logprobs is None:
            raise RuntimeError("model did not return proposal action log-probabilities")
        return PlannerCandidateEvaluation(
            base_action_logprobs=policy_evaluation.action_logprobs,
            proposal_action_logprobs=proposal_action_logprobs,
            reranker_residuals=residuals,
            candidate_counts=candidate_counts,
        )

    def evaluate_action_values(
        self,
        states: StateBatch,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        *,
        ordered_rows: Tensor | None = None,
        decks: DeckBatch | None = None,
        policy_temperature: float | Tensor = 1.0,
        validate_candidate_actions: bool = True,
    ) -> ActionValuePrediction:
        """Evaluate ragged complete-action groups through the persistent critic."""
        conditioned = self._encode_conditioned_state(states, decks)
        context = self._policy_context_from_conditioned(conditioned, options)
        return self.evaluate_action_values_from_context(
            context,
            options,
            candidate_actions,
            ordered_rows=ordered_rows,
            decks=decks,
            policy_temperature=policy_temperature,
            validate_candidate_actions=validate_candidate_actions,
        )

    def evaluate_action_values_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        candidate_actions: Sequence[Sequence[Sequence[int]]],
        *,
        ordered_rows: Tensor | None = None,
        decks: DeckBatch | None = None,
        policy_temperature: float | Tensor = 1.0,
        validate_candidate_actions: bool = True,
    ) -> ActionValuePrediction:
        """Reuse one policy context to score only retained legal candidates."""
        head = self._required_action_value_head()
        batch_size = int(context.policy_global.shape[0])
        if len(candidate_actions) != batch_size:
            raise ValueError("action-value candidate groups must align with states")
        if ordered_rows is not None and (
            ordered_rows.shape != (batch_size,) or ordered_rows.dtype != torch.bool
        ):
            raise ValueError("ordered_rows must be a bool tensor with shape [batch]")
        candidate_counts = tuple(len(group) for group in candidate_actions)
        if any(count <= 0 for count in candidate_counts):
            raise ValueError("each action-value row must retain candidates")
        flattened_actions = tuple(
            action for group in candidate_actions for action in group
        )
        if all(count == 1 for count in candidate_counts):
            # Real Retrace evaluates exactly the behavior action for every row.
            # Reusing the existing context avoids a redundant full-batch D2D
            # copy, deck-signature round trip, and private-route resolution.
            expanded_options = options
            expanded_decks = decks
            expanded_context = context
            expanded_ordered = ordered_rows
        else:
            source_rows = tuple(
                row
                for row, count in enumerate(candidate_counts)
                for _index in range(count)
            )
            source_indices = torch.tensor(
                source_rows,
                dtype=torch.long,
                device=context.policy_global.device,
            )
            expanded_options = _select_option_rows(
                options,
                source_indices.to(device=options.valid_options.device),
            )
            expanded_decks = None if decks is None else decks.select(source_rows)
            expanded_context = self.select_policy_context(
                context,
                source_indices,
                decks=expanded_decks,
            )
            expanded_ordered = (
                None
                if ordered_rows is None
                else ordered_rows.index_select(
                    0,
                    source_indices.to(device=ordered_rows.device),
                ).to(device=expanded_options.valid_options.device)
            )
        evaluation = self._teacher_forced_action_scores_from_context(
            expanded_context,
            expanded_options,
            flattened_actions,
            include_completion=True,
            ordered_rows=expanded_ordered,
            temperature=policy_temperature,
            validate_actions=validate_candidate_actions,
        )
        prediction = cast(
            ActionValuePrediction,
            head(
                expanded_context.policy_global,
                _required_completed_action_latents(evaluation),
                candidate_counts=candidate_counts,
            ),
        )
        if expanded_context.route_plan is not None and _is_compositional_config(
            self.config
        ):
            prediction = _apply_exact_action_value_calibration(
                prediction,
                expanded_context.policy_global,
                _required_completed_action_latents(evaluation),
                route_plan=expanded_context.route_plan,
                capsules=self.exact_capsules,
            )
        return replace(
            prediction,
            action_logprobs=evaluation.action_logprobs,
        )

    def action_value_state_logits_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
    ) -> Tensor:
        """Return actor-perspective categorical state-value logits."""
        policy_global = self._policy_global(conditioned)
        logits = self._required_action_value_head().state_logits(policy_global)
        if conditioned.route_plan is not None and _is_compositional_config(self.config):
            logits = logits + _exact_capsule_calibrator(
                policy_global,
                conditioned.route_plan,
                self.exact_capsules,
                target="action_state",
                output_features=3,
            )
        return logits

    def _required_action_value_head(self) -> CompleteActionValueHead:
        head = self.action_value_head
        if head is None:
            raise RuntimeError("complete-action value head is disabled")
        return head

    def first_step_logits_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
    ) -> Tensor:
        """Return initial option/STOP logits through the conditioned policy path."""
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        return cast(
            Tensor,
            self.policy_head(
                self._policy_global(conditioned),
                option_embeddings,
                options,
                route_plan=conditioned.route_plan,
            ),
        )

    def root_values_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
    ) -> Tensor:
        """Evaluate the shared plus exact-deck private root critic."""
        return self._root_values(conditioned)

    def root_information_values_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        *,
        actor_relations: Tensor,
        endpoints: Tensor,
        belief_summaries: Tensor,
    ) -> Tensor:
        """Evaluate semantic leaves without routing through an opponent critic."""
        base_values = self._root_values(conditioned)
        return cast(
            Tensor,
            self.root_perspective_value_adapter(
                base_values,
                self._decision_embedding(conditioned),
                actor_relations,
                endpoints,
                belief_summaries,
            ),
        )

    def macro_outcomes(
        self,
        completed_action_latents: Tensor,
        continuation_summaries: Tensor,
    ) -> MacroOutcomePrediction:
        """Evaluate schema-10 heads without another state-trunk forward."""
        return cast(
            MacroOutcomePrediction,
            self.macro_outcome_heads(
                completed_action_latents,
                continuation_summaries,
            ),
        )

    def evaluate_actions_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Teacher-force actions through conditioned policy and value paths."""
        policy_context = self._policy_context_from_conditioned(conditioned, options)
        policy_evaluation = self._teacher_forced_policy_from_context(
            policy_context,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )
        values = self._root_values(conditioned)
        return ActionEvaluation(
            action_logprobs=policy_evaluation.action_logprobs,
            entropies=policy_evaluation.entropies,
            values=values,
            step_logits=policy_evaluation.step_logits,
            first_logits=policy_evaluation.first_logits,
            token_logprobs=policy_evaluation.token_logprobs,
            token_entropies=policy_evaluation.token_entropies,
            token_mask=policy_evaluation.token_mask,
            prefix_values=self._prefix_values(
                values,
                policy_evaluation.prefix_queries,
                conditioned.route_plan,
            ),
            stop_sampled=policy_evaluation.stop_sampled,
            completed_action_latents=policy_evaluation.completed_action_latents,
            factual_presence_logits=policy_evaluation.factual_presence_logits,
            factual_magnitude_predictions=(
                policy_evaluation.factual_magnitude_predictions
            ),
            factual_actor_relation_logits=(
                policy_evaluation.factual_actor_relation_logits
            ),
            factual_next_context_logits=(policy_evaluation.factual_next_context_logits),
            policy_context=policy_context,
        )

    def evaluate_action_step_logits(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
    ) -> tuple[Tensor, ...]:
        """Return only teacher-forced logits, omitting unused value heads."""
        conditioned = self._encode_conditioned_state(states, decks)
        return self._teacher_forced_policy_from_conditioned(
            conditioned,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            include_completion=False,
            include_factual=False,
            validate_temperature=validate_temperature,
        ).step_logits

    def evaluate_action_logprobs(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
    ) -> Tensor:
        """Return only teacher-forced action log-probabilities."""
        conditioned = self._encode_conditioned_state(states, decks)
        context = self._policy_context_from_conditioned(conditioned, options)
        return self._teacher_forced_action_scores_from_context(
            context,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
        ).action_logprobs

    def evaluate_action_logprobs_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
    ) -> Tensor:
        """Score auxiliary actions without recomputing their source states."""
        return self._teacher_forced_action_scores_from_context(
            context,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
        ).action_logprobs

    def select_policy_context(
        self,
        context: PolicyEvaluationContext,
        indices: Tensor,
        *,
        decks: DeckBatch | None = None,
    ) -> PolicyEvaluationContext:
        """Select or repeat reusable policy rows and rebuild their route plan."""
        selected_route_plan: DeckRoutePlan | None = None
        if context.route_plan is not None:
            conditioning = self.config.deck_conditioning
            if conditioning is None or decks is None:
                raise RuntimeError("deck conditioning context is incomplete")
            selected_route_plan = resolve_deck_route_plan(decks, conditioning)
            source_embeddings = context.route_plan.deck_embeddings
            if (
                conditioning.architecture_version
                == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            ):
                if source_embeddings is None:
                    raise RuntimeError(
                        "compositional policy context has no deck embeddings"
                    )
                selected_route_plan = replace(
                    selected_route_plan,
                    deck_embeddings=source_embeddings.index_select(
                        0,
                        indices.to(device=source_embeddings.device),
                    ),
                    exact_capsules=self.exact_capsules,
                )
        return PolicyEvaluationContext(
            policy_global=context.policy_global.index_select(
                0,
                indices.to(device=context.policy_global.device),
            ),
            option_embeddings=context.option_embeddings.index_select(
                0,
                indices.to(device=context.option_embeddings.device),
            ),
            projected_options=context.projected_options.index_select(
                0,
                indices.to(device=context.projected_options.device),
            ),
            route_plan=selected_route_plan,
        )

    def _teacher_forced_policy_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        include_completion: bool = True,
        include_factual: bool = True,
        validate_temperature: bool = True,
    ) -> TeacherForcedEvaluation:
        """Decode actions from one state context with selectable output heads."""
        context = self._policy_context_from_conditioned(conditioned, options)
        return self._teacher_forced_policy_from_context(
            context,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            include_completion=include_completion,
            include_factual=include_factual,
            validate_temperature=validate_temperature,
        )

    def _policy_context_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
    ) -> PolicyEvaluationContext:
        """Build the action-independent inputs to teacher-forced decoding."""
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        return PolicyEvaluationContext(
            policy_global=self._policy_global(conditioned),
            option_embeddings=option_embeddings,
            projected_options=self.policy_head.project_option_embeddings(
                option_embeddings,
                route_plan=conditioned.route_plan,
            ),
            route_plan=conditioned.route_plan,
        )

    def _teacher_forced_policy_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        include_completion: bool = True,
        include_factual: bool = True,
        validate_temperature: bool = True,
        include_proposal: bool = False,
        ordered_rows: Tensor | None = None,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> TeacherForcedEvaluation:
        """Decode actions using previously computed policy inputs."""
        return self.policy_head.teacher_forced_decode(
            context.policy_global,
            context.option_embeddings,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            route_plan=context.route_plan,
            projected_options=context.projected_options,
            include_completion=include_completion,
            include_factual=include_factual,
            validate_temperature=validate_temperature,
            include_proposal=include_proposal,
            ordered_rows=ordered_rows,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )

    def _teacher_forced_action_scores_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        include_completion: bool = False,
        validate_temperature: bool = True,
        include_proposal: bool = False,
        ordered_rows: Tensor | None = None,
        validate_actions: bool = True,
    ) -> TeacherForcedActionScores:
        """Score actions from cached policy inputs without diagnostic traces."""
        return self.policy_head.teacher_forced_action_scores(
            context.policy_global,
            context.option_embeddings,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            route_plan=context.route_plan,
            projected_options=context.projected_options,
            include_completion=include_completion,
            validate_temperature=validate_temperature,
            include_proposal=include_proposal,
            ordered_rows=ordered_rows,
            validate_actions=validate_actions,
        )

    def opponent_logits_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate auxiliary belief heads from the conditioned state."""
        global_embedding = conditioned.encoded_state.global_embedding
        return (
            cast(Tensor, self.opponent_card_head(global_embedding)),
            cast(Tensor, self.opponent_hand_head(global_embedding)),
        )

    def auxiliary_outputs_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate prize, opponent, and effect heads from one cached context."""
        encoded_state = conditioned.encoded_state
        option_embeddings = self.policy_head.option_embeddings(
            encoded_state.token_embeddings,
            options,
            self.card_encoder,
            state_padding_mask=encoded_state.padding_mask,
            route_plan=conditioned.route_plan,
        )
        global_embedding = encoded_state.global_embedding
        return (
            cast(Tensor, self.prize_diff_head(global_embedding)).squeeze(-1),
            cast(Tensor, self.opponent_card_head(global_embedding)),
            cast(Tensor, self.opponent_hand_head(global_embedding)),
            cast(Tensor, self.effect_head(option_embeddings)),
        )

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample autoregressive actions with log-probs and value predictions."""
        trace = self.sample_decode_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            generator=generator,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTrace:
        """Sample actions and retain per-token behavior evidence."""
        with torch.no_grad():
            conditioned = self._encode_conditioned_state(states, decks)
            context = self._policy_context_from_conditioned(conditioned, options)
            return self.sample_decode_with_trace_from_context(
                conditioned,
                context,
                options,
                temperature=temperature,
                generator=generator,
            )

    def policy_context_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
    ) -> PolicyEvaluationContext:
        """Expose the reusable option/pointer context for a model lease."""
        return self._policy_context_from_conditioned(conditioned, options)

    def sample_action_candidates_from_context(
        self,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        candidate_counts: tuple[int, ...],
        *,
        max_select_steps: int,
        decks: DeckBatch | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SampledActionCandidates:
        """Sample ragged repeated actions in one expanded policy batch.

        Rows with a zero count are skipped. This lets exhaustive action spaces
        avoid behavior sampling while non-exhaustive rows share one GPU launch
        instead of issuing one autoregressive decode per requested candidate.
        """
        batch_size = int(context.policy_global.shape[0])
        if len(candidate_counts) != batch_size:
            raise ValueError("candidate counts must align with policy rows")
        if any(count < 0 for count in candidate_counts):
            raise ValueError("candidate sample counts must be non-negative")
        if max_select_steps < 0:
            raise ValueError("max_select_steps must be non-negative")
        if int(options.valid_options.shape[0]) != batch_size:
            raise ValueError("candidate options must align with policy rows")
        if decks is not None and len(decks) != batch_size:
            raise ValueError("candidate decks must align with policy rows")

        total_candidates = sum(candidate_counts)
        if total_candidates == 0:
            return SampledActionCandidates(
                actions=tuple(() for _ in candidate_counts),
                action_logprobs=torch.empty(
                    0,
                    dtype=torch.float32,
                    device=context.policy_global.device,
                ),
                candidate_counts=candidate_counts,
            )

        source_rows = tuple(
            row for row, count in enumerate(candidate_counts) for _ in range(count)
        )
        source_indices = torch.tensor(
            source_rows,
            dtype=torch.long,
            device=context.policy_global.device,
        )
        expanded_options = _select_option_rows(
            options,
            source_indices.to(device=options.valid_options.device),
        )
        expanded_decks = None if decks is None else decks.select(source_rows)
        expanded_context = self.select_policy_context(
            context,
            source_indices,
            decks=expanded_decks,
        )
        output = self.policy_head.sample_action_tensors_from_embeddings(
            expanded_context.policy_global,
            expanded_context.option_embeddings,
            expanded_options,
            max_select_steps=max_select_steps,
            temperature=temperature,
            route_plan=expanded_context.route_plan,
            projected_options=expanded_context.projected_options,
            generator=generator,
        )
        flattened_actions = actions_from_decode_tensors(
            output.choice_indices,
            output.append_masks,
        )
        grouped_actions = []
        start = 0
        for count in candidate_counts:
            stop = start + count
            grouped_actions.append(flattened_actions[start:stop])
            start = stop
        return SampledActionCandidates(
            actions=tuple(grouped_actions),
            action_logprobs=output.action_logprobs,
            candidate_counts=candidate_counts,
        )

    def sample_decode_with_trace_from_context(
        self,
        conditioned: ConditionedStateOutput,
        context: PolicyEvaluationContext,
        options: OptionBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTrace:
        """Sample base behavior without recomputing the leased policy context."""
        output = self.policy_head.sample_decode_tensors_from_embeddings(
            context.policy_global,
            context.option_embeddings,
            options,
            temperature=temperature,
            route_plan=context.route_plan,
            projected_options=context.projected_options,
            generator=generator,
        )
        values = self._root_values(conditioned)
        prefix_values = self._prefix_values(
            values,
            output.prefix_queries,
            conditioned.route_plan,
        )
        return SampleDecodeTrace(
            actions=actions_from_decode_tensors(
                output.choice_indices,
                output.append_masks,
            ),
            action_logprobs=output.action_logprobs,
            values=values,
            token_logprobs=output.token_logprobs,
            prefix_values=prefix_values,
            token_mask=output.token_mask,
            stop_sampled=output.stop_sampled,
        )

    def sample_decode_tensors(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample actions and return tensor traces for static-shape serving."""
        trace = self.sample_decode_tensors_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
            generator=generator,
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
        decks: DeckBatch | None = None,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
        validate_decode_cap: bool = True,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTensorTrace:
        """Sample static-shape actions and retain per-token behavior evidence."""
        if validate_decode_cap:
            required_steps = int(options.max_counts.max().item())
            if max_select_steps < required_steps:
                raise ValueError(
                    "max_select_steps must cover every normalized max_count"
                )
        with torch.no_grad():
            conditioned = self._encode_conditioned_state(states, decks)
            encoded_state = conditioned.encoded_state
            output = self.policy_head.sample_decode_tensors(
                self._policy_global(conditioned),
                encoded_state.token_embeddings,
                options,
                self.card_encoder,
                state_padding_mask=encoded_state.padding_mask,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
                route_plan=conditioned.route_plan,
                generator=generator,
            )
            values = self._root_values(conditioned)
            return SampleDecodeTensorTrace(
                choice_indices=output.choice_indices,
                append_masks=output.append_masks,
                action_logprobs=output.action_logprobs,
                values=values,
                token_logprobs=output.token_logprobs,
                prefix_values=self._prefix_values(
                    values,
                    output.prefix_queries,
                    conditioned.route_plan,
                ),
                token_mask=output.token_mask,
                stop_sampled=output.stop_sampled,
            )

    def sample_decode_static(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Sample with a caller-supplied decode step cap and materialize actions."""
        trace = self.sample_decode_static_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
            generator=generator,
        )
        return (trace.actions, trace.action_logprobs, trace.values)

    def sample_decode_static_with_trace(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch | None = None,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int,
        gumbel_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTrace:
        """Sample a fixed-cap decode and materialize its token trace."""
        trace = self.sample_decode_tensors_with_trace(
            states,
            options,
            decks,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
            generator=generator,
        )
        return SampleDecodeTrace(
            actions=actions_from_decode_tensors(
                trace.choice_indices,
                trace.append_masks,
            ),
            action_logprobs=trace.action_logprobs,
            values=trace.values,
            token_logprobs=trace.token_logprobs,
            prefix_values=trace.prefix_values,
            token_mask=trace.token_mask,
            stop_sampled=trace.stop_sampled,
        )

    def evaluate_actions(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Return differentiable sequence log-probs, entropies, and values."""
        conditioned = self._encode_conditioned_state(states, decks)
        return self.evaluate_actions_from_conditioned(
            conditioned,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )

    def evaluate_recurrent_actions(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        events: PublicEventBatch,
        sequence_offsets: Tensor,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Teacher-force actions after replaying each full recurrent history."""
        conditioned = self.recurrent_replay_from_conditioned(
            self._encode_conditioned_state(states, decks),
            events,
            sequence_offsets,
        )
        return self.evaluate_actions_from_conditioned(
            conditioned,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )

    def evaluate_action_sequences_from_conditioned(
        self,
        conditioned: ConditionedStateOutput,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Evaluate complete-action statistics from one prepared context."""
        context = self._policy_context_from_conditioned(conditioned, options)
        policy_scores = self.policy_head.teacher_forced_sequence_scores(
            context.policy_global,
            context.option_embeddings,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            route_plan=context.route_plan,
            projected_options=context.projected_options,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )
        return ActionEvaluation(
            action_logprobs=policy_scores.action_logprobs,
            entropies=policy_scores.entropies,
            values=self._root_values(conditioned),
            step_logits=(),
        )

    def evaluate_recurrent_action_sequences(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        events: PublicEventBatch,
        sequence_offsets: Tensor,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Evaluate joint PPO statistics after full recurrent replay."""
        conditioned = self.recurrent_replay_from_conditioned(
            self._encode_conditioned_state(states, decks),
            events,
            sequence_offsets,
        )
        return self.evaluate_action_sequences_from_conditioned(
            conditioned,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )

    def evaluate_action_sequences(
        self,
        states: StateBatch,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        decks: DeckBatch | None = None,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        validate_temperature: bool = True,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> ActionEvaluation:
        """Evaluate only sequence policy statistics and root value for PPO."""
        conditioned = self._encode_conditioned_state(states, decks)
        return self.evaluate_action_sequences_from_conditioned(
            conditioned,
            options,
            actions,
            action_targets=action_targets,
            temperature=temperature,
            validate_temperature=validate_temperature,
            validate_actions=validate_actions,
            count_first_rows_present=count_first_rows_present,
        )

    def _encode_conditioned_state(
        self,
        states: StateBatch,
        decks: DeckBatch | None,
    ) -> ConditionedStateOutput:
        """Resolve one route plan and produce the unified state context."""
        self._validate_decks(states, decks)
        conditioning = self.config.deck_conditioning
        if conditioning is None or not conditioning.enabled:
            return ConditionedStateOutput(
                encoded_state=self.state_encoder(
                    states,
                    attack_embedding=self.policy_head.attack_embedding,
                ),
                deck_embedding=None,
                route_plan=None,
            )
        if decks is None:
            raise ValueError("enabled deck conditioning requires a DeckBatch")
        if decks.card_ids.device != states.card_ids.device:
            raise ValueError("deck and state tensors must use the same device")
        plan = resolve_deck_route_plan(decks, conditioning)
        deck_embedding: Tensor | None = None
        if conditioning.deck_context_mode == "folded":
            deck_global_residual = torch.zeros(
                (len(decks), self.config.state_encoder.d_model),
                device=states.card_ids.device,
                dtype=_module_dtype(self.state_encoder),
            )
        else:
            if self.deck_encoder is None or self.deck_input_projection is None:
                raise RuntimeError("enabled deck conditioning modules are missing")
            deck_embedding = encode_deck_compositions(
                decks,
                plan,
                self.deck_encoder,
                card_encoder=self.card_encoder,
            )
            deck_global_residual = self.deck_input_projection(deck_embedding)
        if (
            conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        ):
            plan = replace(
                plan,
                deck_embeddings=deck_embedding,
                exact_capsules=self.exact_capsules,
            )
        return ConditionedStateOutput(
            encoded_state=self.state_encoder.forward_conditioned(
                states,
                deck_global_residual=deck_global_residual,
                route_plan=plan,
                exact_capsules=self.exact_capsules,
                attack_embedding=self.policy_head.attack_embedding,
            ),
            deck_embedding=deck_embedding,
            route_plan=plan,
        )

    def _validate_decks(
        self,
        states: StateBatch,
        decks: DeckBatch | None,
    ) -> None:
        """Validate required deck presence and persistent row alignment."""
        conditioning = self.config.deck_conditioning
        if conditioning is not None and conditioning.enabled and decks is None:
            raise ValueError("enabled deck conditioning requires a DeckBatch")
        if decks is not None and len(decks) != int(states.card_ids.shape[0]):
            raise ValueError("deck batch must align with state batch rows")
        if (
            decks is not None
            and conditioning is not None
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
        ):
            selected_signature = conditioning.expert_routes[0].signature
            if any(signature != selected_signature for signature in decks.signatures):
                raise ValueError(
                    "fixed deck-specialized model cannot evaluate a different deck"
                )

    def _decision_embedding(self, conditioned: ConditionedStateOutput) -> Tensor:
        """Return the one shared vector consumed by policy and value heads."""
        snapshot = conditioned.encoded_state.global_embedding
        decision = conditioned.decision_embedding
        if self.recurrent_policy is not None and decision is None:
            raise RuntimeError(
                "recurrent policy requires an explicit event/state transition"
            )
        if decision is None:
            return snapshot
        if decision.shape != snapshot.shape:
            raise ValueError("decision and snapshot embedding shapes differ")
        if decision.device != snapshot.device:
            raise ValueError("decision and snapshot embedding devices differ")
        if decision.dtype != snapshot.dtype:
            raise ValueError("decision and snapshot embedding dtypes differ")
        return decision

    def _required_recurrent_policy(self) -> RecurrentPolicyCore:
        core = self.recurrent_policy
        if core is None:
            raise RuntimeError("recurrent policy is disabled")
        return core

    def _policy_global(self, conditioned: ConditionedStateOutput) -> Tensor:
        """Apply active-only private policy-query residuals once per decode."""
        global_embedding = self._decision_embedding(conditioned)
        if conditioned.route_plan is None:
            return global_embedding
        conditioning = self.config.deck_conditioning
        if conditioning is not None and conditioning.architecture_version in {
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
        }:
            return global_embedding + self.policy_head.apply_private_global_residual(
                global_embedding,
                conditioned.route_plan,
            )
        return global_embedding + apply_private_residual(
            global_embedding,
            conditioned.route_plan,
            self.private_policy_adapters,
        )

    def _root_values(self, conditioned: ConditionedStateOutput) -> Tensor:
        """Return bounded shared plus active private pre-tanh values."""
        conditioning = self.config.deck_conditioning
        fixed_dense_private = (
            conditioning is not None
            and conditioning.dense_private is not None
            and conditioning.dense_private.export_mode == "fixed"
        )
        dtype_module: nn.Module = self.value_head
        if fixed_dense_private:
            selected_key = (
                cast(
                    DeckConditioningConfig,
                    conditioning,
                )
                .expert_routes[0]
                .module_key
            )
            dtype_module = self.dense_private_root_value_heads[selected_key]
        value_input = self._decision_embedding(conditioned).to(
            dtype=_module_dtype(dtype_module)
        )
        with _autocast_disabled(value_input):
            if (
                conditioned.route_plan is not None
                and conditioning is not None
                and conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            ):
                pre_tanh = value_input.new_zeros((value_input.shape[0],))
                generic_rows = conditioned.route_plan.generic_row_indices
                if int(generic_rows.numel()) > 0:
                    if fixed_dense_private:
                        raise ValueError(
                            "fixed dense-private value head cannot use a generic route"
                        )
                    generic_inputs = value_input.index_select(0, generic_rows)
                    generic_hidden = self.value_head[1](
                        self.value_head[0](generic_inputs)
                    )
                    generic_values = cast(
                        Tensor,
                        self.value_head[2](generic_hidden),
                    ).squeeze(-1)
                    pre_tanh = pre_tanh.index_copy(
                        0,
                        generic_rows,
                        generic_values,
                    )
                for group in conditioned.route_plan.groups:
                    private_inputs = value_input.index_select(0, group.row_indices)
                    private_values = cast(
                        Tensor,
                        self.dense_private_root_value_heads[group.module_key](
                            private_inputs
                        ),
                    )
                    pre_tanh = pre_tanh.index_copy(
                        0,
                        group.row_indices,
                        private_values,
                    )
                return torch.tanh(pre_tanh)
            hidden = self.value_head[1](self.value_head[0](value_input))
            pre_tanh = cast(Tensor, self.value_head[2](hidden))
            if (
                conditioned.route_plan is not None
                and conditioning is not None
                and conditioning.architecture_version
                == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
            ):
                pre_tanh = pre_tanh + _exact_value_residual(
                    value_input,
                    conditioned.route_plan,
                    self.exact_capsules,
                    target="root",
                )
            elif conditioned.route_plan is not None:
                pre_tanh = pre_tanh + apply_private_residual(
                    value_input,
                    conditioned.route_plan,
                    self.private_root_value_heads,
                    output_dim=1,
                )
            return cast(Tensor, self.value_head[3](pre_tanh)).squeeze(-1)

    def _prefix_values(
        self,
        root_values: Tensor,
        prefix_queries: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Return bounded prefix values with the root as the exact first prefix."""
        conditioning = self.config.deck_conditioning
        fixed_dense_private = (
            conditioning is not None
            and conditioning.dense_private is not None
            and conditioning.dense_private.export_mode == "fixed"
        )
        dtype_module: nn.Module = self.prefix_value_delta_head
        if fixed_dense_private:
            selected_key = (
                cast(
                    DeckConditioningConfig,
                    conditioning,
                )
                .expert_routes[0]
                .module_key
            )
            dtype_module = self.dense_private_prefix_value_heads[selected_key]
        prefix_input = prefix_queries.to(dtype=_module_dtype(dtype_module))
        with _autocast_disabled(prefix_input):
            if (
                route_plan is not None
                and conditioning is not None
                and conditioning.architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
            ):
                deltas = prefix_input.new_zeros(prefix_input.shape[:2])
                generic_rows = route_plan.generic_row_indices
                if int(generic_rows.numel()) > 0:
                    if fixed_dense_private:
                        raise ValueError(
                            "fixed dense-private prefix head cannot use a generic route"
                        )
                    generic_inputs = prefix_input.index_select(0, generic_rows)
                    generic_deltas = cast(
                        Tensor,
                        self.prefix_value_delta_head(generic_inputs),
                    ).squeeze(-1)
                    deltas = deltas.index_copy(
                        0,
                        generic_rows,
                        generic_deltas,
                    )
                for group in route_plan.groups:
                    private_inputs = prefix_input.index_select(
                        0,
                        group.row_indices,
                    )
                    private_deltas = cast(
                        Tensor,
                        self.dense_private_prefix_value_heads[group.module_key](
                            private_inputs
                        ),
                    )
                    deltas = deltas.index_copy(
                        0,
                        group.row_indices,
                        private_deltas,
                    )
            else:
                deltas = cast(
                    Tensor,
                    self.prefix_value_delta_head(prefix_input),
                ).squeeze(-1)
                if (
                    route_plan is not None
                    and conditioning is not None
                    and conditioning.architecture_version
                    == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
                ):
                    deltas = deltas + _exact_value_residual(
                        prefix_input,
                        route_plan,
                        self.exact_capsules,
                        target="prefix",
                    ).squeeze(-1)
                elif route_plan is not None:
                    private_deltas = apply_private_residual(
                        prefix_input,
                        route_plan,
                        self.private_prefix_value_heads,
                        output_dim=1,
                    ).squeeze(-1)
                    deltas = deltas + private_deltas
        if int(deltas.shape[1]) > 0:
            deltas = torch.cat(
                (torch.zeros_like(deltas[:, :1]), deltas[:, 1:]),
                dim=1,
            )
        return (root_values.unsqueeze(1) + deltas).clamp(min=-1.0, max=1.0)


@dataclass(frozen=True)
class _PolicyContextProposalScorer:
    """Bind one reusable policy context to the generic prefix search."""

    model: AgentPolicyValueNet
    context: PolicyEvaluationContext
    options: OptionBatch
    ordered_rows: Tensor
    decks: DeckBatch | None

    def score_prefixes(
        self,
        queries: tuple[PlannerProposalPrefixQuery, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Return learned proposal log-probabilities for one prefix wave."""
        return self.model._planner_proposal_prefix_logprobs_from_context(
            self.context,
            self.options,
            queries,
            ordered_rows=self.ordered_rows,
            decks=self.decks,
        )


def _select_state_rows(states: StateBatch, indices: Tensor) -> StateBatch:
    """Select or repeat state rows while preserving optional feature tensors."""

    def select(tensor: Tensor) -> Tensor:
        return tensor.index_select(0, indices.to(device=tensor.device))

    def optional_select(tensor: Tensor | None) -> Tensor | None:
        return None if tensor is None else select(tensor)

    selected_fingerprints: tuple[str, ...] = ()
    if states.root_input_fingerprints:
        selected_fingerprints = tuple(
            states.root_input_fingerprints[int(index)]
            for index in indices.detach().cpu().tolist()
        )

    return StateBatch(
        card_ids=select(states.card_ids),
        areas=select(states.areas),
        owner_roles=select(states.owner_roles),
        token_kinds=select(states.token_kinds),
        scalars=select(states.scalars),
        last_attack_ids=select(states.last_attack_ids),
        padding_mask=select(states.padding_mask),
        attachment_card_ids=optional_select(states.attachment_card_ids),
        attachment_parent_indices=optional_select(states.attachment_parent_indices),
        attachment_kinds=optional_select(states.attachment_kinds),
        entity_slots=optional_select(states.entity_slots),
        root_input_fingerprints=selected_fingerprints,
    )


def _select_option_rows(options: OptionBatch, indices: Tensor) -> OptionBatch:
    """Select or repeat all aligned option tensors."""

    def select(tensor: Tensor) -> Tensor:
        return tensor.index_select(0, indices.to(device=tensor.device))

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
    )


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _required_proposal_action_logprobs(
    evaluation: TeacherForcedEvaluation | TeacherForcedActionScores,
) -> Tensor:
    values = evaluation.proposal_action_logprobs
    if values is None:
        raise RuntimeError("model did not return proposal action log-probabilities")
    return values


def _required_completed_action_latents(
    evaluation: TeacherForcedEvaluation | TeacherForcedActionScores,
) -> Tensor:
    values = evaluation.completed_action_latents
    if values is None:
        raise RuntimeError("model did not return completed-action latents")
    return values


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _exact_value_residual(
    inputs: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    *,
    target: str,
) -> Tensor:
    """Apply one exact-deck value calibrator only to its owning rows."""
    residual = inputs.new_zeros((*inputs.shape[:-1], 1))
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        if target not in capsule.value_calibrators:
            raise KeyError(f"missing exact value calibrator {target!r}")
        row_indices.append(group.row_indices)
        contributions.append(
            capsule.value_calibrators[target](
                inputs.index_select(0, group.row_indices)
            ).to(dtype=residual.dtype)
        )
    if not contributions:
        return residual
    return residual.index_copy(
        0,
        torch.cat(row_indices),
        torch.cat(contributions),
    )


def _exact_capsule_calibrator(
    inputs: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    *,
    target: str,
    output_features: int,
) -> Tensor:
    """Apply an arbitrary-width exact value/Q/WDL calibration branch."""
    residual = inputs.new_zeros((*inputs.shape[:-1], output_features))
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        if target not in capsule.value_calibrators:
            raise KeyError(f"missing exact value calibrator {target!r}")
        row_indices.append(group.row_indices)
        contributions.append(
            capsule.value_calibrators[target](
                inputs.index_select(0, group.row_indices)
            ).to(dtype=residual.dtype)
        )
    if not contributions:
        return residual
    return residual.index_copy(
        0,
        torch.cat(row_indices),
        torch.cat(contributions),
    )


def _apply_exact_action_value_calibration(
    prediction: ActionValuePrediction,
    state_latents: Tensor,
    action_latents: Tensor,
    *,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
) -> ActionValuePrediction:
    """Add exact state and complete-action WDL residuals consistently."""
    state_delta = _exact_capsule_calibrator(
        state_latents,
        route_plan,
        capsules,
        target="action_state",
        output_features=3,
    )
    action_delta = _exact_capsule_calibrator(
        torch.cat((state_latents, action_latents), dim=-1),
        route_plan,
        capsules,
        target="action",
        output_features=3,
    )
    action_delta = action_delta - action_delta.mean(dim=-1, keepdim=True)
    logits = prediction.logits + state_delta + action_delta
    state_logits = prediction.state_logits + state_delta
    probabilities = torch.softmax(logits.float(), dim=-1)
    first_candidate_rows: list[int] = []
    offset = 0
    for count in prediction.candidate_counts:
        first_candidate_rows.append(offset)
        offset += count
    row_indices = torch.tensor(
        first_candidate_rows,
        dtype=torch.long,
        device=state_logits.device,
    )
    return replace(
        prediction,
        logits=logits,
        probabilities=probabilities,
        expected_scores=wdl_expected_score(probabilities),
        state_logits=state_logits,
        information_set_state_logits=state_logits.index_select(0, row_indices),
    )


def _is_compositional_config(config: AgentNetworkConfig) -> bool:
    conditioning = config.deck_conditioning
    return (
        conditioning is not None
        and conditioning.enabled
        and conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
    )


def _autocast_disabled(tensor: Tensor) -> AbstractContextManager[None]:
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def load_agent_policy_value_state_dict(
    model: AgentPolicyValueNet,
    state_dict: Mapping[str, Tensor],
    *,
    source_config: AgentNetworkConfig | None,
    additional_allowed_missing: Sequence[str] = (),
) -> None:
    """Strictly import a compatible legacy or conditioned model state."""
    target_conditioning = _enabled_conditioning(model.config)
    source_conditioning = (
        None if source_config is None else _enabled_conditioning(source_config)
    )
    if source_conditioning is not None and target_conditioning is None:
        raise RuntimeError(
            "cannot import a deck-conditioned checkpoint into a legacy model"
        )

    allowed_missing = set(additional_allowed_missing)
    if target_conditioning is not None and source_conditioning is None:
        allowed_missing.update(
            name
            for name in model.state_dict()
            if name.startswith(DECK_CONDITIONING_STATE_PREFIXES)
        )
    elif target_conditioning is not None and source_conditioning is not None:
        if (
            source_conditioning.architecture_version
            > target_conditioning.architecture_version
        ):
            raise RuntimeError(
                "cannot import a newer deck architecture into an older one"
            )
        if (
            source_conditioning.architecture_version
            == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
            and target_conditioning.architecture_version
            == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        ):
            _validate_lora_warm_start_compatibility(
                source_config=cast(AgentNetworkConfig, source_config),
                source_conditioning=source_conditioning,
                target_config=model.config,
                target_conditioning=target_conditioning,
            )
        source_profiles = {
            profile.signature: profile for profile in source_conditioning.active_routes
        }
        target_profiles = {
            profile.signature: profile for profile in target_conditioning.active_routes
        }
        if any(
            target_profiles.get(signature) is None
            or target_profiles[signature].module_key != profile.module_key
            for signature, profile in source_profiles.items()
        ):
            raise RuntimeError(
                "target deck registry must retain every source private profile"
            )
        added_module_keys = {
            profile.module_key
            for signature, profile in target_profiles.items()
            if signature not in source_profiles
        }
        allowed_missing.update(
            name
            for name in model.state_dict()
            if name.startswith(DECK_CONDITIONING_STATE_PREFIXES)
            and any(f".{module_key}." in name for module_key in added_module_keys)
        )
        if (
            source_conditioning.architecture_version
            < target_conditioning.architecture_version
        ):
            allowed_missing.update(
                name
                for name in model.state_dict()
                if name.startswith(
                    (
                        "state_encoder.private_lora.",
                        "policy_head.private_lora.",
                    )
                )
            )

    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = sorted(missing - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint state dict is incompatible with target architecture: "
            f"missing={disallowed_missing}, unexpected={sorted(unexpected)}"
        )


def _validate_lora_warm_start_compatibility(
    *,
    source_config: AgentNetworkConfig,
    source_conditioning: DeckConditioningConfig,
    target_config: AgentNetworkConfig,
    target_conditioning: DeckConditioningConfig,
) -> None:
    """Reject v2 imports that would reinterpret routed LoRA parameters."""
    source_lora = source_conditioning.lora
    target_lora = target_conditioning.lora
    if source_lora is None or target_lora is None:
        raise RuntimeError("architecture-v2 checkpoints require LoRA configuration")
    source_semantics = {
        "rank": source_lora.rank,
        "alpha": source_lora.alpha,
        "dropout": source_lora.dropout,
        "transformer_layers": source_lora.resolved_transformer_layers(
            num_layers=source_config.state_encoder.num_layers
        ),
        "transformer_targets": source_lora.transformer_targets,
        "policy_targets": source_lora.policy_targets,
        "export_mode": source_lora.export_mode,
    }
    target_semantics = {
        "rank": target_lora.rank,
        "alpha": target_lora.alpha,
        "dropout": target_lora.dropout,
        "transformer_layers": target_lora.resolved_transformer_layers(
            num_layers=target_config.state_encoder.num_layers
        ),
        "transformer_targets": target_lora.transformer_targets,
        "policy_targets": target_lora.policy_targets,
        "export_mode": target_lora.export_mode,
    }
    if source_semantics != target_semantics or source_lora.export_mode != "routed":
        raise RuntimeError("source and target LoRA semantics must match exactly")


def _enabled_conditioning(
    config: AgentNetworkConfig,
) -> DeckConditioningConfig | None:
    conditioning = config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        return None
    return conditioning


def build_agent_policy_value_net(config: AgentNetworkConfig) -> AgentPolicyValueNet:
    """Build the policy/value network and load configured card static features."""
    card_encoder: CardEncoder | None = None
    if config.card_encoder is not None:
        if config.card_encoder.d_model != config.state_encoder.d_model:
            raise ValueError("card_encoder.d_model must match state_encoder.d_model")
        card_encoder = build_card_encoder(config.card_encoder)
    return AgentPolicyValueNet(config, card_encoder=card_encoder)
