"""Pointer policy head for variable select-option sets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

from ptcg_rl.actions.encoding import (
    ATTACHMENT_IDENTITY_FEATURE_SIZE,
    LEGACY_SCALAR_FEATURE_SIZE,
    SCALAR_FEATURE_SIZE,
    EncodedOption,
    EncodedOptionArrayFeatures,
    EncodedOptionInput,
)
from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.factual_schema import (
    FACTUAL_ACTOR_RELATION_COUNT,
    FACTUAL_NEXT_CONTEXT_COUNT,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.compositional_capsule import (
    ProjectionShape,
    exact_capsule,
)
from ptcg_rl.model.compositional_projection import (
    DeckConditionedFiLM,
    FixedCompositionalLinear,
    FixedDeckFiLM,
    SharedCompositionalLinear,
)
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DeckConditioningConfig,
    DeckRoutePlan,
    PrivateResidualAdapter,
)
from ptcg_rl.model.deck_lora import RoutedLinearLoRA
from ptcg_rl.model.private_strategy import (
    DensePrivatePolicyStrategy,
    OptionSetInteractionBlock,
)

MAX_ENTITY_SLOTS = 2
KNOWN_OPTION_TYPE_COUNT = 17
OPTION_TYPE_OOV_INDEX = KNOWN_OPTION_TYPE_COUNT
OPTION_TYPE_EMBEDDING_COUNT = OPTION_TYPE_OOV_INDEX + 1
KNOWN_CONTEXT_COUNT = 49
CONTEXT_OOV_INDEX = KNOWN_CONTEXT_COUNT
CONTEXT_EMBEDDING_COUNT = CONTEXT_OOV_INDEX + 1
DEFAULT_MAX_ATTACK_ID = 1556
TEACHER_FORCED_PADDING_TARGET = -1
DECODER_CARDINALITY_FEATURE_SIZE = 4
COUNT_FIRST_FEATURE_SIZE = 4
ORDERED_HISTORY_DECAY = 0.5
_LEGACY_ZERO_POINTER_POLICY_PARAMETER_NAMES = (
    "attachment_identity_projection.weight",
    "ordered_history_projection.weight",
    "decoder_cardinality_projection.weight",
    "proposal_query_projection.weight",
)
_FACTUAL_POINTER_POLICY_PARAMETER_NAMES = (
    "factual_presence_head.0.weight",
    "factual_presence_head.0.bias",
    "factual_presence_head.2.weight",
    "factual_presence_head.2.bias",
    "factual_magnitude_head.0.weight",
    "factual_magnitude_head.0.bias",
    "factual_magnitude_head.2.weight",
    "factual_magnitude_head.2.bias",
    "factual_actor_relation_head.0.weight",
    "factual_actor_relation_head.0.bias",
    "factual_actor_relation_head.2.weight",
    "factual_actor_relation_head.2.bias",
    "factual_next_context_head.0.weight",
    "factual_next_context_head.0.bias",
    "factual_next_context_head.2.weight",
    "factual_next_context_head.2.bias",
)
_COUNT_FIRST_POINTER_POLICY_PARAMETER_NAMES = (
    "count_state_projection.weight",
    "count_state_projection.bias",
    "count_feature_projection.weight",
    "count_feature_projection.bias",
    "count_output_projection.weight",
    "count_output_projection.bias",
)
_RETIRED_FACTUAL_V1_PARAMETER_NAMES = (
    "factual_effect_head.0.weight",
    "factual_effect_head.0.bias",
    "factual_effect_head.2.weight",
    "factual_effect_head.2.bias",
    "factual_endpoint_head.0.weight",
    "factual_endpoint_head.0.bias",
    "factual_endpoint_head.2.weight",
    "factual_endpoint_head.2.bias",
    "factual_q_head.0.weight",
    "factual_q_head.0.bias",
    "factual_q_head.2.weight",
    "factual_q_head.2.bias",
)
_LEGACY_POINTER_POLICY_PARAMETER_NAMES = (
    *_LEGACY_ZERO_POINTER_POLICY_PARAMETER_NAMES,
    *_FACTUAL_POINTER_POLICY_PARAMETER_NAMES,
    *_COUNT_FIRST_POINTER_POLICY_PARAMETER_NAMES,
)
LEGACY_POINTER_POLICY_MISSING_KEYS = frozenset(
    f"policy_head.{name}" for name in _LEGACY_POINTER_POLICY_PARAMETER_NAMES
)


class _ZeroInitializedLinear(nn.Linear):
    """Linear residual whose construction does not advance model RNG."""

    def reset_parameters(self) -> None:
        """Initialize the compatibility residual as exact zeros."""
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


@dataclass(frozen=True)
class OptionBatch:
    """Padded batch of encoded legal select options."""

    option_types: Tensor
    contexts: Tensor
    entity_slots: Tensor
    entity_slot_mask: Tensor
    attack_ids: Tensor
    card_ids: Tensor
    scalars: Tensor
    dynamic_effect_features: Tensor
    dynamic_effect_masks: Tensor
    valid_options: Tensor
    min_counts: Tensor
    max_counts: Tensor
    option_lengths: tuple[int, ...] = ()
    maximum_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class TeacherForcedStep:
    """One active teacher-forced autoregressive decode step."""

    logits: Tensor
    active_indices: Tensor
    targets: Tensor


@dataclass(frozen=True)
class TeacherForcedEvaluation:
    """Teacher-forced log-probs and diagnostics for selected actions."""

    action_logprobs: Tensor
    entropies: Tensor
    step_logits: tuple[Tensor, ...]
    steps: tuple[TeacherForcedStep, ...]
    first_logits: Tensor
    token_logprobs: Tensor
    token_entropies: Tensor
    token_mask: Tensor
    prefix_queries: Tensor
    stop_sampled: Tensor
    completed_action_latents: Tensor | None
    factual_presence_logits: Tensor | None
    factual_magnitude_predictions: Tensor | None
    factual_actor_relation_logits: Tensor | None
    factual_next_context_logits: Tensor | None
    proposal_action_logprobs: Tensor | None
    proposal_entropies: Tensor | None


@dataclass(frozen=True)
class TeacherForcedActionScores:
    """Minimal differentiable outputs from complete-action replay."""

    action_logprobs: Tensor
    completed_action_latents: Tensor | None
    proposal_action_logprobs: Tensor | None


@dataclass(frozen=True)
class TeacherForcedSequenceScores:
    """Minimal sequence-level policy outputs for decision-credit PPO."""

    action_logprobs: Tensor
    entropies: Tensor


@dataclass(frozen=True)
class SampleDecodeTensorOutput:
    """Tensor-only sampled decode output before Python action materialization."""

    choice_indices: Tensor
    append_masks: Tensor
    action_logprobs: Tensor
    token_logprobs: Tensor
    token_mask: Tensor
    prefix_queries: Tensor
    stop_sampled: Tensor


@dataclass(frozen=True)
class SampleActionTensorOutput:
    """Minimal tensor outputs needed by behavior candidate sampling."""

    choice_indices: Tensor
    append_masks: Tensor
    action_logprobs: Tensor


@dataclass(frozen=True)
class BaseProposalStepLogits:
    """Base and proposal logits produced from one shared prefix forward."""

    base: Tensor
    proposal: Tensor
    prefix_query: Tensor


class PointerPolicyConfig(BaseModel):
    """Config for pointer-style candidate scoring."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = 128
    hidden_dim: int | None = None
    dropout: float = 0.0
    max_attack_id: int = DEFAULT_MAX_ATTACK_ID
    dynamic_effect_feature_size: int = DYNAMIC_EFFECT_FEATURE_SIZE
    unordered_set_policy: Literal["autoregressive_stop", "count_first"] = (
        "autoregressive_stop"
    )

    @field_validator("d_model", "max_attack_id", "dynamic_effect_feature_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid positive integer fields."""
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("hidden_dim")
    @classmethod
    def valid_hidden_dim(cls, value: int | None) -> int | None:
        """Reject non-positive hidden dimensions."""
        if value is not None and value <= 0:
            raise ValueError("hidden_dim must be positive when set")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Reject invalid dropout rates."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value


def collate_encoded_options(
    options: Sequence[EncodedOptionInput],
    *,
    min_counts: Sequence[int],
    max_counts: Sequence[int],
    device: torch.device | str | None = None,
) -> OptionBatch:
    """Pad encoded option rows and normalized cardinality bounds."""
    if len(options) != len(min_counts) or len(options) != len(max_counts):
        raise ValueError("options, min_counts, and max_counts must align")
    batch_size = len(options)
    if batch_size <= 0:
        raise ValueError("options must be non-empty")
    max_options = max((len(row) for row in options), default=0)

    option_type_rows = np.zeros((batch_size, max_options), dtype=np.int64)
    context_rows = np.zeros((batch_size, max_options), dtype=np.int64)
    entity_rows = np.zeros(
        (batch_size, max_options, MAX_ENTITY_SLOTS),
        dtype=np.int64,
    )
    entity_mask_rows = np.zeros(
        (batch_size, max_options, MAX_ENTITY_SLOTS),
        dtype=np.bool_,
    )
    attack_rows = np.zeros((batch_size, max_options), dtype=np.int64)
    card_rows = np.zeros((batch_size, max_options), dtype=np.int64)
    scalar_rows = np.zeros(
        (batch_size, max_options, SCALAR_FEATURE_SIZE),
        dtype=np.float32,
    )
    dynamic_rows = np.zeros(
        (batch_size, max_options, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    dynamic_mask_rows = np.zeros((batch_size, max_options), dtype=np.bool_)
    valid_rows = np.zeros((batch_size, max_options), dtype=np.bool_)
    normalized_mins = np.zeros(batch_size, dtype=np.int64)
    normalized_maxes = np.zeros(batch_size, dtype=np.int64)
    for row_index, (row, min_count, max_count) in enumerate(
        zip(options, min_counts, max_counts, strict=True)
    ):
        option_count = len(row)
        normalized_min = min(option_count, max(0, int(min_count)))
        normalized_max = min(option_count, max(normalized_min, int(max_count)))
        normalized_mins[row_index] = normalized_min
        normalized_maxes[row_index] = normalized_max

        if isinstance(row, EncodedOptionArrayFeatures):
            _copy_option_array_row(
                row,
                row_index=row_index,
                option_count=option_count,
                option_type_rows=option_type_rows,
                context_rows=context_rows,
                entity_rows=entity_rows,
                entity_mask_rows=entity_mask_rows,
                attack_rows=attack_rows,
                card_rows=card_rows,
                scalar_rows=scalar_rows,
                dynamic_rows=dynamic_rows,
                dynamic_mask_rows=dynamic_mask_rows,
                valid_rows=valid_rows,
            )
            continue

        for option_index, option in enumerate(row):
            _copy_option_row(
                option,
                row_index=row_index,
                option_index=option_index,
                option_type_rows=option_type_rows,
                context_rows=context_rows,
                entity_rows=entity_rows,
                entity_mask_rows=entity_mask_rows,
                attack_rows=attack_rows,
                card_rows=card_rows,
                scalar_rows=scalar_rows,
                dynamic_rows=dynamic_rows,
                dynamic_mask_rows=dynamic_mask_rows,
                valid_rows=valid_rows,
            )

    return OptionBatch(
        option_types=torch.as_tensor(option_type_rows, device=device),
        contexts=torch.as_tensor(context_rows, device=device),
        entity_slots=torch.as_tensor(entity_rows, device=device),
        entity_slot_mask=torch.as_tensor(
            entity_mask_rows,
            device=device,
        ),
        attack_ids=torch.as_tensor(attack_rows, device=device),
        card_ids=torch.as_tensor(card_rows, device=device),
        scalars=torch.as_tensor(scalar_rows, device=device),
        dynamic_effect_features=torch.as_tensor(
            dynamic_rows,
            device=device,
        ),
        dynamic_effect_masks=torch.as_tensor(
            dynamic_mask_rows,
            device=device,
        ),
        valid_options=torch.as_tensor(valid_rows, device=device),
        min_counts=torch.as_tensor(normalized_mins, device=device),
        max_counts=torch.as_tensor(normalized_maxes, device=device),
        option_lengths=tuple(len(row) for row in options),
        maximum_counts=tuple(int(value) for value in normalized_maxes),
    )


def _copy_option_array_row(
    row: EncodedOptionArrayFeatures,
    *,
    row_index: int,
    option_count: int,
    option_type_rows: np.ndarray,
    context_rows: np.ndarray,
    entity_rows: np.ndarray,
    entity_mask_rows: np.ndarray,
    attack_rows: np.ndarray,
    card_rows: np.ndarray,
    scalar_rows: np.ndarray,
    dynamic_rows: np.ndarray,
    dynamic_mask_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> None:
    option_type_rows[row_index, :option_count] = row.option_types
    context_rows[row_index, :option_count] = row.contexts
    slot_width = min(int(row.entity_slots.shape[1]), MAX_ENTITY_SLOTS)
    entity_rows[row_index, :option_count, :slot_width] = row.entity_slots[
        :,
        :slot_width,
    ]
    entity_mask_rows[row_index, :option_count, :slot_width] = row.entity_slot_mask[
        :,
        :slot_width,
    ]
    attack_rows[row_index, :option_count] = np.maximum(row.attack_ids, 0)
    card_rows[row_index, :option_count] = np.maximum(row.card_ids, 0)
    scalar_rows[row_index, :option_count, :] = row.scalars
    dynamic_rows[row_index, :option_count, :] = row.dynamic_effect_features
    dynamic_mask_rows[row_index, :option_count] = row.dynamic_effect_masks
    valid_rows[row_index, :option_count] = True


def _copy_option_row(
    option: EncodedOption,
    *,
    row_index: int,
    option_index: int,
    option_type_rows: np.ndarray,
    context_rows: np.ndarray,
    entity_rows: np.ndarray,
    entity_mask_rows: np.ndarray,
    attack_rows: np.ndarray,
    card_rows: np.ndarray,
    scalar_rows: np.ndarray,
    dynamic_rows: np.ndarray,
    dynamic_mask_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> None:
    option_type_rows[row_index, option_index] = int(option.option_type)
    context_rows[row_index, option_index] = int(option.context)
    slots = tuple(int(slot) for slot in option.entity_slots[:MAX_ENTITY_SLOTS])
    if slots:
        entity_rows[row_index, option_index, : len(slots)] = slots
        entity_mask_rows[row_index, option_index, : len(slots)] = True
    attack_rows[row_index, option_index] = max(0, int(option.attack_id))
    card_rows[row_index, option_index] = max(0, int(option.card_id))
    scalar_rows[row_index, option_index, :] = option.scalars
    dynamic_rows[row_index, option_index, :] = _dynamic_feature_list(
        option.dynamic_effect_features
    )
    dynamic_mask_rows[row_index, option_index] = bool(option.dynamic_effect_mask)
    valid_rows[row_index, option_index] = True


class _PrivateOnlyLinear(nn.Linear):
    """Shape metadata for a generic Linear removed from a fixed runtime."""

    def __init__(self, source: nn.Linear) -> None:
        """Retain dispatch dimensions without retaining any parameters."""
        nn.Module.__init__(self)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Reject an impossible generic route in a fixed-deck artifact."""
        del inputs
        raise RuntimeError("fixed dense-private policy cannot use a generic Linear")


class PointerPolicyHead(nn.Module):
    """Score legal select options and an autoregressive STOP token."""

    def __init__(
        self,
        config: PointerPolicyConfig | None = None,
        *,
        deck_conditioning: DeckConditioningConfig | None = None,
    ) -> None:
        """Initialize pointer candidate scoring layers."""
        super().__init__()
        self.config = config or PointerPolicyConfig()
        d_model = self.config.d_model
        hidden_dim = self.config.hidden_dim or d_model
        self.option_type_embedding = nn.Embedding(
            OPTION_TYPE_EMBEDDING_COUNT,
            d_model,
        )
        self.context_embedding = nn.Embedding(CONTEXT_EMBEDDING_COUNT, d_model)
        self.attack_embedding = nn.Embedding(
            self.config.max_attack_id + 1,
            d_model,
            padding_idx=0,
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(LEGACY_SCALAR_FEATURE_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.attachment_identity_projection = _ZeroInitializedLinear(
            ATTACHMENT_IDENTITY_FEATURE_SIZE,
            d_model,
            bias=False,
        )
        self.dynamic_effect_projection = nn.Sequential(
            nn.Linear(self.config.dynamic_effect_feature_size + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.option_projection = nn.Linear(d_model, d_model, bias=False)
        self.selected_projection = nn.Linear(d_model, d_model)
        self.ordered_history_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.decoder_cardinality_projection = nn.Linear(
            DECODER_CARDINALITY_FEATURE_SIZE,
            d_model,
            bias=False,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.proposal_query_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.stop_embedding = nn.Parameter(torch.empty(d_model))
        self.factual_presence_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, DYNAMIC_EFFECT_FEATURE_SIZE),
        )
        self.factual_magnitude_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, DYNAMIC_EFFECT_FEATURE_SIZE),
        )
        self.factual_actor_relation_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, FACTUAL_ACTOR_RELATION_COUNT),
        )
        self.factual_next_context_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, FACTUAL_NEXT_CONTEXT_COUNT),
        )
        # New count-head initialization must not move the legacy policy's
        # process-global random stream. This keeps non-count rows and old
        # initialization-seed fixtures bit-exact across the migration.
        with torch.random.fork_rng(devices=[]):
            self.count_state_projection = nn.Linear(d_model, hidden_dim)
            self.count_feature_projection = nn.Linear(
                COUNT_FIRST_FEATURE_SIZE,
                hidden_dim,
            )
            self.count_output_projection = nn.Linear(hidden_dim, 1)
        self.private_lora = nn.ModuleDict()
        self.private_strategies = nn.ModuleDict()
        self.compositional_linears = nn.ModuleDict()
        self.compositional_option_set: OptionSetInteractionBlock | None = None
        self.compositional_option_film = nn.ModuleDict()
        self.compositional_count_set_projection: nn.Linear | None = None
        self._compositional_fixed = False
        conditioning = (
            deck_conditioning
            if deck_conditioning is not None and deck_conditioning.enabled
            else None
        )
        if (
            conditioning is not None
            and conditioning.lora is not None
            and conditioning.lora.export_mode == "routed"
        ):
            with torch.random.fork_rng(devices=[]):
                self.private_lora = _policy_lora_modules(
                    conditioning,
                    target_linears=self._decision_linears(),
                )
        self.reset_parameters()
        if (
            conditioning is not None
            and conditioning.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ):
            dense_config = conditioning.dense_private
            if dense_config is None:
                raise ValueError("architecture v3 requires dense-private configuration")
            with torch.random.fork_rng(devices=[]):
                self.private_strategies = nn.ModuleDict(
                    {
                        route.module_key: DensePrivatePolicyStrategy(
                            self._decision_linears(),
                            self.stop_embedding,
                            PrivateResidualAdapter(
                                d_model,
                                conditioning.policy_bottleneck_dim,
                                dropout=conditioning.adapter_dropout,
                            ),
                            count_hidden_dim=hidden_dim,
                            option_set_bottleneck_dim=(dense_config.option_set_width),
                            option_set_attention_heads=(
                                dense_config.option_set_attention_heads
                            ),
                            option_set_feedforward_dim=(
                                dense_config.option_set_feedforward_dim
                            ),
                            dropout=self.config.dropout,
                        )
                        for route in conditioning.active_routes
                    }
                )
            if dense_config.export_mode == "fixed":
                self._remove_fixed_generic_decision_parameters()
        elif (
            conditioning is not None
            and conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        ):
            compositional = conditioning.compositional
            if compositional is None:
                raise ValueError("architecture v4 requires compositional config")
            self._compositional_fixed = compositional.export_mode == "fixed"
            decision_linears = self._decision_linears()
            self.compositional_linears = nn.ModuleDict(
                {
                    target: (
                        FixedCompositionalLinear(
                            in_features=linear.in_features,
                            out_features=linear.out_features,
                        )
                        if self._compositional_fixed
                        else SharedCompositionalLinear(
                            target=target,
                            domain="policy",
                            in_features=linear.in_features,
                            out_features=linear.out_features,
                            deck_dim=d_model,
                            basis_count=compositional.shared_basis_count,
                            shared_rank=compositional.policy_shared_rank,
                            router_hidden_dim=compositional.router_hidden_dim,
                        )
                    )
                    for target, linear in decision_linears.items()
                }
            )
            self.compositional_option_set = OptionSetInteractionBlock(
                d_model,
                bottleneck_dim=compositional.option_set_width,
                attention_heads=compositional.option_set_attention_heads,
                feedforward_dim=compositional.option_set_feedforward_dim,
                dropout=self.config.dropout,
            )
            self.compositional_option_film = nn.ModuleDict(
                {
                    site: (
                        FixedDeckFiLM(compositional.option_set_width)
                        if self._compositional_fixed
                        else DeckConditionedFiLM(
                            compositional.option_set_width,
                            d_model,
                            compositional.router_hidden_dim,
                            domain="policy",
                            site=f"option_{site}",
                        )
                    )
                    for site in ("attention", "feedforward")
                }
            )
            self.compositional_count_set_projection = nn.Linear(
                d_model,
                hidden_dim,
            )
            _zero_linear(self.compositional_count_set_projection)

    def _remove_fixed_generic_decision_parameters(self) -> None:
        """Remove generic decision weights duplicated by one fixed strategy."""
        self.scalar_projection[0] = _PrivateOnlyLinear(
            cast(nn.Linear, self.scalar_projection[0])
        )
        self.scalar_projection[3] = _PrivateOnlyLinear(
            cast(nn.Linear, self.scalar_projection[3])
        )
        self.dynamic_effect_projection[0] = _PrivateOnlyLinear(
            cast(nn.Linear, self.dynamic_effect_projection[0])
        )
        self.dynamic_effect_projection[3] = _PrivateOnlyLinear(
            cast(nn.Linear, self.dynamic_effect_projection[3])
        )
        self.option_projection = _PrivateOnlyLinear(self.option_projection)
        self.selected_projection = _PrivateOnlyLinear(self.selected_projection)
        self.ordered_history_projection = _PrivateOnlyLinear(
            self.ordered_history_projection
        )
        self.decoder_cardinality_projection = _PrivateOnlyLinear(
            self.decoder_cardinality_projection
        )
        self.query_projection[0] = _PrivateOnlyLinear(
            cast(nn.Linear, self.query_projection[0])
        )
        self.query_projection[3] = _PrivateOnlyLinear(
            cast(nn.Linear, self.query_projection[3])
        )
        self.count_state_projection = _PrivateOnlyLinear(self.count_state_projection)
        self.count_feature_projection = _PrivateOnlyLinear(
            self.count_feature_projection
        )
        self.count_output_projection = _PrivateOnlyLinear(self.count_output_projection)
        del self.stop_embedding
        self.register_parameter("stop_embedding", None)

    def _decision_linears(self) -> dict[str, nn.Linear]:
        """Return stable names for every routed policy/count projection."""
        return {
            "scalar_input": cast(nn.Linear, self.scalar_projection[0]),
            "scalar_output": cast(nn.Linear, self.scalar_projection[3]),
            "dynamic_input": cast(nn.Linear, self.dynamic_effect_projection[0]),
            "dynamic_output": cast(nn.Linear, self.dynamic_effect_projection[3]),
            "option_projection": self.option_projection,
            "selected_projection": self.selected_projection,
            "ordered_history_projection": self.ordered_history_projection,
            "decoder_cardinality_projection": self.decoder_cardinality_projection,
            "query_input": cast(nn.Linear, self.query_projection[0]),
            "query_output": cast(nn.Linear, self.query_projection[3]),
            "count_state_projection": self.count_state_projection,
            "count_feature_projection": self.count_feature_projection,
            "count_output_projection": self.count_output_projection,
        }

    def compositional_policy_shapes(self) -> dict[str, ProjectionShape]:
        """Return stable projection shapes used by root-owned exact capsules."""
        return {
            target: ProjectionShape(
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
            )
            for target, linear in self._decision_linears().items()
        }

    def reset_parameters(self) -> None:
        """Initialize STOP, adapters, and inert factual output layers."""
        nn.init.normal_(self.stop_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.ordered_history_projection.weight)
        nn.init.zeros_(self.decoder_cardinality_projection.weight)
        nn.init.zeros_(self.proposal_query_projection.weight)
        nn.init.zeros_(self.count_output_projection.weight)
        nn.init.zeros_(self.count_output_projection.bias)
        _zero_sequential_output(self.factual_presence_head)
        _zero_sequential_output(self.factual_magnitude_head)
        _zero_sequential_output(self.factual_actor_relation_head)
        _zero_sequential_output(self.factual_next_context_head)

    def _routed_linear(
        self,
        target: str,
        base: nn.Linear,
        inputs: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Apply one generic, dense-private, or legacy LoRA Linear route."""
        if route_plan is not None and target in self.compositional_linears:
            return cast(
                Tensor,
                cast(Any, self.compositional_linears[target])(
                    base,
                    inputs,
                    route_plan=route_plan,
                    capsules=_required_exact_capsules(route_plan),
                ),
            )
        if route_plan is not None and self.private_strategies:
            return self._dense_private_linear(
                target,
                base,
                inputs,
                route_plan,
            )
        if (
            route_plan is None
            or not route_plan.groups
            or target not in self.private_lora
        ):
            return cast(Tensor, base(inputs))
        bank = cast(RoutedLinearLoRA, self.private_lora[target])
        return cast(
            Tensor,
            bank(
                inputs,
                route_plan.groups,
                base=base,
                dispatch=route_plan.lora_dispatch,
                cache_stacked_weights=True,
            ),
        )

    def _dense_private_linear(
        self,
        target: str,
        base: nn.Linear,
        inputs: Tensor,
        route_plan: DeckRoutePlan,
    ) -> Tensor:
        """Dispatch rows without evaluating generic or inactive private paths."""
        if int(inputs.shape[0]) != route_plan.batch_size:
            raise ValueError("private policy inputs must align with route plan")
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        if int(route_plan.generic_row_indices.numel()) > 0:
            row_indices.append(route_plan.generic_row_indices)
            contributions.append(
                cast(
                    Tensor,
                    base(inputs.index_select(0, route_plan.generic_row_indices)),
                )
            )
        for group in route_plan.groups:
            strategy = cast(
                DensePrivatePolicyStrategy,
                self.private_strategies[group.module_key],
            )
            row_indices.append(group.row_indices)
            contributions.append(
                strategy.decision_linear(
                    target,
                    inputs.index_select(0, group.row_indices),
                )
            )
        if not contributions:
            raise ValueError("route plan does not cover any policy rows")
        covered_rows = sum(int(indices.numel()) for indices in row_indices)
        if covered_rows != route_plan.batch_size:
            raise ValueError("route plan must partition every policy row exactly once")
        output = contributions[0].new_empty((*inputs.shape[:-1], base.out_features))
        return output.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )

    def apply_private_global_residual(
        self,
        global_embedding: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Return active dense-strategy global residuals and generic zeros."""
        residual = torch.zeros_like(global_embedding)
        if route_plan is not None and self.compositional_linears:
            return _routed_capsule_residual(
                global_embedding,
                route_plan,
                _required_exact_capsules(route_plan),
                module_name="policy_global_residual",
            )
        if route_plan is None or not self.private_strategies:
            return residual
        if int(global_embedding.shape[0]) != route_plan.batch_size:
            raise ValueError("private policy inputs must align with route plan")
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            strategy = cast(
                DensePrivatePolicyStrategy,
                self.private_strategies[group.module_key],
            )
            row_indices.append(group.row_indices)
            contributions.append(
                strategy.global_residual(
                    global_embedding.index_select(0, group.row_indices)
                ).to(dtype=residual.dtype, device=residual.device)
            )
        if not contributions:
            return residual
        return residual.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )

    def _routed_stop_embeddings(
        self,
        inputs: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Return the generic or selected strategy STOP vector per row."""
        batch_size = int(inputs.shape[0])
        if route_plan is not None and self.compositional_linears:
            if batch_size != route_plan.batch_size:
                raise ValueError("compositional policy inputs must align with route")
            stop = self.stop_embedding.unsqueeze(0).expand(batch_size, -1)
            if self._compositional_fixed:
                return stop.to(device=inputs.device, dtype=inputs.dtype)
            exact_deltas = _routed_capsule_vector(
                route_plan,
                _required_exact_capsules(route_plan),
                parameter_name="stop_delta",
                width=self.config.d_model,
                reference=inputs,
            )
            return stop.to(device=inputs.device, dtype=inputs.dtype) + exact_deltas
        if route_plan is None or not self.private_strategies:
            if not isinstance(self.stop_embedding, Tensor):
                raise RuntimeError(
                    "fixed dense-private policy requires its exact deck route"
                )
            return self.stop_embedding.unsqueeze(0).expand(batch_size, -1)
        if batch_size != route_plan.batch_size:
            raise ValueError("private policy inputs must align with route plan")
        stop_embeddings = inputs.new_empty((batch_size, self.config.d_model))
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        if int(route_plan.generic_row_indices.numel()) > 0:
            generic_rows = route_plan.generic_row_indices
            if not isinstance(self.stop_embedding, Tensor):
                raise ValueError(
                    "fixed dense-private policy cannot use a generic route"
                )
            row_indices.append(generic_rows)
            contributions.append(
                self.stop_embedding.unsqueeze(0)
                .expand(
                    int(generic_rows.numel()),
                    -1,
                )
                .to(dtype=inputs.dtype, device=inputs.device)
            )
        for group in route_plan.groups:
            strategy = cast(
                DensePrivatePolicyStrategy,
                self.private_strategies[group.module_key],
            )
            row_indices.append(group.row_indices)
            contributions.append(
                strategy.stop_embedding.unsqueeze(0)
                .expand(
                    int(group.row_indices.numel()),
                    -1,
                )
                .to(dtype=inputs.dtype, device=inputs.device)
            )
        if not contributions:
            raise ValueError("route plan does not cover any policy rows")
        covered_rows = sum(int(indices.numel()) for indices in row_indices)
        if covered_rows != route_plan.batch_size:
            raise ValueError("route plan must partition every policy row exactly once")
        return stop_embeddings.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )

    def _private_option_deltas(
        self,
        option_embeddings: Tensor,
        state_embeddings: Tensor,
        options: OptionBatch,
        *,
        state_padding_mask: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Evaluate each active set-aware block once for its routed rows."""
        residual = torch.zeros_like(option_embeddings)
        if route_plan is not None and self.compositional_option_set is not None:
            capsules = _required_exact_capsules(route_plan)
            attention_film = cast(Any, self.compositional_option_film["attention"])
            feedforward_film = cast(
                Any,
                self.compositional_option_film["feedforward"],
            )
            modulators = {
                "attention": lambda values: attention_film(
                    values,
                    route_plan=route_plan,
                    capsules=capsules,
                ),
                "feedforward": lambda values: feedforward_film(
                    values,
                    route_plan=route_plan,
                    capsules=capsules,
                ),
            }
            shared = self.compositional_option_set(
                option_embeddings,
                state_embeddings,
                option_valid_mask=options.valid_options,
                state_padding_mask=state_padding_mask,
                residual_modulators=modulators,
            )
            exact = _routed_capsule_residual(
                option_embeddings,
                route_plan,
                capsules,
                module_name="option_adapter",
            )
            return cast(
                Tensor,
                (shared + exact).masked_fill(
                    ~options.valid_options.unsqueeze(-1),
                    0.0,
                ),
            )
        if route_plan is None or not self.private_strategies:
            return residual
        if int(option_embeddings.shape[0]) != route_plan.batch_size:
            raise ValueError("private option inputs must align with route plan")
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            strategy = cast(
                DensePrivatePolicyStrategy,
                self.private_strategies[group.module_key],
            )
            indices = group.row_indices
            row_indices.append(indices)
            contributions.append(
                strategy.option_delta(
                    option_embeddings.index_select(0, indices),
                    state_embeddings.index_select(0, indices),
                    option_valid_mask=options.valid_options.index_select(0, indices),
                    state_padding_mask=state_padding_mask.index_select(0, indices),
                ).to(dtype=residual.dtype, device=residual.device)
            )
        if not contributions:
            return residual
        return residual.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )

    def _private_count_set_hidden(
        self,
        option_embeddings: Tensor,
        options: OptionBatch,
        route_plan: DeckRoutePlan | None,
        *,
        reference: Tensor,
    ) -> Tensor:
        """Pool legal options invariantly for active private count heads."""
        residual = torch.zeros_like(reference)
        if (
            route_plan is not None
            and self.compositional_count_set_projection is not None
        ):
            pooled = _masked_option_mean(
                option_embeddings,
                options.valid_options,
            )
            shared = self.compositional_count_set_projection(pooled)
            if self._compositional_fixed:
                return cast(Tensor, shared.to(dtype=reference.dtype))
            exact = _routed_capsule_linear(
                pooled,
                route_plan,
                _required_exact_capsules(route_plan),
                module_name="count_set_projection",
                output_features=int(reference.shape[-1]),
            )
            return cast(Tensor, (shared + exact).to(dtype=reference.dtype))
        if route_plan is None or not self.private_strategies:
            return residual
        row_indices: list[Tensor] = []
        contributions: list[Tensor] = []
        for group in route_plan.groups:
            strategy = cast(
                DensePrivatePolicyStrategy,
                self.private_strategies[group.module_key],
            )
            indices = group.row_indices
            row_indices.append(indices)
            contributions.append(
                strategy.count_set_hidden(
                    option_embeddings.index_select(0, indices),
                    option_valid_mask=options.valid_options.index_select(0, indices),
                ).to(dtype=residual.dtype, device=residual.device)
            )
        if not contributions:
            return residual
        return residual.index_copy(
            0,
            torch.cat(row_indices),
            torch.cat(contributions),
        )

    def _routed_mlp(
        self,
        prefix: str,
        module: nn.Sequential,
        inputs: Tensor,
        route_plan: DeckRoutePlan | None,
    ) -> Tensor:
        """Apply a two-Linear policy MLP without changing legacy state keys."""
        first = cast(nn.Linear, module[0])
        second = cast(nn.Linear, module[3])
        hidden = self._routed_linear(
            f"{prefix}_input",
            first,
            inputs,
            route_plan,
        )
        hidden = module[2](module[1](hidden))
        return self._routed_linear(
            f"{prefix}_output",
            second,
            hidden,
            route_plan,
        )

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
        """Load legacy checkpoints with the new decoder adapters kept inert.

        Checkpoints written before the multi-select/proposal fix do not contain
        the zero-initialized projections. Supplying explicit zeros here
        makes both strict model-state loading and non-strict warm-start preserve
        the legacy policy exactly until training updates the new parameters.
        Shape errors for checkpoints that do contain either key remain hard
        failures in the standard ``nn.Module`` loader.
        """
        for name in _LEGACY_ZERO_POINTER_POLICY_PARAMETER_NAMES:
            key = f"{prefix}{name}"
            parameter = dict(self.named_parameters()).get(name)
            if key not in state_dict and parameter is not None:
                state_dict[key] = torch.zeros_like(parameter)
        for name in _RETIRED_FACTUAL_V1_PARAMETER_NAMES:
            state_dict.pop(f"{prefix}{name}", None)
        factual_keys = tuple(
            f"{prefix}{name}" for name in _FACTUAL_POINTER_POLICY_PARAMETER_NAMES
        )
        if not any(key in state_dict for key in factual_keys):
            for name, key in zip(
                _FACTUAL_POINTER_POLICY_PARAMETER_NAMES,
                factual_keys,
                strict=True,
            ):
                state_dict[key] = self.get_parameter(name).detach().clone()
        count_keys = tuple(
            f"{prefix}{name}" for name in _COUNT_FIRST_POINTER_POLICY_PARAMETER_NAMES
        )
        if not any(key in state_dict for key in count_keys):
            for name, key in zip(
                _COUNT_FIRST_POINTER_POLICY_PARAMETER_NAMES,
                count_keys,
                strict=True,
            ):
                parameter = dict(self.named_parameters()).get(name)
                if parameter is not None:
                    state_dict[key] = parameter.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def option_embeddings(
        self,
        token_embeddings: Tensor,
        options: OptionBatch,
        card_encoder: CardEncoder,
        *,
        route_plan: DeckRoutePlan | None = None,
        state_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return dense option embeddings aligned to ``options.valid_options``."""
        if state_padding_mask is None:
            state_padding_mask = torch.zeros(
                token_embeddings.shape[:2],
                dtype=torch.bool,
                device=token_embeddings.device,
            )
        elif state_padding_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("state_padding_mask must align with state tokens")
        else:
            state_padding_mask = state_padding_mask.to(
                device=token_embeddings.device,
                dtype=torch.bool,
            )
        entity_embeddings = _gather_entity_embeddings(
            token_embeddings,
            options.entity_slots,
            options.entity_slot_mask,
        )
        safe_option_types = _safe_indices(
            options.option_types,
            max_known=KNOWN_OPTION_TYPE_COUNT,
            oov_index=OPTION_TYPE_OOV_INDEX,
        )
        safe_contexts = _safe_indices(
            options.contexts,
            max_known=KNOWN_CONTEXT_COUNT,
            oov_index=CONTEXT_OOV_INDEX,
        )
        safe_attack_ids = torch.where(
            (options.attack_ids >= 0)
            & (options.attack_ids <= self.config.max_attack_id),
            options.attack_ids,
            torch.zeros_like(options.attack_ids),
        )
        if int(options.scalars.shape[-1]) != SCALAR_FEATURE_SIZE:
            raise ValueError(
                "option scalar width does not match the current public input schema"
            )
        embeddings = (
            entity_embeddings
            + self.option_type_embedding(safe_option_types)
            + self.context_embedding(safe_contexts)
            + self.attack_embedding(safe_attack_ids)
            + card_encoder(options.card_ids)
            + self._routed_mlp(
                "scalar",
                self.scalar_projection,
                options.scalars[..., :LEGACY_SCALAR_FEATURE_SIZE],
                route_plan,
            )
            + self.attachment_identity_projection(
                options.scalars[..., LEGACY_SCALAR_FEATURE_SIZE:]
            )
            + self._routed_mlp(
                "dynamic",
                self.dynamic_effect_projection,
                _dynamic_effect_inputs(
                    options.dynamic_effect_features,
                    options.dynamic_effect_masks,
                    feature_size=self.config.dynamic_effect_feature_size,
                ),
                route_plan,
            )
        )
        embeddings = embeddings + self._private_option_deltas(
            embeddings,
            token_embeddings,
            options,
            state_padding_mask=state_padding_mask,
            route_plan=route_plan,
        )
        return cast(
            Tensor,
            embeddings.masked_fill(~options.valid_options.unsqueeze(-1), 0.0),
        )

    def forward(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Return initial-step logits over ``options`` plus STOP."""
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = self.initial_ordered_history(option_embeddings)
        return self.step_logits(
            global_embedding,
            option_embeddings,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            route_plan=route_plan,
        )

    def step_logits(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor | None = None,
        ordered_rows: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Score the next autoregressive pick plus STOP."""
        projected_options = self._routed_linear(
            "option_projection",
            self.option_projection,
            option_embeddings,
            route_plan,
        )
        return self._step_logits_with_projection(
            global_embedding,
            option_embeddings,
            projected_options,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            ordered_rows=ordered_rows,
            route_plan=route_plan,
        )

    def project_option_embeddings(
        self,
        option_embeddings: Tensor,
        *,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Project one option batch once for repeated prefix scoring."""
        return self._routed_linear(
            "option_projection",
            self.option_projection,
            option_embeddings,
            route_plan,
        )

    def _step_logits_with_projection(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        projected_options: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor | None = None,
        ordered_rows: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Score one decode step using precomputed option projections."""
        _query, logits = self._step_query_and_logits_with_projection(
            global_embedding,
            option_embeddings,
            projected_options,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            ordered_rows=ordered_rows,
            route_plan=route_plan,
        )
        return logits

    def _step_query_and_logits_with_projection(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        projected_options: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor | None = None,
        ordered_rows: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
        proposal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return one shared prefix query and base or proposal logits."""
        if ordered_history is None:
            ordered_history = self.initial_ordered_history(option_embeddings)
        query = self._query_embedding(
            global_embedding,
            option_embeddings,
            options,
            selected_mask,
            selected_counts,
            ordered_history,
            route_plan,
        )
        option_logits = (projected_options * query.unsqueeze(1)).sum(dim=-1)
        option_logits = option_logits / math.sqrt(float(self.config.d_model))

        available = options.valid_options & ~selected_mask
        if int(available.shape[1]) > 0:
            option_indices = torch.arange(
                available.shape[1],
                device=available.device,
            ).unsqueeze(0)
            last_selected = torch.where(
                selected_mask,
                option_indices,
                torch.full_like(option_indices, -1),
            ).amax(dim=1)
            order_sensitive = _order_sensitive_rows(
                options,
                ordered_rows,
                enable_unordered_sets=(
                    self.config.unordered_set_policy == "count_first"
                ),
            )
            canonical_next = option_indices > last_selected.unsqueeze(1)
            available_counts_to_right = torch.flip(
                torch.cumsum(
                    torch.flip(available.to(dtype=torch.long), dims=(1,)),
                    dim=1,
                ),
                dims=(1,),
            ) - available.to(dtype=torch.long)
            required_after_pick = (options.min_counts - selected_counts - 1).clamp_min(
                0
            )
            canonical_next = canonical_next & available_counts_to_right.ge(
                required_after_pick.unsqueeze(1)
            )
            available = available & (order_sensitive.unsqueeze(1) | canonical_next)
        max_reached = selected_counts >= options.max_counts
        option_logits = option_logits.masked_fill(~available, -torch.inf)
        option_logits = option_logits.masked_fill(max_reached.unsqueeze(-1), -torch.inf)

        stop_embeddings = self._routed_stop_embeddings(query, route_plan)
        stop_logits = (query * stop_embeddings).sum(dim=-1)
        stop_logits = stop_logits / math.sqrt(float(self.config.d_model))
        stop_allowed = selected_counts >= options.min_counts
        stop_logits = stop_logits.masked_fill(~stop_allowed, -torch.inf)
        logits = torch.cat([option_logits, stop_logits.unsqueeze(-1)], dim=1)
        if proposal:
            logits = self._proposal_logits_from_shared_trunk(
                query,
                projected_options,
                logits,
                route_plan=route_plan,
            )
        return (query, logits)

    def base_and_proposal_step_logits(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor | None = None,
        ordered_rows: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
        projected_options: Tensor | None = None,
    ) -> BaseProposalStepLogits:
        """Score both policies with one pointer-decoder trunk evaluation."""
        if projected_options is None:
            projected_options = self.project_option_embeddings(
                option_embeddings,
                route_plan=route_plan,
            )
        elif projected_options.shape != option_embeddings.shape:
            raise ValueError("projected_options must align with option_embeddings")
        query, base_logits = self._step_query_and_logits_with_projection(
            global_embedding,
            option_embeddings,
            projected_options,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            ordered_rows=ordered_rows,
            route_plan=route_plan,
        )
        return BaseProposalStepLogits(
            base=base_logits,
            proposal=self._proposal_logits_from_shared_trunk(
                query,
                projected_options,
                base_logits,
                route_plan=route_plan,
            ),
            prefix_query=query,
        )

    def _proposal_logits_from_shared_trunk(
        self,
        query: Tensor,
        projected_options: Tensor,
        base_logits: Tensor,
        *,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Apply the lightweight zero-initialized proposal residual."""
        residual_query = cast(Tensor, self.proposal_query_projection(query))
        option_residual = (projected_options * residual_query.unsqueeze(1)).sum(dim=-1)
        option_residual = option_residual / math.sqrt(float(self.config.d_model))
        stop_embeddings = self._routed_stop_embeddings(residual_query, route_plan)
        stop_residual = (residual_query * stop_embeddings).sum(dim=-1)
        stop_residual = stop_residual / math.sqrt(float(self.config.d_model))
        residual = torch.cat(
            [option_residual, stop_residual.unsqueeze(-1)],
            dim=1,
        )
        return base_logits + residual

    def count_first_rows(
        self,
        options: OptionBatch,
        *,
        ordered_rows: Tensor | None = None,
    ) -> Tensor:
        """Return rows using a separate cardinality decision before set picks."""
        enabled = self.config.unordered_set_policy == "count_first"
        rows = _engine_proven_unordered_rows(options) & options.min_counts.lt(
            options.max_counts
        )
        if ordered_rows is not None:
            explicit_ordered = _validated_ordered_rows(options, ordered_rows)
            rows = rows & ~explicit_ordered
        return rows if enabled else torch.zeros_like(rows)

    def count_first_logits(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        projected_options: Tensor,
        options: OptionBatch,
        *,
        ordered_rows: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        """Score legal selection counts using a stop-hazard prior and residual."""
        batch_size, max_options = options.valid_options.shape
        selected_mask = torch.zeros_like(options.valid_options)
        ordered_history = self.initial_ordered_history(option_embeddings)
        hazard_query, hazard_logits = self._step_query_and_logits_with_projection(
            global_embedding,
            option_embeddings,
            projected_options,
            options,
            selected_mask=selected_mask,
            selected_counts=options.min_counts,
            ordered_history=ordered_history,
            ordered_rows=ordered_rows,
            route_plan=route_plan,
        )
        probability_dtype = (
            torch.float32
            if hazard_logits.dtype in (torch.float16, torch.bfloat16)
            else hazard_logits.dtype
        )
        hazard_logprobs = torch.log_softmax(
            hazard_logits.to(dtype=probability_dtype),
            dim=1,
        )
        stop_logprobs = hazard_logprobs[:, max_options]
        option_hazard_logprobs = hazard_logprobs[:, :max_options]
        if max_options == 0:
            continue_logprobs = torch.zeros_like(stop_logprobs)
        else:
            # Fixed-cardinality rows reach this shared batched path with every
            # item masked and only STOP legal. logsumexp([-inf, ...]) has a
            # finite-looking replacement under nan_to_num but an undefined
            # backward (0 / 0), contaminating the entire PPO gradient. Replace
            # those inactive inputs before the reduction; variable-cardinality
            # count-first rows always retain at least one legal item here.
            has_continue = torch.isfinite(option_hazard_logprobs).any(dim=1)
            safe_option_hazard_logprobs = torch.where(
                has_continue.unsqueeze(1),
                option_hazard_logprobs,
                torch.zeros_like(option_hazard_logprobs),
            )
            continue_logprobs = torch.logsumexp(
                safe_option_hazard_logprobs,
                dim=1,
            )
            continue_logprobs = torch.where(
                has_continue,
                continue_logprobs,
                torch.zeros_like(continue_logprobs),
            )
        stop_logprobs = torch.nan_to_num(
            stop_logprobs,
            nan=0.0,
            neginf=0.0,
            posinf=0.0,
        )
        continue_logprobs = torch.nan_to_num(
            continue_logprobs,
            nan=0.0,
            neginf=0.0,
            posinf=0.0,
        )

        counts = (
            torch.arange(
                max_options + 1,
                device=options.valid_options.device,
                dtype=options.min_counts.dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        minimums = options.min_counts.unsqueeze(1)
        maximums = options.max_counts.unsqueeze(1)
        option_counts = options.valid_options.sum(dim=1).unsqueeze(1)
        additional = (counts - minimums).clamp_min(0).to(dtype=probability_dtype)
        base_logits = additional * continue_logprobs.unsqueeze(1)
        base_logits = base_logits + torch.where(
            counts < maximums,
            stop_logprobs.unsqueeze(1),
            torch.zeros_like(base_logits),
        )

        count_features = _count_first_features(
            counts,
            options,
            dtype=hazard_query.dtype,
        )
        state_hidden = self._routed_linear(
            "count_state_projection",
            self.count_state_projection,
            hazard_query,
            route_plan,
        )
        state_hidden = state_hidden + self._private_count_set_hidden(
            option_embeddings,
            options,
            route_plan,
            reference=state_hidden,
        )
        count_hidden = self._routed_linear(
            "count_feature_projection",
            self.count_feature_projection,
            count_features,
            route_plan,
        )
        residuals = self._routed_linear(
            "count_output_projection",
            self.count_output_projection,
            torch.nn.functional.gelu(state_hidden.unsqueeze(1) + count_hidden),
            route_plan,
        ).squeeze(-1)
        logits = base_logits + residuals.to(dtype=base_logits.dtype)
        legal = counts.ge(minimums) & counts.le(maximums) & counts.le(option_counts)
        return logits.masked_fill(~legal, -torch.inf)

    def initial_ordered_history(self, option_embeddings: Tensor) -> Tensor:
        """Return the empty ordered-selection history for a decode batch."""
        return option_embeddings.new_zeros(
            (option_embeddings.shape[0], option_embeddings.shape[-1])
        )

    def advance_ordered_history(
        self,
        ordered_history: Tensor,
        option_embeddings: Tensor,
        choice_indices: Tensor,
        append_mask: Tensor,
    ) -> Tensor:
        """Append selected options to a recency-weighted ordered history."""
        _batch_size, max_options, d_model = option_embeddings.shape
        if max_options <= 0:
            return ordered_history
        safe_choices = choice_indices.clamp(min=0, max=max_options - 1)
        chosen_embeddings = torch.gather(
            option_embeddings,
            dim=1,
            index=safe_choices.view(-1, 1, 1).expand(-1, 1, d_model),
        ).squeeze(1)
        updated = ordered_history * ORDERED_HISTORY_DECAY + chosen_embeddings
        return torch.where(append_mask.unsqueeze(-1), updated, ordered_history)

    def greedy_decode(
        self,
        global_embedding: Tensor,
        token_embeddings: Tensor,
        options: OptionBatch,
        card_encoder: CardEncoder,
        *,
        route_plan: DeckRoutePlan | None = None,
        state_padding_mask: Tensor | None = None,
        proposal: bool = False,
        ordered_rows: Tensor | None = None,
        max_select_steps: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Decode base or proposal logits greedily with autoregressive STOP."""
        actions, _logprobs = self.sample_decode(
            global_embedding,
            token_embeddings,
            options,
            card_encoder,
            temperature=0.0,
            route_plan=route_plan,
            state_padding_mask=state_padding_mask,
            proposal=proposal,
            ordered_rows=ordered_rows,
            max_select_steps=max_select_steps,
        )
        return actions

    def sample_decode(
        self,
        global_embedding: Tensor,
        token_embeddings: Tensor,
        options: OptionBatch,
        card_encoder: CardEncoder,
        *,
        temperature: float = 1.0,
        route_plan: DeckRoutePlan | None = None,
        state_padding_mask: Tensor | None = None,
        proposal: bool = False,
        ordered_rows: Tensor | None = None,
        generator: torch.Generator | None = None,
        max_select_steps: int | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor]:
        """Sample legal index sequences and return their summed log-probs."""
        output = self.sample_decode_tensors(
            global_embedding,
            token_embeddings,
            options,
            card_encoder,
            temperature=temperature,
            route_plan=route_plan,
            state_padding_mask=state_padding_mask,
            proposal=proposal,
            ordered_rows=ordered_rows,
            generator=generator,
            max_select_steps=max_select_steps,
        )
        actions = actions_from_decode_tensors(
            output.choice_indices,
            output.append_masks,
        )
        return (actions, output.action_logprobs)

    def sample_decode_tensors(
        self,
        global_embedding: Tensor,
        token_embeddings: Tensor,
        options: OptionBatch,
        card_encoder: CardEncoder,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int | None = None,
        gumbel_noise: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
        state_padding_mask: Tensor | None = None,
        proposal: bool = False,
        ordered_rows: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTensorOutput:
        """Embed legal options once, then decode through the shared trunk."""
        option_embeddings = self.option_embeddings(
            token_embeddings,
            options,
            card_encoder,
            route_plan=route_plan,
            state_padding_mask=state_padding_mask,
        )
        return self.sample_decode_tensors_from_embeddings(
            global_embedding,
            option_embeddings,
            options,
            temperature=temperature,
            max_select_steps=max_select_steps,
            gumbel_noise=gumbel_noise,
            route_plan=route_plan,
            proposal=proposal,
            ordered_rows=ordered_rows,
            generator=generator,
        )

    def sample_decode_tensors_from_embeddings(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        temperature: float | Tensor = 1.0,
        max_select_steps: int | None = None,
        gumbel_noise: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
        proposal: bool = False,
        ordered_rows: Tensor | None = None,
        projected_options: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SampleDecodeTensorOutput:
        """Sample legal index sequences and keep decode choices on device.

        ``max_select_steps`` fixes the decode loop trip count for bucketed inference.
        When it is provided by the caller, this path avoids synchronizing on
        ``options.max_counts.max().item()`` inside the CUDA hot path.
        ``gumbel_noise`` may be provided to make positive-temperature sampling
        deterministic and capturable via Gumbel-max.
        """
        return cast(
            SampleDecodeTensorOutput,
            self._sample_decode_tensors_from_embeddings(
                global_embedding,
                option_embeddings,
                options,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
                route_plan=route_plan,
                proposal=proposal,
                ordered_rows=ordered_rows,
                projected_options=projected_options,
                generator=generator,
                collect_trace=True,
            ),
        )

    def sample_action_tensors_from_embeddings(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        max_select_steps: int,
        temperature: float | Tensor = 1.0,
        gumbel_noise: Tensor | None = None,
        route_plan: DeckRoutePlan | None = None,
        proposal: bool = False,
        ordered_rows: Tensor | None = None,
        projected_options: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SampleActionTensorOutput:
        """Sample actions without constructing learner-only decode traces.

        The caller supplies the CPU-known selection bound so collection does not
        synchronize on device-resident action-space metadata.
        """
        return cast(
            SampleActionTensorOutput,
            self._sample_decode_tensors_from_embeddings(
                global_embedding,
                option_embeddings,
                options,
                temperature=temperature,
                max_select_steps=max_select_steps,
                gumbel_noise=gumbel_noise,
                route_plan=route_plan,
                proposal=proposal,
                ordered_rows=ordered_rows,
                projected_options=projected_options,
                generator=generator,
                collect_trace=False,
            ),
        )

    def _sample_decode_tensors_from_embeddings(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        temperature: float | Tensor,
        max_select_steps: int | None,
        gumbel_noise: Tensor | None,
        route_plan: DeckRoutePlan | None,
        proposal: bool,
        ordered_rows: Tensor | None,
        projected_options: Tensor | None,
        generator: torch.Generator | None,
        collect_trace: bool,
    ) -> SampleDecodeTensorOutput | SampleActionTensorOutput:
        temperature_tensor: Tensor | None = None
        temperature_float: float | None
        if isinstance(temperature, Tensor):
            if temperature.ndim != 0:
                raise ValueError("tensor temperature must be scalar")
            if gumbel_noise is None:
                raise ValueError("tensor temperature requires gumbel_noise")
            temperature_tensor = temperature
            temperature_float = None
        else:
            temperature_float = float(temperature)
            if not math.isfinite(temperature_float) or temperature_float < 0.0:
                raise ValueError("temperature must be finite and non-negative")
        if max_select_steps is not None and max_select_steps < 0:
            raise ValueError("max_select_steps must be non-negative")
        if projected_options is None:
            projected_options = self.project_option_embeddings(
                option_embeddings,
                route_plan=route_plan,
            )
        elif projected_options.shape != option_embeddings.shape:
            raise ValueError("projected_options must align with option_embeddings")
        batch_size, max_options = options.valid_options.shape
        if max_select_steps is None:
            max_select_steps = int(options.max_counts.max().item())
        max_steps = min(max_options, int(max_select_steps))
        if gumbel_noise is not None:
            expected_noise_shape = (batch_size, max_steps + 1, max_options + 1)
            if tuple(gumbel_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "gumbel_noise must have shape "
                    f"{expected_noise_shape}, got {tuple(gumbel_noise.shape)}"
                )
            if not gumbel_noise.is_floating_point():
                raise ValueError("gumbel_noise must be floating point")

        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = self.initial_ordered_history(option_embeddings)
        probability_dtype = (
            torch.float32
            if global_embedding.dtype in (torch.float16, torch.bfloat16)
            else global_embedding.dtype
        )
        count_rows = self.count_first_rows(
            options,
            ordered_rows=ordered_rows,
        )
        if self.config.unordered_set_policy != "count_first":
            sampled_counts = options.max_counts
            count_logprobs = torch.zeros(
                batch_size,
                dtype=probability_dtype,
                device=options.valid_options.device,
            )
        else:
            count_logits = self.count_first_logits(
                global_embedding,
                option_embeddings,
                projected_options,
                options,
                ordered_rows=ordered_rows,
                route_plan=route_plan,
            )
            if temperature_float == 0.0:
                sampled_counts = count_logits.argmax(dim=1)
                count_logprobs = torch.zeros(
                    batch_size,
                    dtype=probability_dtype,
                    device=options.valid_options.device,
                )
            else:
                if temperature_tensor is None:
                    scaled_count_logits = count_logits / float(
                        cast(float, temperature_float)
                    )
                else:
                    scaled_count_logits = count_logits / temperature_tensor.to(
                        device=count_logits.device,
                        dtype=count_logits.dtype,
                    )
                count_logprobabilities = torch.log_softmax(
                    scaled_count_logits.to(dtype=probability_dtype),
                    dim=1,
                )
                if gumbel_noise is None:
                    sampled_counts = torch.multinomial(
                        count_logprobabilities.exp(),
                        num_samples=1,
                        generator=generator,
                    ).squeeze(1)
                else:
                    count_noise = gumbel_noise[:, max_steps, :].to(
                        dtype=scaled_count_logits.dtype
                    )
                    sampled_counts = (scaled_count_logits + count_noise).argmax(dim=1)
                count_logprobs = count_logprobabilities.gather(
                    1,
                    sampled_counts.unsqueeze(1),
                ).squeeze(1)
        target_counts = torch.where(
            count_rows,
            sampled_counts.to(dtype=options.max_counts.dtype),
            options.max_counts,
        )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, target_counts, options.min_counts),
            max_counts=torch.where(count_rows, target_counts, options.max_counts),
        )
        stopped = selected_counts >= decode_options.max_counts
        action_logprobs = torch.where(
            count_rows,
            count_logprobs.to(dtype=probability_dtype),
            torch.zeros(
                batch_size,
                dtype=probability_dtype,
                device=options.valid_options.device,
            ),
        )
        action_logprobs = action_logprobs.to(
            dtype=probability_dtype,
            device=options.valid_options.device,
        )
        count_logprobs = count_logprobs.to(
            dtype=probability_dtype,
            device=options.valid_options.device,
        )
        row_indices = torch.arange(
            batch_size,
            device=options.valid_options.device,
        )
        selected_steps: list[Tensor] = []
        selected_step_masks: list[Tensor] = []
        token_logprob_steps: list[Tensor] = []
        token_masks: list[Tensor] = []
        prefix_queries: list[Tensor] = []
        stop_sampled = (
            torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=options.valid_options.device,
            )
            if collect_trace
            else None
        )

        for step in range(max_steps + 1):
            query, logits = self._step_query_and_logits_with_projection(
                global_embedding,
                option_embeddings,
                projected_options,
                decode_options,
                selected_mask=selected_mask,
                selected_counts=selected_counts,
                ordered_history=ordered_history,
                ordered_rows=ordered_rows,
                route_plan=route_plan,
                proposal=proposal,
            )
            active = ~stopped
            if temperature_float == 0.0:
                choices = logits.argmax(dim=1)
                step_logprobs = torch.zeros_like(action_logprobs)
            else:
                if temperature_tensor is None:
                    scaled_logits = logits / float(cast(float, temperature_float))
                else:
                    scaled_logits = logits / temperature_tensor.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                log_probabilities = torch.log_softmax(
                    scaled_logits.to(dtype=probability_dtype),
                    dim=1,
                )
                if gumbel_noise is None:
                    probabilities = log_probabilities.exp()
                    choices = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    ).squeeze(1)
                else:
                    step_noise = gumbel_noise[:, step, :].to(dtype=scaled_logits.dtype)
                    choices = (scaled_logits + step_noise).argmax(dim=1)
                step_logprobs = log_probabilities.gather(
                    1,
                    choices.unsqueeze(1),
                ).squeeze(1)
                step_logprobs = step_logprobs.to(dtype=action_logprobs.dtype)
            masked_step_logprobs = torch.where(
                active,
                step_logprobs,
                torch.zeros_like(step_logprobs),
            )
            action_logprobs = torch.where(
                active,
                action_logprobs + masked_step_logprobs,
                action_logprobs,
            )
            option_choice = choices < max_options
            clamped_choices = choices.clamp(max=max(max_options - 1, 0))
            valid_choice = torch.zeros_like(option_choice)
            if max_options > 0:
                valid_choice = options.valid_options[row_indices, clamped_choices]
            append_mask = active & option_choice & valid_choice
            if max_options > 0:
                ordered_history = self.advance_ordered_history(
                    ordered_history,
                    option_embeddings,
                    clamped_choices,
                    append_mask,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1, clamped_choices.unsqueeze(1), append_mask.unsqueeze(1)
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + append_mask.to(
                    dtype=selected_counts.dtype
                )
            selected_steps.append(clamped_choices)
            selected_step_masks.append(append_mask)
            sampled_stop = active & choices.eq(max_options)
            if collect_trace:
                token_logprob_steps.append(masked_step_logprobs)
                token_masks.append(active)
                prefix_queries.append(query)
                assert stop_sampled is not None
                stop_sampled = stop_sampled | sampled_stop
            stopped = stopped | sampled_stop
            stopped = stopped | (active & option_choice & ~valid_choice)
            stopped = stopped | (selected_counts >= decode_options.max_counts)

        choice_indices = torch.stack(selected_steps, dim=1)
        append_masks = torch.stack(selected_step_masks, dim=1)
        if not collect_trace:
            return SampleActionTensorOutput(
                choice_indices=choice_indices,
                append_masks=append_masks,
                action_logprobs=action_logprobs,
            )

        assert stop_sampled is not None
        token_logprobs = torch.stack(token_logprob_steps, dim=1)
        token_mask = torch.stack(token_masks, dim=1)
        stacked_prefix_queries = torch.stack(prefix_queries, dim=1)
        early_count_stop = count_rows & target_counts.lt(options.max_counts)
        count_slots = torch.where(
            early_count_stop,
            target_counts,
            torch.zeros_like(target_counts),
        ).clamp(min=0, max=max_steps)
        count_contributions = torch.zeros_like(token_logprobs).scatter_add(
            1,
            count_slots.unsqueeze(1),
            torch.where(
                count_rows,
                count_logprobs,
                torch.zeros_like(count_logprobs),
            ).unsqueeze(1),
        )
        token_logprobs = token_logprobs + count_contributions
        count_token_mask = torch.zeros_like(token_mask).scatter(
            1,
            count_slots.unsqueeze(1),
            count_rows.unsqueeze(1),
        )
        token_mask = token_mask | count_token_mask
        stop_sampled = stop_sampled | early_count_stop

        return SampleDecodeTensorOutput(
            choice_indices=choice_indices,
            append_masks=append_masks,
            action_logprobs=action_logprobs,
            token_logprobs=token_logprobs,
            token_mask=token_mask,
            prefix_queries=stacked_prefix_queries,
            stop_sampled=stop_sampled,
        )

    def teacher_forced_decode(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        route_plan: DeckRoutePlan | None = None,
        projected_options: Tensor | None = None,
        include_completion: bool = True,
        include_factual: bool = True,
        validate_temperature: bool = True,
        proposal: bool = False,
        include_proposal: bool = False,
        ordered_rows: Tensor | None = None,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> TeacherForcedEvaluation:
        """Replay actions with the full learner-facing diagnostic trace."""
        return cast(
            TeacherForcedEvaluation,
            self._teacher_forced_replay(
                global_embedding,
                option_embeddings,
                options,
                actions,
                action_targets=action_targets,
                temperature=temperature,
                route_plan=route_plan,
                projected_options=projected_options,
                include_completion=include_completion,
                include_factual=include_factual,
                validate_temperature=validate_temperature,
                proposal=proposal,
                include_proposal=include_proposal,
                ordered_rows=ordered_rows,
                collect_diagnostics=True,
                validate_actions=validate_actions,
                count_first_rows_present=count_first_rows_present,
            ),
        )

    def teacher_forced_action_scores(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        route_plan: DeckRoutePlan | None = None,
        projected_options: Tensor | None = None,
        include_completion: bool = False,
        validate_temperature: bool = True,
        proposal: bool = False,
        include_proposal: bool = False,
        ordered_rows: Tensor | None = None,
        validate_actions: bool = True,
    ) -> TeacherForcedActionScores:
        """Replay actions without materializing token-level diagnostics.

        ``validate_actions=False`` is reserved for internal callers whose
        candidate construction already proved count-first canonicality. Public
        replay remains strict by default.
        """
        return cast(
            TeacherForcedActionScores,
            self._teacher_forced_replay(
                global_embedding,
                option_embeddings,
                options,
                actions,
                action_targets=action_targets,
                temperature=temperature,
                route_plan=route_plan,
                projected_options=projected_options,
                include_completion=include_completion,
                include_factual=False,
                validate_temperature=validate_temperature,
                proposal=proposal,
                include_proposal=include_proposal,
                ordered_rows=ordered_rows,
                collect_diagnostics=False,
                validate_actions=validate_actions,
            ),
        )

    def teacher_forced_sequence_scores(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        route_plan: DeckRoutePlan | None = None,
        projected_options: Tensor | None = None,
        validate_temperature: bool = True,
        ordered_rows: Tensor | None = None,
        validate_actions: bool = True,
        count_first_rows_present: bool | None = None,
    ) -> TeacherForcedSequenceScores:
        """Replay only sequence log-probability and entropy for lean PPO."""
        return cast(
            TeacherForcedSequenceScores,
            self._teacher_forced_replay(
                global_embedding,
                option_embeddings,
                options,
                actions,
                action_targets=action_targets,
                temperature=temperature,
                route_plan=route_plan,
                projected_options=projected_options,
                include_completion=False,
                include_factual=False,
                validate_temperature=validate_temperature,
                proposal=False,
                include_proposal=False,
                ordered_rows=ordered_rows,
                collect_diagnostics=False,
                validate_actions=validate_actions,
                count_first_rows_present=count_first_rows_present,
                collect_sequence_entropies=True,
            ),
        )

    def _teacher_forced_replay(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        actions: Sequence[Sequence[int]],
        *,
        action_targets: Tensor | None = None,
        temperature: float | Tensor = 1.0,
        route_plan: DeckRoutePlan | None = None,
        projected_options: Tensor | None = None,
        include_completion: bool = True,
        include_factual: bool = True,
        validate_temperature: bool = True,
        proposal: bool = False,
        include_proposal: bool = False,
        ordered_rows: Tensor | None = None,
        collect_diagnostics: bool,
        validate_actions: bool,
        count_first_rows_present: bool | None = None,
        collect_sequence_entropies: bool = False,
    ) -> (
        TeacherForcedEvaluation
        | TeacherForcedActionScores
        | TeacherForcedSequenceScores
    ):
        """Shared legality and autoregressive core for teacher-forced replay."""
        if proposal and include_proposal:
            raise ValueError(
                "proposal-only decoding cannot also collect the proposal companion"
            )
        if include_factual and not include_completion:
            raise ValueError("factual predictions require a completed-action latent")
        batch_size, max_options = options.valid_options.shape
        if len(actions) != batch_size:
            raise ValueError("actions must align with batch size")
        temperature_rows = _teacher_forced_temperature_rows(
            temperature,
            batch_size=batch_size,
            device=global_embedding.device,
            dtype=global_embedding.dtype,
            validate=validate_temperature,
        )

        if projected_options is None:
            projected_options = self.project_option_embeddings(
                option_embeddings,
                route_plan=route_plan,
            )
        elif projected_options.shape != option_embeddings.shape:
            raise ValueError("projected_options must align with option_embeddings")
        action_targets = _prepare_teacher_forced_action_targets(
            actions,
            action_targets=action_targets,
            batch_size=batch_size,
            max_options=max_options,
            device=options.valid_options.device,
        )
        max_steps = int(action_targets.shape[1])
        target_counts = torch.tensor(
            [len(action) for action in actions],
            dtype=options.max_counts.dtype,
            device=options.valid_options.device,
        )
        count_rows = self.count_first_rows(
            options,
            ordered_rows=ordered_rows,
        )
        if validate_actions:
            _validate_count_first_actions(
                actions,
                options,
                count_rows=count_rows,
            )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, target_counts, options.min_counts),
            max_counts=torch.where(count_rows, target_counts, options.max_counts),
        )
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = self.initial_ordered_history(option_embeddings)
        done = count_rows & target_counts.eq(0)
        probability_dtype = (
            torch.float32
            if global_embedding.dtype in (torch.float16, torch.bfloat16)
            else global_embedding.dtype
        )
        action_logprobs = torch.zeros(
            batch_size,
            dtype=probability_dtype,
            device=options.valid_options.device,
        )
        entropies = torch.zeros_like(action_logprobs)
        scaled_count_logits: Tensor | None = None
        count_logprobs = torch.zeros_like(action_logprobs)
        count_entropies = torch.zeros_like(entropies)
        if self.config.unordered_set_policy == "count_first":
            raw_count_logits = self.count_first_logits(
                global_embedding,
                option_embeddings,
                projected_options,
                options,
                ordered_rows=ordered_rows,
                route_plan=route_plan,
            )
            scaled_count_logits = (
                raw_count_logits
                if temperature_rows is None
                else raw_count_logits / temperature_rows.unsqueeze(1)
            )
            all_count_logprobs = torch.log_softmax(
                scaled_count_logits.to(dtype=probability_dtype),
                dim=1,
            )
            selected_count_logprobs = all_count_logprobs.gather(
                1,
                target_counts.unsqueeze(1),
            ).squeeze(1)
            count_logprobs = torch.where(
                count_rows,
                selected_count_logprobs.to(dtype=action_logprobs.dtype),
                count_logprobs,
            )
            action_logprobs = action_logprobs + count_logprobs
            if collect_diagnostics or collect_sequence_entropies:
                all_count_entropies = _entropy_from_logprobs(all_count_logprobs)
                count_entropies = torch.where(
                    count_rows,
                    all_count_entropies.to(dtype=entropies.dtype),
                    count_entropies,
                )
                entropies = entropies + count_entropies
        proposal_action_logprobs = count_logprobs.clone() if include_proposal else None
        proposal_entropies = (
            count_entropies.clone()
            if include_proposal and collect_diagnostics
            else None
        )
        first_logits: Tensor | None = None
        step_logits: list[Tensor] = []
        steps: list[TeacherForcedStep] = []
        token_logprob_steps: list[Tensor] = []
        token_entropy_steps: list[Tensor] = []
        token_masks: list[Tensor] = []
        prefix_queries: list[Tensor] = []
        stop_sampled = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=options.valid_options.device,
        )
        for step in range(max_steps):
            step_targets = action_targets[:, step]
            legacy_active = (
                ~count_rows
                & ~done
                & step_targets.ge(0)
                & selected_counts.lt(options.max_counts)
            )
            count_active = (
                count_rows
                & ~done
                & step_targets.ge(0)
                & step_targets.lt(max_options)
                & selected_counts.lt(target_counts)
            )
            active = legacy_active | count_active
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            if int(active_indices.numel()) == 0:
                break

            query, raw_logits = self._step_query_and_logits_with_projection(
                global_embedding,
                option_embeddings,
                projected_options,
                decode_options,
                selected_mask=selected_mask,
                selected_counts=selected_counts,
                ordered_history=ordered_history,
                ordered_rows=ordered_rows,
                route_plan=route_plan,
                proposal=proposal,
            )
            proposal_raw_logits = (
                self._proposal_logits_from_shared_trunk(
                    query,
                    projected_options,
                    raw_logits,
                    route_plan=route_plan,
                )
                if include_proposal
                else None
            )
            logits = (
                raw_logits
                if temperature_rows is None
                else raw_logits / temperature_rows.unsqueeze(1)
            )
            if collect_diagnostics and first_logits is None:
                first_logits = (
                    logits
                    if scaled_count_logits is None
                    else torch.where(
                        count_rows.unsqueeze(1),
                        scaled_count_logits,
                        logits,
                    )
                )
            if collect_diagnostics:
                step_logits.append(logits)
                prefix_queries.append(query)

            targets = step_targets.index_select(0, active_indices)
            active_logits = logits.index_select(0, active_indices)
            logprobs = torch.log_softmax(
                active_logits.to(dtype=probability_dtype),
                dim=1,
            )
            step_logprobs = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
            action_logprobs = action_logprobs.scatter_add(
                0,
                active_indices,
                step_logprobs.to(dtype=action_logprobs.dtype),
            )
            dense_logprobs: Tensor | None = None
            dense_entropies: Tensor | None = None
            if collect_diagnostics:
                dense_logprobs = torch.zeros_like(action_logprobs).scatter_add(
                    0,
                    active_indices,
                    step_logprobs.to(dtype=action_logprobs.dtype),
                )
            if collect_diagnostics or collect_sequence_entropies:
                step_entropies = _entropy_from_logprobs(logprobs)
                entropies = entropies.scatter_add(
                    0,
                    active_indices,
                    step_entropies.to(dtype=entropies.dtype),
                )
                if collect_diagnostics:
                    dense_entropies = torch.zeros_like(entropies).scatter_add(
                        0,
                        active_indices,
                        step_entropies.to(dtype=entropies.dtype),
                    )
            if proposal_raw_logits is not None:
                if proposal_action_logprobs is None:
                    raise RuntimeError("proposal log-prob accumulator is unavailable")
                scaled_proposal_logits = (
                    proposal_raw_logits
                    if temperature_rows is None
                    else proposal_raw_logits / temperature_rows.unsqueeze(1)
                )
                active_proposal_logits = scaled_proposal_logits.index_select(
                    0, active_indices
                )
                proposal_logprobs = torch.log_softmax(
                    active_proposal_logits.to(dtype=probability_dtype),
                    dim=1,
                )
                selected_proposal_logprobs = proposal_logprobs.gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)
                proposal_action_logprobs = proposal_action_logprobs.scatter_add(
                    0,
                    active_indices,
                    selected_proposal_logprobs.to(dtype=proposal_action_logprobs.dtype),
                )
                if collect_diagnostics:
                    if proposal_entropies is None:
                        raise RuntimeError(
                            "proposal entropy accumulator is unavailable"
                        )
                    proposal_entropies = proposal_entropies.scatter_add(
                        0,
                        active_indices,
                        _entropy_from_logprobs(proposal_logprobs).to(
                            dtype=proposal_entropies.dtype
                        ),
                    )
            if collect_diagnostics:
                if dense_logprobs is None or dense_entropies is None:
                    raise RuntimeError("teacher-forced diagnostics are unavailable")
                token_logprob_steps.append(dense_logprobs)
                token_entropy_steps.append(dense_entropies)
                token_masks.append(active)
                steps.append(
                    TeacherForcedStep(
                        logits=logits,
                        active_indices=active_indices,
                        targets=targets,
                    )
                )

            selected = active & (step_targets < max_options)
            if max_options > 0:
                safe_targets = step_targets.clamp(min=0, max=max_options - 1)
                ordered_history = self.advance_ordered_history(
                    ordered_history,
                    option_embeddings,
                    safe_targets,
                    selected,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1, safe_targets.unsqueeze(1), selected.unsqueeze(1)
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + selected.to(
                    dtype=selected_counts.dtype
                )
            sampled_stop = legacy_active & step_targets.eq(max_options)
            if collect_diagnostics:
                stop_sampled = stop_sampled | sampled_stop
            done = done | step_targets.lt(0) | sampled_stop
            done = done | selected_counts.ge(decode_options.max_counts)

        if not collect_diagnostics:
            if collect_sequence_entropies:
                if include_completion or include_factual or include_proposal:
                    raise ValueError(
                        "sequence-only replay cannot request auxiliary outputs"
                    )
                return TeacherForcedSequenceScores(
                    action_logprobs=action_logprobs,
                    entropies=entropies,
                )
            score_action_latent: Tensor | None = None
            if include_completion:
                completed_query = self._query_embedding(
                    global_embedding,
                    option_embeddings,
                    options,
                    selected_mask,
                    selected_counts,
                    ordered_history,
                    route_plan,
                )
                completed_stop_embeddings = self._routed_stop_embeddings(
                    completed_query,
                    route_plan,
                )
                score_action_latent = (
                    completed_query + completed_stop_embeddings
                ) / math.sqrt(2.0)
            return TeacherForcedActionScores(
                action_logprobs=action_logprobs,
                completed_action_latents=score_action_latent,
                proposal_action_logprobs=proposal_action_logprobs,
            )

        if first_logits is None:
            first_logits = (
                scaled_count_logits
                if scaled_count_logits is not None
                else torch.empty(
                    (batch_size, max_options + 1),
                    dtype=global_embedding.dtype,
                    device=global_embedding.device,
                )
            )
        raw_token_width = len(token_logprob_steps)
        if token_logprob_steps:
            token_logprobs = torch.stack(token_logprob_steps, dim=1)
            token_entropies = torch.stack(token_entropy_steps, dim=1)
            token_mask = torch.stack(token_masks, dim=1)
            stacked_prefix_queries = torch.stack(prefix_queries, dim=1)
        else:
            token_logprobs = action_logprobs.new_zeros((batch_size, 0))
            token_entropies = entropies.new_zeros((batch_size, 0))
            token_mask = torch.zeros(
                (batch_size, 0),
                dtype=torch.bool,
                device=options.valid_options.device,
            )
            stacked_prefix_queries = global_embedding.new_zeros(
                (batch_size, 0, global_embedding.shape[-1])
            )
        has_count_rows = (
            bool(count_rows.any().item())
            if count_first_rows_present is None
            else bool(
                count_first_rows_present
                and self.config.unordered_set_policy == "count_first"
            )
        )
        if has_count_rows and raw_token_width < max_steps:
            padding = max_steps - raw_token_width
            token_logprobs = torch.nn.functional.pad(
                token_logprobs,
                (0, padding),
            )
            token_entropies = torch.nn.functional.pad(
                token_entropies,
                (0, padding),
            )
            token_mask = torch.nn.functional.pad(
                token_mask,
                (0, padding),
                value=False,
            )
            stacked_prefix_queries = torch.nn.functional.pad(
                stacked_prefix_queries,
                (0, 0, 0, padding),
            )
        completed_query = self._query_embedding(
            global_embedding,
            option_embeddings,
            options,
            selected_mask,
            selected_counts,
            ordered_history,
            route_plan,
        )
        if has_count_rows:
            early_count_stop = count_rows & target_counts.lt(options.max_counts)
            count_slots = torch.where(
                early_count_stop,
                target_counts,
                torch.zeros_like(target_counts),
            ).clamp(min=0, max=max_steps - 1)
            count_token_values = torch.where(
                count_rows,
                count_logprobs,
                torch.zeros_like(count_logprobs),
            )
            count_entropy_values = torch.where(
                count_rows,
                count_entropies,
                torch.zeros_like(count_entropies),
            )
            token_logprobs = token_logprobs.scatter_add(
                1,
                count_slots.unsqueeze(1),
                count_token_values.unsqueeze(1),
            )
            token_entropies = token_entropies.scatter_add(
                1,
                count_slots.unsqueeze(1),
                count_entropy_values.unsqueeze(1),
            )
            count_token_mask = torch.zeros_like(token_mask).scatter(
                1,
                count_slots.unsqueeze(1),
                count_rows.unsqueeze(1),
            )
            token_mask = token_mask | count_token_mask
            early_slot_mask = torch.zeros_like(token_mask).scatter(
                1,
                count_slots.unsqueeze(1),
                early_count_stop.unsqueeze(1),
            )
            stacked_prefix_queries = torch.where(
                early_slot_mask.unsqueeze(-1),
                completed_query.unsqueeze(1),
                stacked_prefix_queries,
            )
            stop_sampled = stop_sampled | early_count_stop
        completed_action_latent: Tensor | None = None
        if include_completion:
            completed_stop_embeddings = self._routed_stop_embeddings(
                completed_query,
                route_plan,
            )
            completed_action_latent = (
                completed_query + completed_stop_embeddings
            ) / math.sqrt(2.0)
        factual_presence_logits: Tensor | None = None
        factual_magnitude_predictions: Tensor | None = None
        factual_actor_relation_logits: Tensor | None = None
        factual_next_context_logits: Tensor | None = None
        if include_factual:
            if completed_action_latent is None:
                raise RuntimeError("completed-action latent is unavailable")
            factual_presence_logits = cast(
                Tensor,
                self.factual_presence_head(completed_action_latent),
            )
            factual_magnitude_predictions = torch.tanh(
                cast(Tensor, self.factual_magnitude_head(completed_action_latent))
            )
            factual_actor_relation_logits = cast(
                Tensor,
                self.factual_actor_relation_head(completed_action_latent),
            )
            factual_next_context_logits = cast(
                Tensor,
                self.factual_next_context_head(completed_action_latent),
            )
        return TeacherForcedEvaluation(
            action_logprobs=action_logprobs,
            entropies=entropies,
            step_logits=tuple(step_logits),
            steps=tuple(steps),
            first_logits=first_logits,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            token_mask=token_mask,
            prefix_queries=stacked_prefix_queries,
            stop_sampled=stop_sampled,
            completed_action_latents=completed_action_latent,
            factual_presence_logits=factual_presence_logits,
            factual_magnitude_predictions=factual_magnitude_predictions,
            factual_actor_relation_logits=factual_actor_relation_logits,
            factual_next_context_logits=factual_next_context_logits,
            proposal_action_logprobs=proposal_action_logprobs,
            proposal_entropies=proposal_entropies,
        )

    def _query_embedding(
        self,
        global_embedding: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor,
        route_plan: DeckRoutePlan | None = None,
    ) -> Tensor:
        selected_weights = selected_mask.to(dtype=option_embeddings.dtype).unsqueeze(-1)
        selected_sum = (option_embeddings * selected_weights).sum(dim=1)
        selected_count = selected_weights.sum(dim=1).clamp_min(1.0)
        selected_pool = selected_sum / selected_count
        cardinality_features = _decoder_cardinality_features(
            options,
            selected_counts,
            dtype=option_embeddings.dtype,
        )
        query_input = global_embedding + self._routed_linear(
            "selected_projection",
            self.selected_projection,
            selected_pool,
            route_plan,
        )
        query_input = query_input + self._routed_linear(
            "ordered_history_projection",
            self.ordered_history_projection,
            ordered_history,
            route_plan,
        )
        query_input = query_input + self._routed_linear(
            "decoder_cardinality_projection",
            self.decoder_cardinality_projection,
            cardinality_features,
            route_plan,
        )
        return self._routed_mlp(
            "query",
            self.query_projection,
            query_input,
            route_plan,
        )


def _policy_lora_modules(
    conditioning: DeckConditioningConfig,
    *,
    target_linears: dict[str, nn.Linear],
) -> nn.ModuleDict:
    """Build policy LoRA banks for the configured stable expert keys."""
    config = conditioning.lora
    if config is None:
        return nn.ModuleDict()
    return nn.ModuleDict(
        {
            target: RoutedLinearLoRA(
                conditioning.active_routes,
                in_features=target_linears[target].in_features,
                out_features=target_linears[target].out_features,
                rank=config.rank,
                alpha=config.alpha,
            )
            for target in config.policy_targets
        }
    )


def _decoder_cardinality_features(
    options: OptionBatch,
    selected_counts: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Encode selected, remaining, and min/max distances for decoding."""
    option_counts = options.valid_options.sum(dim=1).to(dtype=dtype)
    selected = selected_counts.to(dtype=dtype)
    scale = option_counts.clamp_min(1.0)
    remaining = (option_counts - selected).clamp_min(0.0)
    min_distance = options.min_counts.to(dtype=dtype) - selected
    max_distance = options.max_counts.to(dtype=dtype) - selected
    return torch.stack(
        (
            selected / scale,
            remaining / scale,
            min_distance / scale,
            max_distance / scale,
        ),
        dim=1,
    )


def _count_first_features(
    counts: Tensor,
    options: OptionBatch,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Encode each candidate count relative to one prompt's legal range."""
    option_counts = options.valid_options.sum(dim=1).to(dtype=dtype).unsqueeze(1)
    scale = option_counts.clamp_min(1.0)
    candidates = counts.to(dtype=dtype)
    minimums = options.min_counts.to(dtype=dtype).unsqueeze(1)
    maximums = options.max_counts.to(dtype=dtype).unsqueeze(1)
    return torch.stack(
        (
            candidates / scale,
            (option_counts - candidates).clamp_min(0.0) / scale,
            (candidates - minimums) / scale,
            (maximums - candidates) / scale,
        ),
        dim=2,
    )


def _zero_sequential_output(module: nn.Sequential) -> None:
    """Keep a newly introduced auxiliary path inert at checkpoint migration."""
    output = cast(nn.Linear, module[-1])
    nn.init.zeros_(output.weight)
    if output.bias is not None:
        nn.init.zeros_(output.bias)


def _gather_entity_embeddings(
    token_embeddings: Tensor,
    entity_slots: Tensor,
    entity_slot_mask: Tensor,
) -> Tensor:
    batch_size, max_options, max_slots = entity_slots.shape
    d_model = token_embeddings.shape[-1]
    safe_slots = entity_slots.clamp(min=0, max=max(0, token_embeddings.shape[1] - 1))
    expanded_tokens = token_embeddings.unsqueeze(1).expand(
        batch_size,
        max_options,
        token_embeddings.shape[1],
        d_model,
    )
    gathered = torch.gather(
        expanded_tokens,
        dim=2,
        index=safe_slots.unsqueeze(-1).expand(
            batch_size, max_options, max_slots, d_model
        ),
    )
    masked = gathered * entity_slot_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    return masked.sum(dim=2)


def _dynamic_feature_list(values: Sequence[float]) -> list[float]:
    if len(values) != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise ValueError("dynamic effect feature vector has invalid width")
    return [float(value) for value in values]


def actions_from_decode_tensors(
    choice_indices: Tensor,
    append_masks: Tensor,
) -> tuple[tuple[int, ...], ...]:
    """Materialize Python action tuples from tensor decode traces."""
    choice_rows = choice_indices.detach().cpu()
    append_rows = append_masks.detach().cpu()
    actions = [
        [
            int(choice)
            for choice, should_append in zip(
                choice_row.tolist(),
                append_row.tolist(),
                strict=True,
            )
            if bool(should_append)
        ]
        for choice_row, append_row in zip(
            choice_rows,
            append_rows,
            strict=True,
        )
    ]
    return tuple(tuple(action) for action in actions)


def _dynamic_effect_inputs(
    features: Tensor,
    masks: Tensor,
    *,
    feature_size: int,
) -> Tensor:
    if features.shape[-1] != feature_size:
        raise ValueError("dynamic effect feature batch has invalid width")
    mask = masks.to(dtype=features.dtype).unsqueeze(-1)
    return torch.cat([features * mask, mask], dim=-1)


def _safe_indices(values: Tensor, *, max_known: int, oov_index: int) -> Tensor:
    return torch.where(
        (values >= 0) & (values < max_known),
        values,
        torch.full_like(values, oov_index),
    )


def _required_exact_capsules(route_plan: DeckRoutePlan) -> nn.ModuleDict:
    """Return the root-owned capsule bank attached to a v4 route plan."""
    capsules = route_plan.exact_capsules
    if capsules is None:
        raise ValueError("compositional policy route has no exact capsule bank")
    return capsules


def _routed_capsule_residual(
    inputs: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    *,
    module_name: str,
) -> Tensor:
    """Apply one same-width exact capsule module to its own route rows."""
    residual = torch.zeros_like(inputs)
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        module = getattr(capsule, module_name)
        if not isinstance(module, nn.Module):
            raise TypeError(f"exact capsule field {module_name!r} is not a module")
        row_indices.append(group.row_indices)
        contributions.append(
            module(inputs.index_select(0, group.row_indices)).to(
                device=inputs.device,
                dtype=inputs.dtype,
            )
        )
    if not contributions:
        return residual
    return residual.index_copy(
        0,
        torch.cat(row_indices),
        torch.cat(contributions),
    )


def _routed_capsule_linear(
    inputs: Tensor,
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    *,
    module_name: str,
    output_features: int,
) -> Tensor:
    """Apply one arbitrary-width exact capsule Linear to routed rows."""
    residual = inputs.new_zeros((*inputs.shape[:-1], output_features))
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        module = getattr(capsule, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"exact capsule field {module_name!r} is not Linear")
        row_indices.append(group.row_indices)
        contributions.append(module(inputs.index_select(0, group.row_indices)))
    if not contributions:
        return residual
    return residual.index_copy(
        0,
        torch.cat(row_indices),
        torch.cat(contributions).to(dtype=residual.dtype),
    )


def _routed_capsule_vector(
    route_plan: DeckRoutePlan,
    capsules: nn.ModuleDict,
    *,
    parameter_name: str,
    width: int,
    reference: Tensor,
) -> Tensor:
    """Gather one exact vector for every routed batch row."""
    result = reference.new_zeros((route_plan.batch_size, width))
    row_indices: list[Tensor] = []
    contributions: list[Tensor] = []
    for group in route_plan.groups:
        capsule = exact_capsule(capsules, group.module_key)
        parameter = getattr(capsule, parameter_name)
        if not isinstance(parameter, Tensor) or tuple(parameter.shape) != (width,):
            raise TypeError(f"exact capsule field {parameter_name!r} is invalid")
        row_indices.append(group.row_indices)
        contributions.append(
            parameter.unsqueeze(0)
            .expand(int(group.row_indices.numel()), -1)
            .to(device=reference.device, dtype=reference.dtype)
        )
    if not contributions:
        return result
    return result.index_copy(
        0,
        torch.cat(row_indices),
        torch.cat(contributions),
    )


def _masked_option_mean(inputs: Tensor, valid_mask: Tensor) -> Tensor:
    """Pool legal options without making padding affect count calibration."""
    if inputs.ndim != 3 or valid_mask.shape != inputs.shape[:2]:
        raise ValueError("option mean inputs and mask do not align")
    weights = valid_mask.to(dtype=inputs.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (inputs * weights).sum(dim=1) / denominator


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    probabilities = torch.softmax(logprobs, dim=1)
    safe_logprobs = torch.where(
        torch.isfinite(logprobs),
        logprobs,
        torch.zeros_like(logprobs),
    )
    return -(probabilities * safe_logprobs).sum(dim=1)


def _teacher_forced_temperature_rows(
    temperature: float | Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    validate: bool = True,
) -> Tensor | None:
    """Return validated per-row temperatures, eliding scalar one."""
    if not isinstance(temperature, Tensor):
        scalar_temperature = float(temperature)
        if not math.isfinite(scalar_temperature) or scalar_temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if scalar_temperature == 1.0:
            return None
        return torch.full(
            (batch_size,),
            scalar_temperature,
            device=device,
            dtype=dtype,
        )

    if temperature.ndim == 0:
        temperature_rows = temperature.expand(batch_size)
    elif temperature.ndim == 1 and temperature.shape[0] == batch_size:
        temperature_rows = temperature
    else:
        raise ValueError("temperature tensor must be scalar or have shape [batch_size]")
    temperature_rows = temperature_rows.detach().to(device=device, dtype=dtype)
    if validate and (
        not bool(torch.isfinite(temperature_rows).all().item())
        or not bool(temperature_rows.gt(0.0).all().item())
    ):
        raise ValueError("temperature must be finite and positive")
    return temperature_rows


def teacher_forced_action_targets(
    actions: Sequence[Sequence[int]],
    *,
    max_options: int,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return padded teacher-forcing targets with STOP at ``max_options``."""
    if max_steps is None:
        max_steps = max((len(action) for action in actions), default=0) + 1
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    rows = np.full(
        (len(actions), max_steps),
        TEACHER_FORCED_PADDING_TARGET,
        dtype=np.int64,
    )
    for row_index, action in enumerate(actions):
        if len(action) >= max_steps:
            raise ValueError("max_steps must include a STOP target for every action")
        for step_index, target in enumerate(action):
            target_index = int(target)
            if target_index < 0 or target_index >= max_options:
                raise ValueError("action target must index a valid option")
            rows[row_index, step_index] = target_index
        rows[row_index, len(action)] = max_options
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def _order_sensitive_rows(
    options: OptionBatch,
    ordered_rows: Tensor | None,
    *,
    enable_unordered_sets: bool = False,
) -> Tensor:
    """Return explicit ordering semantics or the conservative engine default."""
    if ordered_rows is not None:
        return _validated_ordered_rows(options, ordered_rows)
    if enable_unordered_sets:
        return ~_engine_proven_unordered_rows(options)
    return (
        options.valid_options & options.contexts.eq(int(SelectContext.SKILL_ORDER))
    ).any(dim=1) | options.max_counts.gt(1)


def _validated_ordered_rows(options: OptionBatch, ordered_rows: Tensor) -> Tensor:
    """Move an explicit order-semantics mask onto the option batch device."""
    if ordered_rows.ndim != 1 or ordered_rows.shape != options.max_counts.shape:
        raise ValueError("ordered_rows must have shape [batch_size]")
    return ordered_rows.to(
        device=options.valid_options.device,
        dtype=torch.bool,
    )


def _engine_proven_unordered_rows(options: OptionBatch) -> Tensor:
    """Return prompt rows whose context has native permutation parity."""
    rows = torch.zeros_like(options.valid_options.any(dim=1))
    for context in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS:
        rows = rows | (options.valid_options & options.contexts.eq(int(context))).any(
            dim=1
        )
    return rows


def _validate_count_first_actions(
    actions: Sequence[Sequence[int]],
    options: OptionBatch,
    *,
    count_rows: Tensor,
) -> None:
    """Require the unique canonical representative of every unordered set."""
    active_rows = tuple(bool(value) for value in count_rows.detach().cpu().tolist())
    minimums = tuple(int(value) for value in options.min_counts.detach().cpu().tolist())
    maximums = tuple(int(value) for value in options.max_counts.detach().cpu().tolist())
    valid_counts = tuple(
        int(value) for value in options.valid_options.sum(dim=1).detach().cpu().tolist()
    )
    for action, active, minimum, maximum, valid_count in zip(
        actions,
        active_rows,
        minimums,
        maximums,
        valid_counts,
        strict=True,
    ):
        if not active:
            continue
        canonical = tuple(int(index) for index in action)
        if len(canonical) < minimum or len(canonical) > maximum:
            raise ValueError("count-first action violates prompt cardinality")
        if any(index < 0 or index >= valid_count for index in canonical):
            raise ValueError("count-first action contains an invalid option index")
        if any(
            left >= right for left, right in zip(canonical, canonical[1:], strict=False)
        ):
            raise ValueError(
                "count-first unordered actions must be strictly increasing"
            )


def _prepare_teacher_forced_action_targets(
    actions: Sequence[Sequence[int]],
    *,
    action_targets: Tensor | None,
    batch_size: int,
    max_options: int,
    device: torch.device,
) -> Tensor:
    """Return caller-supplied or freshly built teacher-forcing targets."""
    if action_targets is None:
        return teacher_forced_action_targets(
            actions,
            max_options=max_options,
            device=device,
        )
    if action_targets.ndim != 2 or action_targets.shape[0] != batch_size:
        raise ValueError("action_targets must have shape [batch_size, max_steps]")
    if action_targets.shape[1] <= 0:
        raise ValueError("action_targets must include at least one decode step")
    return action_targets.to(device=device, dtype=torch.long)
