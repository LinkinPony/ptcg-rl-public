"""Autoregressive pointer policy and root/prefix value heads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias, cast

import torch
from torch import Tensor, nn

from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless.config import (
    SimpleStatelessModelConfig,
    uses_exact_v2_topology,
    uses_generalist_sequence,
    uses_temporal_prefusion,
    uses_wdl_critic,
)
from ptcg_rl.model.simple_stateless.layers import initialize_simple_stateless_module
from ptcg_rl.model.simple_stateless.options import RoleAwareOptionEncoder
from ptcg_rl.model.simple_stateless.routing import (
    SimpleExactRoutePlan,
    apply_exact_residual,
)
from ptcg_rl.model.simple_stateless.temporal_fusion import (
    GatedTemporalContentFusion,
)
from ptcg_rl.model.simple_stateless.value import (
    DistributionalWdlCritic,
    ExactWdlValueResidual,
    wdl_value_from_logits,
)
from ptcg_rl.model.tensor_validation import require_tensor_condition

_ORDERED_HISTORY_DECAY = 0.5
_CARDINALITY_FEATURES = 4
_TEMPORAL_FUSION_HIDDEN_DIM = 1024
_WDL_CRITIC_HIDDEN_DIM = 1024


class ExactPolicyQueryResidual(nn.Module):
    """One exact deck's zero-output query residual."""

    def __init__(self, *, d_model: int, bottleneck_dim: int) -> None:
        """Initialize a narrow policy-query adapter."""
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the route-private query delta."""
        return cast(Tensor, self.up(self.activation(self.down(self.norm(inputs)))))

    def zero_output(self) -> None:
        """Make the initial exact route equal the shared content path."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


class ExactOptionEmbeddingResidual(ExactPolicyQueryResidual):
    """One exact deck's zero-output legal-option calibration."""


class ExactScalarValueResidual(nn.Module):
    """One exact deck's zero-output scalar root-value residual."""

    def __init__(self, *, d_model: int, bottleneck_dim: int) -> None:
        """Initialize a narrow scalar critic adapter."""
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(bottleneck_dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the route-private pre-tanh scalar delta."""
        return cast(Tensor, self.output(self.activation(self.down(inputs))))

    def zero_output(self) -> None:
        """Make the initial exact route equal the shared critic."""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


@dataclass(frozen=True)
class SimplePolicyStep:
    """One pointer decode step with its corrected prefix value."""

    logits: Tensor
    prefix_query: Tensor
    prefix_value: Tensor


@dataclass(frozen=True)
class SimpleTeacherForcedEvaluation:
    """Decode-token evidence consumed by the new PPO learner."""

    action_logprobs: Tensor
    token_logprobs: Tensor
    token_entropies: Tensor
    token_mask: Tensor
    prefix_queries: Tensor
    prefix_values: Tensor
    step_logits: tuple[Tensor, ...]
    count_logits: Tensor


@dataclass(frozen=True)
class SimpleTeacherForcedActionBatch:
    """Padded tensor actions for allocation-free teacher-forced replay.

    ``choice_indices`` has shape ``[batch, padded_choices]`` and ``lengths`` has
    shape ``[batch]``. Only the first ``lengths[row]`` choices are meaningful;
    padding values are ignored. Integer tensors may use compact replay dtypes
    and are converted to the model device once per call.
    """

    choice_indices: Tensor
    lengths: Tensor
    maximum_length: int | None = None

    def __post_init__(self) -> None:
        """Validate reusable structural invariants without synchronizing values."""
        if self.choice_indices.ndim != 2:
            raise ValueError("choice_indices must have shape [batch, choices]")
        if self.lengths.ndim != 1:
            raise ValueError("lengths must have shape [batch]")
        if int(self.choice_indices.shape[0]) != int(self.lengths.shape[0]):
            raise ValueError("choice_indices and lengths must share a batch size")
        if not _is_integer_tensor(self.choice_indices):
            raise TypeError("choice_indices must use an integer dtype")
        if not _is_integer_tensor(self.lengths):
            raise TypeError("lengths must use an integer dtype")
        if self.maximum_length is not None and (
            self.maximum_length < 0
            or self.maximum_length > int(self.choice_indices.shape[1])
        ):
            raise ValueError("maximum action length exceeds padded choices")


@dataclass(frozen=True)
class SimpleGreedyDecodeTrace:
    """Device-resident greedy actions plus exact STOP semantics.

    This intentionally carries no probability or value evidence.  It is for
    deployment/evaluation action-only inference and must never be converted
    into a PPO behavior trace.
    """

    actions: SimpleTeacherForcedActionBatch
    stop_sampled: Tensor

    def __post_init__(self) -> None:
        """Validate aligned action and STOP tensors without a device sync."""
        batch_size = int(self.actions.lengths.shape[0])
        if self.stop_sampled.shape != (batch_size,):
            raise ValueError("greedy STOP flags must align with actions")
        if self.stop_sampled.dtype != torch.bool:
            raise TypeError("greedy STOP flags must use bool dtype")
        if self.stop_sampled.device != self.actions.lengths.device:
            raise ValueError("greedy actions and STOP flags must share one device")


SimpleTeacherForcedActionInput: TypeAlias = (
    Sequence[Sequence[int]] | SimpleTeacherForcedActionBatch
)


@dataclass(frozen=True)
class SimpleSampledDecodeTrace:
    """Tensor-native behavior evidence from one autoregressive decode."""

    actions: SimpleTeacherForcedActionBatch
    action_logprobs: Tensor
    token_logprobs: Tensor
    token_mask: Tensor
    prefix_values: Tensor
    stop_sampled: Tensor

    def __post_init__(self) -> None:
        """Validate trace shapes without reading device-resident values."""
        batch_size = int(self.actions.lengths.shape[0])
        token_shape = self.token_logprobs.shape
        if len(token_shape) != 2 or int(token_shape[0]) != batch_size:
            raise ValueError("token_logprobs must have shape [batch, tokens]")
        if self.token_mask.shape != token_shape:
            raise ValueError("token_mask must align with token_logprobs")
        if self.prefix_values.shape != token_shape:
            raise ValueError("prefix_values must align with token_logprobs")
        if self.action_logprobs.shape != (batch_size,):
            raise ValueError("action_logprobs must have shape [batch]")
        if self.stop_sampled.shape != (batch_size,):
            raise ValueError("stop_sampled must have shape [batch]")
        if self.token_mask.dtype != torch.bool:
            raise TypeError("token_mask must use bool dtype")
        if self.stop_sampled.dtype != torch.bool:
            raise TypeError("stop_sampled must use bool dtype")


@dataclass(frozen=True)
class _SimpleDecodePlan:
    """Static projections and legality tensors shared by every decode step."""

    policy_context: Tensor
    option_keys: Tensor
    option_indices: Tensor
    valid_options_to_right: Tensor
    ordered_rows: Tensor


class SimpleStatelessPolicyValueHeads(nn.Module):
    """Only the heads retained by the clean stateless architecture."""

    def __init__(self, config: SimpleStatelessModelConfig) -> None:
        """Initialize shared option/pointer/value paths and exact residuals."""
        super().__init__()
        self.config = config
        self._exact_module_keys = frozenset(
            route.module_key for route in config.exact_routes
        )
        d_model = config.d_model
        hidden = config.feedforward_dim
        self.option_encoder = RoleAwareOptionEncoder(
            d_model=d_model,
            num_heads=config.attention_heads,
            feedforward_dim=config.feedforward_dim,
            zero_gated_dynamic_effects=uses_generalist_sequence(config),
        )
        self.entity_temporal_fusion: GatedTemporalContentFusion | None
        self.option_temporal_fusion: GatedTemporalContentFusion | None
        if uses_temporal_prefusion(config):
            self.entity_temporal_fusion = GatedTemporalContentFusion(
                d_model=d_model,
                hidden_dim=_TEMPORAL_FUSION_HIDDEN_DIM,
            )
            self.option_temporal_fusion = GatedTemporalContentFusion(
                d_model=d_model,
                hidden_dim=_TEMPORAL_FUSION_HIDDEN_DIM,
            )
        else:
            self.entity_temporal_fusion = None
            self.option_temporal_fusion = None
        self.selected_projection = nn.Linear(d_model, d_model, bias=False)
        self.ordered_history_projection = nn.Linear(d_model, d_model, bias=False)
        self.cardinality_projection = nn.Linear(
            _CARDINALITY_FEATURES,
            d_model,
            bias=False,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.option_key_projection = nn.Linear(d_model, d_model, bias=False)
        self.stop_key = nn.Parameter(torch.empty(d_model))
        self.policy_belief_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.value_belief_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.count_state_projection = nn.Linear(d_model, d_model)
        self.count_option_projection = nn.Linear(d_model, d_model)
        self.count_feature_projection = nn.Linear(
            _CARDINALITY_FEATURES,
            d_model,
        )
        self.count_output = nn.Linear(d_model, 1)
        self.shared_root_value: nn.Sequential | None
        self.shared_root_wdl: DistributionalWdlCritic | None
        if uses_wdl_critic(config):
            self.shared_root_value = None
            self.shared_root_wdl = DistributionalWdlCritic(
                d_model=d_model,
                hidden_dim=_WDL_CRITIC_HIDDEN_DIM,
            )
        else:
            self.shared_root_value = nn.Sequential(
                nn.Linear(d_model, config.exact_residual_bottleneck_dim),
                nn.GELU(),
                nn.Linear(config.exact_residual_bottleneck_dim, 1),
            )
            self.shared_root_wdl = None
        self.prefix_value = nn.Sequential(
            nn.Linear(d_model, config.exact_residual_bottleneck_dim),
            nn.GELU(),
            nn.Linear(config.exact_residual_bottleneck_dim, 1),
        )
        self.policy_residuals = nn.ModuleDict(
            {
                route.module_key: ExactPolicyQueryResidual(
                    d_model=d_model,
                    bottleneck_dim=config.exact_residual_bottleneck_dim,
                )
                for route in config.exact_routes
            }
        )
        self.option_residuals = nn.ModuleDict(
            {
                route.module_key: ExactOptionEmbeddingResidual(
                    d_model=d_model,
                    bottleneck_dim=config.exact_residual_bottleneck_dim,
                )
                for route in config.exact_routes
            }
            if uses_exact_v2_topology(config)
            else {}
        )
        self.value_residuals = nn.ModuleDict(
            {
                route.module_key: (
                    ExactWdlValueResidual(
                        d_model=d_model,
                        bottleneck_dim=config.exact_residual_bottleneck_dim,
                    )
                    if uses_wdl_critic(config)
                    else ExactScalarValueResidual(
                        d_model=d_model,
                        bottleneck_dim=config.exact_residual_bottleneck_dim,
                    )
                )
                for route in config.exact_routes
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply fresh shared initialization and inert exact outputs."""
        initialize_simple_stateless_module(self)
        self.option_encoder.zero_engine_factual_output()
        for fusion in (self.entity_temporal_fusion, self.option_temporal_fusion):
            if fusion is not None:
                fusion.zero_output()
        nn.init.orthogonal_(
            self.option_key_projection.weight,
            gain=self.config.policy_output_gain,
        )
        nn.init.orthogonal_(
            self.count_output.weight,
            gain=self.config.policy_output_gain,
        )
        nn.init.normal_(
            self.stop_key,
            mean=0.0,
            std=self.config.policy_output_gain,
        )
        for residual in self.policy_residuals.values():
            cast(ExactPolicyQueryResidual, residual).zero_output()
        for residual in self.option_residuals.values():
            cast(ExactOptionEmbeddingResidual, residual).zero_output()
        for residual in self.value_residuals.values():
            if isinstance(residual, (ExactScalarValueResidual, ExactWdlValueResidual)):
                residual.zero_output()
            else:  # pragma: no cover - ModuleDict is constructed above.
                raise TypeError("unsupported exact value residual module")

    def encode_options(
        self,
        entity_embeddings: Tensor,
        options: OptionBatch,
        *,
        card_encoder: CardEncoder,
        temporal_context: Tensor | None = None,
        entity_valid_mask: Tensor | None = None,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> Tensor:
        """Encode and compare the complete engine-legal option set."""
        if uses_exact_v2_topology(self.config):
            if route_plan is None:
                raise ValueError(
                    "simple_stateless_v2 options require an exact route plan"
                )
            if (
                route_plan.resolved_registry_sha256
                != self.config.resolved_registry_sha256
            ):
                raise ValueError("v2 option route plan registry differs from model")
            route_plan.validate_exact_partition(
                expected_module_keys=self._exact_module_keys,
                device=entity_embeddings.device,
                allow_unrouted_rows=allow_unrouted_rows,
            )
        conditioned_entities = entity_embeddings
        entity_fusion = self.entity_temporal_fusion
        option_fusion = self.option_temporal_fusion
        if temporal_context is not None and entity_fusion is not None:
            if entity_valid_mask is None:
                raise ValueError("temporal entity fusion requires a valid-row mask")
            conditioned_entities = entity_fusion(
                entity_embeddings,
                temporal_context,
                valid_mask=entity_valid_mask,
            )
        shared = cast(
            Tensor,
            self.option_encoder(
                conditioned_entities,
                options,
                card_encoder=card_encoder,
            ),
        )
        if temporal_context is not None and option_fusion is not None:
            shared = option_fusion(
                shared,
                temporal_context,
                valid_mask=options.valid_options,
            )
        if not uses_exact_v2_topology(self.config):
            return shared
        if route_plan is None:
            raise RuntimeError("validated v2 option route plan unexpectedly missing")
        return shared + apply_exact_residual(
            shared,
            route_plan,
            self.option_residuals,
        )

    def route_private_banks(self) -> tuple[tuple[str, nn.ModuleDict], ...]:
        """Enumerate route-keyed output banks for transitions and exports."""
        banks = [
            ("policy_residuals", self.policy_residuals),
            ("value_residuals", self.value_residuals),
        ]
        if self.option_residuals:
            banks.append(("option_residuals", self.option_residuals))
        return tuple(banks)

    def inert_option_named_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return V2-only option outputs that must be zero after migration."""
        parameters = [
            (f"option_residuals.{module_key}.{name}", parameter)
            for module_key, module in self.option_residuals.items()
            for name, parameter in module.named_parameters()
            if name in {"up.weight", "up.bias"}
        ]
        parameters.extend(
            (
                f"option_encoder.{name}",
                parameter,
            )
            for name, parameter in self.option_encoder.inert_engine_named_parameters()
        )
        for module_name, fusion in (
            ("entity_temporal_fusion", self.entity_temporal_fusion),
            ("option_temporal_fusion", self.option_temporal_fusion),
        ):
            if fusion is None:
                continue
            parameters.extend(
                (f"{module_name}.{name}", parameter)
                for name, parameter in fusion.inert_output_named_parameters()
            )
        return tuple(parameters)

    def root_value(
        self,
        value_token: Tensor,
        opponent_belief: Tensor,
        *,
        route_plan: SimpleExactRoutePlan,
    ) -> Tensor:
        """Return the shared plus exact zero-sum root value in ``[-1, 1]``."""
        if uses_wdl_critic(self.config):
            return wdl_value_from_logits(
                self.root_wdl_logits(
                    value_token,
                    opponent_belief,
                    route_plan=route_plan,
                )
            )
        shared_input = value_token + self.value_belief_projection(opponent_belief)
        shared_head = self.shared_root_value
        if shared_head is None:
            raise RuntimeError("scalar root-value head is missing")
        shared = cast(Tensor, shared_head(shared_input))
        exact = apply_exact_residual(
            value_token,
            route_plan,
            self.value_residuals,
        )
        return torch.tanh(shared + exact).squeeze(-1)

    def root_wdl_logits(
        self,
        value_token: Tensor,
        opponent_belief: Tensor,
        *,
        route_plan: SimpleExactRoutePlan,
    ) -> Tensor:
        """Return route-conditioned ordered loss/draw/win root logits."""
        shared_head = self.shared_root_wdl
        if shared_head is None:
            raise ValueError("model architecture has no distributional WDL critic")
        shared_input = value_token + self.value_belief_projection(opponent_belief)
        shared = shared_head(shared_input)
        exact = apply_exact_residual(
            value_token,
            route_plan,
            self.value_residuals,
        )
        return cast(Tensor, shared + exact)

    def step(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor,
        route_plan: SimpleExactRoutePlan,
        ordered_rows: Tensor | None = None,
    ) -> SimplePolicyStep:
        """Score the next legal option or STOP from one corrected prefix query."""
        logits, query = self._step_logits(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            route_plan=route_plan,
            ordered_rows=ordered_rows,
        )
        prefix_value = torch.tanh(cast(Tensor, self.prefix_value(query))).squeeze(-1)
        return SimplePolicyStep(
            logits=logits,
            prefix_query=query,
            prefix_value=prefix_value,
        )

    def count_first_logits(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score legal cardinalities before canonical unordered selection."""
        logits, query = self._count_first_logits(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
        )
        prefix_value = torch.tanh(cast(Tensor, self.prefix_value(query))).squeeze(-1)
        return (logits, query, prefix_value)

    def _count_first_logits(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        decode_plan: _SimpleDecodePlan | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score legal cardinalities without evaluating the critic."""
        batch_size, max_options = options.valid_options.shape
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        history = option_embeddings.new_zeros((batch_size, self.config.d_model))
        query = self._query(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=history,
            route_plan=route_plan,
            policy_context=(
                None if decode_plan is None else decode_plan.policy_context
            ),
            selected_pool=(
                None
                if decode_plan is None
                else option_embeddings.new_zeros((batch_size, self.config.d_model))
            ),
        )
        valid = options.valid_options.to(dtype=option_embeddings.dtype).unsqueeze(-1)
        option_pool = (option_embeddings * valid).sum(dim=1)
        option_pool = option_pool / valid.sum(dim=1).clamp_min(1.0)
        counts = torch.arange(
            max_options + 1,
            dtype=options.min_counts.dtype,
            device=options.min_counts.device,
        ).unsqueeze(0)
        counts = counts.expand(batch_size, -1)
        features = _count_features(counts, options, dtype=policy_token.dtype)
        hidden = (
            self.count_state_projection(query).unsqueeze(1)
            + self.count_option_projection(option_pool).unsqueeze(1)
            + self.count_feature_projection(features)
        )
        logits = self.count_output(torch.nn.functional.gelu(hidden)).squeeze(-1)
        valid_count = options.valid_options.sum(dim=1).unsqueeze(1)
        legal = (
            counts.ge(options.min_counts.unsqueeze(1))
            & counts.le(options.max_counts.unsqueeze(1))
            & counts.le(valid_count)
        )
        return (logits.masked_fill(~legal, -torch.inf), query)

    def _prepare_decode_plan(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        ordered_rows: Tensor | None,
    ) -> _SimpleDecodePlan:
        """Precompute immutable tensors reused by count and pointer steps."""
        max_options = int(options.valid_options.shape[1])
        valid_as_counts = options.valid_options.to(dtype=torch.long)
        valid_options_to_right = (
            torch.flip(
                torch.cumsum(
                    torch.flip(valid_as_counts, dims=(1,)),
                    dim=1,
                ),
                dims=(1,),
            )
            - valid_as_counts
        )
        return _SimpleDecodePlan(
            policy_context=(
                policy_token + self.policy_belief_projection(opponent_belief)
            ),
            option_keys=self.option_key_projection(option_embeddings),
            option_indices=torch.arange(
                max_options,
                device=options.valid_options.device,
            ).unsqueeze(0),
            valid_options_to_right=valid_options_to_right,
            ordered_rows=_ordered_rows(options, ordered_rows=ordered_rows),
        )

    def teacher_forced(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        actions: SimpleTeacherForcedActionInput,
        *,
        route_plan: SimpleExactRoutePlan,
        temperature: float = 1.0,
        ordered_rows: Tensor | None = None,
        evaluate_prefix_values: bool = True,
    ) -> SimpleTeacherForcedEvaluation:
        """Replay action sequences with per-decode-token log-probs and values."""
        _validate_temperature(temperature)
        batch_size, max_options = options.valid_options.shape
        action_batch, max_action_length = _prepare_teacher_forced_actions(
            actions,
            batch_size=batch_size,
            device=policy_token.device,
            length_dtype=options.max_counts.dtype,
        )
        action_lengths = action_batch.lengths
        count_rows = _count_first_rows(options, ordered_rows=ordered_rows)
        _validate_action_batch(
            action_batch,
            options,
            count_rows=count_rows,
            max_action_length=max_action_length,
        )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, action_lengths, options.min_counts),
            max_counts=torch.where(count_rows, action_lengths, options.max_counts),
        )
        targets = _teacher_forced_targets(
            action_batch,
            max_options=max_options,
            max_action_length=max_action_length,
        )
        token_width = max_action_length + 1
        probability_dtype = (
            torch.float32
            if policy_token.dtype in {torch.bfloat16, torch.float16}
            else policy_token.dtype
        )
        token_logprobs = torch.zeros(
            (batch_size, token_width),
            dtype=probability_dtype,
            device=policy_token.device,
        )
        token_entropies = torch.zeros_like(token_logprobs)
        token_mask = torch.zeros(
            (batch_size, token_width),
            dtype=torch.bool,
            device=policy_token.device,
        )
        prefix_queries = policy_token.new_zeros(
            (batch_size, token_width, self.config.d_model)
        )
        prefix_values = policy_token.new_zeros((batch_size, token_width))

        if evaluate_prefix_values:
            count_logits, count_query, count_values = self.count_first_logits(
                policy_token,
                opponent_belief,
                option_embeddings,
                options,
                route_plan=route_plan,
            )
        else:
            count_logits, count_query = self._count_first_logits(
                policy_token,
                opponent_belief,
                option_embeddings,
                options,
                route_plan=route_plan,
            )
            count_values = policy_token.new_zeros((batch_size,))
        count_logprobs = torch.log_softmax(
            count_logits.to(dtype=probability_dtype) / temperature,
            dim=1,
        )
        selected_count_logprobs = count_logprobs.gather(
            1,
            action_lengths.unsqueeze(1),
        ).squeeze(1)
        count_entropies = _entropy(count_logprobs)
        rows = torch.nonzero(count_rows, as_tuple=False).flatten()
        if int(rows.numel()) > 0:
            token_logprobs[rows, 0] = selected_count_logprobs.index_select(0, rows)
            token_entropies[rows, 0] = count_entropies.index_select(0, rows)
            token_mask[rows, 0] = True
            prefix_queries[rows, 0] = count_query.index_select(0, rows).to(
                dtype=prefix_queries.dtype
            )
            prefix_values[rows, 0] = count_values.index_select(0, rows).to(
                dtype=prefix_values.dtype
            )

        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = option_embeddings.new_zeros((batch_size, self.config.d_model))
        done = count_rows & action_lengths.eq(0)
        step_logits: list[Tensor] = []
        for step_index in range(token_width):
            step_targets = targets[:, step_index]
            active = ~done
            active = active & torch.where(
                count_rows,
                step_index < action_lengths,
                step_index <= action_lengths,
            )
            active = active & selected_counts.lt(decode_options.max_counts)
            active_rows = torch.nonzero(active, as_tuple=False).flatten()
            if int(active_rows.numel()) == 0:
                continue
            if evaluate_prefix_values:
                result = self.step(
                    policy_token,
                    opponent_belief,
                    option_embeddings,
                    decode_options,
                    selected_mask=selected_mask,
                    selected_counts=selected_counts,
                    ordered_history=ordered_history,
                    route_plan=route_plan,
                    ordered_rows=ordered_rows,
                )
                logits = result.logits
                prefix_query = result.prefix_query
                prefix_value = result.prefix_value
            else:
                logits, prefix_query = self._step_logits(
                    policy_token,
                    opponent_belief,
                    option_embeddings,
                    decode_options,
                    selected_mask=selected_mask,
                    selected_counts=selected_counts,
                    ordered_history=ordered_history,
                    route_plan=route_plan,
                    ordered_rows=ordered_rows,
                )
                prefix_value = policy_token.new_zeros((batch_size,))
            step_logits.append(logits)
            active_logits = logits.index_select(0, active_rows)
            active_targets = step_targets.index_select(0, active_rows)
            selected_logits = active_logits.gather(
                1,
                active_targets.unsqueeze(1),
            ).squeeze(1)
            require_tensor_condition(
                torch.isfinite(selected_logits).all(),
                "teacher-forced action violates pointer legality",
            )
            logprobs = torch.log_softmax(
                active_logits.to(dtype=probability_dtype) / temperature,
                dim=1,
            )
            contributions = logprobs.gather(
                1,
                active_targets.unsqueeze(1),
            ).squeeze(1)
            entropies = _entropy(logprobs)
            slots = torch.full_like(active_rows, step_index) + count_rows.index_select(
                0, active_rows
            ).to(dtype=torch.long)
            token_logprobs[active_rows, slots] = contributions
            token_entropies[active_rows, slots] = entropies
            token_mask[active_rows, slots] = True
            prefix_queries[active_rows, slots] = prefix_query.index_select(
                0,
                active_rows,
            ).to(dtype=prefix_queries.dtype)
            prefix_values[active_rows, slots] = prefix_value.index_select(
                0,
                active_rows,
            ).to(dtype=prefix_values.dtype)

            selected = active & step_targets.lt(max_options)
            if max_options > 0:
                safe_targets = step_targets.clamp(min=0, max=max_options - 1)
                chosen = torch.gather(
                    option_embeddings,
                    dim=1,
                    index=safe_targets[:, None, None].expand(
                        -1,
                        1,
                        self.config.d_model,
                    ),
                ).squeeze(1)
                updated_history = ordered_history * _ORDERED_HISTORY_DECAY + chosen
                ordered_history = torch.where(
                    selected.unsqueeze(1),
                    updated_history,
                    ordered_history,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1,
                    safe_targets.unsqueeze(1),
                    selected.unsqueeze(1),
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + selected.to(
                    dtype=selected_counts.dtype
                )
            sampled_stop = active & step_targets.eq(max_options)
            done = done | sampled_stop
            done = done | selected_counts.ge(decode_options.max_counts)

        return SimpleTeacherForcedEvaluation(
            action_logprobs=(token_logprobs * token_mask).sum(dim=1),
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            token_mask=token_mask,
            prefix_queries=prefix_queries,
            prefix_values=prefix_values,
            step_logits=tuple(step_logits),
            count_logits=count_logits,
        )

    def greedy_decode(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        ordered_rows: Tensor | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Greedily decode legal option sequences while preserving STOP semantics."""
        return _action_sequences(
            self.greedy_decode_actions(
                policy_token,
                opponent_belief,
                option_embeddings,
                options,
                route_plan=route_plan,
                ordered_rows=ordered_rows,
            )
        )

    def greedy_decode_actions(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        ordered_rows: Tensor | None = None,
    ) -> SimpleTeacherForcedActionBatch:
        """Greedily decode on device and transfer the packed actions once."""
        return self.greedy_decode_trace(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
            ordered_rows=ordered_rows,
        ).actions

    def greedy_decode_trace(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        ordered_rows: Tensor | None = None,
    ) -> SimpleGreedyDecodeTrace:
        """Greedily decode actions and retain only engine-required semantics."""
        batch_size, max_options = options.valid_options.shape
        count_rows = _count_first_rows(options, ordered_rows=ordered_rows)
        count_logits, _query = self._count_first_logits(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
        )
        target_counts = torch.where(
            count_rows,
            count_logits.argmax(dim=1).to(dtype=options.max_counts.dtype),
            options.max_counts,
        )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, target_counts, options.min_counts),
            max_counts=torch.where(count_rows, target_counts, options.max_counts),
        )
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = option_embeddings.new_zeros((batch_size, self.config.d_model))
        stopped = selected_counts.ge(decode_options.max_counts)
        action_choices = torch.full(
            (batch_size, max_options),
            -1,
            dtype=torch.long,
            device=policy_token.device,
        )
        action_lengths = torch.zeros_like(options.max_counts)
        stop_sampled = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=policy_token.device,
        )
        maximum_steps = (
            max(options.maximum_counts, default=0)
            if options.maximum_counts
            else max_options + 1
        )
        for _step_index in range(maximum_steps):
            logits, _query = self._step_logits(
                policy_token,
                opponent_belief,
                option_embeddings,
                decode_options,
                selected_mask=selected_mask,
                selected_counts=selected_counts,
                ordered_history=ordered_history,
                route_plan=route_plan,
                ordered_rows=ordered_rows,
            )
            choices = logits.argmax(dim=1)
            active = ~stopped
            selected = active & choices.lt(max_options)
            safe_choices = choices.clamp(min=0, max=max(max_options - 1, 0))
            if max_options > 0:
                write_slots = action_lengths.clamp(
                    min=0,
                    max=max_options - 1,
                ).to(dtype=torch.long)
                previous_choices = action_choices.gather(
                    1,
                    write_slots.unsqueeze(1),
                ).squeeze(1)
                action_choices.scatter_(
                    1,
                    write_slots.unsqueeze(1),
                    torch.where(
                        selected,
                        safe_choices,
                        previous_choices,
                    ).unsqueeze(1),
                )
                chosen = torch.gather(
                    option_embeddings,
                    dim=1,
                    index=safe_choices[:, None, None].expand(
                        -1,
                        1,
                        self.config.d_model,
                    ),
                ).squeeze(1)
                updated_history = ordered_history * _ORDERED_HISTORY_DECAY + chosen
                ordered_history = torch.where(
                    selected.unsqueeze(1),
                    updated_history,
                    ordered_history,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1,
                    safe_choices.unsqueeze(1),
                    selected.unsqueeze(1),
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + selected.to(
                    dtype=selected_counts.dtype
                )
                action_lengths = action_lengths + selected.to(
                    dtype=action_lengths.dtype
                )
            sampled_stop = active & choices.eq(max_options)
            stop_sampled = stop_sampled | sampled_stop
            stopped = stopped | sampled_stop
            stopped = stopped | selected_counts.ge(decode_options.max_counts)
        return SimpleGreedyDecodeTrace(
            actions=SimpleTeacherForcedActionBatch(
                choice_indices=action_choices,
                lengths=action_lengths,
            ),
            stop_sampled=stop_sampled,
        )

    def sample_decode(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        temperature: float,
        generator: torch.Generator | None = None,
        ordered_rows: Tensor | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Sample legal count-first/pointer sequences for behavior rollout."""
        actions = self.sample_decode_actions(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
            temperature=temperature,
            generator=generator,
            ordered_rows=ordered_rows,
        )
        return _action_sequences(actions)

    def sample_decode_actions(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        temperature: float,
        generator: torch.Generator | None = None,
        ordered_rows: Tensor | None = None,
        sampling_uniforms: Tensor | None = None,
    ) -> SimpleTeacherForcedActionBatch:
        """Sample only engine actions while preserving behavior RNG semantics."""
        _validate_temperature(temperature)
        batch_size, max_options = options.valid_options.shape
        _validate_sampling_uniforms(
            sampling_uniforms,
            batch_size=batch_size,
            device=policy_token.device,
        )
        maximum_steps = (
            max(options.maximum_counts, default=0)
            if options.maximum_counts
            else max_options + 1
        )
        if (
            sampling_uniforms is not None
            and int(sampling_uniforms.shape[1]) < maximum_steps + 1
        ):
            raise ValueError("sampling uniforms do not cover every decode step")
        sampling_uniforms = _materialize_sampling_uniforms(
            sampling_uniforms,
            batch_size=batch_size,
            draw_count=maximum_steps + 1,
            device=policy_token.device,
            generator=generator,
        )
        decode_plan = self._prepare_decode_plan(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            ordered_rows=ordered_rows,
        )
        count_rows = ~decode_plan.ordered_rows & options.min_counts.lt(
            options.max_counts
        )
        count_logits, _count_query = self._count_first_logits(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
            decode_plan=decode_plan,
        )
        sampled_counts = _sample_logits(
            count_logits,
            temperature=temperature,
            generator=None,
            uniforms=sampling_uniforms[:, 0],
        ).to(dtype=options.max_counts.dtype)
        target_counts = torch.where(
            count_rows,
            sampled_counts,
            options.max_counts,
        )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, target_counts, options.min_counts),
            max_counts=torch.where(count_rows, target_counts, options.max_counts),
        )
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = option_embeddings.new_zeros((batch_size, self.config.d_model))
        stopped = selected_counts.ge(decode_options.max_counts)
        action_choices = torch.full(
            (batch_size, max_options),
            -1,
            dtype=torch.long,
            device=policy_token.device,
        )
        action_lengths = torch.zeros_like(options.max_counts)
        for _step_index in range(maximum_steps):
            logits, _query = self._step_logits(
                policy_token,
                opponent_belief,
                option_embeddings,
                decode_options,
                selected_mask=selected_mask,
                selected_counts=selected_counts,
                ordered_history=ordered_history,
                route_plan=route_plan,
                ordered_rows=ordered_rows,
                decode_plan=decode_plan,
            )
            choices = _sample_logits(
                logits,
                temperature=temperature,
                generator=None,
                uniforms=sampling_uniforms[:, _step_index + 1],
            )
            active = ~stopped
            selected = active & choices.lt(max_options)
            safe_choices = choices.clamp(min=0, max=max(max_options - 1, 0))
            if max_options > 0:
                write_slots = action_lengths.clamp(
                    min=0,
                    max=max_options - 1,
                ).to(dtype=torch.long)
                previous_choices = action_choices.gather(
                    1,
                    write_slots.unsqueeze(1),
                ).squeeze(1)
                action_choices.scatter_(
                    1,
                    write_slots.unsqueeze(1),
                    torch.where(
                        selected,
                        safe_choices,
                        previous_choices,
                    ).unsqueeze(1),
                )
                chosen = torch.gather(
                    option_embeddings,
                    dim=1,
                    index=safe_choices[:, None, None].expand(
                        -1,
                        1,
                        self.config.d_model,
                    ),
                ).squeeze(1)
                ordered_history = torch.where(
                    selected.unsqueeze(1),
                    ordered_history * _ORDERED_HISTORY_DECAY + chosen,
                    ordered_history,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1,
                    safe_choices.unsqueeze(1),
                    selected.unsqueeze(1),
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + selected.to(
                    dtype=selected_counts.dtype
                )
                action_lengths = action_lengths + selected.to(
                    dtype=action_lengths.dtype
                )
            stopped = stopped | (active & choices.eq(max_options))
            stopped = stopped | selected_counts.ge(decode_options.max_counts)
        return SimpleTeacherForcedActionBatch(
            choice_indices=action_choices,
            lengths=action_lengths,
        )

    def sample_decode_with_trace(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan,
        temperature: float,
        generator: torch.Generator | None = None,
        ordered_rows: Tensor | None = None,
        sampling_uniforms: Tensor | None = None,
    ) -> SimpleSampledDecodeTrace:
        """Sample actions and exact PPO behavior evidence in one decode pass."""
        _validate_temperature(temperature)
        batch_size, max_options = options.valid_options.shape
        _validate_sampling_uniforms(
            sampling_uniforms,
            batch_size=batch_size,
            device=policy_token.device,
        )
        maximum_steps = (
            max(options.maximum_counts, default=0)
            if options.maximum_counts
            else max_options + 1
        )
        if (
            sampling_uniforms is not None
            and int(sampling_uniforms.shape[1]) < maximum_steps + 1
        ):
            raise ValueError("sampling uniforms do not cover every decode step")
        sampling_uniforms = _materialize_sampling_uniforms(
            sampling_uniforms,
            batch_size=batch_size,
            draw_count=maximum_steps + 1,
            device=policy_token.device,
            generator=generator,
        )
        decode_plan = self._prepare_decode_plan(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            ordered_rows=ordered_rows,
        )
        count_rows = ~decode_plan.ordered_rows & options.min_counts.lt(
            options.max_counts
        )
        count_logits, count_query = self._count_first_logits(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            route_plan=route_plan,
            decode_plan=decode_plan,
        )
        count_values = torch.tanh(cast(Tensor, self.prefix_value(count_query))).squeeze(
            -1
        )
        sampled_count_indices, count_logprobs = _sample_logits_with_logprobs(
            count_logits,
            temperature=temperature,
            generator=None,
            uniforms=sampling_uniforms[:, 0],
        )
        sampled_counts = sampled_count_indices.to(dtype=options.max_counts.dtype)
        target_counts = torch.where(
            count_rows,
            sampled_counts,
            options.max_counts,
        )
        decode_options = replace(
            options,
            min_counts=torch.where(count_rows, target_counts, options.min_counts),
            max_counts=torch.where(count_rows, target_counts, options.max_counts),
        )
        selected_mask = torch.zeros_like(options.valid_options)
        selected_counts = torch.zeros_like(options.min_counts)
        ordered_history = option_embeddings.new_zeros((batch_size, self.config.d_model))
        stopped = selected_counts.ge(decode_options.max_counts)
        action_choices = torch.full(
            (batch_size, max_options),
            -1,
            dtype=torch.long,
            device=policy_token.device,
        )
        action_lengths = torch.zeros_like(options.max_counts)
        token_width = max_options + 1
        probability_dtype = (
            torch.float32
            if policy_token.dtype in {torch.bfloat16, torch.float16}
            else policy_token.dtype
        )
        token_logprob_storage = torch.zeros(
            (batch_size, token_width + 1),
            dtype=probability_dtype,
            device=policy_token.device,
        )
        token_mask_storage = torch.zeros(
            (batch_size, token_width + 1),
            dtype=torch.bool,
            device=policy_token.device,
        )
        prefix_value_storage = torch.zeros(
            (batch_size, token_width + 1),
            dtype=probability_dtype,
            device=policy_token.device,
        )
        selected_count_logprobs = count_logprobs.gather(
            1,
            sampled_count_indices.unsqueeze(1),
        ).squeeze(1)
        token_logprob_storage[:, 1] = torch.where(
            count_rows,
            selected_count_logprobs,
            torch.zeros_like(selected_count_logprobs),
        )
        token_mask_storage[:, 1] = count_rows
        prefix_value_storage[:, 1] = torch.where(
            count_rows,
            count_values,
            torch.zeros_like(count_values),
        ).to(dtype=prefix_value_storage.dtype)
        stop_sampled = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=policy_token.device,
        )
        decode_slot_offsets = count_rows.to(dtype=torch.long)
        for step_index in range(maximum_steps):
            logits, prefix_query = self._step_logits(
                policy_token,
                opponent_belief,
                option_embeddings,
                decode_options,
                selected_mask=selected_mask,
                selected_counts=selected_counts,
                ordered_history=ordered_history,
                route_plan=route_plan,
                ordered_rows=ordered_rows,
                decode_plan=decode_plan,
            )
            choices, logprobs = _sample_logits_with_logprobs(
                logits,
                temperature=temperature,
                generator=None,
                uniforms=sampling_uniforms[:, step_index + 1],
            )
            active = ~stopped
            selected = active & choices.lt(max_options)
            safe_choices = choices.clamp(min=0, max=max(max_options - 1, 0))
            contributions = logprobs.gather(
                1,
                choices.unsqueeze(1),
            ).squeeze(1)
            step_slots = decode_slot_offsets + step_index
            storage_indices = torch.where(
                active,
                step_slots + 1,
                torch.zeros_like(step_slots),
            ).unsqueeze(1)
            token_logprob_storage.scatter_(
                1,
                storage_indices,
                contributions.unsqueeze(1),
            )
            token_mask_storage.scatter_(
                1,
                storage_indices,
                active.unsqueeze(1),
            )
            step_prefix_values = torch.tanh(
                cast(Tensor, self.prefix_value(prefix_query))
            ).squeeze(-1)
            prefix_value_storage.scatter_(
                1,
                storage_indices,
                step_prefix_values.to(dtype=prefix_value_storage.dtype).unsqueeze(1),
            )
            if max_options > 0:
                write_slots = action_lengths.clamp(
                    min=0,
                    max=max_options - 1,
                ).to(dtype=torch.long)
                previous_choices = action_choices.gather(
                    1,
                    write_slots.unsqueeze(1),
                ).squeeze(1)
                action_choices.scatter_(
                    1,
                    write_slots.unsqueeze(1),
                    torch.where(
                        selected,
                        safe_choices,
                        previous_choices,
                    ).unsqueeze(1),
                )
                chosen = torch.gather(
                    option_embeddings,
                    dim=1,
                    index=safe_choices[:, None, None].expand(
                        -1,
                        1,
                        self.config.d_model,
                    ),
                ).squeeze(1)
                ordered_history = torch.where(
                    selected.unsqueeze(1),
                    ordered_history * _ORDERED_HISTORY_DECAY + chosen,
                    ordered_history,
                )
                new_selected = torch.zeros_like(selected_mask)
                new_selected.scatter_(
                    1,
                    safe_choices.unsqueeze(1),
                    selected.unsqueeze(1),
                )
                selected_mask = selected_mask | new_selected
                selected_counts = selected_counts + selected.to(
                    dtype=selected_counts.dtype
                )
                action_lengths = action_lengths + selected.to(
                    dtype=action_lengths.dtype
                )
            sampled_stop = active & choices.eq(max_options)
            stop_sampled = stop_sampled | sampled_stop
            stopped = stopped | sampled_stop
            stopped = stopped | selected_counts.ge(decode_options.max_counts)
        token_logprobs = token_logprob_storage[:, 1:]
        token_mask = token_mask_storage[:, 1:]
        prefix_values = prefix_value_storage[:, 1:]
        return SimpleSampledDecodeTrace(
            actions=SimpleTeacherForcedActionBatch(
                choice_indices=action_choices,
                lengths=action_lengths,
            ),
            action_logprobs=(token_logprobs * token_mask).sum(dim=1),
            token_logprobs=token_logprobs,
            token_mask=token_mask,
            prefix_values=prefix_values,
            stop_sampled=stop_sampled,
        )

    def _step_logits(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor,
        route_plan: SimpleExactRoutePlan,
        ordered_rows: Tensor | None,
        option_keys: Tensor | None = None,
        decode_plan: _SimpleDecodePlan | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one pointer step without evaluating the prefix critic."""
        query = self._query(
            policy_token,
            opponent_belief,
            option_embeddings,
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_history=ordered_history,
            route_plan=route_plan,
            policy_context=(
                None if decode_plan is None else decode_plan.policy_context
            ),
        )
        keys = (
            decode_plan.option_keys
            if decode_plan is not None
            else (
                self.option_key_projection(option_embeddings)
                if option_keys is None
                else option_keys
            )
        )
        option_logits = (keys * query.unsqueeze(1)).sum(dim=-1)
        option_logits = option_logits / math.sqrt(float(self.config.d_model))
        available = _available_options(
            options,
            selected_mask=selected_mask,
            selected_counts=selected_counts,
            ordered_rows=ordered_rows,
            option_indices=(
                None if decode_plan is None else decode_plan.option_indices
            ),
            valid_options_to_right=(
                None if decode_plan is None else decode_plan.valid_options_to_right
            ),
            resolved_ordered_rows=(
                None if decode_plan is None else decode_plan.ordered_rows
            ),
        )
        option_logits = option_logits.masked_fill(~available, -torch.inf)
        stop_logits = (query * self.stop_key.to(dtype=query.dtype)).sum(dim=-1)
        stop_logits = stop_logits / math.sqrt(float(self.config.d_model))
        stop_logits = stop_logits.masked_fill(
            selected_counts < options.min_counts,
            -torch.inf,
        )
        return (
            torch.cat((option_logits, stop_logits.unsqueeze(1)), dim=1),
            query,
        )

    def _query(
        self,
        policy_token: Tensor,
        opponent_belief: Tensor,
        option_embeddings: Tensor,
        options: OptionBatch,
        *,
        selected_mask: Tensor,
        selected_counts: Tensor,
        ordered_history: Tensor,
        route_plan: SimpleExactRoutePlan,
        policy_context: Tensor | None = None,
        selected_pool: Tensor | None = None,
    ) -> Tensor:
        """Build a shared prefix query, then add only its exact route residual."""
        if selected_pool is None:
            weights = selected_mask.to(dtype=option_embeddings.dtype).unsqueeze(-1)
            selected_pool = (option_embeddings * weights).sum(dim=1)
            selected_pool = selected_pool / weights.sum(dim=1).clamp_min(1.0)
        cardinality = _decoder_cardinality_features(
            options,
            selected_counts,
            dtype=policy_token.dtype,
        )
        shared_input = (
            (
                policy_token + self.policy_belief_projection(opponent_belief)
                if policy_context is None
                else policy_context
            )
            + self.selected_projection(selected_pool)
            + self.ordered_history_projection(ordered_history)
            + self.cardinality_projection(cardinality)
        )
        shared_query = cast(Tensor, self.query_projection(shared_input))
        return shared_query + apply_exact_residual(
            shared_query,
            route_plan,
            self.policy_residuals,
        )


def _available_options(
    options: OptionBatch,
    *,
    selected_mask: Tensor,
    selected_counts: Tensor,
    ordered_rows: Tensor | None,
    option_indices: Tensor | None = None,
    valid_options_to_right: Tensor | None = None,
    resolved_ordered_rows: Tensor | None = None,
) -> Tensor:
    """Apply engine-preserving uniqueness, order, and cardinality masks."""
    available = options.valid_options & ~selected_mask
    if int(available.shape[1]) > 0:
        if option_indices is None:
            option_indices = torch.arange(
                available.shape[1],
                device=available.device,
            ).unsqueeze(0)
        last_selected = torch.where(
            selected_mask,
            option_indices,
            torch.full_like(option_indices, -1),
        ).amax(dim=1)
        canonical_next = option_indices > last_selected.unsqueeze(1)
        required_after = (options.min_counts - selected_counts - 1).clamp_min(0)
        if valid_options_to_right is None:
            available_as_counts = available.to(dtype=torch.long)
            valid_options_to_right = (
                torch.flip(
                    torch.cumsum(
                        torch.flip(available_as_counts, dims=(1,)),
                        dim=1,
                    ),
                    dims=(1,),
                )
                - available_as_counts
            )
        canonical_next = canonical_next & valid_options_to_right.ge(
            required_after.unsqueeze(1)
        )
        order_sensitive = (
            _ordered_rows(options, ordered_rows=ordered_rows)
            if resolved_ordered_rows is None
            else resolved_ordered_rows
        )
        available = available & (order_sensitive.unsqueeze(1) | canonical_next)
    return available & selected_counts.lt(options.max_counts).unsqueeze(1)


def _ordered_rows(options: OptionBatch, *, ordered_rows: Tensor | None) -> Tensor:
    """Use explicit order semantics or conservative engine-context evidence."""
    if ordered_rows is not None:
        if ordered_rows.shape != options.max_counts.shape:
            raise ValueError("ordered_rows must have shape [batch]")
        return ordered_rows.to(device=options.max_counts.device, dtype=torch.bool)
    return ~_engine_proven_unordered_rows(options)


def _count_first_rows(
    options: OptionBatch,
    *,
    ordered_rows: Tensor | None,
) -> Tensor:
    """Use count-first only for engine-proven unordered variable-cardinality rows."""
    return ~_ordered_rows(options, ordered_rows=ordered_rows) & options.min_counts.lt(
        options.max_counts
    )


def simple_count_first_rows(
    options: OptionBatch,
    *,
    ordered_rows: Tensor | None = None,
) -> Tensor:
    """Return rows using the clean policy's count-first action contract."""
    return _count_first_rows(options, ordered_rows=ordered_rows)


def _engine_proven_unordered_rows(options: OptionBatch) -> Tensor:
    """Return rows whose engine context has native permutation parity."""
    rows = torch.zeros_like(options.valid_options.any(dim=1))
    for context in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS:
        rows = rows | (options.valid_options & options.contexts.eq(int(context))).any(
            dim=1
        )
    return rows


def _decoder_cardinality_features(
    options: OptionBatch,
    selected_counts: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Encode selected/remaining and distances to legal cardinality bounds."""
    option_counts = options.valid_options.sum(dim=1).to(dtype=dtype)
    selected = selected_counts.to(dtype=dtype)
    scale = option_counts.clamp_min(1.0)
    return torch.stack(
        (
            selected / scale,
            (option_counts - selected).clamp_min(0.0) / scale,
            (options.min_counts.to(dtype=dtype) - selected) / scale,
            (options.max_counts.to(dtype=dtype) - selected) / scale,
        ),
        dim=1,
    )


def _count_features(
    counts: Tensor,
    options: OptionBatch,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Encode candidate cardinalities relative to each legal prompt."""
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


def _prepare_teacher_forced_actions(
    actions: SimpleTeacherForcedActionInput,
    *,
    batch_size: int,
    device: torch.device,
    length_dtype: torch.dtype,
) -> tuple[SimpleTeacherForcedActionBatch, int]:
    """Materialize padded choices and lengths at most once for a replay call."""
    if isinstance(actions, SimpleTeacherForcedActionBatch):
        if int(actions.choice_indices.shape[0]) != batch_size:
            raise ValueError("actions must align with the option batch")
        if actions.maximum_length is not None:
            max_action_length = actions.maximum_length
        elif int(actions.lengths.numel()) == 0:
            max_action_length = 0
        else:
            max_action_length = int(actions.lengths.max().item())
        action_batch = SimpleTeacherForcedActionBatch(
            choice_indices=actions.choice_indices.to(
                device=device,
                dtype=torch.long,
            ),
            lengths=actions.lengths.to(
                device=device,
                dtype=length_dtype,
            ),
            maximum_length=max_action_length,
        )
    else:
        if len(actions) != batch_size:
            raise ValueError("actions must align with the option batch")
        python_lengths = tuple(len(action) for action in actions)
        max_action_length = max(python_lengths, default=0)
        if batch_size == 0:
            padded_choices: list[list[int]] = []
        else:
            padded_choices = [
                [int(index) for index in action]
                + [0] * (max_action_length - len(action))
                for action in actions
            ]
        action_batch = SimpleTeacherForcedActionBatch(
            choice_indices=torch.tensor(
                padded_choices,
                dtype=torch.long,
                device=device,
            ).reshape(batch_size, max_action_length),
            lengths=torch.tensor(
                python_lengths,
                dtype=length_dtype,
                device=device,
            ),
            maximum_length=max_action_length,
        )
    if max_action_length < 0:
        raise ValueError("action lengths cannot be negative")
    if max_action_length > int(action_batch.choice_indices.shape[1]):
        raise ValueError("action lengths exceed the padded choice width")
    return (action_batch, max_action_length)


def _action_sequences(
    actions: SimpleTeacherForcedActionBatch,
) -> tuple[tuple[int, ...], ...]:
    """Materialize the legacy Python action boundary with one host transfer."""
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
        tuple(int(choice) for choice in packed[row, : int(packed[row, choice_width])])
        for row in range(int(packed.shape[0]))
    )


def _teacher_forced_targets(
    actions: SimpleTeacherForcedActionBatch,
    *,
    max_options: int,
    max_action_length: int,
) -> Tensor:
    """Return all pointer targets, including per-row STOP, in one tensor."""
    targets = torch.full(
        (int(actions.lengths.shape[0]), max_action_length + 1),
        max_options,
        dtype=torch.long,
        device=actions.choice_indices.device,
    )
    if max_action_length == 0:
        return targets
    positions = torch.arange(
        max_action_length,
        device=actions.choice_indices.device,
    ).unsqueeze(0)
    active = positions < actions.lengths.unsqueeze(1)
    targets[:, :max_action_length] = torch.where(
        active,
        actions.choice_indices[:, :max_action_length],
        max_options,
    )
    return targets


def _validate_action_batch(
    actions: SimpleTeacherForcedActionBatch,
    options: OptionBatch,
    *,
    count_rows: Tensor,
    max_action_length: int,
) -> None:
    """Validate all action rows with one vectorized device-to-host decision."""
    lengths = actions.lengths
    choices = actions.choice_indices[:, :max_action_length]
    cardinality_violation = (
        lengths.lt(options.min_counts) | lengths.gt(options.max_counts)
    ).any()
    if max_action_length == 0:
        duplicate_violation = cardinality_violation.new_zeros(())
        invalid_index_violation = cardinality_violation.new_zeros(())
        order_violation = cardinality_violation.new_zeros(())
    else:
        positions = torch.arange(
            max_action_length,
            device=choices.device,
        ).unsqueeze(0)
        active = positions < lengths.unsqueeze(1)
        valid_counts = options.valid_options.sum(dim=1).unsqueeze(1)
        invalid_index_violation = (
            active & (choices.lt(0) | choices.ge(valid_counts))
        ).any()
        if max_action_length == 1:
            duplicate_violation = cardinality_violation.new_zeros(())
            order_violation = cardinality_violation.new_zeros(())
        else:
            sorted_choices, permutation = choices.sort(dim=1)
            sorted_active = active.gather(1, permutation)
            duplicate_violation = (
                sorted_active[:, 1:]
                & sorted_active[:, :-1]
                & sorted_choices[:, 1:].eq(sorted_choices[:, :-1])
            ).any()
            adjacent_active = positions[:, 1:] < lengths.unsqueeze(1)
            order_violation = (
                count_rows.unsqueeze(1)
                & adjacent_active
                & choices[:, 1:].le(choices[:, :-1])
            ).any()
    require_tensor_condition(
        ~cardinality_violation,
        "action violates engine cardinality bounds",
    )
    require_tensor_condition(
        ~duplicate_violation,
        "action selects the same legal option more than once",
    )
    require_tensor_condition(
        ~invalid_index_violation,
        "action indexes an invalid option",
    )
    require_tensor_condition(
        ~order_violation,
        "count-first action must use the canonical option order",
    )


def _is_integer_tensor(tensor: Tensor) -> bool:
    """Return whether a tensor can represent exact action indices/counts."""
    return (
        tensor.dtype != torch.bool
        and not tensor.dtype.is_floating_point
        and not tensor.dtype.is_complex
    )


def _entropy(logprobs: Tensor) -> Tensor:
    """Return categorical entropy without multiplying zero by ``-inf``."""
    finite = torch.isfinite(logprobs)
    safe_logprobs = torch.where(
        finite,
        logprobs,
        torch.zeros_like(logprobs),
    )
    probabilities = safe_logprobs.exp() * finite
    terms = probabilities * safe_logprobs
    return -terms.sum(dim=1)


def _sample_logits(
    logits: Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
    uniforms: Tensor | None = None,
) -> Tensor:
    choices, _logprobs = _sample_logits_with_logprobs(
        logits,
        temperature=temperature,
        generator=generator,
        uniforms=uniforms,
    )
    return choices


def _sample_logits_with_logprobs(
    logits: Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
    uniforms: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample from FP32 log-softmax probabilities and return the same evidence."""
    logprobs = torch.log_softmax(logits.float() / temperature, dim=1)
    probabilities = logprobs.exp()
    require_tensor_condition(
        torch.isfinite(probabilities).all(),
        "policy sampling probabilities are invalid",
    )
    if uniforms is not None:
        if uniforms.shape != (int(logits.shape[0]),):
            raise ValueError("sampling uniforms must align with logits")
        if uniforms.device != logits.device:
            raise ValueError("sampling uniforms must share the logits device")
        cumulative_probabilities = probabilities.cumsum(dim=1)
        cumulative_totals = cumulative_probabilities[:, -1]
        sampling_targets = uniforms.to(dtype=probabilities.dtype) * cumulative_totals
        sampling_targets = torch.minimum(
            sampling_targets,
            torch.nextafter(
                cumulative_totals,
                torch.zeros_like(cumulative_totals),
            ),
        )
        choices = torch.searchsorted(
            cumulative_probabilities,
            sampling_targets.unsqueeze(1),
            right=True,
        ).squeeze(1)
        return (
            choices.clamp_max(int(logits.shape[1]) - 1),
            logprobs,
        )
    return (
        torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1),
        logprobs,
    )


def _materialize_sampling_uniforms(
    uniforms: Tensor | None,
    *,
    batch_size: int,
    draw_count: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Generate one schedule-independent random slab for the whole decode."""
    if uniforms is not None:
        return uniforms
    return torch.rand(
        (batch_size, draw_count),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def _validate_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sampling temperature must be finite and positive")


def _validate_sampling_uniforms(
    uniforms: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    """Validate optional schedule-independent per-row sampling draws."""
    if uniforms is None:
        return
    if uniforms.ndim != 2 or int(uniforms.shape[0]) != batch_size:
        raise ValueError("sampling uniforms must have shape [batch, draws]")
    if int(uniforms.shape[1]) < 1:
        raise ValueError("sampling uniforms must contain at least one draw")
    if uniforms.device != device:
        raise ValueError("sampling uniforms must share the policy device")
    require_tensor_condition(
        torch.isfinite(uniforms).all()
        & uniforms.ge(0.0).all()
        & uniforms.lt(1.0).all(),
        "sampling uniforms must be finite values in [0, 1)",
    )
