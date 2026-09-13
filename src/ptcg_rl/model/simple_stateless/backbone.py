"""Shared input encoding and Transformer backbone for the stateless policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.card_encoder import CardEncoder, build_card_encoder
from ptcg_rl.decks.registry import validate_active_exact_strategy_routes
from ptcg_rl.model.simple_stateless.belief import (
    PublicBeliefSummaryBatch,
    PublicBeliefSummaryEncoder,
)
from ptcg_rl.model.simple_stateless.config import (
    SimpleStatelessModelConfig,
    uses_exact_v2_topology,
    uses_family_private_topology,
    uses_temporal_prefusion,
)
from ptcg_rl.model.simple_stateless.family_private import (
    FAMILY_PRIVATE_INHERITED_LAYERS,
    FAMILY_PRIVATE_SHARED_LAYERS,
    FamilyPrivateStrategyBank,
)
from ptcg_rl.model.simple_stateless.layers import (
    PackedTransformerTrunk,
    initialize_simple_stateless_module,
)
from ptcg_rl.model.simple_stateless.packed import (
    PackedTokenBatch,
    SpecialTokenPositions,
    prepend_special_tokens,
)
from ptcg_rl.model.simple_stateless.routing import SimpleExactRoutePlan
from ptcg_rl.model.simple_stateless.state import SimpleStatelessRawStateEncoder
from ptcg_rl.model.simple_stateless.v2 import (
    GENERALIST_SEQUENCE_V2_CAPSULE_STAGES,
    SIMPLE_STATELESS_V2_CAPSULE_STAGES,
    SimpleStatelessV2Adapters,
)
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.model.tensor_validation import require_tensor_condition


class CountAwareDeckEncoder(nn.Module):
    """Permutation-invariant exact-deck multiset encoder."""

    def __init__(self, *, d_model: int, hidden_dim: int, deck_size: int) -> None:
        """Initialize count-aware DeepSet projections."""
        super().__init__()
        self.deck_size = deck_size
        self.card_projection = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        card_embeddings: Tensor,
        counts: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Encode unique card rows and their multiplicities into one deck token."""
        if card_embeddings.ndim != 3:
            raise ValueError("card_embeddings must have shape [batch, cards, d_model]")
        if (
            counts.shape != card_embeddings.shape[:2]
            or valid_mask.shape != counts.shape
        ):
            raise ValueError("deck counts and mask must align with card embeddings")
        if valid_mask.dtype != torch.bool:
            raise ValueError("deck valid_mask must be boolean")
        count_values = counts.to(dtype=card_embeddings.dtype)
        if not count_values.is_meta:
            require_tensor_condition(
                ~(count_values < 0).any(),
                "deck card counts cannot be negative",
            )
            totals = (count_values * valid_mask).sum(dim=1)
            require_tensor_condition(
                torch.isclose(
                    totals,
                    totals.new_full(totals.shape, 60.0),
                ).all(),
                "every encoded exact deck must contain 60 cards",
            )
        count_feature = torch.log1p(count_values).unsqueeze(-1)
        encoded = self.card_projection(
            torch.cat((card_embeddings, count_feature), dim=-1)
        )
        weighted = encoded * count_values.unsqueeze(-1) * valid_mask.unsqueeze(-1)
        pooled = weighted.sum(dim=1) / float(self.deck_size)
        return cast(Tensor, self.output_projection(pooled))


class SimpleStatelessInputEncoder(nn.Module):
    """Sole CardEncoder owner plus fixed learned special tokens."""

    def __init__(
        self,
        config: SimpleStatelessModelConfig,
        *,
        static_features: npt.NDArray[np.float32] | Tensor | None = None,
        load_static_features: bool = True,
    ) -> None:
        """Initialize shared card/deck encoders and special-token parameters."""
        super().__init__()
        self.config = config
        if static_features is None and load_static_features:
            self.card_encoder = build_card_encoder(config.card_encoder)
        elif static_features is None:
            self.card_encoder = CardEncoder(
                d_model=config.d_model,
                hidden_dim=config.card_encoder.hidden_dim,
                dropout=0.0,
                embedding_l2=config.card_encoder.embedding_l2,
            )
        else:
            self.card_encoder = CardEncoder.from_feature_table(
                static_features,
                d_model=config.d_model,
                hidden_dim=config.card_encoder.hidden_dim,
                dropout=0.0,
                embedding_l2=config.card_encoder.embedding_l2,
            )
        self.deck_encoder = CountAwareDeckEncoder(
            d_model=config.d_model,
            hidden_dim=config.deck_hidden_dim,
            deck_size=config.deck_size,
        )
        self.belief_summary_encoder = PublicBeliefSummaryEncoder(d_model=config.d_model)
        self.belief_public_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.raw_state_encoder = SimpleStatelessRawStateEncoder(d_model=config.d_model)
        self.policy_token = nn.Parameter(torch.empty(config.d_model))
        self.value_token = nn.Parameter(torch.empty(config.d_model))
        self.scratch_tokens = nn.Parameter(
            torch.empty(config.scratch_tokens, config.d_model)
        )

    def reset_special_tokens(self) -> None:
        """Initialize learned tokens independently with a small normal prior."""
        nn.init.normal_(self.policy_token, mean=0.0, std=0.02)
        nn.init.normal_(self.value_token, mean=0.0, std=0.02)
        nn.init.normal_(self.scratch_tokens, mean=0.0, std=0.02)
        self.raw_state_encoder.reset_non_linear_parameters()

    def encode_decks(
        self,
        unique_card_ids: Tensor,
        counts: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Encode exact deck multisets through the sole shared CardEncoder."""
        card_embeddings = self.card_encoder(unique_card_ids)
        return cast(Tensor, self.deck_encoder(card_embeddings, counts, valid_mask))

    def encode_public_belief(
        self,
        summary: PublicBeliefSummaryBatch,
        *,
        public_context: Tensor,
    ) -> Tensor:
        """Encode raw catalog counts in-graph with the current CardEncoder."""
        summary_rows = (
            int(summary.row_indices.numel())
            if summary.row_indices is not None
            else int(summary.card_ids.shape[0])
        )
        if summary_rows != public_context.shape[0]:
            raise ValueError("public belief summary batch differs from state batch")
        expected_fingerprint = self.config.public_deck_catalog_fingerprint
        if (
            expected_fingerprint is not None
            and summary.catalog_fingerprint != expected_fingerprint
        ):
            raise ValueError("public belief summary catalog fingerprint mismatch")
        card_embeddings = self.card_encoder(summary.card_ids)
        summary_token = self.belief_summary_encoder(card_embeddings, summary)
        if summary.row_indices is not None:
            summary_token = summary_token.index_select(0, summary.row_indices)
        if summary_token.shape[0] != public_context.shape[0]:
            raise ValueError("public belief row mapping differs from state batch")
        return cast(
            Tensor,
            summary_token + self.belief_public_projection(public_context),
        )

    def compose(
        self,
        *,
        public_context: Tensor,
        deck_tokens: Tensor,
        opponent_belief: Tensor,
        entity_tokens: PackedTokenBatch,
    ) -> tuple[PackedTokenBatch, SpecialTokenPositions]:
        """Create the fixed public/deck/belief/policy/value/scratch layout."""
        batch_size = entity_tokens.batch_size
        expected = (batch_size, self.config.d_model)
        for name, tensor in (
            ("public_context", public_context),
            ("deck_tokens", deck_tokens),
            ("opponent_belief", opponent_belief),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        learned = torch.cat(
            (
                self.policy_token.unsqueeze(0),
                self.value_token.unsqueeze(0),
                self.scratch_tokens,
            ),
            dim=0,
        )
        learned = preserve_cuda_bfloat16_activation(learned)
        learned = learned.unsqueeze(0).expand(batch_size, -1, -1)
        specials = preserve_cuda_bfloat16_activation(
            torch.cat(
                (
                    public_context.unsqueeze(1),
                    deck_tokens.unsqueeze(1),
                    opponent_belief.unsqueeze(1),
                    learned,
                ),
                dim=1,
            )
        )
        packed, positions = prepend_special_tokens(entity_tokens, specials)
        return (
            packed.with_tokens(preserve_cuda_bfloat16_activation(packed.tokens)),
            positions,
        )


@dataclass(frozen=True)
class SimpleStatelessBackboneOutput:
    """Semantic token outputs plus all contextualized packed tokens."""

    packed: PackedTokenBatch
    positions: SpecialTokenPositions
    public: Tensor
    deck: Tensor
    opponent_belief: Tensor
    policy: Tensor
    value: Tensor
    scratch: Tensor
    temporal_context: Tensor | None = None


def select_simple_stateless_backbone_rows(
    output: SimpleStatelessBackboneOutput,
    indices: Tensor,
) -> SimpleStatelessBackboneOutput:
    """Select or repeat packed snapshot rows without cross-row attention."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("backbone row indices must be a one-dimensional long tensor")
    if indices.device != output.packed.tokens.device:
        raise ValueError("backbone row indices must use the packed-token device")
    if not indices.is_meta and bool(
        ((indices < 0) | (indices >= output.packed.batch_size)).any()
    ):
        raise IndexError("backbone row index is out of range")
    source_rows = tuple(int(value) for value in indices.detach().cpu().tolist())
    packed = PackedTokenBatch.from_sequences(
        tuple(
            output.packed.tokens[
                output.packed.offsets[row] : output.packed.offsets[row + 1]
            ]
            for row in source_rows
        )
    )
    starts = packed.cu_seqlens[:-1].to(dtype=torch.long)
    scratch_width = int(output.scratch.shape[1])
    scratch_offsets = torch.arange(
        5,
        5 + scratch_width,
        dtype=torch.long,
        device=indices.device,
    )
    positions = SpecialTokenPositions(
        public=starts,
        deck=starts + 1,
        opponent_belief=starts + 2,
        policy=starts + 3,
        value=starts + 4,
        scratch=starts[:, None] + scratch_offsets[None, :],
    )
    return SimpleStatelessBackboneOutput(
        packed=packed,
        positions=positions,
        public=output.public.index_select(0, indices),
        deck=output.deck.index_select(0, indices),
        opponent_belief=output.opponent_belief.index_select(0, indices),
        policy=output.policy.index_select(0, indices),
        value=output.value.index_select(0, indices),
        scratch=output.scratch.index_select(0, indices),
        temporal_context=(
            None
            if output.temporal_context is None
            else output.temporal_context.index_select(0, indices)
        ),
    )


class SimpleStatelessBackbone(nn.Module):
    """Architecture-bound shared stateless Transformer backbone."""

    def __init__(
        self,
        config: SimpleStatelessModelConfig,
        *,
        static_features: npt.NDArray[np.float32] | Tensor | None = None,
        load_static_features: bool = True,
        initialize: bool = True,
    ) -> None:
        """Build the sole input owner and the deep shared trunk."""
        super().__init__()
        if uses_exact_v2_topology(config) and (
            not config.exact_routes or config.resolved_registry_sha256 is None
        ):
            raise ValueError("exact-route model requires a resolved exact registry")
        if uses_exact_v2_topology(config):
            validate_active_exact_strategy_routes(config.exact_routes)
        self.config = config
        self._exact_module_keys = frozenset(
            route.module_key for route in config.exact_routes
        )
        self.input_encoder = SimpleStatelessInputEncoder(
            config,
            static_features=static_features,
            load_static_features=load_static_features,
        )
        self.trunk = PackedTransformerTrunk(
            d_model=config.d_model,
            num_layers=(
                FAMILY_PRIVATE_SHARED_LAYERS
                if uses_family_private_topology(config)
                else config.num_layers
            ),
            num_heads=config.attention_heads,
            feedforward_dim=config.feedforward_dim,
            residual_scale_layers=(
                FAMILY_PRIVATE_INHERITED_LAYERS
                if uses_family_private_topology(config)
                else None
            ),
        )
        self.v2_adapters: SimpleStatelessV2Adapters | None
        if uses_exact_v2_topology(config):
            self.v2_adapters = SimpleStatelessV2Adapters(
                d_model=config.d_model,
                module_keys=tuple(route.module_key for route in config.exact_routes),
                stage_layers=(
                    GENERALIST_SEQUENCE_V2_CAPSULE_STAGES
                    if uses_temporal_prefusion(config)
                    else SIMPLE_STATELESS_V2_CAPSULE_STAGES
                ),
            )
        else:
            self.v2_adapters = None
        self.family_private: FamilyPrivateStrategyBank | None
        if uses_family_private_topology(config):
            self.family_private = FamilyPrivateStrategyBank(
                d_model=config.d_model,
                num_heads=config.attention_heads,
                feedforward_dim=config.feedforward_dim,
                exact_module_keys_by_digest={
                    route.deck_digest: route.module_key
                    for route in config.exact_routes
                },
                family_routes=config.family_routes,
                include_generic_upper=config.export_mode == "routed",
            )
        else:
            self.family_private = None
        if initialize:
            initialize_simple_stateless_module(self)
            self.input_encoder.reset_special_tokens()
            if self.v2_adapters is not None:
                self.v2_adapters.zero_output()
            if self.family_private is not None:
                self.family_private.zero_appended_outputs()

    def forward(
        self,
        *,
        public_context: Tensor,
        unique_deck_card_ids: Tensor,
        deck_counts: Tensor,
        deck_valid_mask: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        entity_tokens: PackedTokenBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Encode one stateless batch and return fixed semantic token rows."""
        deck_tokens = self.input_encoder.encode_decks(
            unique_deck_card_ids,
            deck_counts,
            deck_valid_mask,
        )
        return self.forward_with_deck_tokens(
            public_context=public_context,
            deck_tokens=deck_tokens,
            belief_summary=belief_summary,
            entity_tokens=entity_tokens,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def forward_with_deck_tokens(
        self,
        *,
        public_context: Tensor,
        deck_tokens: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        entity_tokens: PackedTokenBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Encode dynamic public inputs with preencoded exact-deck tokens."""
        active_route_plan = self._validated_v2_route_plan(
            route_plan,
            batch_size=entity_tokens.batch_size,
            device=entity_tokens.tokens.device,
            allow_unrouted_rows=allow_unrouted_rows,
        )
        opponent_belief = self.input_encoder.encode_public_belief(
            belief_summary,
            public_context=public_context,
        )
        packed, positions = self.input_encoder.compose(
            public_context=public_context,
            deck_tokens=deck_tokens,
            opponent_belief=opponent_belief,
            entity_tokens=entity_tokens,
        )
        v2_adapters = self.v2_adapters
        if v2_adapters is None:
            encoded = self.trunk(packed)
        else:
            if active_route_plan is None:
                raise RuntimeError("validated v2 route plan unexpectedly missing")
            packed = v2_adapters.apply_prompt(
                packed,
                positions,
                route_plan=active_route_plan,
                allow_unrouted_rows=allow_unrouted_rows,
            )

            def apply_v2_stage(
                completed_layers: int,
                stage_input: PackedTokenBatch,
            ) -> PackedTokenBatch:
                return v2_adapters.apply_stage(
                    completed_layers,
                    stage_input,
                    positions,
                    route_plan=active_route_plan,
                    allow_unrouted_rows=allow_unrouted_rows,
                )

            family_private = self.family_private
            if family_private is None:
                encoded = self.trunk(
                    packed,
                    layer_boundary_callback=apply_v2_stage,
                )
            else:
                encoded = self.trunk(
                    packed,
                    layer_boundary_callback=apply_v2_stage,
                    normalize_output=False,
                )
                encoded = family_private.apply_cloned(
                    encoded,
                    route_plan=active_route_plan,
                    allow_unrouted_rows=allow_unrouted_rows,
                )
                encoded = v2_adapters.apply_stage(
                    20,
                    encoded,
                    positions,
                    route_plan=active_route_plan,
                    allow_unrouted_rows=allow_unrouted_rows,
                )
                encoded = family_private.apply_appended(
                    encoded,
                    route_plan=active_route_plan,
                )
                encoded = self.trunk.normalize(encoded)
        tokens = encoded.tokens
        return SimpleStatelessBackboneOutput(
            packed=encoded,
            positions=positions,
            public=tokens[positions.public],
            deck=tokens[positions.deck],
            opponent_belief=tokens[positions.opponent_belief],
            policy=tokens[positions.policy],
            value=tokens[positions.value],
            scratch=tokens[positions.scratch],
        )

    def forward_observation(
        self,
        *,
        state: StateBatch,
        unique_deck_card_ids: Tensor,
        deck_counts: Tensor,
        deck_valid_mask: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Encode raw public observation arrays through the sole card owner."""
        public_context, entity_tokens = self.input_encoder.raw_state_encoder(
            state,
            card_encoder=self.input_encoder.card_encoder,
        )
        return self.forward(
            public_context=public_context,
            unique_deck_card_ids=unique_deck_card_ids,
            deck_counts=deck_counts,
            deck_valid_mask=deck_valid_mask,
            belief_summary=belief_summary,
            entity_tokens=entity_tokens,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def forward_observation_with_deck_tokens(
        self,
        *,
        state: StateBatch,
        deck_tokens: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Encode one observation while reusing immutable exact-deck tokens."""
        public_context, entity_tokens = self.input_encoder.raw_state_encoder(
            state,
            card_encoder=self.input_encoder.card_encoder,
        )
        return self.forward_with_deck_tokens(
            public_context=public_context,
            deck_tokens=deck_tokens,
            belief_summary=belief_summary,
            entity_tokens=entity_tokens,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def _validated_v2_route_plan(
        self,
        route_plan: SimpleExactRoutePlan | None,
        *,
        batch_size: int,
        device: torch.device,
        allow_unrouted_rows: bool,
    ) -> SimpleExactRoutePlan | None:
        """Require the exact registry route only for architecture v2."""
        if self.v2_adapters is None:
            return None
        if route_plan is None:
            raise ValueError("simple_stateless_v2 requires an exact route plan")
        if route_plan.batch_size != batch_size:
            raise ValueError("v2 route plan and packed batch differ")
        if route_plan.resolved_registry_sha256 != self.config.resolved_registry_sha256:
            raise ValueError("v2 route plan registry differs from model identity")
        route_plan.validate_exact_partition(
            expected_module_keys=self._exact_module_keys,
            device=device,
            allow_unrouted_rows=allow_unrouted_rows,
        )
        return route_plan
