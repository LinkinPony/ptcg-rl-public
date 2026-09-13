"""Role-aware legal-option encoding and shared set comparison."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.policy import (
    CONTEXT_EMBEDDING_COUNT,
    CONTEXT_OOV_INDEX,
    DEFAULT_MAX_ATTACK_ID,
    KNOWN_CONTEXT_COUNT,
    KNOWN_OPTION_TYPE_COUNT,
    MAX_ENTITY_SLOTS,
    OPTION_TYPE_EMBEDDING_COUNT,
    OPTION_TYPE_OOV_INDEX,
    OptionBatch,
)
from ptcg_rl.model.simple_stateless.layers import PackedTransformerBlock
from ptcg_rl.model.simple_stateless.packed import PackedTokenBatch


class RoleAwareOptionEncoder(nn.Module):
    """Encode engine-legal options with distinct source/target entity roles."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        max_attack_id: int = DEFAULT_MAX_ATTACK_ID,
        zero_gated_dynamic_effects: bool = False,
    ) -> None:
        """Initialize shared option features and one comparator block."""
        super().__init__()
        self.d_model = d_model
        self.max_attack_id = max_attack_id
        self.zero_gated_dynamic_effects = zero_gated_dynamic_effects
        self.option_type_embedding = nn.Embedding(
            OPTION_TYPE_EMBEDDING_COUNT,
            d_model,
        )
        self.context_embedding = nn.Embedding(CONTEXT_EMBEDDING_COUNT, d_model)
        self.attack_embedding = nn.Embedding(
            max_attack_id + 1,
            d_model,
            padding_idx=0,
        )
        self.entity_role_embedding = nn.Embedding(MAX_ENTITY_SLOTS, d_model)
        self.entity_role_projections = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(MAX_ENTITY_SLOTS)
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(SCALAR_FEATURE_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dynamic_effect_projection = nn.Sequential(
            nn.Linear(DYNAMIC_EFFECT_FEATURE_SIZE + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.engine_factual_residual = (
            nn.Sequential(
                nn.Linear(DYNAMIC_EFFECT_FEATURE_SIZE + 1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if zero_gated_dynamic_effects
            else None
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.comparator = PackedTransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            residual_scale=2.0**-0.5,
        )
        if self.engine_factual_residual is not None:
            self.zero_engine_factual_output()

    def zero_engine_factual_output(self) -> None:
        """Restore exact incumbent parity after parent-wide initialization."""
        if self.engine_factual_residual is None:
            return
        output = cast(nn.Linear, self.engine_factual_residual[-1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def inert_engine_named_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the zero-output gate that protects incumbent parity."""
        if self.engine_factual_residual is None:
            return ()
        output = cast(nn.Linear, self.engine_factual_residual[-1])
        return (
            (
                "engine_factual_residual.2.weight",
                cast(nn.Parameter, output.weight),
            ),
            (
                "engine_factual_residual.2.bias",
                output.bias,
            ),
        )

    def forward(
        self,
        entity_embeddings: Tensor,
        options: OptionBatch,
        *,
        card_encoder: CardEncoder,
    ) -> Tensor:
        """Return contextualized option rows aligned to the padded option batch."""
        _validate_option_batch(options, batch_size=int(entity_embeddings.shape[0]))
        if entity_embeddings.ndim != 3:
            raise ValueError(
                "entity_embeddings must have shape [batch, entities, d_model]"
            )
        if int(entity_embeddings.shape[-1]) != self.d_model:
            raise ValueError("entity embedding width differs from option encoder")
        entity_features = self._role_aware_entities(entity_embeddings, options)
        safe_types = _safe_indices(
            options.option_types,
            max_known=KNOWN_OPTION_TYPE_COUNT,
            oov_index=OPTION_TYPE_OOV_INDEX,
        )
        safe_contexts = _safe_indices(
            options.contexts,
            max_known=KNOWN_CONTEXT_COUNT,
            oov_index=CONTEXT_OOV_INDEX,
        )
        safe_attacks = torch.where(
            (options.attack_ids >= 0) & (options.attack_ids <= self.max_attack_id),
            options.attack_ids,
            torch.zeros_like(options.attack_ids),
        )
        feature_dtype = entity_embeddings.dtype
        dynamic_features = options.dynamic_effect_features.to(dtype=feature_dtype)
        dynamic_mask = options.dynamic_effect_masks.to(dtype=feature_dtype).unsqueeze(
            -1
        )
        dynamic_inputs = torch.cat(
            (
                dynamic_features * dynamic_mask,
                dynamic_mask,
            ),
            dim=-1,
        )
        legacy_dynamic_inputs = (
            torch.zeros_like(dynamic_inputs)
            if self.zero_gated_dynamic_effects
            else dynamic_inputs
        )
        engine_residual = (
            torch.zeros_like(entity_features)
            if self.engine_factual_residual is None
            else self.engine_factual_residual(dynamic_inputs)
        )
        encoded = (
            entity_features
            + self.option_type_embedding(safe_types)
            + self.context_embedding(safe_contexts)
            + self.attack_embedding(safe_attacks)
            + card_encoder(options.card_ids)
            + self.scalar_projection(options.scalars.to(dtype=feature_dtype))
            + self.dynamic_effect_projection(legacy_dynamic_inputs)
            + engine_residual
        )
        encoded = preserve_cuda_bfloat16_activation(self.input_norm(encoded))
        encoded = encoded.masked_fill(~options.valid_options.unsqueeze(-1), 0.0)
        return self._compare_valid_options(
            encoded,
            options.valid_options,
            option_lengths=options.option_lengths,
        )

    def _role_aware_entities(
        self,
        entity_embeddings: Tensor,
        options: OptionBatch,
    ) -> Tensor:
        """Gather two entity slots without erasing source/target semantics."""
        batch_size, max_options, slot_count = options.entity_slots.shape
        if slot_count != MAX_ENTITY_SLOTS:
            raise ValueError("option entity slot count differs from action schema")
        entity_count = int(entity_embeddings.shape[1])
        safe_slots = options.entity_slots.clamp(
            min=0,
            max=max(entity_count - 1, 0),
        )
        expanded = entity_embeddings.unsqueeze(1).expand(
            batch_size,
            max_options,
            entity_count,
            self.d_model,
        )
        gathered = torch.gather(
            expanded,
            dim=2,
            index=safe_slots.unsqueeze(-1).expand(
                batch_size,
                max_options,
                slot_count,
                self.d_model,
            ),
        )
        combined = torch.zeros_like(gathered[:, :, 0])
        for slot, projection in enumerate(self.entity_role_projections):
            role = self.entity_role_embedding.weight[slot]
            projected = projection(gathered[:, :, slot] + role)
            combined = combined + projected * options.entity_slot_mask[
                :, :, slot
            ].unsqueeze(-1)
        return combined

    def _compare_valid_options(
        self,
        encoded: Tensor,
        valid_options: Tensor,
        *,
        option_lengths: tuple[int, ...],
    ) -> Tensor:
        """Run one permutation-equivariant block without padded option rows."""
        packed = PackedTokenBatch.from_padded(
            encoded,
            valid_options,
            lengths=option_lengths,
        )
        contextualized = self.comparator(packed).tokens
        output = torch.zeros_like(encoded)
        output[valid_options] = contextualized
        return output


def _validate_option_batch(options: OptionBatch, *, batch_size: int) -> None:
    """Validate only reusable action-schema shape invariants."""
    if options.valid_options.ndim != 2 or options.valid_options.shape[0] != batch_size:
        raise ValueError("valid_options must have shape [batch, options]")
    shape = options.valid_options.shape
    scalar_shape = (*shape, SCALAR_FEATURE_SIZE)
    dynamic_shape = (*shape, DYNAMIC_EFFECT_FEATURE_SIZE)
    if options.scalars.shape != scalar_shape:
        raise ValueError("option scalar features have the wrong schema width")
    if options.dynamic_effect_features.shape != dynamic_shape:
        raise ValueError("option dynamic-effect features have the wrong schema width")
    if options.entity_slots.shape != (*shape, MAX_ENTITY_SLOTS):
        raise ValueError("option entity slots have the wrong schema width")
    if options.entity_slot_mask.shape != options.entity_slots.shape:
        raise ValueError("option entity slot mask differs from slot tensor")
    if options.min_counts.shape != (batch_size,) or options.max_counts.shape != (
        batch_size,
    ):
        raise ValueError("option cardinality bounds must have shape [batch]")
    if options.option_lengths and (
        len(options.option_lengths) != batch_size
        or any(
            length <= 0 or length > int(shape[1]) for length in options.option_lengths
        )
    ):
        raise ValueError("option lengths must align with legal option rows")
    if options.maximum_counts and (
        len(options.maximum_counts) != batch_size
        or any(value < 0 for value in options.maximum_counts)
    ):
        raise ValueError("maximum counts must align with legal option rows")


def _safe_indices(values: Tensor, *, max_known: int, oov_index: int) -> Tensor:
    """Map unknown categorical IDs to the explicit OOV row."""
    return torch.where(
        (values >= 0) & (values < max_known),
        values,
        torch.full_like(values, oov_index),
    )
