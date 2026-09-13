"""Raw public-state projections for the clean stateless Transformer."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.model.policy import DEFAULT_MAX_ATTACK_ID
from ptcg_rl.model.simple_stateless.packed import PackedTokenBatch
from ptcg_rl.model.state_encoder import (
    AREA_EMBEDDING_COUNT,
    ATTACHMENT_KIND_COUNT,
    ENTITY_SLOT_COUNT,
    GLOBAL_CONTEXT_SCALAR_SIZE,
    LEGACY_TOKEN_SCALAR_SIZE,
    OWNER_ROLE_COUNT,
    PUBLIC_STATE_SCALAR_SIZE,
    PUBLIC_STATE_SCALAR_START,
    TOKEN_KIND_COUNT,
    TOKEN_KIND_TO_INDEX,
    TOKEN_SCALAR_SIZE,
    StateBatch,
)
from ptcg_rl.model.tensor_validation import require_tensor_condition


class SimpleStatelessRawStateEncoder(nn.Module):
    """Project public observation arrays without owning a second CardEncoder."""

    def __init__(self, *, d_model: int) -> None:
        """Initialize categorical, scalar, attack, and attachment projections."""
        super().__init__()
        self.d_model = d_model
        self.area_embedding = nn.Embedding(AREA_EMBEDDING_COUNT, d_model)
        self.owner_embedding = nn.Embedding(OWNER_ROLE_COUNT, d_model)
        self.kind_embedding = nn.Embedding(TOKEN_KIND_COUNT, d_model)
        self.entity_slot_embedding = nn.Embedding(ENTITY_SLOT_COUNT, d_model)
        self.attack_embedding = nn.Embedding(DEFAULT_MAX_ATTACK_ID + 1, d_model)
        self.scalar_projection = nn.Sequential(
            nn.Linear(LEGACY_TOKEN_SCALAR_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.global_context_projection = nn.Linear(
            GLOBAL_CONTEXT_SCALAR_SIZE,
            d_model,
            bias=False,
        )
        self.public_state_projection = nn.Linear(
            PUBLIC_STATE_SCALAR_SIZE,
            d_model,
            bias=False,
        )
        self.attachment_kind_gates = nn.Parameter(
            torch.empty(ATTACHMENT_KIND_COUNT, d_model)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.reset_non_linear_parameters()

    def reset_non_linear_parameters(self) -> None:
        """Initialize embeddings and multiplicative attachment gates."""
        for embedding in (
            self.area_embedding,
            self.owner_embedding,
            self.kind_embedding,
            self.entity_slot_embedding,
            self.attack_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.entity_slot_embedding.weight)
        nn.init.zeros_(self.attachment_kind_gates)

    def forward(
        self,
        batch: StateBatch,
        *,
        card_encoder: CardEncoder,
    ) -> tuple[Tensor, PackedTokenBatch]:
        """Return public/global rows and packed public entity representations."""
        _validate_state_batch(batch)
        card_embeddings = card_encoder(batch.card_ids)
        attack_ids = batch.last_attack_ids.clamp(
            min=0,
            max=DEFAULT_MAX_ATTACK_ID,
        )
        entity_slots = cast(Tensor, batch.entity_slots).to(dtype=torch.long)
        shared_scalars, global_scalars, public_scalars = _split_scalars(batch)
        token_inputs = (
            card_embeddings
            + self.area_embedding(batch.areas)
            + self.owner_embedding(batch.owner_roles)
            + self.kind_embedding(batch.token_kinds)
            + self.entity_slot_embedding(entity_slots)
            + self.attack_embedding(attack_ids)
            + self.scalar_projection(shared_scalars.to(dtype=card_embeddings.dtype))
            + self.global_context_projection(
                global_scalars.to(dtype=card_embeddings.dtype)
            )
            + self.public_state_projection(
                public_scalars.to(dtype=card_embeddings.dtype)
            )
        )
        token_inputs = token_inputs + _attachment_embeddings(
            batch,
            card_encoder=card_encoder,
            gates=self.attachment_kind_gates,
            dtype=card_embeddings.dtype,
        )
        token_inputs = preserve_cuda_bfloat16_activation(self.output_norm(token_inputs))
        return (
            token_inputs[:, 0],
            PackedTokenBatch.from_padded(
                token_inputs,
                ~batch.padding_mask,
                lengths=batch.sequence_lengths,
            ),
        )


def _validate_state_batch(batch: StateBatch) -> None:
    shape = batch.card_ids.shape
    if batch.card_ids.ndim != 2 or int(shape[1]) <= 0:
        raise ValueError("state card IDs must have shape [batch, tokens]")
    for name, value in (
        ("areas", batch.areas),
        ("owner_roles", batch.owner_roles),
        ("token_kinds", batch.token_kinds),
        ("last_attack_ids", batch.last_attack_ids),
        ("padding_mask", batch.padding_mask),
    ):
        if value.shape != shape:
            raise ValueError(f"state {name} must align with card IDs")
    if batch.scalars.shape[:2] != shape:
        raise ValueError("state scalars must align with card IDs")
    if batch.scalars.shape[-1] != TOKEN_SCALAR_SIZE:
        raise ValueError("state scalar width differs from the public input schema")
    if batch.padding_mask.dtype != torch.bool:
        raise ValueError("state padding mask must be boolean")
    require_tensor_condition(
        ~batch.padding_mask[:, 0].any(),
        "every state row must begin with a public/global token",
    )
    if batch.sequence_lengths and len(batch.sequence_lengths) != int(shape[0]):
        raise ValueError("state sequence lengths must align with state rows")
    if batch.entity_slots is None or batch.entity_slots.shape != shape:
        raise ValueError("state entity slots must align with card IDs")
    for name, attachment_value in (
        ("attachment_card_ids", batch.attachment_card_ids),
        ("attachment_parent_indices", batch.attachment_parent_indices),
        ("attachment_kinds", batch.attachment_kinds),
    ):
        if (
            attachment_value is None
            or attachment_value.ndim != 2
            or attachment_value.shape[0] != shape[0]
        ):
            raise ValueError(f"state {name} must have shape [batch, attachments]")


def _split_scalars(batch: StateBatch) -> tuple[Tensor, Tensor, Tensor]:
    context = batch.scalars[..., :GLOBAL_CONTEXT_SCALAR_SIZE]
    global_rows = batch.token_kinds.eq(TOKEN_KIND_TO_INDEX["global"]).unsqueeze(-1)
    zero_context = torch.zeros_like(context)
    shared = torch.cat(
        (
            torch.where(global_rows, zero_context, context),
            batch.scalars[
                ...,
                GLOBAL_CONTEXT_SCALAR_SIZE:LEGACY_TOKEN_SCALAR_SIZE,
            ],
        ),
        dim=-1,
    )
    public = batch.scalars[
        ...,
        PUBLIC_STATE_SCALAR_START : (
            PUBLIC_STATE_SCALAR_START + PUBLIC_STATE_SCALAR_SIZE
        ),
    ]
    return (
        shared,
        torch.where(global_rows, context, zero_context),
        torch.where(global_rows, public, torch.zeros_like(public)),
    )


def _attachment_embeddings(
    batch: StateBatch,
    *,
    card_encoder: CardEncoder,
    gates: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    card_ids = cast(Tensor, batch.attachment_card_ids).long()
    parent_indices = cast(Tensor, batch.attachment_parent_indices).long()
    kinds = cast(Tensor, batch.attachment_kinds).long()
    batch_size, token_count = batch.card_ids.shape
    encoded = card_encoder(card_ids)
    valid = (
        card_ids.gt(0)
        & parent_indices.ge(0)
        & parent_indices.lt(token_count)
        & kinds.ge(1)
        & kinds.le(ATTACHMENT_KIND_COUNT)
    )
    safe_kinds = (kinds - 1).clamp(min=0, max=ATTACHMENT_KIND_COUNT - 1)
    values = encoded * gates[safe_kinds].to(dtype=dtype)
    values = values * valid.unsqueeze(-1)
    output = encoded.new_zeros((batch_size, token_count, encoded.shape[-1]))
    output.scatter_add_(
        1,
        parent_indices.clamp(min=0, max=token_count - 1)
        .unsqueeze(-1)
        .expand_as(values),
        values,
    )
    return cast(Tensor, output)
