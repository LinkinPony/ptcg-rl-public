"""Full-snapshot pooling for temporal STATE tokens."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.model.simple_stateless.backbone import SimpleStatelessBackboneOutput
from ptcg_rl.model.simple_stateless.packed import (
    PackedTokenBatch,
    unpack_after_prefix,
)

varlen_attn: Any
try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - only older unsupported torch builds.
    varlen_attn = None


class TemporalStateEncoder(nn.Module):
    """Read all contextualized snapshot tokens plus semantic direct paths."""

    def __init__(self, *, d_model: int, num_heads: int) -> None:
        """Build one learned full-snapshot attention pool."""
        super().__init__()
        self.d_model = d_model
        self.semantic_projection = nn.Linear(6 * d_model, d_model)
        self.query = nn.Parameter(torch.empty(d_model))
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, state: SimpleStatelessBackboneOutput) -> Tensor:
        """Return one STATE token while retaining policy/value direct paths."""
        scratch = state.scratch.mean(dim=1)
        semantic = preserve_cuda_bfloat16_activation(
            torch.cat(
                (
                    state.public,
                    state.deck,
                    state.opponent_belief,
                    state.policy,
                    state.value,
                    scratch,
                ),
                dim=-1,
            )
        )
        query = preserve_cuda_bfloat16_activation(
            self.query.unsqueeze(0)
        ) + self.semantic_projection(semantic)
        query = preserve_cuda_bfloat16_activation(query)
        if (
            query.is_cuda
            and query.dtype in {torch.bfloat16, torch.float16}
            and varlen_attn is not None
        ):
            attended = _packed_state_attention(
                self.attention,
                query,
                state.packed,
            )
        else:
            tokens, padding_mask = unpack_after_prefix(
                state.packed,
                prefix_tokens=0,
            )
            tokens = preserve_cuda_bfloat16_activation(tokens)
            attended, _weights = self.attention(
                query.unsqueeze(1),
                tokens,
                tokens,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            attended = attended.squeeze(1)
        return preserve_cuda_bfloat16_activation(
            self.output_norm(query + attended)
        )


def _packed_state_attention(
    attention: nn.MultiheadAttention,
    query: Tensor,
    packed: PackedTokenBatch,
) -> Tensor:
    """Pool packed state tokens with one varlen query per sequence."""
    batch_size, d_model = query.shape
    if packed.batch_size != batch_size or packed.d_model != d_model:
        raise ValueError("packed state attention inputs are misaligned")
    in_projection_bias = attention.in_proj_bias
    query_bias: Tensor | None
    key_value_bias: Tensor | None
    if in_projection_bias is None:
        query_bias = key_value_bias = None
    else:
        query_bias = in_projection_bias[:d_model]
        key_value_bias = in_projection_bias[d_model:]
    query_weight = attention.in_proj_weight[:d_model]
    key_value_weight = attention.in_proj_weight[d_model:]
    num_heads = attention.num_heads
    head_dim = d_model // num_heads
    projected_query = functional.linear(
        query,
        query_weight,
        query_bias,
    ).reshape(batch_size, num_heads, head_dim)
    projected_key_value = functional.linear(
        packed.tokens,
        key_value_weight,
        key_value_bias,
    ).reshape(-1, 2, num_heads, head_dim)
    projected_key, projected_value = projected_key_value.unbind(dim=1)
    query_offsets = torch.arange(
        batch_size + 1,
        dtype=torch.int32,
        device=query.device,
    )
    attended = cast(
        Tensor,
        varlen_attn(
            projected_query,
            projected_key,
            projected_value,
            query_offsets,
            packed.cu_seqlens,
            1,
            packed.max_seqlen,
        ),
    )
    return cast(
        Tensor,
        attention.out_proj(attended.reshape(batch_size, d_model)),
    )


__all__ = ["TemporalStateEncoder"]
