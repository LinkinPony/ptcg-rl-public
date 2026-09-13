"""Block-local causal Transformer with full replay and incremental KV paths."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.model.sequence.config import GeneralistSequenceConfig

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

EVENT_TOKEN = 0
STATE_TOKEN = 1
ACTION_TOKEN = 2
TEMPORAL_TOKEN_TYPE_COUNT = 3
_FLEX_ATTENTION_BLOCK_SIZE = 128


@lru_cache(maxsize=1)
def _flex_attention_ops() -> tuple[
    Callable[..., BlockMask],
    Callable[..., Tensor],
]:
    """Load FlexAttention only for CUDA full-sequence replay."""
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    return create_block_mask, cast(
        Callable[..., Tensor],
        torch.compile(flex_attention, dynamic=True),
    )


@dataclass(frozen=True)
class TemporalKvSlotPool:
    """Preallocated fixed-capacity KV storage owned by one sequence actor."""

    keys: Tensor
    values: Tensor
    block_indices: Tensor
    token_positions: Tensor
    capacity_tokens: int

    @classmethod
    def allocate(
        cls,
        config: GeneralistSequenceConfig,
        *,
        slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemporalKvSlotPool:
        """Allocate all layer KV slots once for an actor artifact."""
        if slots <= 0:
            raise ValueError("temporal KV slot capacity must be positive")
        # Keep one provisional block outside the committed context ring.
        # EVENT/STATE/ACTION writes may later be aborted, so they must not
        # overwrite the oldest still-committed block before metadata advances.
        capacity = (
            config.max_context_blocks + 1
        ) * TEMPORAL_TOKEN_TYPE_COUNT
        shape = (
            config.num_layers,
            slots,
            capacity,
            config.attention_heads,
            config.d_model // config.attention_heads,
        )
        return cls(
            keys=torch.zeros(shape, dtype=dtype, device=device),
            values=torch.zeros(shape, dtype=dtype, device=device),
            block_indices=torch.empty(
                (slots, capacity),
                dtype=torch.long,
                device=device,
            ),
            token_positions=torch.empty(
                (slots, capacity),
                dtype=torch.long,
                device=device,
            ),
            capacity_tokens=capacity,
        )

    @property
    def slot_count(self) -> int:
        """Return the number of independently owned sequence slots."""
        return int(self.block_indices.shape[0])

    def empty_cache(self, slot_index: int) -> TemporalKvCache:
        """Create zero-length committed metadata for one allocated slot."""
        if slot_index < 0 or slot_index >= self.slot_count:
            raise ValueError("temporal KV slot index is outside the pool")
        return TemporalKvCache(
            layers=(),
            block_indices_host=(),
            slot_pool=self,
            slot_index=slot_index,
            slot_start=0,
            slot_length=0,
        )


@dataclass(frozen=True)
class _TemporalSlotAppendPlan:
    """Layer-invariant device coordinates for one pooled temporal append."""

    slot_indices: Tensor
    retained_lengths: Tensor
    retained_width: int
    retained_columns: Tensor
    slot_rows: Tensor
    attention_mask: Tensor
    write_columns: Tensor
    write_slots: Tensor


@dataclass(frozen=True)
class TemporalLayerCache:
    """One layer's immutable incremental attention keys and values."""

    keys: Tensor
    values: Tensor
    block_indices: Tensor
    token_positions: Tensor

    def __post_init__(self) -> None:
        """Validate one sequence-local KV cache."""
        if self.keys.ndim != 3 or self.values.shape != self.keys.shape:
            raise ValueError("temporal KV tensors must have shape [tokens, heads, dim]")
        tokens = int(self.keys.shape[0])
        if self.block_indices.shape != (tokens,):
            raise ValueError("temporal KV block indices are misaligned")
        if self.token_positions.shape != (tokens,):
            raise ValueError("temporal KV positions are misaligned")
        if self.keys.device != self.values.device:
            raise ValueError("temporal keys and values use different devices")

    @property
    def token_count(self) -> int:
        """Return retained temporal tokens."""
        return int(self.keys.shape[0])


@dataclass(frozen=True)
class TemporalKvCache:
    """All layer caches for one immutable policy artifact and sequence."""

    layers: tuple[TemporalLayerCache, ...]
    block_indices_host: tuple[int, ...] = ()
    slot_pool: TemporalKvSlotPool | None = None
    slot_index: int | None = None
    slot_start: int = 0
    slot_length: int | None = None

    @property
    def token_count(self) -> int:
        """Return the common retained token count."""
        if self.slot_pool is not None:
            if self.slot_length is None:
                raise RuntimeError("preallocated temporal cache lost its length")
            return self.slot_length
        return 0 if not self.layers else self.layers[0].token_count

    def __post_init__(self) -> None:
        """Require all layers to retain identical token coordinates."""
        if self.slot_pool is not None:
            if self.layers:
                raise ValueError("preallocated temporal cache cannot own row tensors")
            if self.slot_index is None or self.slot_length is None:
                raise ValueError("preallocated temporal cache metadata is incomplete")
            if (
                self.slot_index < 0
                or self.slot_index >= self.slot_pool.slot_count
                or self.slot_start < 0
                or self.slot_start >= self.slot_pool.capacity_tokens
                or self.slot_length < 0
                or self.slot_length > self.slot_pool.capacity_tokens
            ):
                raise ValueError("preallocated temporal cache metadata is invalid")
            if len(self.block_indices_host) != self.slot_length:
                raise ValueError(
                    "preallocated temporal host coordinates are misaligned"
                )
            return
        if (
            self.slot_index is not None
            or self.slot_length is not None
            or self.slot_start != 0
        ):
            raise ValueError("row-local temporal cache has slot metadata")
        if not self.layers:
            if self.block_indices_host:
                raise ValueError("empty temporal cache has block coordinates")
            return
        reference = self.layers[0]
        if len(self.block_indices_host) != reference.token_count:
            raise ValueError("temporal cache host coordinates are misaligned")
        for layer in self.layers[1:]:
            if layer.token_count != reference.token_count:
                raise ValueError("temporal KV layers retain different token counts")
            if (
                layer.block_indices.shape != reference.block_indices.shape
                or layer.token_positions.shape != reference.token_positions.shape
                or layer.block_indices.device != reference.block_indices.device
                or layer.token_positions.device != reference.token_positions.device
            ):
                raise ValueError("temporal KV layer coordinates are misaligned")

    def detach(self) -> TemporalKvCache:
        """Detach actor cache tensors at the train/serve boundary."""
        if self.slot_pool is not None:
            return self
        return TemporalKvCache(
            layers=tuple(
                TemporalLayerCache(
                    keys=layer.keys.detach(),
                    values=layer.values.detach(),
                    block_indices=layer.block_indices,
                    token_positions=layer.token_positions,
                )
                for layer in self.layers
            ),
            block_indices_host=self.block_indices_host,
        )


class TemporalCausalLayer(nn.Module):
    """One pre-LN causal self-attention and feedforward layer."""

    def __init__(self, config: GeneralistSequenceConfig) -> None:
        """Initialize projections for both full and incremental execution."""
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.attention_heads
        self.head_dim = config.d_model // config.attention_heads
        self.max_context_blocks = config.max_context_blocks
        self.residual_scale = 1.0 / math.sqrt(2.0 * config.num_layers)
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attention_output = nn.Linear(config.d_model, config.d_model)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.d_model),
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        valid_mask: Tensor,
        block_indices: Tensor,
        token_positions: Tensor,
        flex_block_mask: BlockMask | None = None,
    ) -> Tensor:
        """Run padded independent sequences with runtime-identical masking."""
        batch_size, width, _ = inputs.shape
        normalized = self.attention_norm(inputs)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch_size, width, self.num_heads, self.head_dim)
        key = key.view(batch_size, width, self.num_heads, self.head_dim)
        value = value.view(batch_size, width, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if flex_block_mask is None:
            allowed = _full_attention_mask(
                valid_mask=valid_mask,
                block_indices=block_indices,
                token_positions=token_positions,
                max_context_blocks=self.max_context_blocks,
            )
            additive = torch.zeros(
                (batch_size, 1, width, width),
                dtype=query.dtype,
                device=query.device,
            ).masked_fill(~allowed.unsqueeze(1), -torch.inf)
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            _unused_create_block_mask, compiled_flex_attention = _flex_attention_ops()
            attended = compiled_flex_attention(
                query,
                key,
                value,
                block_mask=flex_block_mask,
            )
        attended = attended.transpose(1, 2).reshape(batch_size, width, self.d_model)
        attended = self.attention_output(attended)
        attended = attended.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        output = inputs + self.residual_scale * attended
        feedforward = self.feedforward(self.feedforward_norm(output))
        feedforward = feedforward.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return cast(Tensor, output + self.residual_scale * feedforward)

    def append(
        self,
        inputs: Tensor,
        *,
        block_indices: Tensor,
        token_positions: Tensor,
        cache: TemporalLayerCache | None,
    ) -> tuple[Tensor, TemporalLayerCache]:
        """Append one provisional token group to one sequence-local KV cache."""
        if inputs.ndim != 2 or int(inputs.shape[1]) != self.d_model:
            raise ValueError("incremental temporal inputs must have shape [tokens, D]")
        tokens = int(inputs.shape[0])
        if tokens <= 0:
            raise ValueError("incremental temporal append requires tokens")
        if block_indices.shape != (tokens,) or token_positions.shape != (tokens,):
            raise ValueError("incremental temporal coordinates are misaligned")
        normalized = self.attention_norm(inputs)
        query, new_keys, new_values = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(tokens, self.num_heads, self.head_dim)
        new_keys = new_keys.view(tokens, self.num_heads, self.head_dim)
        new_values = new_values.view(tokens, self.num_heads, self.head_dim)
        if cache is None:
            all_keys = new_keys
            all_values = new_values
            all_blocks = block_indices
            all_positions = token_positions
        else:
            all_keys = torch.cat((cache.keys, new_keys), dim=0)
            all_values = torch.cat((cache.values, new_values), dim=0)
            all_blocks = torch.cat((cache.block_indices, block_indices), dim=0)
            all_positions = torch.cat((cache.token_positions, token_positions), dim=0)
        minimum_block = int(block_indices[-1]) - self.max_context_blocks + 1
        retained = all_blocks >= minimum_block
        all_keys = all_keys[retained]
        all_values = all_values[retained]
        all_blocks = all_blocks[retained]
        all_positions = all_positions[retained]
        allowed = (all_positions.unsqueeze(0) <= token_positions.unsqueeze(1)) & (
            all_blocks.unsqueeze(0)
            >= block_indices.unsqueeze(1) - self.max_context_blocks + 1
        )
        additive = torch.zeros(
            (1, self.num_heads, tokens, int(all_keys.shape[0])),
            dtype=query.dtype,
            device=query.device,
        ).masked_fill(~allowed.unsqueeze(0).unsqueeze(0), -torch.inf)
        attended = functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            all_keys.transpose(0, 1).unsqueeze(0),
            all_values.transpose(0, 1).unsqueeze(0),
            attn_mask=additive,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.squeeze(0).transpose(0, 1).reshape(tokens, self.d_model)
        output = inputs + self.residual_scale * self.attention_output(attended)
        output = output + self.residual_scale * self.feedforward(
            self.feedforward_norm(output)
        )
        return (
            output,
            TemporalLayerCache(
                keys=all_keys,
                values=all_values,
                block_indices=all_blocks,
                token_positions=all_positions,
            ),
        )

    def append_many(
        self,
        inputs: Tensor,
        *,
        block_indices: Tensor,
        token_positions: Tensor,
        caches: Sequence[TemporalLayerCache | None],
        retained_starts_host: Sequence[int],
        retained_lengths_host: Sequence[int],
    ) -> tuple[Tensor, tuple[TemporalLayerCache, ...]]:
        """Append equal-width token groups to independent variable KV caches."""
        if inputs.ndim != 3 or int(inputs.shape[2]) != self.d_model:
            raise ValueError(
                "batched incremental temporal inputs must have shape [B, T, D]"
            )
        batch_size, tokens, _ = inputs.shape
        if batch_size <= 0 or tokens <= 0:
            raise ValueError("batched incremental temporal append requires tokens")
        expected_coordinates = (batch_size, tokens)
        if (
            block_indices.shape != expected_coordinates
            or token_positions.shape != expected_coordinates
            or len(caches) != batch_size
            or len(retained_starts_host) != batch_size
            or len(retained_lengths_host) != batch_size
        ):
            raise ValueError("batched incremental temporal coordinates are misaligned")

        normalized = self.attention_norm(inputs)
        query, new_keys, new_values = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        new_keys = new_keys.view(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        new_values = new_values.view(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        )

        empty_keys = new_keys.new_empty((0, self.num_heads, self.head_dim))
        empty_coordinates = block_indices.new_empty((0,))
        old_lengths_host = tuple(
            0 if cache is None else cache.token_count for cache in caches
        )
        if any(
            start < 0 or length < 0 or start + length != old_length
            for start, length, old_length in zip(
                retained_starts_host,
                retained_lengths_host,
                old_lengths_host,
                strict=True,
            )
        ):
            raise ValueError("batched temporal retention plan is invalid")
        old_keys: list[Tensor] = []
        old_values: list[Tensor] = []
        old_blocks: list[Tensor] = []
        old_positions: list[Tensor] = []
        for cache in caches:
            if cache is None:
                old_keys.append(empty_keys)
                old_values.append(empty_keys)
                old_blocks.append(empty_coordinates)
                old_positions.append(empty_coordinates)
            else:
                old_keys.append(cache.keys)
                old_values.append(cache.values)
                old_blocks.append(cache.block_indices)
                old_positions.append(cache.token_positions)
        padded_old_keys = pad_sequence(
            old_keys,
            batch_first=True,
        )
        padded_old_values = pad_sequence(
            old_values,
            batch_first=True,
        )
        padded_old_blocks = pad_sequence(
            old_blocks,
            batch_first=True,
        )
        padded_old_positions = pad_sequence(
            old_positions,
            batch_first=True,
        )
        old_width = int(padded_old_keys.shape[1])
        if old_width:
            retained_width = max(retained_lengths_host)
            retained_starts = torch.tensor(
                retained_starts_host,
                dtype=torch.long,
                device=inputs.device,
            )
            retained_columns = (
                retained_starts.unsqueeze(1)
                + torch.arange(retained_width, device=inputs.device).unsqueeze(0)
            ).clamp(max=old_width - 1)
            key_indices = (
                retained_columns.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(
                    -1,
                    -1,
                    self.num_heads,
                    self.head_dim,
                )
            )
            retained_keys = padded_old_keys.gather(1, key_indices)
            retained_values = padded_old_values.gather(1, key_indices)
            retained_blocks = padded_old_blocks.gather(1, retained_columns)
            retained_positions = padded_old_positions.gather(
                1,
                retained_columns,
            )
        else:
            retained_width = 0
            retained_keys = padded_old_keys
            retained_values = padded_old_values
            retained_blocks = padded_old_blocks
            retained_positions = padded_old_positions

        lengths_host = tuple(length + tokens for length in retained_lengths_host)
        retained_lengths = torch.tensor(
            retained_lengths_host,
            dtype=torch.long,
            device=inputs.device,
        )
        lengths = torch.tensor(
            lengths_host,
            dtype=torch.long,
            device=inputs.device,
        )
        key_width = retained_width + tokens
        padded_keys = new_keys.new_zeros(
            (batch_size, key_width, self.num_heads, self.head_dim)
        )
        padded_values = new_values.new_zeros(
            (batch_size, key_width, self.num_heads, self.head_dim)
        )
        padded_blocks = block_indices.new_zeros((batch_size, key_width))
        padded_positions = token_positions.new_zeros((batch_size, key_width))
        if retained_width:
            padded_keys[:, :retained_width] = retained_keys
            padded_values[:, :retained_width] = retained_values
            padded_blocks[:, :retained_width] = retained_blocks
            padded_positions[:, :retained_width] = retained_positions
        new_columns = retained_lengths.unsqueeze(1) + torch.arange(
            tokens,
            device=inputs.device,
        ).unsqueeze(0)
        batch_rows = (
            torch.arange(
                batch_size,
                device=inputs.device,
            )
            .unsqueeze(1)
            .expand(-1, tokens)
        )
        padded_keys[batch_rows, new_columns] = new_keys
        padded_values[batch_rows, new_columns] = new_values
        padded_blocks[batch_rows, new_columns] = block_indices
        padded_positions[batch_rows, new_columns] = token_positions
        valid_keys = torch.arange(key_width, device=inputs.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        allowed = (
            valid_keys.unsqueeze(1)
            & (padded_positions.unsqueeze(1) <= token_positions.unsqueeze(2))
            & (
                padded_blocks.unsqueeze(1)
                >= block_indices.unsqueeze(2) - self.max_context_blocks + 1
            )
        )
        additive = torch.zeros(
            (batch_size, 1, tokens, key_width),
            dtype=query.dtype,
            device=query.device,
        ).masked_fill(~allowed.unsqueeze(1), -torch.inf)
        attended = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            padded_keys.transpose(1, 2),
            padded_values.transpose(1, 2),
            attn_mask=additive,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            tokens,
            self.d_model,
        )
        output = inputs + self.residual_scale * self.attention_output(attended)
        output = output + self.residual_scale * self.feedforward(
            self.feedforward_norm(output)
        )
        return (
            output,
            tuple(
                TemporalLayerCache(
                    keys=padded_keys[row, :length].clone(),
                    values=padded_values[row, :length].clone(),
                    block_indices=padded_blocks[row, :length].clone(),
                    token_positions=padded_positions[row, :length].clone(),
                )
                for row, length in enumerate(lengths_host)
            ),
        )

    def append_many_slots(
        self,
        inputs: Tensor,
        *,
        plan: _TemporalSlotAppendPlan,
        retained_keys: Tensor,
        retained_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append a batch into fixed slots without per-row pad or cache clones."""
        batch_size, tokens, _ = inputs.shape
        if (
            plan.slot_indices.shape != (batch_size,)
            or plan.retained_lengths.shape != (batch_size,)
            or plan.write_columns.shape != (batch_size, tokens)
        ):
            raise ValueError("preallocated temporal slot rows are misaligned")
        if (
            plan.slot_indices.dtype != torch.long
            or plan.retained_lengths.dtype != torch.long
            or plan.slot_indices.device != inputs.device
            or plan.attention_mask.device != inputs.device
        ):
            raise ValueError("preallocated temporal slot plan is invalid")
        if plan.retained_width < 0:
            raise ValueError("preallocated temporal retained width is invalid")
        expected_retained = (
            batch_size,
            plan.retained_width,
            self.num_heads,
            self.head_dim,
        )
        if (
            retained_keys.shape != expected_retained
            or retained_values.shape != expected_retained
            or retained_keys.device != inputs.device
            or retained_values.device != inputs.device
        ):
            raise ValueError("preallocated temporal retained rows are misaligned")
        normalized = self.attention_norm(inputs)
        query, new_keys, new_values = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch_size, tokens, self.num_heads, self.head_dim)
        new_keys = new_keys.view(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        new_values = new_values.view(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        padded_keys = torch.cat((retained_keys, new_keys), dim=1)
        padded_values = torch.cat((retained_values, new_values), dim=1)
        attended = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            padded_keys.transpose(1, 2),
            padded_values.transpose(1, 2),
            attn_mask=plan.attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            tokens,
            self.d_model,
        )
        output = inputs + self.residual_scale * self.attention_output(attended)
        output = output + self.residual_scale * self.feedforward(
            self.feedforward_norm(output)
        )

        return cast(Tensor, output), new_keys, new_values


def _prepare_temporal_slot_append_plan(
    *,
    block_indices: Tensor,
    token_positions: Tensor,
    pool: TemporalKvSlotPool,
    slot_indices_host: Sequence[int],
    retained_slot_starts_host: Sequence[int],
    retained_lengths_host: Sequence[int],
    max_context_blocks: int,
    dtype: torch.dtype,
) -> _TemporalSlotAppendPlan:
    """Build layer-invariant pooled KV coordinates and the attention mask once."""
    batch_size, tokens = block_indices.shape
    slot_indices = torch.tensor(
        slot_indices_host,
        dtype=torch.long,
        device=block_indices.device,
    )
    retained_slot_starts = torch.tensor(
        retained_slot_starts_host,
        dtype=torch.long,
        device=block_indices.device,
    )
    retained_lengths = torch.tensor(
        retained_lengths_host,
        dtype=torch.long,
        device=block_indices.device,
    )
    retained_width = max(retained_lengths_host, default=0)
    retained_offsets = torch.arange(
        retained_width,
        device=block_indices.device,
    ).unsqueeze(0)
    retained_columns = (
        retained_slot_starts.unsqueeze(1) + retained_offsets
    ) % pool.capacity_tokens
    slot_rows = slot_indices.unsqueeze(1).expand(-1, retained_width)
    retained_valid = retained_offsets < retained_lengths.unsqueeze(1)
    retained_blocks = pool.block_indices[slot_rows, retained_columns]
    retained_positions = pool.token_positions[slot_rows, retained_columns]

    token_offsets = torch.arange(tokens, device=block_indices.device).unsqueeze(0)
    retained_allowed = (
        retained_valid.unsqueeze(1)
        & (retained_positions.unsqueeze(1) <= token_positions.unsqueeze(2))
        & (
            retained_blocks.unsqueeze(1)
            >= block_indices.unsqueeze(2) - max_context_blocks + 1
        )
    )
    new_allowed = (
        token_positions.unsqueeze(1) <= token_positions.unsqueeze(2)
    ) & (
        block_indices.unsqueeze(1)
        >= block_indices.unsqueeze(2) - max_context_blocks + 1
    )
    allowed = torch.cat((retained_allowed, new_allowed), dim=2)
    key_width = retained_width + tokens
    attention_mask = torch.zeros(
        (batch_size, 1, tokens, key_width),
        dtype=dtype,
        device=block_indices.device,
    ).masked_fill(~allowed.unsqueeze(1), -torch.inf)
    write_columns = (
        retained_slot_starts.unsqueeze(1) + retained_lengths.unsqueeze(1)
        + token_offsets
    ) % pool.capacity_tokens
    return _TemporalSlotAppendPlan(
        slot_indices=slot_indices,
        retained_lengths=retained_lengths,
        retained_width=retained_width,
        retained_columns=retained_columns,
        slot_rows=slot_rows,
        attention_mask=attention_mask,
        write_columns=write_columns,
        write_slots=slot_indices.unsqueeze(1).expand(-1, tokens),
    )


class GeneralistTemporalCore(nn.Module):
    """Deterministic temporal Transformer with exact raw-replay semantics."""

    def __init__(self, config: GeneralistSequenceConfig) -> None:
        """Build token-type identity and causal layers."""
        super().__init__()
        self.config = config
        self.token_type_embedding = nn.Embedding(
            TEMPORAL_TOKEN_TYPE_COUNT,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            TemporalCausalLayer(config) for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        tokens: Tensor,
        *,
        token_types: Tensor,
        block_indices: Tensor,
        sequence_offsets: tuple[int, ...],
    ) -> Tensor:
        """Replay packed complete or provisional sequences under a causal mask."""
        _validate_packed_coordinates(
            tokens,
            token_types=token_types,
            block_indices=block_indices,
            sequence_offsets=sequence_offsets,
            d_model=self.config.d_model,
        )
        positions = block_indices * TEMPORAL_TOKEN_TYPE_COUNT + token_types
        padded, types, blocks, padded_positions, valid = _pad_sequences(
            tokens,
            token_types,
            block_indices,
            positions,
            sequence_offsets=sequence_offsets,
        )
        padded = preserve_cuda_bfloat16_activation(
            padded
            + self.token_type_embedding(types)
            + _sinusoidal_positions(
                padded_positions,
                d_model=self.config.d_model,
                dtype=padded.dtype,
            )
        )
        flex_block_mask = (
            _full_attention_block_mask(
                valid_mask=valid,
                block_indices=blocks,
                token_positions=padded_positions,
                max_context_blocks=self.config.max_context_blocks,
            )
            if padded.is_cuda
            else None
        )
        for layer in self.layers:
            padded = layer(
                padded,
                valid_mask=valid,
                block_indices=blocks,
                token_positions=padded_positions,
                flex_block_mask=flex_block_mask,
            )
        padded = preserve_cuda_bfloat16_activation(self.output_norm(padded))
        return torch.cat(
            tuple(
                padded[row, : end - start]
                for row, (start, end) in enumerate(pairwise(sequence_offsets))
            ),
            dim=0,
        )

    def append(
        self,
        tokens: Tensor,
        *,
        token_types: Tensor,
        block_indices: Tensor,
        cache: TemporalKvCache | None,
    ) -> tuple[Tensor, TemporalKvCache]:
        """Incrementally append EVENT/STATE or ACTION tokens to one cache."""
        if tokens.ndim != 2 or int(tokens.shape[1]) != self.config.d_model:
            raise ValueError("incremental temporal tokens must have shape [tokens, D]")
        if token_types.shape != (int(tokens.shape[0]),):
            raise ValueError("incremental temporal token types are misaligned")
        if block_indices.shape != token_types.shape:
            raise ValueError("incremental temporal block indices are misaligned")
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError("temporal KV cache layer count differs from the model")
        positions = block_indices * TEMPORAL_TOKEN_TYPE_COUNT + token_types
        hidden = preserve_cuda_bfloat16_activation(
            tokens
            + self.token_type_embedding(token_types)
            + _sinusoidal_positions(
                positions,
                d_model=self.config.d_model,
                dtype=tokens.dtype,
            )
        )
        layers: list[TemporalLayerCache] = []
        for index, untyped_layer in enumerate(self.layers):
            layer = cast(TemporalCausalLayer, untyped_layer)
            hidden, layer_cache = layer.append(
                hidden,
                block_indices=block_indices,
                token_positions=positions,
                cache=None if cache is None else cache.layers[index],
            )
            layers.append(layer_cache)
        cache_blocks = tuple(
            int(value) for value in layers[0].block_indices.detach().cpu().tolist()
        )
        return preserve_cuda_bfloat16_activation(
            cast(Tensor, self.output_norm(hidden))
        ), TemporalKvCache(
            tuple(layers),
            block_indices_host=cache_blocks,
        )

    def append_many(
        self,
        tokens: Tensor,
        *,
        token_types: Tensor,
        block_indices: Tensor,
        caches: Sequence[TemporalKvCache | None],
        block_indices_host: Sequence[Sequence[int]] | None = None,
    ) -> tuple[Tensor, tuple[TemporalKvCache, ...]]:
        """Incrementally append one equal-width group to many independent caches."""
        if tokens.ndim != 3 or int(tokens.shape[2]) != self.config.d_model:
            raise ValueError(
                "batched incremental temporal tokens must have shape [B, T, D]"
            )
        batch_size, width, _ = tokens.shape
        expected_coordinates = (batch_size, width)
        if (
            token_types.shape != expected_coordinates
            or block_indices.shape != expected_coordinates
            or len(caches) != batch_size
        ):
            raise ValueError("batched incremental temporal coordinates are misaligned")
        if any(
            cache is not None
            and cache.slot_pool is None
            and len(cache.layers) != len(self.layers)
            for cache in caches
        ):
            raise ValueError("batched temporal KV cache layer count differs")
        if block_indices_host is None:
            block_indices_host = tuple(
                tuple(int(value) for value in row)
                for row in block_indices.detach().cpu().tolist()
            )
        new_blocks_host = tuple(
            tuple(int(value) for value in row) for row in block_indices_host
        )
        if (
            len(new_blocks_host) != batch_size
            or any(len(row) != width for row in new_blocks_host)
            or any(not row for row in new_blocks_host)
        ):
            raise ValueError("batched temporal host coordinates are misaligned")
        old_blocks_host = tuple(
            () if cache is None else cache.block_indices_host for cache in caches
        )
        if any(
            cache is not None and len(blocks) != cache.token_count
            for cache, blocks in zip(caches, old_blocks_host, strict=True)
        ):
            raise ValueError("batched temporal cache lost host coordinates")
        retained_starts_host: list[int] = []
        retained_lengths_host: list[int] = []
        output_blocks_host: list[tuple[int, ...]] = []
        for old_blocks, new_blocks in zip(
            old_blocks_host,
            new_blocks_host,
            strict=True,
        ):
            minimum_block = new_blocks[-1] - self.config.max_context_blocks + 1
            retained_start = next(
                (
                    index
                    for index, old_block in enumerate(old_blocks)
                    if old_block >= minimum_block
                ),
                len(old_blocks),
            )
            retained_starts_host.append(retained_start)
            retained_lengths_host.append(len(old_blocks) - retained_start)
            output_blocks_host.append(old_blocks[retained_start:] + new_blocks)
        pooled = tuple(
            cache is not None and cache.slot_pool is not None for cache in caches
        )
        if any(pooled) and not all(pooled):
            raise ValueError("batched temporal caches mix slot and row storage")
        slot_pool = cast(TemporalKvCache, caches[0]).slot_pool if all(pooled) else None
        if slot_pool is not None and any(
            cast(TemporalKvCache, cache).slot_pool is not slot_pool for cache in caches
        ):
            raise ValueError("batched temporal caches cross preallocated pools")
        slot_indices_host: tuple[int, ...] = ()
        retained_slot_starts_host: tuple[int, ...] = ()
        if slot_pool is not None:
            slot_indices_host = tuple(
                cast(int, cast(TemporalKvCache, cache).slot_index) for cache in caches
            )
            retained_slot_starts_host = tuple(
                (cast(TemporalKvCache, cache).slot_start + retained_start)
                % slot_pool.capacity_tokens
                for cache, retained_start in zip(
                    caches,
                    retained_starts_host,
                    strict=True,
                )
            )
            if len(set(slot_indices_host)) != batch_size:
                raise ValueError("batched temporal rows reuse one preallocated slot")
            if any(
                len(blocks) > slot_pool.capacity_tokens for blocks in output_blocks_host
            ):
                raise RuntimeError("temporal append exceeded slot capacity")
        positions = block_indices * TEMPORAL_TOKEN_TYPE_COUNT + token_types
        hidden = preserve_cuda_bfloat16_activation(
            tokens
            + self.token_type_embedding(token_types)
            + _sinusoidal_positions(
                positions,
                d_model=self.config.d_model,
                dtype=tokens.dtype,
            )
        )
        row_layers: list[list[TemporalLayerCache]] = [[] for _ in range(batch_size)]
        slot_plan = (
            None
            if slot_pool is None
            else _prepare_temporal_slot_append_plan(
                block_indices=block_indices,
                token_positions=positions,
                pool=slot_pool,
                slot_indices_host=slot_indices_host,
                retained_slot_starts_host=retained_slot_starts_host,
                retained_lengths_host=retained_lengths_host,
                max_context_blocks=self.config.max_context_blocks,
                dtype=hidden.dtype,
            )
        )
        retained_slot_keys: Tensor | None = None
        retained_slot_values: Tensor | None = None
        new_slot_keys: list[Tensor] = []
        new_slot_values: list[Tensor] = []
        if slot_pool is not None:
            if slot_plan is None:
                raise RuntimeError("preallocated temporal slot plan is absent")
            if slot_plan.retained_width:
                retained_slot_keys = slot_pool.keys[
                    :, slot_plan.slot_rows, slot_plan.retained_columns
                ]
                retained_slot_values = slot_pool.values[
                    :, slot_plan.slot_rows, slot_plan.retained_columns
                ]
            else:
                retained_shape = (
                    len(self.layers),
                    batch_size,
                    0,
                    self.config.attention_heads,
                    self.config.d_model // self.config.attention_heads,
                )
                retained_slot_keys = hidden.new_empty(retained_shape)
                retained_slot_values = hidden.new_empty(retained_shape)
        for layer_index, untyped_layer in enumerate(self.layers):
            layer = cast(TemporalCausalLayer, untyped_layer)
            if slot_pool is not None:
                if (
                    slot_plan is None
                    or retained_slot_keys is None
                    or retained_slot_values is None
                ):
                    raise RuntimeError("preallocated temporal slot plan is absent")
                hidden, layer_keys, layer_values = layer.append_many_slots(
                    hidden,
                    plan=slot_plan,
                    retained_keys=retained_slot_keys[layer_index],
                    retained_values=retained_slot_values[layer_index],
                )
                new_slot_keys.append(layer_keys)
                new_slot_values.append(layer_values)
                continue
            hidden, layer_caches = layer.append_many(
                hidden,
                block_indices=block_indices,
                token_positions=positions,
                caches=tuple(
                    None if cache is None else cache.layers[layer_index]
                    for cache in caches
                ),
                retained_starts_host=retained_starts_host,
                retained_lengths_host=retained_lengths_host,
            )
            for row, layer_cache in enumerate(layer_caches):
                row_layers[row].append(layer_cache)
        if slot_pool is not None:
            if slot_plan is None:
                raise RuntimeError("preallocated temporal slot plan is absent")
            slot_pool.keys[:, slot_plan.write_slots, slot_plan.write_columns] = (
                torch.stack(new_slot_keys)
            )
            slot_pool.values[:, slot_plan.write_slots, slot_plan.write_columns] = (
                torch.stack(new_slot_values)
            )
            slot_pool.block_indices[
                slot_plan.write_slots, slot_plan.write_columns
            ] = block_indices
            slot_pool.token_positions[
                slot_plan.write_slots, slot_plan.write_columns
            ] = positions
            return (
                preserve_cuda_bfloat16_activation(
                    cast(Tensor, self.output_norm(hidden))
                ),
                tuple(
                    TemporalKvCache(
                        layers=(),
                        block_indices_host=output_blocks_host[row],
                        slot_pool=slot_pool,
                        slot_index=slot_indices_host[row],
                        slot_start=retained_slot_starts_host[row],
                        slot_length=len(output_blocks_host[row]),
                    )
                    for row in range(batch_size)
                ),
            )
        return (
            preserve_cuda_bfloat16_activation(cast(Tensor, self.output_norm(hidden))),
            tuple(
                TemporalKvCache(
                    tuple(layers),
                    block_indices_host=output_blocks_host[row],
                )
                for row, layers in enumerate(row_layers)
            ),
        )


def _full_attention_mask(
    *,
    valid_mask: Tensor,
    block_indices: Tensor,
    token_positions: Tensor,
    max_context_blocks: int,
) -> Tensor:
    query_positions = token_positions.unsqueeze(2)
    key_positions = token_positions.unsqueeze(1)
    query_blocks = block_indices.unsqueeze(2)
    key_blocks = block_indices.unsqueeze(1)
    allowed = (
        (key_positions <= query_positions)
        & (key_blocks >= query_blocks - max_context_blocks + 1)
        & valid_mask.unsqueeze(1)
        & valid_mask.unsqueeze(2)
    )
    # Invalid padded queries need one finite attention cell to keep SDPA finite;
    # their outputs are masked immediately after attention.
    invalid_queries = ~valid_mask
    if bool(invalid_queries.any()):
        allowed = allowed.clone()
        rows, queries = torch.nonzero(invalid_queries, as_tuple=True)
        allowed[rows, queries, 0] = True
    return allowed


def _full_attention_block_mask(
    *,
    valid_mask: Tensor,
    block_indices: Tensor,
    token_positions: Tensor,
    max_context_blocks: int,
) -> BlockMask:
    """Build the exact causal sliding mask without a dense score tensor."""
    if not valid_mask.is_cuda:
        raise ValueError("flex temporal attention requires CUDA coordinates")
    batch_size, width = valid_mask.shape
    create_block_mask, _unused_flex_attention = _flex_attention_ops()

    def mask_mod(
        batch: Tensor,
        _head: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tensor:
        query_valid = valid_mask[batch, query]
        allowed = (
            query_valid
            & valid_mask[batch, key]
            & (token_positions[batch, key] <= token_positions[batch, query])
            & (
                block_indices[batch, key]
                >= block_indices[batch, query] - max_context_blocks + 1
            )
        )
        return allowed | (~query_valid & (key == 0))

    return create_block_mask(
        mask_mod,
        batch_size,
        None,
        width,
        width,
        device=valid_mask.device,
        BLOCK_SIZE=_FLEX_ATTENTION_BLOCK_SIZE,
        _compile=True,
    )


def _pad_sequences(
    tokens: Tensor,
    token_types: Tensor,
    block_indices: Tensor,
    positions: Tensor,
    *,
    sequence_offsets: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    lengths = tuple(end - start for start, end in pairwise(sequence_offsets))
    batch_size = len(lengths)
    width = max(lengths)
    padded = tokens.new_zeros((batch_size, width, tokens.shape[1]))
    types = torch.zeros(
        (batch_size, width),
        dtype=torch.long,
        device=tokens.device,
    )
    blocks = torch.zeros_like(types)
    padded_positions = torch.zeros_like(types)
    valid = torch.zeros(
        (batch_size, width),
        dtype=torch.bool,
        device=tokens.device,
    )
    for row, (start, end) in enumerate(pairwise(sequence_offsets)):
        length = end - start
        padded[row, :length] = tokens[start:end]
        types[row, :length] = token_types[start:end]
        blocks[row, :length] = block_indices[start:end]
        padded_positions[row, :length] = positions[start:end]
        valid[row, :length] = True
    return padded, types, blocks, padded_positions, valid


def _sinusoidal_positions(
    positions: Tensor,
    *,
    d_model: int,
    dtype: torch.dtype,
) -> Tensor:
    if positions.dtype == torch.bool or positions.is_floating_point():
        raise TypeError("temporal positions must use an integer dtype")
    frequency_indices = torch.arange(
        0,
        d_model,
        2,
        dtype=torch.float32,
        device=positions.device,
    )
    frequencies = torch.exp(frequency_indices * (-math.log(10_000.0) / d_model))
    angles = positions.to(dtype=torch.float32).unsqueeze(-1) * frequencies
    output = torch.zeros(
        (*positions.shape, d_model),
        dtype=torch.float32,
        device=positions.device,
    )
    output[..., 0::2] = torch.sin(angles)
    if d_model > 1:
        output[..., 1::2] = torch.cos(angles[..., : output[..., 1::2].shape[-1]])
    return output.to(dtype=dtype)


def _validate_packed_coordinates(
    tokens: Tensor,
    *,
    token_types: Tensor,
    block_indices: Tensor,
    sequence_offsets: tuple[int, ...],
    d_model: int,
) -> None:
    if tokens.ndim != 2 or int(tokens.shape[1]) != d_model:
        raise ValueError("packed temporal tokens must have shape [tokens, D]")
    count = int(tokens.shape[0])
    if token_types.shape != (count,) or block_indices.shape != (count,):
        raise ValueError("packed temporal coordinates are misaligned")
    if (
        len(sequence_offsets) < 2
        or sequence_offsets[0] != 0
        or sequence_offsets[-1] != count
        or any(left >= right for left, right in pairwise(sequence_offsets))
    ):
        raise ValueError("temporal sequence offsets must partition all tokens")
    if bool(((token_types < 0) | (token_types >= TEMPORAL_TOKEN_TYPE_COUNT)).any()):
        raise ValueError("temporal token type is invalid")
    for start, end in pairwise(sequence_offsets):
        positions = (
            block_indices[start:end] * TEMPORAL_TOKEN_TYPE_COUNT
            + token_types[start:end]
        )
        if bool((positions[1:] <= positions[:-1]).any()):
            raise ValueError("temporal token clock must be strictly increasing")


__all__ = [
    "ACTION_TOKEN",
    "EVENT_TOKEN",
    "STATE_TOKEN",
    "GeneralistTemporalCore",
    "TemporalKvCache",
    "TemporalKvSlotPool",
    "TemporalLayerCache",
]
