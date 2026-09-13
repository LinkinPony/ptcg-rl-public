"""Packed variable-length token batches and fixed special-token layout."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedTokenBatch:
    """Concatenated non-empty token sequences with FlashAttention offsets."""

    tokens: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate packed storage without synchronizing token values."""
        if self.tokens.ndim != 2:
            raise ValueError("packed tokens must have shape [total_tokens, d_model]")
        if self.cu_seqlens.ndim != 1 or self.cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must have shape [batch_size + 1]")
        if self.cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must use int32 FlashAttention offsets")
        if self.cu_seqlens.device != self.tokens.device:
            raise ValueError("packed tokens and offsets must use the same device")
        if self.max_seqlen <= 0:
            raise ValueError("max_seqlen must be positive")
        if len(self.offsets) != self.cu_seqlens.numel():
            raise ValueError("packed offsets differ from cu_seqlens width")
        if self.offsets[0] != 0 or self.offsets[-1] != self.tokens.shape[0]:
            raise ValueError("offsets do not cover the packed token storage")
        lengths = [end - start for start, end in pairwise(self.offsets)]
        if any(length <= 0 for length in lengths):
            raise ValueError("packed sequences must be non-empty")
        if max(lengths) != self.max_seqlen:
            raise ValueError("max_seqlen does not match packed sequence lengths")
        if (
            not self.cu_seqlens.is_meta
            and not self.cu_seqlens.is_cuda
            and tuple(self.cu_seqlens.tolist()) != self.offsets
        ):
            raise ValueError("cu_seqlens values differ from packed offsets")

    @property
    def batch_size(self) -> int:
        """Return the number of packed sequences."""
        return int(self.cu_seqlens.numel()) - 1

    @property
    def d_model(self) -> int:
        """Return the token representation width."""
        return int(self.tokens.shape[1])

    @property
    def lengths(self) -> Tensor:
        """Return per-sequence token counts on the packed device."""
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]

    @classmethod
    def from_sequences(cls, sequences: tuple[Tensor, ...]) -> PackedTokenBatch:
        """Pack a non-empty tuple of non-empty token matrices."""
        if not sequences:
            raise ValueError("at least one token sequence is required")
        d_model = int(sequences[0].shape[-1])
        device = sequences[0].device
        lengths: list[int] = []
        for sequence in sequences:
            if sequence.ndim != 2 or int(sequence.shape[1]) != d_model:
                raise ValueError(
                    "all token sequences must share shape [length, d_model]"
                )
            if sequence.device != device:
                raise ValueError("all token sequences must use the same device")
            length = int(sequence.shape[0])
            if length <= 0:
                raise ValueError("packed sequences must be non-empty")
            lengths.append(length)
        offset_values = [0]
        for length in lengths:
            offset_values.append(offset_values[-1] + length)
        offsets = torch.tensor(offset_values, dtype=torch.int32, device=device)
        return cls(
            tokens=torch.cat(sequences, dim=0),
            cu_seqlens=offsets,
            max_seqlen=max(lengths),
            offsets=tuple(offset_values),
        )

    @classmethod
    def from_padded(
        cls,
        tokens: Tensor,
        valid_mask: Tensor,
        *,
        lengths: tuple[int, ...] = (),
    ) -> PackedTokenBatch:
        """Pack row-major valid tokens without constructing per-row tensors."""
        if tokens.ndim != 3:
            raise ValueError("padded tokens must have shape [batch, tokens, d_model]")
        if valid_mask.shape != tokens.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid mask must be boolean and align with padded tokens")
        if valid_mask.device != tokens.device:
            raise ValueError("valid mask and padded tokens must use the same device")
        resolved_lengths = lengths or tuple(
            int(value) for value in valid_mask.sum(dim=1).detach().cpu().tolist()
        )
        if (
            len(resolved_lengths) != int(tokens.shape[0])
            or not resolved_lengths
            or any(
                length <= 0 or length > int(tokens.shape[1])
                for length in resolved_lengths
            )
        ):
            raise ValueError("packed sequences must be non-empty")
        offset_values = [0]
        for length in resolved_lengths:
            offset_values.append(offset_values[-1] + length)
        return cls(
            tokens=tokens[valid_mask],
            cu_seqlens=torch.tensor(
                offset_values,
                dtype=torch.int32,
                device=tokens.device,
            ),
            max_seqlen=max(resolved_lengths),
            offsets=tuple(offset_values),
        )

    def with_tokens(self, tokens: Tensor) -> PackedTokenBatch:
        """Reuse the immutable packing identity with transformed token values."""
        if tokens.shape != self.tokens.shape:
            raise ValueError("transformed packed tokens changed shape")
        return PackedTokenBatch(
            tokens=tokens,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.max_seqlen,
            offsets=self.offsets,
        )


@dataclass(frozen=True)
class SpecialTokenPositions:
    """Packed indices for fixed semantic tokens in every sequence."""

    public: Tensor
    deck: Tensor
    opponent_belief: Tensor
    policy: Tensor
    value: Tensor
    scratch: Tensor


def prepend_special_tokens(
    entity_tokens: PackedTokenBatch,
    special_tokens: Tensor,
) -> tuple[PackedTokenBatch, SpecialTokenPositions]:
    """Prepend fixed-layout special tokens to each packed entity sequence."""
    if special_tokens.ndim != 3:
        raise ValueError("special_tokens must have shape [batch, special, d_model]")
    batch_size, special_count, d_model = special_tokens.shape
    if batch_size != entity_tokens.batch_size:
        raise ValueError("special-token batch differs from entity-token batch")
    if d_model != entity_tokens.d_model:
        raise ValueError("special and entity token widths differ")
    if special_count < 5:
        raise ValueError("special-token layout requires five semantic tokens")

    output_offsets = tuple(
        source_offset + row * special_count
        for row, source_offset in enumerate(entity_tokens.offsets)
    )
    output_tokens = entity_tokens.tokens.new_zeros(
        (output_offsets[-1], entity_tokens.d_model)
    )
    row_extra = (
        torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device=entity_tokens.tokens.device,
        )
        * special_count
    )
    output_cu_seqlens = entity_tokens.cu_seqlens + row_extra
    starts = output_cu_seqlens[:-1].to(dtype=torch.long)
    special_columns = torch.arange(
        special_count,
        dtype=torch.long,
        device=starts.device,
    )
    special_destinations = (starts[:, None] + special_columns[None, :]).reshape(-1)
    output_tokens.index_copy_(
        0,
        special_destinations,
        special_tokens.reshape(-1, d_model),
    )
    entity_rows = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=starts.device),
        entity_tokens.lengths.to(dtype=torch.long),
        output_size=int(entity_tokens.tokens.shape[0]),
    )
    entity_destinations = (
        torch.arange(
            entity_tokens.tokens.shape[0],
            dtype=torch.long,
            device=starts.device,
        )
        + (entity_rows + 1) * special_count
    )
    output_tokens.index_copy_(
        0,
        entity_destinations,
        entity_tokens.tokens,
    )
    packed = PackedTokenBatch(
        tokens=output_tokens,
        cu_seqlens=output_cu_seqlens,
        max_seqlen=entity_tokens.max_seqlen + special_count,
        offsets=output_offsets,
    )
    starts = packed.cu_seqlens[:-1].to(dtype=torch.long)
    scratch_offsets = torch.arange(
        5,
        special_count,
        dtype=torch.long,
        device=starts.device,
    )
    return (
        packed,
        SpecialTokenPositions(
            public=starts,
            deck=starts + 1,
            opponent_belief=starts + 2,
            policy=starts + 3,
            value=starts + 4,
            scratch=starts[:, None] + scratch_offsets[None, :],
        ),
    )


def unpack_after_prefix(
    batch: PackedTokenBatch,
    *,
    prefix_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Pad only tokens following a fixed per-sequence prefix."""
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    lengths_host = [
        int(end) - int(start) - prefix_tokens for start, end in pairwise(batch.offsets)
    ]
    if any(length < 0 for length in lengths_host):
        raise ValueError("fixed prefix exceeds a packed sequence length")
    max_length = batch.max_seqlen - prefix_tokens
    if max_length == 0:
        return (
            batch.tokens.new_zeros((batch.batch_size, 0, batch.d_model)),
            torch.ones(
                (batch.batch_size, 0),
                dtype=torch.bool,
                device=batch.tokens.device,
            ),
        )
    length_tensor = (
        batch.cu_seqlens[1:].to(dtype=torch.long)
        - batch.cu_seqlens[:-1].to(dtype=torch.long)
        - prefix_tokens
    )
    columns = torch.arange(
        max_length,
        dtype=torch.long,
        device=batch.tokens.device,
    )
    valid = columns.unsqueeze(0) < length_tensor.unsqueeze(1)
    source_starts = batch.cu_seqlens[:-1].to(dtype=torch.long) + prefix_tokens
    source_indices = source_starts.unsqueeze(1) + columns.unsqueeze(0)
    safe_indices = torch.where(
        valid,
        source_indices,
        torch.zeros_like(source_indices),
    )
    padded = batch.tokens[safe_indices]
    padded = padded.masked_fill(~valid.unsqueeze(-1), 0.0)
    return (padded, ~valid)
