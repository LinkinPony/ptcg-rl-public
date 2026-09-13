"""Raw public-catalog encoding and learner-only sparse belief supervision."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.belief.public_catalog import (
    PublicDeckPosterior,
    PublicDeckPosteriorArrays,
)
from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.decks.identity import DECK_SIZE
from ptcg_rl.model.tensor_validation import require_tensor_condition

_SUMMARY_SCALAR_SIZE = 4
_BELIEF_LOSS_COEFFICIENT = 0.05


@dataclass(frozen=True)
class PublicBeliefSummaryBatch:
    """Padded tensor view of raw sparse catalog posterior summaries."""

    card_ids: Tensor
    expected_counts: Tensor
    valid_mask: Tensor
    scalars: Tensor
    catalog_fingerprint: str
    row_indices: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate reusable tensor-shape invariants."""
        if self.card_ids.ndim != 2:
            raise ValueError("belief summary card_ids must have shape [batch, cards]")
        if self.expected_counts.shape != self.card_ids.shape:
            raise ValueError("belief expected counts must align with card IDs")
        if self.valid_mask.shape != self.card_ids.shape:
            raise ValueError("belief valid mask must align with card IDs")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("belief valid mask must be boolean")
        if self.scalars.shape != (self.card_ids.shape[0], _SUMMARY_SCALAR_SIZE):
            raise ValueError("belief summary scalar width is invalid")
        if not self.catalog_fingerprint:
            raise ValueError("belief summary requires a catalog fingerprint")
        if self.row_indices is not None:
            if (
                self.row_indices.ndim != 1
                or self.row_indices.dtype != torch.long
                or self.row_indices.device != self.card_ids.device
            ):
                raise ValueError(
                    "belief summary row indices must be an aligned int64 vector"
                )
            if (
                not self.row_indices.is_meta
                and not self.row_indices.is_cuda
                and int(self.row_indices.numel()) > 0
                and (
                    bool((self.row_indices < 0).any())
                    or bool((self.row_indices >= self.card_ids.shape[0]).any())
                )
            ):
                raise ValueError("belief summary row index is out of range")


def collate_public_belief_summaries(
    summaries: Sequence[PublicDeckPosterior | PublicDeckPosteriorArrays],
    *,
    catalog_fingerprint: str,
    device: torch.device | str | None = None,
    deduplicate: bool = False,
) -> PublicBeliefSummaryBatch:
    """Collate full sparse expected-multiset summaries without top-k truncation."""
    if not summaries:
        raise ValueError("at least one public belief summary is required")
    selected_summaries: list[PublicDeckPosterior | PublicDeckPosteriorArrays] = []
    inverse: list[int] = []
    if deduplicate:
        unique_by_content: dict[tuple[bytes, bytes, tuple[float, ...]], int] = {}
        for summary in summaries:
            card_ids, expected_counts = _summary_arrays(summary)
            key = (
                np.asarray(card_ids, dtype=np.int32).tobytes(),
                np.asarray(expected_counts, dtype=np.float32).tobytes(),
                _summary_scalars(summary),
            )
            unique_index = unique_by_content.get(key)
            if unique_index is None:
                unique_index = len(selected_summaries)
                unique_by_content[key] = unique_index
                selected_summaries.append(summary)
            inverse.append(unique_index)
    else:
        selected_summaries.extend(summaries)
        inverse.extend(range(len(summaries)))
    width = max(
        1,
        max(
            (
                len(summary.card_ids)
                if isinstance(summary, PublicDeckPosteriorArrays)
                else len(summary.expected_remaining)
            )
            for summary in selected_summaries
        ),
    )
    batch_size = len(selected_summaries)
    card_id_rows = np.zeros((batch_size, width), dtype=np.int64)
    expected_rows = np.zeros((batch_size, width), dtype=np.float32)
    valid_rows = np.zeros((batch_size, width), dtype=np.bool_)
    scalar_rows = np.zeros(
        (batch_size, _SUMMARY_SCALAR_SIZE),
        dtype=np.float32,
    )
    for row, summary in enumerate(selected_summaries):
        summary_card_ids, summary_expected_counts = _summary_arrays(summary)
        count = len(summary_card_ids)
        if count:
            card_id_rows[row, :count] = summary_card_ids
            expected_rows[row, :count] = summary_expected_counts
            valid_rows[row, :count] = True
        scalar_rows[row] = _summary_scalars(summary)
    row_indices = None
    if len(selected_summaries) != len(summaries):
        row_indices = torch.as_tensor(
            inverse,
            dtype=torch.long,
            device=device,
        )
    return PublicBeliefSummaryBatch(
        card_ids=torch.as_tensor(card_id_rows, device=device),
        expected_counts=torch.as_tensor(expected_rows, device=device),
        valid_mask=torch.as_tensor(valid_rows, device=device),
        scalars=torch.as_tensor(scalar_rows, device=device),
        catalog_fingerprint=catalog_fingerprint,
        row_indices=row_indices,
    )


def _summary_arrays(
    summary: PublicDeckPosterior | PublicDeckPosteriorArrays,
) -> tuple[Any, Any]:
    if isinstance(summary, PublicDeckPosteriorArrays):
        return (summary.card_ids, summary.expected_counts)
    return (
        [item.card_id for item in summary.expected_remaining],
        [item.expected_count for item in summary.expected_remaining],
    )


def _summary_scalars(
    summary: PublicDeckPosterior | PublicDeckPosteriorArrays,
) -> tuple[float, ...]:
    return (
        float(summary.entropy),
        float(summary.compatible_deck_count),
        float(summary.public_evidence_count),
        float(summary.unknown_probability),
    )


class PublicBeliefSummaryEncoder(nn.Module):
    """Encode raw expected card counts using current shared card embeddings."""

    def __init__(self, *, d_model: int) -> None:
        """Initialize count-aware set and scalar projections."""
        super().__init__()
        self.card_projection = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(_SUMMARY_SCALAR_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        card_embeddings: Tensor,
        summary: PublicBeliefSummaryBatch,
    ) -> Tensor:
        """Return one learned catalog-summary token per decision."""
        if card_embeddings.shape[:2] != summary.card_ids.shape:
            raise ValueError("belief card embeddings differ from raw summary")
        counts = summary.expected_counts.to(dtype=card_embeddings.dtype)
        count_features = torch.log1p(counts.clamp_min(0.0)).unsqueeze(-1)
        encoded = self.card_projection(
            torch.cat((card_embeddings, count_features), dim=-1)
        )
        weights = counts * summary.valid_mask
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        scalars = summary.scalars.to(dtype=card_embeddings.dtype)
        normalized_scalars = torch.stack(
            (
                scalars[:, 0],
                torch.log1p(scalars[:, 1]),
                scalars[:, 2] / float(DECK_SIZE),
                scalars[:, 3],
            ),
            dim=1,
        )
        return preserve_cuda_bfloat16_activation(
            self.output_norm(pooled + self.scalar_projection(normalized_scalars)),
        )


class OpponentCardBeliefHead(nn.Module):
    """Diagnostic card-vocabulary logits computed only on explicit request."""

    def __init__(self, *, d_model: int) -> None:
        """Initialize the belief-token query projection."""
        super().__init__()
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        belief_token: Tensor,
        *,
        card_encoder: CardEncoder,
    ) -> Tensor:
        """Score every canonical card using shared CardEncoder key embeddings."""
        card_ids = torch.arange(
            1,
            card_encoder.num_card_ids + 1,
            dtype=torch.long,
            device=belief_token.device,
        )
        card_embeddings = card_encoder(card_ids)
        query = self.query_projection(belief_token)
        return torch.matmul(query, card_embeddings.transpose(0, 1)) / math.sqrt(
            float(self.d_model)
        )


@dataclass(frozen=True)
class SparseBeliefTargets:
    """CSR-like learner-only unidentified-card counts."""

    offsets: Tensor
    card_ids: Tensor
    counts: Tensor

    def __post_init__(self) -> None:
        """Validate compact target alignment."""
        if self.offsets.ndim != 1 or self.offsets.numel() < 2:
            raise ValueError("belief target offsets must have shape [batch + 1]")
        if self.offsets.dtype != torch.long:
            raise ValueError("belief target offsets must use int64")
        if self.card_ids.ndim != 1 or self.counts.shape != self.card_ids.shape:
            raise ValueError("belief target card IDs and counts must align")
        if (
            self.offsets.device != self.card_ids.device
            or self.counts.device != self.card_ids.device
        ):
            raise ValueError("belief target tensors must share one device")
        if not self.offsets.is_meta:
            require_tensor_condition(
                self.offsets[0].eq(0) & self.offsets[-1].eq(self.card_ids.numel()),
                "belief target offsets do not cover sparse values",
            )
            require_tensor_condition(
                (self.offsets[1:] >= self.offsets[:-1]).all(),
                "belief target offsets must be non-decreasing",
            )
            require_tensor_condition(
                (self.counts > 0).all(),
                "belief target counts must be positive",
            )

    @property
    def batch_size(self) -> int:
        """Return the number of decision targets."""
        return int(self.offsets.numel()) - 1

    @property
    def valid_rows(self) -> Tensor:
        """Return decisions that still have unidentified cards."""
        return self.offsets[1:] > self.offsets[:-1]


def build_sparse_belief_targets(
    opponent_decks: Sequence[Sequence[int]],
    known_counts: Sequence[Counter[int]],
    *,
    card_vocab_size: int,
    device: torch.device | str | None = None,
) -> SparseBeliefTargets:
    """Subtract public evidence from exact learner-side opponent decks."""
    if len(opponent_decks) != len(known_counts):
        raise ValueError("opponent decks and public evidence must align")
    offsets = [0]
    card_ids: list[int] = []
    counts: list[float] = []
    for deck_cards, known in zip(opponent_decks, known_counts, strict=True):
        if len(deck_cards) != DECK_SIZE:
            raise ValueError("learner-side opponent deck must contain 60 cards")
        exact = Counter(int(card_id) for card_id in deck_cards)
        if any(card_id <= 0 or card_id > card_vocab_size for card_id in exact):
            raise ValueError("learner-side opponent card exceeds vocabulary")
        unidentified = exact.copy()
        for raw_card_id, raw_count in known.items():
            card_id = int(raw_card_id)
            count = int(raw_count)
            if (
                card_id <= 0
                or card_id > card_vocab_size
                or count <= 0
                or exact[card_id] < count
            ):
                raise ValueError(
                    "public evidence cannot be subtracted from exact opponent deck"
                )
            unidentified[card_id] -= count
        for card_id in sorted(unidentified):
            count = unidentified[card_id]
            if count <= 0:
                continue
            card_ids.append(card_id)
            counts.append(float(count))
        offsets.append(len(card_ids))
    return SparseBeliefTargets(
        offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        card_ids=torch.tensor(card_ids, dtype=torch.long, device=device),
        counts=torch.tensor(counts, dtype=torch.float32, device=device),
    )


def normalized_sparse_belief_loss(
    logits: Tensor,
    targets: SparseBeliefTargets,
) -> Tensor:
    """Return fixed-weight sparse soft-target CE normalized by log vocabulary."""
    row_losses, valid = normalized_sparse_belief_row_losses(logits, targets)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return row_losses[valid].mean()


def normalized_sparse_belief_row_losses(
    logits: Tensor,
    targets: SparseBeliefTargets,
) -> tuple[Tensor, Tensor]:
    """Return fixed-weight normalized loss and validity for every decision."""
    if logits.ndim != 2 or logits.shape[0] != targets.batch_size:
        raise ValueError("belief logits must have shape [target_batch, vocabulary]")
    vocabulary_size = int(logits.shape[1])
    if vocabulary_size <= 1:
        raise ValueError("belief vocabulary must contain more than one card")
    if targets.card_ids.numel() == 0:
        return (
            logits.new_zeros(targets.batch_size, dtype=torch.float32),
            logits.new_zeros(targets.batch_size, dtype=torch.bool),
        )
    require_tensor_condition(
        ~((targets.card_ids < 1) | (targets.card_ids > vocabulary_size)).any(),
        "belief target card ID exceeds logits vocabulary",
    )
    lengths = (targets.offsets[1:] - targets.offsets[:-1]).to(device=logits.device)
    row_indices = torch.repeat_interleave(
        torch.arange(
            targets.batch_size,
            dtype=torch.long,
            device=logits.device,
        ),
        lengths,
    )
    logprobabilities = torch.log_softmax(logits.float(), dim=1)
    selected = logprobabilities[
        row_indices,
        targets.card_ids.to(device=logits.device) - 1,
    ]
    weighted = -selected * targets.counts.to(device=logits.device)
    row_losses = logits.new_zeros(targets.batch_size, dtype=torch.float32)
    row_weights = logits.new_zeros(targets.batch_size, dtype=torch.float32)
    row_losses.index_add_(0, row_indices, weighted)
    row_weights.index_add_(
        0,
        row_indices,
        targets.counts.to(device=logits.device),
    )
    valid = row_weights > 0.0
    normalized = torch.where(
        valid,
        row_losses / row_weights.clamp_min(1.0),
        torch.zeros_like(row_losses),
    )
    normalized = (
        normalized * _BELIEF_LOSS_COEFFICIENT / math.log(float(vocabulary_size))
    )
    return (normalized, valid)
