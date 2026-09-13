"""Early temporal conditioning for snapshot entities and legal options."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
from torch import Tensor, nn


class GatedTemporalContentFusion(nn.Module):
    """Condition content rows on one temporal context with an inert output gate."""

    def __init__(self, *, d_model: int, hidden_dim: int) -> None:
        """Build content-aware context modulation at a fixed shared width."""
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("temporal fusion dimensions must be positive")
        self.d_model = d_model
        self.content_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.content_projection = nn.Linear(d_model, hidden_dim)
        self.context_projection = nn.Linear(d_model, hidden_dim, bias=False)
        self.content_gate = nn.Linear(d_model, hidden_dim)
        self.context_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, d_model)

    def forward(
        self,
        content: Tensor,
        context: Tensor,
        *,
        valid_mask: Tensor,
    ) -> Tensor:
        """Fuse aligned batch contexts without changing invalid padded rows."""
        if content.ndim != 3 or int(content.shape[-1]) != self.d_model:
            raise ValueError("fusion content must have shape [batch, rows, d_model]")
        expected_context = (int(content.shape[0]), self.d_model)
        if tuple(context.shape) != expected_context:
            raise ValueError(f"fusion context must have shape {expected_context}")
        if valid_mask.shape != content.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("fusion valid mask must align with content rows")
        if valid_mask.device != content.device or context.device != content.device:
            raise ValueError("fusion tensors must share one device")
        normalized_content = self.content_norm(content)
        normalized_context = self.context_norm(context).unsqueeze(1)
        hidden = torch.nn.functional.gelu(
            self.content_projection(normalized_content)
            + self.context_projection(normalized_context)
        )
        gate = torch.sigmoid(
            self.content_gate(normalized_content)
            + self.context_gate(normalized_context)
        )
        delta = self.output(hidden * gate)
        return cast(
            Tensor,
            content + delta * valid_mask.unsqueeze(-1).to(dtype=delta.dtype),
        )

    def zero_output(self) -> None:
        """Make the fusion exactly preserve its content input."""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def inert_output_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield the output gate parameters used by migration audits."""
        yield ("output.weight", cast(nn.Parameter, self.output.weight))
        yield ("output.bias", self.output.bias)


__all__ = ["GatedTemporalContentFusion"]
