"""Root-perspective value adaptation for semantic engine endpoints."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

from ptcg_rl.agent.search.root_information import RootActorRelation
from ptcg_rl.engine.compact_consequence import SemanticEndpoint

ROOT_PERSPECTIVE_VALUE_ARCHITECTURE_VERSION: Literal[1] = 1


class RootPerspectiveValueAdapterConfig(BaseModel):
    """Validated shape contract for the root-information residual adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: Literal[1] = ROOT_PERSPECTIVE_VALUE_ARCHITECTURE_VERSION
    belief_summary_dim: int = 0
    hidden_dim: int = 64

    @field_validator("belief_summary_dim")
    @classmethod
    def valid_belief_summary_dim(cls, value: int) -> int:
        """Allow an empty summary while rejecting negative dimensions."""
        if value < 0:
            raise ValueError("belief_summary_dim must be non-negative")
        return value

    @field_validator("hidden_dim")
    @classmethod
    def valid_hidden_dim(cls, value: int) -> int:
        """Require a non-empty residual bottleneck."""
        if value <= 0:
            raise ValueError("hidden_dim must be positive")
        return value


class RootPerspectiveValueAdapter(nn.Module):
    """Adapt one critic scalar without changing its initial predictions.

    The existing state/value trunk remains the base.  Relation, endpoint, and
    belief inputs only contribute through a residual whose output layer is
    initialized to exact zeros, so same-seat values initially equal the
    current critic bit-for-bit.
    """

    def __init__(
        self,
        state_embedding_dim: int,
        config: RootPerspectiveValueAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        if state_embedding_dim <= 0:
            raise ValueError("state_embedding_dim must be positive")
        self.config = config or RootPerspectiveValueAdapterConfig()
        input_dim = state_embedding_dim + self.config.belief_summary_dim + 4
        self.residual = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        output = cast(nn.Linear, self.residual[-1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        base_values: Tensor,
        state_embeddings: Tensor,
        actor_relations: Tensor,
        endpoints: Tensor,
        belief_summaries: Tensor,
    ) -> Tensor:
        """Return bounded values in the immutable root player's perspective."""
        batch_size = _validate_shapes(
            base_values=base_values,
            state_embeddings=state_embeddings,
            actor_relations=actor_relations,
            endpoints=endpoints,
            belief_summaries=belief_summaries,
            expected_belief_dim=self.config.belief_summary_dim,
        )
        relation_indices = actor_relations.to(
            device=state_embeddings.device,
            dtype=torch.long,
        )
        if bool(
            ((relation_indices < 0) | (relation_indices > 1)).any().item()
        ):
            raise ValueError("actor_relations contains an unsupported value")
        endpoint_indices, expected_relations = _endpoint_indices(
            endpoints.to(device=state_embeddings.device, dtype=torch.long)
        )
        if bool((relation_indices != expected_relations).any().item()):
            raise ValueError("actor relation does not match semantic endpoint")
        relation_features = torch.nn.functional.one_hot(
            relation_indices,
            num_classes=2,
        )
        endpoint_features = torch.nn.functional.one_hot(
            endpoint_indices,
            num_classes=2,
        )
        module_dtype = next(self.residual.parameters()).dtype
        state_features = state_embeddings.to(dtype=module_dtype)
        relation_features = relation_features.to(dtype=module_dtype)
        endpoint_features = endpoint_features.to(dtype=module_dtype)
        belief = belief_summaries.to(
            device=state_embeddings.device,
            dtype=module_dtype,
        )
        inputs = torch.cat(
            (state_features, belief, relation_features, endpoint_features),
            dim=1,
        )
        with _autocast_disabled(inputs):
            residual = cast(Tensor, self.residual(inputs)).reshape(batch_size)
        return (
            base_values.to(device=residual.device, dtype=residual.dtype) + residual
        ).clamp(min=-1.0, max=1.0)


def _validate_shapes(
    *,
    base_values: Tensor,
    state_embeddings: Tensor,
    actor_relations: Tensor,
    endpoints: Tensor,
    belief_summaries: Tensor,
    expected_belief_dim: int,
) -> int:
    if state_embeddings.ndim != 2:
        raise ValueError("state_embeddings must have shape [batch, feature]")
    batch_size = int(state_embeddings.shape[0])
    if base_values.shape != (batch_size,):
        raise ValueError("base_values must have shape [batch]")
    if actor_relations.shape != (batch_size,):
        raise ValueError("actor_relations must have shape [batch]")
    if endpoints.shape != (batch_size,):
        raise ValueError("endpoints must have shape [batch]")
    if belief_summaries.shape != (batch_size, expected_belief_dim):
        raise ValueError("belief_summaries has the wrong shape")
    if not bool(torch.isfinite(base_values).all().item()):
        raise ValueError("base_values must be finite")
    if bool((base_values.abs() > 1.0).any().item()):
        raise ValueError("base_values must already be bounded to [-1, 1]")
    if not bool(torch.isfinite(state_embeddings).all().item()):
        raise ValueError("state_embeddings must be finite")
    if not bool(torch.isfinite(belief_summaries).all().item()):
        raise ValueError("belief_summaries must be finite")
    return batch_size


def _endpoint_indices(endpoints: Tensor) -> tuple[Tensor, Tensor]:
    same_seat = endpoints.eq(int(SemanticEndpoint.SAME_SEAT_MAIN))
    handoff = endpoints.eq(int(SemanticEndpoint.TURN_HANDOFF))
    if bool((~(same_seat | handoff)).any().item()):
        raise ValueError("value adapter received a non-value semantic endpoint")
    indices = handoff.to(dtype=torch.long)
    expected_relations = torch.where(
        handoff,
        torch.full_like(endpoints, int(RootActorRelation.OTHER_SEAT)),
        torch.full_like(endpoints, int(RootActorRelation.SAME_SEAT)),
    )
    return indices, expected_relations


def _autocast_disabled(tensor: Tensor) -> AbstractContextManager[None]:
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


__all__ = [
    "ROOT_PERSPECTIVE_VALUE_ARCHITECTURE_VERSION",
    "RootPerspectiveValueAdapter",
    "RootPerspectiveValueAdapterConfig",
]
