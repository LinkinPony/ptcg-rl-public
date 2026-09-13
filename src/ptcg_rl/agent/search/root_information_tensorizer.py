"""Production root-information tensorization and batched value inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor

from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.agent.search.root_information import (
    RootInformationLeaf,
    RootInformationStateTensorBatch,
)
from ptcg_rl.agent.search.root_information_context import (
    ROOT_INFORMATION_PRODUCER_CONTEXT_FINGERPRINT,
    decode_root_information_observation,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.model.state_encoder import (
    TOKEN_SCALAR_SIZE,
    StateBatch,
    collate_state_tokens,
)

_TENSOR_SCHEMA_DESCRIPTOR = (
    "root-information-tensor/v1;state=canonical_policy_state_tokens;"
    f"token_scalar_width={TOKEN_SCALAR_SIZE};"
    f"producer={ROOT_INFORMATION_PRODUCER_CONTEXT_FINGERPRINT};"
    "extras=actor_relation,endpoint,belief_summary"
)
ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT = hashlib.sha256(
    _TENSOR_SCHEMA_DESCRIPTOR.encode("ascii")
).hexdigest()


class RootInformationTensorizerConfig(BaseModel):
    """Bounded decode/tensor geometry shared by training and serving."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_unique_leaves: int
    max_observation_bytes_per_leaf: int
    max_producer_context_bytes_per_leaf: int
    belief_summary_dim: int

    @field_validator(
        "max_unique_leaves",
        "max_observation_bytes_per_leaf",
        "max_producer_context_bytes_per_leaf",
    )
    @classmethod
    def positive_bound(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("root-information tensorizer bounds must be positive")
        return value

    @field_validator("belief_summary_dim")
    @classmethod
    def nonnegative_belief_width(cls, value: int) -> int:
        if value < 0:
            raise ValueError("belief_summary_dim must be non-negative")
        return value


@dataclass(frozen=True, slots=True)
class RootInformationModelInputBatch:
    """Unique CPU/GPU model inputs aligned with root-information leaves."""

    states: StateBatch
    actor_relations: Tensor
    endpoints: Tensor
    belief_summaries: Tensor
    tensor_schema_fingerprint: str = ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT

    def __post_init__(self) -> None:
        batch_size = int(self.states.card_ids.shape[0])
        if self.actor_relations.shape != (batch_size,):
            raise ValueError("actor_relations must align with unique leaves")
        if self.endpoints.shape != (batch_size,):
            raise ValueError("endpoints must align with unique leaves")
        if (
            self.belief_summaries.ndim != 2
            or int(self.belief_summaries.shape[0]) != batch_size
        ):
            raise ValueError("belief_summaries must align with unique leaves")
        if self.actor_relations.dtype != torch.long:
            raise TypeError("actor_relations must use torch.long")
        if self.endpoints.dtype != torch.long:
            raise TypeError("endpoints must use torch.long")
        if self.belief_summaries.dtype != torch.float32:
            raise TypeError("belief_summaries must use torch.float32")
        if not bool(torch.isfinite(self.belief_summaries).all().item()):
            raise ValueError("belief_summaries must be finite")
        if self.tensor_schema_fingerprint != (
            ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
        ):
            raise ValueError("root-information tensor schema fingerprint mismatch")


class ProductionRootInformationTensorizer:
    """Decode each unique native leaf once through the serving state encoder."""

    def __init__(self, config: RootInformationTensorizerConfig) -> None:
        self.config = config

    @property
    def tensor_schema_fingerprint(self) -> str:
        """Return the immutable input layout used for batching/cache identity."""
        return ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT

    def tensorize(
        self,
        leaves: tuple[RootInformationLeaf, ...],
    ) -> RootInformationModelInputBatch:
        """Build one CPU batch without materializing any scenario-private row."""
        if not leaves:
            raise ValueError("production tensorizer requires at least one leaf")
        if len(leaves) > self.config.max_unique_leaves:
            raise ValueError("unique leaf count exceeds tensorizer capacity")
        fingerprints = tuple(leaf.model_input_fingerprint for leaf in leaves)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("production tensorizer received duplicate leaves")
        states = []
        relations: list[int] = []
        endpoints: list[int] = []
        beliefs: list[tuple[float, ...]] = []
        for leaf in leaves:
            if (
                len(leaf.root_observable_state)
                > self.config.max_observation_bytes_per_leaf
            ):
                raise ValueError("root-visible observation exceeds tensorizer capacity")
            if (
                len(leaf.producer_context)
                > self.config.max_producer_context_bytes_per_leaf
            ):
                raise ValueError("producer context exceeds tensorizer capacity")
            if len(leaf.belief_summary) != self.config.belief_summary_dim:
                raise ValueError("leaf belief summary has the wrong width")
            observation = decode_root_information_observation(
                leaf.root_observable_state,
                leaf.producer_context,
            )
            policy_input = build_canonical_policy_input(
                observation,
                require_options=False,
            )
            if policy_input is None:
                raise ValueError("root-visible leaf could not be tensorized")
            states.append(policy_input.state.without_layout())
            relations.append(int(leaf.actor_relation))
            endpoints.append(int(leaf.endpoint))
            beliefs.append(leaf.belief_summary)
        return RootInformationModelInputBatch(
            states=collate_state_tokens(states, device="cpu"),
            actor_relations=torch.tensor(relations, dtype=torch.long),
            endpoints=torch.tensor(endpoints, dtype=torch.long),
            belief_summaries=torch.tensor(beliefs, dtype=torch.float32).reshape(
                len(leaves), self.config.belief_summary_dim
            ),
        )

    def tensorize_all(
        self,
        leaves: tuple[RootInformationLeaf, ...],
    ) -> RootInformationModelInputBatch:
        """Tensorize an aggregate replay set through bounded input shards.

        ``max_unique_leaves`` remains the hard geometry of each tensorizer
        invocation used by serving. Learner windows may contain more distinct
        actual endpoints, so this path applies that same bound to every shard
        before concatenating their CPU tensors for later minibatch selection.
        """
        if not leaves:
            raise ValueError("production tensorizer requires at least one leaf")
        fingerprints = tuple(leaf.model_input_fingerprint for leaf in leaves)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("production tensorizer received duplicate leaves")
        shard_size = self.config.max_unique_leaves
        shards = tuple(
            self.tensorize(leaves[start : start + shard_size])
            for start in range(0, len(leaves), shard_size)
        )
        return _concatenate_model_input_batches(shards)


class RootInformationDeckProvider(Protocol):
    """Provide root deck rows without reading scenario-private identities."""

    def decks_for(
        self,
        leaves: tuple[RootInformationLeaf, ...],
        *,
        device: torch.device,
    ) -> DeckBatch:
        """Return one root deck row per unique information leaf."""


class ConstantRootDeckProvider:
    """Repeat one deployment-bundle deck across request-local unique leaves."""

    def __init__(self, card_ids: Sequence[int] | CanonicalDeck) -> None:
        self.deck = (
            card_ids
            if isinstance(card_ids, CanonicalDeck)
            else canonicalize_deck(card_ids)
        )

    def decks_for(
        self,
        leaves: tuple[RootInformationLeaf, ...],
        *,
        device: torch.device,
    ) -> DeckBatch:
        """Construct a single batched route input, never one row at a time."""
        return DeckBatch.from_decks((self.deck,) * len(leaves), device=device)


class RootInformationNetwork(Protocol):
    """Read-only model API needed by local root-value batching."""

    def encode_conditioned_state(
        self,
        states: StateBatch,
        decks: DeckBatch,
    ) -> Any:
        """Encode state/deck rows once."""

    def root_information_values_from_conditioned(
        self,
        conditioned: Any,
        *,
        actor_relations: Tensor,
        endpoints: Tensor,
        belief_summaries: Tensor,
    ) -> Tensor:
        """Return root-perspective values for unique leaves."""


class LocalBatchedRootInformationValueProvider:
    """Batch all unique leaf rows through one GPU model invocation."""

    def __init__(
        self,
        *,
        model: RootInformationNetwork,
        deck_provider: RootInformationDeckProvider,
        device: torch.device | str,
        use_bfloat16: bool,
    ) -> None:
        self._model = model
        self._deck_provider = deck_provider
        self._device = torch.device(device)
        self._use_bfloat16 = bool(use_bfloat16)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA root-information inference is unavailable")

    def values(
        self,
        batch: RootInformationStateTensorBatch[RootInformationModelInputBatch],
    ) -> npt.NDArray[np.float32]:
        """Return one detached CPU float32 value per unique leaf."""
        inputs = batch.model_inputs
        states = _move_state_batch(inputs.states, device=self._device)
        decks = self._deck_provider.decks_for(batch.leaves, device=self._device)
        relations = inputs.actor_relations.to(device=self._device, non_blocking=True)
        endpoints = inputs.endpoints.to(device=self._device, non_blocking=True)
        beliefs = inputs.belief_summaries.to(
            device=self._device,
            non_blocking=True,
        )
        with torch.inference_mode(), self._autocast_context():
            conditioned = self._model.encode_conditioned_state(states, decks)
            values = self._model.root_information_values_from_conditioned(
                conditioned,
                actor_relations=relations,
                endpoints=endpoints,
                belief_summaries=beliefs,
            )
        if values.shape != (len(batch.leaves),):
            raise ValueError("root-information model returned the wrong value shape")
        return values.detach().to(device="cpu", dtype=torch.float32).numpy()

    def _autocast_context(self) -> AbstractContextManager[None]:
        if self._device.type == "cuda" and self._use_bfloat16:
            return cast(
                AbstractContextManager[None],
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            )
        return nullcontext()


class ScheduledRootInformationInferenceClient(Protocol):
    """Action-critical remote inference surface implemented by the server."""

    def predict_root_information_values_until(
        self,
        inputs: RootInformationModelInputBatch,
        decks: DeckBatch,
        *,
        deadline_monotonic: float,
        model_version_lease: int,
        tensor_schema_fingerprint: str,
    ) -> Tensor:
        """Batch unique values under one immutable model lease/deadline."""


class ScheduledRootInformationValueProvider:
    """Forward unique CPU leaves to the cross-actor planner GPU scheduler."""

    def __init__(
        self,
        *,
        client: ScheduledRootInformationInferenceClient,
        deck_provider: RootInformationDeckProvider,
        deadline_monotonic: float,
        model_version_lease: int,
    ) -> None:
        if not math.isfinite(deadline_monotonic) or deadline_monotonic <= 0.0:
            raise ValueError(
                "scheduled root-value deadline must be finite and positive"
            )
        if model_version_lease < 0:
            raise ValueError("model_version_lease must be non-negative")
        self._client = client
        self._deck_provider = deck_provider
        self._deadline = float(deadline_monotonic)
        self._model_version_lease = int(model_version_lease)

    def values(
        self,
        batch: RootInformationStateTensorBatch[RootInformationModelInputBatch],
    ) -> npt.NDArray[np.float32]:
        """Request one deadline-aware microbatch, not per-candidate RPCs."""
        decks = self._deck_provider.decks_for(
            batch.leaves,
            device=torch.device("cpu"),
        )
        values = self._client.predict_root_information_values_until(
            batch.model_inputs,
            decks,
            deadline_monotonic=self._deadline,
            model_version_lease=self._model_version_lease,
            tensor_schema_fingerprint=(batch.model_inputs.tensor_schema_fingerprint),
        )
        return values.detach().to(device="cpu", dtype=torch.float32).numpy()


def _move_state_batch(states: StateBatch, *, device: torch.device) -> StateBatch:
    def move(tensor: Tensor | None) -> Tensor | None:
        return None if tensor is None else tensor.to(device=device, non_blocking=True)

    return StateBatch(
        card_ids=states.card_ids.to(device=device, non_blocking=True),
        areas=states.areas.to(device=device, non_blocking=True),
        owner_roles=states.owner_roles.to(device=device, non_blocking=True),
        token_kinds=states.token_kinds.to(device=device, non_blocking=True),
        scalars=states.scalars.to(device=device, non_blocking=True),
        last_attack_ids=states.last_attack_ids.to(
            device=device,
            non_blocking=True,
        ),
        padding_mask=states.padding_mask.to(device=device, non_blocking=True),
        attachment_card_ids=move(states.attachment_card_ids),
        attachment_parent_indices=move(states.attachment_parent_indices),
        attachment_kinds=move(states.attachment_kinds),
        entity_slots=move(states.entity_slots),
        root_input_fingerprints=states.root_input_fingerprints,
    )


def _concatenate_model_input_batches(
    batches: tuple[RootInformationModelInputBatch, ...],
) -> RootInformationModelInputBatch:
    """Concatenate bounded CPU tensorizer shards without changing row order."""
    if not batches:
        raise ValueError("root-information input batches must be non-empty")
    states = tuple(batch.states for batch in batches)
    max_tokens = max(int(state.card_ids.shape[1]) for state in states)
    max_attachments = max(
        _required_optional_width(state.attachment_card_ids) for state in states
    )
    return RootInformationModelInputBatch(
        states=StateBatch(
            card_ids=_pad_and_concatenate(
                tuple(state.card_ids for state in states),
                width=max_tokens,
                fill_value=0,
            ),
            areas=_pad_and_concatenate(
                tuple(state.areas for state in states),
                width=max_tokens,
                fill_value=0,
            ),
            owner_roles=_pad_and_concatenate(
                tuple(state.owner_roles for state in states),
                width=max_tokens,
                fill_value=0,
            ),
            token_kinds=_pad_and_concatenate(
                tuple(state.token_kinds for state in states),
                width=max_tokens,
                fill_value=0,
            ),
            scalars=_pad_and_concatenate(
                tuple(state.scalars for state in states),
                width=max_tokens,
                fill_value=0.0,
            ),
            last_attack_ids=_pad_and_concatenate(
                tuple(state.last_attack_ids for state in states),
                width=max_tokens,
                fill_value=0,
            ),
            padding_mask=_pad_and_concatenate(
                tuple(state.padding_mask for state in states),
                width=max_tokens,
                fill_value=True,
            ),
            attachment_card_ids=_pad_and_concatenate(
                tuple(
                    _require_optional_tensor(state.attachment_card_ids)
                    for state in states
                ),
                width=max_attachments,
                fill_value=0,
            ),
            attachment_parent_indices=_pad_and_concatenate(
                tuple(
                    _require_optional_tensor(state.attachment_parent_indices)
                    for state in states
                ),
                width=max_attachments,
                fill_value=0,
            ),
            attachment_kinds=_pad_and_concatenate(
                tuple(
                    _require_optional_tensor(state.attachment_kinds)
                    for state in states
                ),
                width=max_attachments,
                fill_value=0,
            ),
            entity_slots=_pad_and_concatenate(
                tuple(
                    _require_optional_tensor(state.entity_slots) for state in states
                ),
                width=max_tokens,
                fill_value=0,
            ),
        ),
        actor_relations=torch.cat(
            tuple(batch.actor_relations for batch in batches), dim=0
        ),
        endpoints=torch.cat(tuple(batch.endpoints for batch in batches), dim=0),
        belief_summaries=torch.cat(
            tuple(batch.belief_summaries for batch in batches), dim=0
        ),
    )


def _required_optional_width(values: Tensor | None) -> int:
    return int(_require_optional_tensor(values).shape[1])


def _require_optional_tensor(values: Tensor | None) -> Tensor:
    if values is None:
        raise ValueError("tensorized root inputs require optional state tensors")
    return values


def _pad_and_concatenate(
    values: tuple[Tensor, ...],
    *,
    width: int,
    fill_value: int | float | bool,
) -> Tensor:
    """Right-pad rank-2/3 row tensors to one width and concatenate them."""
    padded: list[Tensor] = []
    for tensor in values:
        if tensor.ndim not in (2, 3):
            raise ValueError("root-information state tensors must have rank 2 or 3")
        if int(tensor.shape[1]) > width:
            raise ValueError("root-information state tensor exceeds padded width")
        if int(tensor.shape[1]) == width:
            padded.append(tensor)
            continue
        shape = list(tensor.shape)
        shape[1] = width - int(tensor.shape[1])
        padding = torch.full(
            shape,
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded.append(torch.cat((tensor, padding), dim=1))
    return torch.cat(tuple(padded), dim=0)


__all__ = [
    "ConstantRootDeckProvider",
    "LocalBatchedRootInformationValueProvider",
    "ProductionRootInformationTensorizer",
    "ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT",
    "RootInformationModelInputBatch",
    "RootInformationTensorizerConfig",
    "ScheduledRootInformationValueProvider",
]
