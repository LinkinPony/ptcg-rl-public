"""Tensor-ready replay rows for actual root-information value endpoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.agent.search.root_information_tensorizer import (
    RootInformationModelInputBatch,
)
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.model.state_encoder import StateBatch


@dataclass(frozen=True, slots=True)
class RootInformationValueReplayBatch:
    """Deduplicated actual endpoint rows assigned to canonical decisions."""

    decision_indices: Tensor
    value_input_indices: Tensor
    model_inputs: RootInformationModelInputBatch
    decks: DeckBatch
    final_root_outcomes: Tensor

    @property
    def row_count(self) -> int:
        """Return the number of unique actual endpoints."""
        return int(self.decision_indices.numel())

    def validate(self, *, batch_size: int | None = None) -> None:
        """Reject misaligned model inputs, labels, or canonical row indices."""
        row_count = self.row_count
        if (
            self.decision_indices.ndim != 1
            or self.decision_indices.dtype == torch.bool
            or self.decision_indices.is_floating_point()
        ):
            raise ValueError("root-information decision indices must be integer rows")
        if (
            self.value_input_indices.shape != (row_count,)
            or self.value_input_indices.dtype == torch.bool
            or self.value_input_indices.is_floating_point()
        ):
            raise ValueError("root-information gather indices must be integer rows")
        if batch_size is not None and (
            bool((self.decision_indices < 0).any())
            or bool((self.decision_indices >= batch_size).any())
        ):
            raise ValueError("root-information decision index is outside the PPO batch")
        ordered = self.decision_indices.sort().values
        if row_count > 1 and bool((ordered[1:] == ordered[:-1]).any()):
            raise ValueError("root-information rows reuse a canonical decision")
        input_count = int(self.model_inputs.states.card_ids.shape[0])
        if (
            input_count <= 0
            or bool((self.value_input_indices < 0).any())
            or bool((self.value_input_indices >= input_count).any())
        ):
            raise ValueError("root-information gather index is outside model inputs")
        if len(self.decks) != input_count:
            raise ValueError("root-information deck rows do not align")
        if (
            self.final_root_outcomes.shape != (row_count,)
            or not self.final_root_outcomes.is_floating_point()
            or not bool(torch.isfinite(self.final_root_outcomes).all())
        ):
            raise ValueError("root-information WDL targets must be finite floats")
        allowed = self.final_root_outcomes.new_tensor((-1.0, 0.0, 1.0))
        if not bool(torch.isin(self.final_root_outcomes, allowed).all()):
            raise ValueError("root-information targets must be W/D/L")


def select_root_information_value_replay(
    replay: RootInformationValueReplayBatch | None,
    decision_indices: Sequence[int],
) -> RootInformationValueReplayBatch | None:
    """Select endpoint rows whose canonical decision is in one minibatch."""
    if replay is None:
        return None
    replay.validate()
    target_positions = {
        int(source): target for target, source in enumerate(decision_indices)
    }
    source_decisions = tuple(
        int(value)
        for value in replay.decision_indices.detach().to(device="cpu").tolist()
    )
    source_rows = tuple(
        row for row, source in enumerate(source_decisions) if source in target_positions
    )
    if not source_rows:
        return None
    indices = torch.tensor(
        source_rows,
        dtype=torch.long,
        device=replay.decision_indices.device,
    )
    source_value_indices = replay.value_input_indices.index_select(
        0,
        indices.to(device=replay.value_input_indices.device),
    )
    unique_source_inputs = tuple(
        dict.fromkeys(
            int(value)
            for value in source_value_indices.detach().to(device="cpu").tolist()
        )
    )
    input_positions = {
        source: target for target, source in enumerate(unique_source_inputs)
    }
    input_indices = torch.tensor(
        unique_source_inputs,
        dtype=torch.long,
        device=replay.decision_indices.device,
    )
    selected = RootInformationValueReplayBatch(
        decision_indices=torch.tensor(
            [target_positions[source_decisions[row]] for row in source_rows],
            dtype=replay.decision_indices.dtype,
            device=replay.decision_indices.device,
        ),
        value_input_indices=torch.tensor(
            [
                input_positions[int(value)]
                for value in source_value_indices.detach().to(device="cpu").tolist()
            ],
            dtype=replay.value_input_indices.dtype,
            device=replay.value_input_indices.device,
        ),
        model_inputs=_select_model_inputs(replay.model_inputs, input_indices),
        decks=replay.decks.select(
            input_indices.to(device=replay.decks.card_ids.device)
        ),
        final_root_outcomes=replay.final_root_outcomes.index_select(
            0,
            indices.to(device=replay.final_root_outcomes.device),
        ),
    )
    selected.validate(batch_size=len(decision_indices))
    return selected


def move_root_information_value_replay(
    replay: RootInformationValueReplayBatch | None,
    *,
    device: torch.device | str,
    non_blocking: bool = False,
) -> RootInformationValueReplayBatch | None:
    """Move endpoint model inputs and targets to one device."""
    if replay is None:
        return None
    return RootInformationValueReplayBatch(
        decision_indices=replay.decision_indices.to(
            device=device, non_blocking=non_blocking
        ),
        value_input_indices=replay.value_input_indices.to(
            device=device, non_blocking=non_blocking
        ),
        model_inputs=_move_model_inputs(
            replay.model_inputs,
            device=device,
            non_blocking=non_blocking,
        ),
        decks=replay.decks.to(device, non_blocking=non_blocking),
        final_root_outcomes=replay.final_root_outcomes.to(
            device=device, non_blocking=non_blocking
        ),
    )


def pin_root_information_value_replay(
    replay: RootInformationValueReplayBatch | None,
) -> RootInformationValueReplayBatch | None:
    """Pin CPU endpoint inputs for asynchronous accelerator transfer."""
    if replay is None:
        return None

    def pin(values: Tensor) -> Tensor:
        if values.device.type != "cpu" or values.is_pinned():
            return values
        return values.pin_memory()

    return RootInformationValueReplayBatch(
        decision_indices=pin(replay.decision_indices),
        value_input_indices=pin(replay.value_input_indices),
        model_inputs=_map_model_input_tensors(replay.model_inputs, pin),
        decks=replay.decks.pin_memory(),
        final_root_outcomes=pin(replay.final_root_outcomes),
    )


def _select_model_inputs(
    inputs: RootInformationModelInputBatch,
    indices: Tensor,
) -> RootInformationModelInputBatch:
    def select(values: Tensor) -> Tensor:
        return values.index_select(0, indices.to(device=values.device))

    return RootInformationModelInputBatch(
        states=_map_state_tensors(inputs.states, select),
        actor_relations=select(inputs.actor_relations),
        endpoints=select(inputs.endpoints),
        belief_summaries=select(inputs.belief_summaries),
    )


def _move_model_inputs(
    inputs: RootInformationModelInputBatch,
    *,
    device: torch.device | str,
    non_blocking: bool,
) -> RootInformationModelInputBatch:
    return _map_model_input_tensors(
        inputs,
        lambda values: values.to(device=device, non_blocking=non_blocking),
    )


def _map_model_input_tensors(
    inputs: RootInformationModelInputBatch,
    transform: Callable[[Tensor], Tensor],
) -> RootInformationModelInputBatch:
    return RootInformationModelInputBatch(
        states=_map_state_tensors(inputs.states, transform),
        actor_relations=transform(inputs.actor_relations),
        endpoints=transform(inputs.endpoints),
        belief_summaries=transform(inputs.belief_summaries),
    )


def _map_state_tensors(
    states: StateBatch,
    transform: Callable[[Tensor], Tensor],
) -> StateBatch:
    def optional(values: Tensor | None) -> Tensor | None:
        return None if values is None else transform(values)

    return StateBatch(
        card_ids=transform(states.card_ids),
        areas=transform(states.areas),
        owner_roles=transform(states.owner_roles),
        token_kinds=transform(states.token_kinds),
        scalars=transform(states.scalars),
        last_attack_ids=transform(states.last_attack_ids),
        padding_mask=transform(states.padding_mask),
        attachment_card_ids=optional(states.attachment_card_ids),
        attachment_parent_indices=optional(states.attachment_parent_indices),
        attachment_kinds=optional(states.attachment_kinds),
        entity_slots=optional(states.entity_slots),
    )


__all__ = [
    "RootInformationValueReplayBatch",
    "move_root_information_value_replay",
    "pin_root_information_value_replay",
    "select_root_information_value_replay",
]
