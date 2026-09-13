"""Reusable encoded-state caching and batched leaf-value inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.agent.search.policy_inputs import (
    CanonicalPolicyInput,
    build_canonical_policy_input,
)


@dataclass(frozen=True)
class CachedPolicyObservation:
    """One strong-referenced observation and its reusable device tensors."""

    observation: Any
    conditioned_state: Any
    options: Any | None
    deck_signature: str | None
    model_version: str


class EncodedObservationCache:
    """Cache state-encoder outputs within exactly one runtime callback."""

    def __init__(self, *, model: Any, device: Any) -> None:
        self._model = model
        self._device = device
        self._enabled = False
        self._entries: dict[tuple[int, str | None, str], CachedPolicyObservation] = {}
        self._decks: Any | None = None
        self._deck_signature: str | None = None
        self._model_version = ""

    @property
    def enabled(self) -> bool:
        """Return whether callback-local reuse is active."""
        return self._enabled

    def configure(self, *, enabled: bool) -> None:
        """Set reuse mode and release tensors from any earlier callback."""
        self._enabled = enabled
        self.clear()

    def clear(self) -> None:
        """Release strong observation references and cached device tensors."""
        self._entries.clear()

    def bind_context(
        self,
        *,
        decks: Any | None,
        deck_signature: str | None,
        model_version: str,
    ) -> None:
        """Bind cache identity to one immutable model and own deck."""
        if (
            deck_signature != self._deck_signature
            or model_version != self._model_version
        ):
            self.clear()
        self._decks = decks
        self._deck_signature = deck_signature
        self._model_version = model_version

    def encode(
        self,
        observation: Any,
        *,
        require_options: bool,
    ) -> CachedPolicyObservation | None:
        """Return one cached or newly encoded observation."""
        from ptcg_rl.model import collate_encoded_options, collate_state_tokens

        cache_key = (id(observation), self._deck_signature, self._model_version)
        cached = self._entries.get(cache_key)
        if (
            cached is not None
            and cached.observation is observation
            and (not require_options or cached.options is not None)
        ):
            return cached
        policy_input = build_canonical_policy_input(
            observation,
            require_options=require_options,
        )
        if policy_input is None:
            return None
        states = collate_state_tokens([policy_input.state], device=self._device)
        conditioned_state = self._model.encode_conditioned_state(
            states,
            self._decks,
        )
        options = None
        if require_options:
            options = collate_encoded_options(
                [policy_input.options],
                min_counts=[policy_input.min_count],
                max_counts=[policy_input.max_count],
                device=self._device,
            )
        cached = CachedPolicyObservation(
            observation=observation,
            conditioned_state=conditioned_state,
            options=options,
            deck_signature=self._deck_signature,
            model_version=self._model_version,
        )
        self._entries[cache_key] = cached
        return cached


def batched_observation_values(
    observations: Sequence[Any],
    *,
    root_player_index: int,
    model: Any,
    device: Any,
    torch_module: Any,
    terminal_value: Callable[[Any, int], float | None],
    current_player_index: Callable[[Any, int], int],
    decks: Any | None,
) -> tuple[float, ...]:
    """Run all nonterminal observations through one state-encoder batch."""
    from ptcg_rl.model import collate_state_tokens

    results: list[float | None] = [None] * len(observations)
    pending_indices: list[int] = []
    pending_inputs: list[CanonicalPolicyInput] = []
    for index, observation in enumerate(observations):
        terminal = terminal_value(observation, root_player_index)
        if terminal is not None:
            results[index] = terminal
            continue
        policy_input = build_canonical_policy_input(
            observation,
            require_options=False,
        )
        if policy_input is None:
            raise ValueError("value observation could not be tensorized")
        pending_indices.append(index)
        pending_inputs.append(policy_input)
    if pending_inputs:
        states = collate_state_tokens(
            [policy_input.state for policy_input in pending_inputs],
            device=device,
        )
        with torch_module.inference_mode():
            conditioned = model.encode_conditioned_state(states, decks)
            raw_values = (
                model.root_values_from_conditioned(conditioned).detach().cpu().tolist()
            )
        for index, raw_value in zip(pending_indices, raw_values, strict=True):
            player_index = current_player_index(
                observations[index],
                root_player_index,
            )
            value = float(raw_value)
            results[index] = value if player_index == root_player_index else -value
    if any(value is None for value in results):
        raise RuntimeError("batched value evaluation left an unset result")
    return tuple(float(value) for value in results if value is not None)


def batched_root_information_values(
    observations: Sequence[Any],
    *,
    actor_relations: Sequence[int],
    semantic_endpoints: Sequence[int],
    root_player_index: int,
    belief_summary_dim: int,
    model: Any,
    device: Any,
    torch_module: Any,
    decks: Any,
    current_player_index: Callable[[Any, int], int],
) -> tuple[float, ...]:
    """Evaluate handoffs with the adapter and same-seat leaves with the critic."""
    from ptcg_rl.agent.search.root_information_producer import (
        root_information_belief_summary,
    )
    from ptcg_rl.context import context_features_from_observation
    from ptcg_rl.engine.compact_consequence import SemanticEndpoint
    from ptcg_rl.model import collate_state_tokens

    row_count = len(observations)
    if len(actor_relations) != row_count or len(semantic_endpoints) != row_count:
        raise ValueError("root-information metadata must align with observations")
    if belief_summary_dim < 0:
        raise ValueError("belief_summary_dim must be non-negative")
    inputs: list[CanonicalPolicyInput] = []
    beliefs: list[tuple[float, ...]] = []
    for observation in observations:
        if current_player_index(observation, root_player_index) != root_player_index:
            raise ValueError("semantic leaf is not projected to the root player")
        policy_input = build_canonical_policy_input(
            observation,
            require_options=False,
        )
        if policy_input is None:
            raise ValueError("semantic leaf could not be tensorized")
        inputs.append(policy_input)
        beliefs.append(
            root_information_belief_summary(
                context_features_from_observation(observation),
                width=belief_summary_dim,
            )
        )
    if not inputs:
        return ()
    states = collate_state_tokens(
        [policy_input.state for policy_input in inputs],
        device=device,
    )
    relations = torch_module.tensor(
        actor_relations,
        dtype=torch_module.long,
        device=device,
    )
    endpoints = torch_module.tensor(
        semantic_endpoints,
        dtype=torch_module.long,
        device=device,
    )
    belief_summaries = torch_module.tensor(
        beliefs,
        dtype=torch_module.float32,
        device=device,
    ).reshape(row_count, belief_summary_dim)
    with torch_module.inference_mode():
        conditioned = model.encode_conditioned_state(states, decks)
        base_values = model.root_values_from_conditioned(conditioned)
        adapter_values = model.root_information_values_from_conditioned(
            conditioned,
            actor_relations=relations,
            endpoints=endpoints,
            belief_summaries=belief_summaries,
        )
        values = torch_module.where(
            endpoints.eq(int(SemanticEndpoint.TURN_HANDOFF)),
            adapter_values,
            base_values,
        )
    if tuple(values.shape) != (row_count,):
        raise ValueError("root adapter returned the wrong value shape")
    return tuple(float(value) for value in values.detach().cpu().tolist())


__all__ = [
    "EncodedObservationCache",
    "batched_observation_values",
    "batched_root_information_values",
]
