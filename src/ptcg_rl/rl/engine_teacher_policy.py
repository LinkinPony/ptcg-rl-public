"""Actor-side policy/value adapter for complete-action engine teaching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import Tensor

from ptcg_rl.actions.selection import normalize_action_order
from ptcg_rl.agent.search.continuation_types import ContinuationValueRequest
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch


class DecodePolicy(Protocol):
    """Minimal local or remote inference surface used by the adapter."""

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Decode complete selections and return root value estimates."""

    def predict_values(
        self,
        states: StateBatch,
        decks: DeckBatch,
    ) -> Tensor:
        """Predict critic values without entering autoregressive decoding."""


class EngineTeacherPolicyAdapter:
    """Expose deterministic proposals and actor-perspective batched values.

    This adapter deliberately uses the same rollout inference policy and deck
    binding as behavior collection. Deterministic proposal calls do not create
    PPO evidence; they only order bounded engine search.
    """

    def __init__(
        self,
        policy: DecodePolicy,
        *,
        own_deck: Sequence[int],
        device: torch.device | str | None,
        value_batch_size: int = 256,
    ) -> None:
        if value_batch_size <= 0:
            raise ValueError("value_batch_size must be positive")
        self._policy = policy
        self._own_deck = canonicalize_deck(own_deck)
        self._device = device
        self._value_batch_size = int(value_batch_size)
        self._inference_version: int | None = None

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return one complete greedy action for proposal ordering."""
        actions, _ = self._decode((observation,))
        select = _field(observation, "select")
        return normalize_action_order(select, actions[0])

    def rank_actions(
        self,
        observation: Any,
        *,
        top_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the available greedy proposal for bounded beam seeding.

        The rollout RPC intentionally does not expose model-specific action
        scoring. Exact prompt spaces require no proposal; larger spaces add
        stratified legal candidates around this deterministic seed.
        """
        if top_k <= 0:
            return ()
        return (self.select_action(observation),)

    def values(
        self,
        requests: Sequence[ContinuationValueRequest],
    ) -> tuple[float, ...]:
        """Return legal current-actor values for explicit sampled deck routes."""
        values: list[float] = []
        for offset in range(0, len(requests), self._value_batch_size):
            chunk = tuple(requests[offset : offset + self._value_batch_size])
            inputs = tuple(
                build_canonical_policy_input(item.observation, require_options=False)
                for item in chunk
            )
            if any(item is None for item in inputs):
                raise ValueError("engine teacher leaf could not be tensorized")
            required = tuple(item for item in inputs if item is not None)
            for request in chunk:
                perspective = _int_field(
                    _field(request.observation, "current"),
                    "yourIndex",
                    -1,
                )
                if perspective != request.perspective_player_index:
                    raise ValueError("engine teacher leaf perspective is misaligned")
            states = collate_state_tokens(
                [item.state for item in required],
                device=self._device,
            )
            decks = DeckBatch.from_decks(
                tuple(canonicalize_deck(item.deck) for item in chunk),
                device=self._device,
            )
            raw_values = self._policy.predict_values(states, decks)
            self._verify_inference_version()
            if raw_values.ndim != 1 or raw_values.numel() != len(chunk):
                raise RuntimeError("engine teacher value inference was misaligned")
            values.extend(
                float(value) for value in raw_values.detach().cpu().tolist()
            )
        return tuple(values)

    def _decode(
        self,
        observations: Sequence[Any],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
        if not observations:
            return (), ()
        inputs = tuple(
            build_canonical_policy_input(observation, require_options=True)
            for observation in observations
        )
        if any(item is None for item in inputs):
            raise ValueError("engine teacher observation could not be tensorized")
        required = tuple(item for item in inputs if item is not None)
        states = collate_state_tokens(
            [item.state for item in required],
            device=self._device,
        )
        options = collate_encoded_options(
            [item.options for item in required],
            min_counts=[item.min_count for item in required],
            max_counts=[item.max_count for item in required],
            device=self._device,
        )
        decks = DeckBatch.from_decks(
            (self._own_deck,) * len(required),
            device=self._device,
        )
        actions, _logprobs, raw_values = self._policy.sample_decode(
            states,
            options,
            decks,
            temperature=0.0,
        )
        self._verify_inference_version()
        if len(actions) != len(required) or raw_values.numel() != len(required):
            raise RuntimeError("engine teacher inference returned a misaligned batch")
        values = tuple(float(value) for value in raw_values.detach().cpu().tolist())
        return tuple(tuple(int(index) for index in action) for action in actions), values

    def _verify_inference_version(self) -> None:
        """Reject one target whose proposal/value calls straddle a reload."""
        version = getattr(self._policy, "last_response_policy_version", None)
        if version is None:
            version = getattr(self._policy, "policy_version", None)
        if version is None:
            return
        observed = int(version)
        if self._inference_version is None:
            self._inference_version = observed
        elif observed != self._inference_version:
            raise RuntimeError("engine teacher inference version changed")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


__all__ = ["DecodePolicy", "EngineTeacherPolicyAdapter"]
