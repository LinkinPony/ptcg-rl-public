"""Verified public-only actor path for the clean stateless policy."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.model.simple_stateless import (
    SimpleSampledDecodeTrace,
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.policy_inputs import (
    SimpleStatelessActorRow,
    SimpleStatelessPolicyInputBatch,
    collate_simple_stateless_actor_rows,
)
from ptcg_rl.rl.stateless_fragment import (
    StatelessFragmentDecision,
    StatelessFragmentIdentity,
)


@dataclass(frozen=True)
class StatelessActorDecisionTrace:
    """Detached behavior evidence for one engine decision."""

    action: tuple[int, ...]
    action_logprob: float
    token_logprobs: tuple[float, ...]
    prefix_values: tuple[float, ...]
    root_value: float
    stop_sampled: bool
    accepted_action: AcceptedActionRecord | None = None
    sequence_request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the compact trace before it can enter a fragment."""
        if not self.token_logprobs:
            raise ValueError("actor decision trace cannot be empty")
        if len(self.token_logprobs) != len(self.prefix_values):
            raise ValueError("actor token log-probs and prefix values must align")
        if not math.isclose(
            self.action_logprob,
            sum(self.token_logprobs),
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("actor action log-prob differs from token sum")

    def fragment_decision(
        self,
        *,
        decision_index: int,
        actor_row: SimpleStatelessActorRow,
        known_opponent_counts: Mapping[int, int] | Counter[int],
    ) -> StatelessFragmentDecision:
        """Attach learner-only public evidence after inference has completed."""
        if any(
            int(card_id) <= 0 or int(count) <= 0
            for card_id, count in known_opponent_counts.items()
        ):
            raise ValueError("known opponent card IDs and counts must be positive")
        known = tuple(
            sorted(
                (int(card_id), int(count))
                for card_id, count in known_opponent_counts.items()
            )
        )
        return StatelessFragmentDecision(
            decision_index=decision_index,
            actor_row=actor_row,
            action=self.action,
            action_logprob=self.action_logprob,
            token_logprobs=self.token_logprobs,
            prefix_values=self.prefix_values,
            root_value=self.root_value,
            stop_sampled=self.stop_sampled,
            known_opponent_counts=known,
            public_event_delta=(
                actor_row.public_event_delta
                if self.accepted_action is not None
                else None
            ),
            accepted_action=self.accepted_action,
        )


@dataclass(frozen=True)
class StatelessActorBatchTrace:
    """One homogeneous sampled actor batch under an immutable behavior."""

    behavior_policy_version: int
    behavior_policy_fingerprint: str
    input_contract_fingerprint: str
    decisions: tuple[StatelessActorDecisionTrace, ...]


class StatelessActorPolicy(Protocol):
    """Minimal local/remote actor contract consumed by engine collection."""

    identity: StatelessFragmentIdentity

    def sample(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> StatelessActorBatchTrace:
        """Sample legal actions with exact behavior evidence."""


class SimpleStatelessActorPolicy:
    """Run the clean model without recurrent state or privileged labels."""

    def __init__(
        self,
        model: SimpleStatelessPolicyValueNet,
        *,
        identity: StatelessFragmentIdentity,
        device: torch.device | str,
        verify_model_state: bool = True,
    ) -> None:
        """Bind one immutable behavior publication to one inference device."""
        self.device = torch.device(device)
        self.identity = identity
        self.model = model.to(self.device).eval()
        self._uses_pure_bfloat16 = model_uses_pure_bfloat16(self.model)
        self._verify_contract()
        if verify_model_state:
            actual = canonical_model_state_fingerprint(self.model)
            if actual != identity.behavior_policy_fingerprint:
                raise ValueError("actor model state differs from behavior identity")

    @property
    def uses_bfloat16_autocast(self) -> bool:
        """Return whether this actor uses the H200 BF16 compute path."""
        return self.device.type == "cuda" and not self._uses_pure_bfloat16

    @property
    def uses_pure_bfloat16(self) -> bool:
        """Return whether all resident floating model state already uses BF16."""
        return self._uses_pure_bfloat16

    def sample(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> StatelessActorBatchTrace:
        """Sample legal actions and persist their exact behavior evidence."""
        if any(row.max_count == 0 for row in rows):
            raise ValueError("zero-card forced prompts must bypass policy sampling")
        batch = self._collate(rows)
        self._validate_generator(generator)
        routes = resolve_simple_exact_routes(
            batch.deck_signatures,
            self.model.config,
            device=self.device,
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            state = self.model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            option_embeddings = self.model.encode_legal_options(
                state,
                batch.options,
                route_plan=routes,
            )
            sampled = self.model.heads.sample_decode_with_trace(
                state.policy,
                state.opponent_belief,
                option_embeddings,
                batch.options,
                route_plan=routes,
                temperature=temperature,
                generator=generator,
            )
            root_values = self.model.heads.root_value(
                state.value,
                state.opponent_belief,
                route_plan=routes,
            )
        decisions = _detach_decisions(
            sampled=sampled,
            root_values=root_values,
        )
        return StatelessActorBatchTrace(
            behavior_policy_version=self.identity.behavior_policy_version,
            behavior_policy_fingerprint=self.identity.behavior_policy_fingerprint,
            input_contract_fingerprint=self.identity.input_contract_fingerprint,
            decisions=decisions,
        )

    def root_values(
        self,
        rows: Sequence[SimpleStatelessActorRow],
    ) -> tuple[float, ...]:
        """Evaluate same-behavior root values for fragment bootstrapping."""
        batch = self._collate(rows)
        routes = resolve_simple_exact_routes(
            batch.deck_signatures,
            self.model.config,
            device=self.device,
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            state = self.model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            values = self.model.heads.root_value(
                state.value,
                state.opponent_belief,
                route_plan=routes,
            )
        return tuple(float(value) for value in values.float().cpu().tolist())

    def _collate(
        self,
        rows: Sequence[SimpleStatelessActorRow],
    ) -> SimpleStatelessPolicyInputBatch:
        if not rows:
            raise ValueError("actor inference requires at least one row")
        if any(
            row.input_contract_fingerprint != self.identity.input_contract_fingerprint
            for row in rows
        ):
            raise ValueError("actor row differs from behavior input contract")
        if any(
            row.catalog_fingerprint != self.identity.public_deck_catalog_fingerprint
            for row in rows
        ):
            raise ValueError("actor row differs from behavior public catalog")
        return collate_simple_stateless_actor_rows(rows, device=self.device)

    def _verify_contract(self) -> None:
        config = self.model.config
        if model_config_fingerprint(config) != self.identity.model_config_fingerprint:
            raise ValueError("actor model config differs from behavior identity")
        if config.resolved_registry_sha256 != self.identity.exact_registry_fingerprint:
            raise ValueError("actor exact registry differs from behavior identity")
        if (
            config.public_deck_catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("actor catalog differs from behavior identity")

    def _validate_generator(self, generator: torch.Generator | None) -> None:
        if generator is None:
            return
        generator_device = torch.device(generator.device)
        if generator_device.type != self.device.type:
            raise ValueError("sampling generator must use the actor device type")


def _detach_decisions(
    *,
    sampled: SimpleSampledDecodeTrace,
    root_values: Tensor,
) -> tuple[StatelessActorDecisionTrace, ...]:
    """Build object-boundary traces after one packed device-to-host transfer."""
    choice_width = int(sampled.actions.choice_indices.shape[1])
    token_width = int(sampled.token_logprobs.shape[1])
    host_rows: list[list[float]] = (
        torch.cat(
            (
                sampled.actions.choice_indices.float(),
                sampled.actions.lengths.float().unsqueeze(1),
                sampled.action_logprobs.float().unsqueeze(1),
                sampled.token_logprobs.float(),
                sampled.token_mask.float(),
                sampled.prefix_values.float(),
                root_values.float().unsqueeze(1),
                sampled.stop_sampled.float().unsqueeze(1),
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )
    length_column = choice_width
    action_logprob_column = length_column + 1
    token_logprob_start = action_logprob_column + 1
    token_mask_start = token_logprob_start + token_width
    prefix_start = token_mask_start + token_width
    root_column = prefix_start + token_width
    stop_column = root_column + 1
    decisions: list[StatelessActorDecisionTrace] = []
    for host_row in host_rows:
        action_length = int(host_row[length_column])
        action = tuple(int(choice) for choice in host_row[:action_length])
        row_token_logs: list[float] = []
        row_prefixes: list[float] = []
        for token_index in range(token_width):
            if not bool(host_row[token_mask_start + token_index]):
                continue
            row_token_logs.append(float(host_row[token_logprob_start + token_index]))
            row_prefixes.append(float(host_row[prefix_start + token_index]))
        decisions.append(
            StatelessActorDecisionTrace(
                action=action,
                action_logprob=float(host_row[action_logprob_column]),
                token_logprobs=tuple(row_token_logs),
                prefix_values=tuple(row_prefixes),
                root_value=float(host_row[root_column]),
                stop_sampled=bool(host_row[stop_column]),
            )
        )
    return tuple(decisions)


def model_uses_pure_bfloat16(model: nn.Module) -> bool:
    """Return whether every resident floating state tensor uses BF16.

    A pure-BF16 frozen model must bypass CUDA autocast so its per-context
    weight-cast cache is not rebuilt for every inference batch. Models without
    inspectable state, FP32 models, and mixed-precision models retain the
    existing autocast behavior.
    """
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return False
    floating_state = tuple(
        tensor
        for tensor in state_dict().values()
        if isinstance(tensor, Tensor) and tensor.is_floating_point()
    )
    return bool(floating_state) and all(
        tensor.dtype == torch.bfloat16 for tensor in floating_state
    )


__all__ = [
    "model_uses_pure_bfloat16",
    "StatelessActorPolicy",
    "SimpleStatelessActorPolicy",
    "StatelessActorBatchTrace",
    "StatelessActorDecisionTrace",
]
