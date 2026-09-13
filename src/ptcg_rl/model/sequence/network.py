"""Post-backbone EVENT/STATE/ACTION temporal policy module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.cards.card_encoder import CardEncoder
from ptcg_rl.context.public_event_arrays import PublicEventBatch
from ptcg_rl.model.recurrent import PublicEventEncoder, RecurrentPolicyConfig
from ptcg_rl.model.sequence.action import AcceptedActionBatch, AcceptedActionEncoder
from ptcg_rl.model.sequence.config import GeneralistSequenceConfig
from ptcg_rl.model.sequence.core import (
    ACTION_TOKEN,
    EVENT_TOKEN,
    STATE_TOKEN,
    GeneralistTemporalCore,
    TemporalKvCache,
)
from ptcg_rl.model.sequence.state import TemporalStateEncoder
from ptcg_rl.model.simple_stateless.backbone import SimpleStatelessBackboneOutput
from ptcg_rl.model.simple_stateless.layers import initialize_simple_stateless_module


@dataclass(frozen=True)
class TemporalPreparedDecision:
    """One provisional EVENT/STATE append and its current temporal context."""

    context: Tensor
    provisional_cache: TemporalKvCache
    block_index: int


class GeneralistSequenceModule(nn.Module):
    """Direct-policy temporal residual over the strong snapshot backbone."""

    def __init__(self, config: GeneralistSequenceConfig) -> None:
        """Build deterministic local encoders, temporal core, and inert outputs."""
        super().__init__()
        self.config = config
        event_config = RecurrentPolicyConfig(
            hidden_size=config.d_model,
            event_hidden_size=config.d_model,
            num_layers=1,
            dropout=0.0,
            serial_hash_buckets=config.serial_hash_buckets,
            attack_hash_buckets=config.attack_hash_buckets,
        )
        self.event_encoder = PublicEventEncoder(config.d_model, event_config)
        self.state_encoder = TemporalStateEncoder(
            d_model=config.d_model,
            num_heads=config.attention_heads,
        )
        self.action_encoder = AcceptedActionEncoder(
            d_model=config.d_model,
            attack_hash_buckets=config.attack_hash_buckets,
        )
        self.core = GeneralistTemporalCore(config)
        self.policy_residual = nn.Linear(config.d_model, config.d_model)
        self.value_residual = nn.Linear(config.d_model, config.d_model)
        self.belief_residual = nn.Linear(config.d_model, config.d_model)
        initialize_simple_stateless_module(self)
        self.zero_outputs()

    def zero_outputs(self) -> None:
        """Make a topology migration exactly preserve incumbent outputs."""
        for projection in (
            self.policy_residual,
            self.value_residual,
            self.belief_residual,
        ):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def assert_output_inert(self) -> None:
        """Fail when any temporal output projection changes migrated behavior."""
        for name, parameter in self.inert_output_named_parameters():
            if parameter.is_meta:
                raise ValueError("cannot validate temporal outputs on meta device")
            if torch.count_nonzero(parameter).item():
                raise ValueError(f"temporal output parameter is not inert: {name}")

    def inert_output_named_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return all zero-output parameters for migration auditing."""
        return tuple(
            (f"{module_name}.{parameter_name}", parameter)
            for module_name, module in (
                ("policy_residual", self.policy_residual),
                ("value_residual", self.value_residual),
                ("belief_residual", self.belief_residual),
            )
            for parameter_name, parameter in module.named_parameters()
        )

    def replay(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        actions: AcceptedActionBatch,
        *,
        sequence_offsets: tuple[int, ...],
        block_indices: Tensor,
        card_encoder: CardEncoder,
    ) -> Tensor:
        """Rebuild current-weight temporal contexts from complete raw blocks."""
        block_count = snapshots.packed.batch_size
        if (
            events.batch_size != block_count
            or actions.batch_size != block_count
            or block_indices.shape != (block_count,)
        ):
            raise ValueError("temporal replay raw block rows are misaligned")
        event_tokens = self.event_encoder(events, card_encoder=card_encoder)
        state_tokens = self.state_encoder(snapshots)
        action_tokens = self.action_encoder(actions, card_encoder=card_encoder)
        tokens = torch.stack((event_tokens, state_tokens, action_tokens), dim=1)
        packed_tokens = tokens.reshape(block_count * 3, self.config.d_model)
        device = packed_tokens.device
        token_types = torch.arange(
            EVENT_TOKEN,
            ACTION_TOKEN + 1,
            dtype=torch.long,
            device=device,
        ).repeat(block_count)
        packed_blocks = block_indices.repeat_interleave(3)
        temporal_offsets = tuple(offset * 3 for offset in sequence_offsets)
        encoded = self.core(
            packed_tokens,
            token_types=token_types,
            block_indices=packed_blocks,
            sequence_offsets=temporal_offsets,
        )
        return cast(
            Tensor,
            encoded.reshape(block_count, 3, self.config.d_model)[:, STATE_TOKEN],
        )

    def prepare_incremental(
        self,
        snapshot: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        *,
        block_index: int,
        card_encoder: CardEncoder,
        cache: TemporalKvCache | None,
    ) -> TemporalPreparedDecision:
        """Fork a provisional cache containing current EVENT and STATE."""
        if snapshot.packed.batch_size != 1 or events.batch_size != 1:
            raise ValueError("incremental temporal prepare requires one sequence row")
        event_token = self.event_encoder(events, card_encoder=card_encoder)
        state_token = self.state_encoder(snapshot)
        tokens = torch.cat((event_token, state_token), dim=0)
        token_types = torch.arange(
            EVENT_TOKEN,
            STATE_TOKEN + 1,
            dtype=torch.long,
            device=tokens.device,
        )
        blocks = torch.full(
            (2,),
            int(block_index),
            dtype=torch.long,
            device=tokens.device,
        )
        encoded, provisional = self.core.append(
            tokens,
            token_types=token_types,
            block_indices=blocks,
            cache=cache,
        )
        return TemporalPreparedDecision(
            context=encoded[STATE_TOKEN : STATE_TOKEN + 1],
            provisional_cache=provisional,
            block_index=int(block_index),
        )

    def prepare_incremental_many(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        *,
        block_indices: Sequence[int],
        card_encoder: CardEncoder,
        caches: Sequence[TemporalKvCache | None],
    ) -> tuple[TemporalPreparedDecision, ...]:
        """Fork batched provisional EVENT/STATE caches for independent games."""
        batch_size = snapshots.packed.batch_size
        if (
            batch_size <= 0
            or events.batch_size != batch_size
            or len(block_indices) != batch_size
            or len(caches) != batch_size
        ):
            raise ValueError("batched incremental temporal inputs are misaligned")
        event_tokens = self.event_encoder(events, card_encoder=card_encoder)
        state_tokens = self.state_encoder(snapshots)
        tokens = torch.stack((event_tokens, state_tokens), dim=1)
        device = tokens.device
        token_types = torch.arange(
            EVENT_TOKEN,
            STATE_TOKEN + 1,
            dtype=torch.long,
            device=device,
        ).expand(batch_size, -1)
        blocks = (
            torch.as_tensor(
                tuple(int(index) for index in block_indices),
                dtype=torch.long,
                device=device,
            )
            .unsqueeze(1)
            .expand(-1, 2)
        )
        encoded, provisional = self.core.append_many(
            tokens,
            token_types=token_types,
            block_indices=blocks,
            caches=caches,
            block_indices_host=tuple(
                (int(index), int(index)) for index in block_indices
            ),
        )
        return tuple(
            TemporalPreparedDecision(
                context=encoded[row, STATE_TOKEN : STATE_TOKEN + 1],
                provisional_cache=provisional[row],
                block_index=int(block_indices[row]),
            )
            for row in range(batch_size)
        )

    def commit_incremental(
        self,
        prepared: TemporalPreparedDecision,
        actions: AcceptedActionBatch,
        *,
        card_encoder: CardEncoder,
    ) -> TemporalKvCache:
        """Append the accepted ACTION token and return a committed cache."""
        if actions.batch_size != 1:
            raise ValueError("incremental temporal commit requires one action")
        token = self.action_encoder(actions, card_encoder=card_encoder)
        token_types = torch.tensor(
            (ACTION_TOKEN,),
            dtype=torch.long,
            device=token.device,
        )
        blocks = torch.tensor(
            (prepared.block_index,),
            dtype=torch.long,
            device=token.device,
        )
        _encoded, committed = self.core.append(
            token,
            token_types=token_types,
            block_indices=blocks,
            cache=prepared.provisional_cache,
        )
        return committed.detach()

    def commit_incremental_many(
        self,
        prepared: Sequence[TemporalPreparedDecision],
        actions: AcceptedActionBatch,
        *,
        card_encoder: CardEncoder,
    ) -> tuple[TemporalKvCache, ...]:
        """Encode and append accepted ACTION tokens for a prepared batch."""
        batch_size = len(prepared)
        if batch_size <= 0 or actions.batch_size != batch_size:
            raise ValueError("batched temporal commit inputs are misaligned")
        action_tokens = self.action_encoder(actions, card_encoder=card_encoder)
        tokens = action_tokens.unsqueeze(1)
        device = tokens.device
        token_types = torch.full(
            (batch_size, 1),
            ACTION_TOKEN,
            dtype=torch.long,
            device=device,
        )
        blocks = torch.as_tensor(
            tuple(item.block_index for item in prepared),
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        _encoded, committed = self.core.append_many(
            tokens,
            token_types=token_types,
            block_indices=blocks,
            caches=tuple(item.provisional_cache for item in prepared),
            block_indices_host=tuple((int(item.block_index),) for item in prepared),
        )
        return tuple(cache.detach() for cache in committed)

    def condition(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        contexts: Tensor,
    ) -> SimpleStatelessBackboneOutput:
        """Inject independent zero-initialized policy/value/belief residuals."""
        expected = (snapshots.packed.batch_size, self.config.d_model)
        if tuple(contexts.shape) != expected:
            raise ValueError(f"temporal contexts must have shape {expected}")
        return replace(
            snapshots,
            policy=snapshots.policy + self.policy_residual(contexts),
            value=snapshots.value + self.value_residual(contexts),
            opponent_belief=(
                snapshots.opponent_belief + self.belief_residual(contexts)
            ),
            temporal_context=contexts,
        )


__all__ = [
    "GeneralistSequenceModule",
    "TemporalPreparedDecision",
]
