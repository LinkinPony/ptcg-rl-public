"""Complete clean policy module without legacy or retired heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from ptcg_rl.context.public_event_arrays import PublicEventBatch
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.sequence.action import AcceptedActionBatch
from ptcg_rl.model.sequence.core import TemporalKvCache
from ptcg_rl.model.sequence.network import (
    GeneralistSequenceModule,
    TemporalPreparedDecision,
)
from ptcg_rl.model.simple_stateless.backbone import (
    SimpleStatelessBackbone,
    SimpleStatelessBackboneOutput,
)
from ptcg_rl.model.simple_stateless.belief import (
    OpponentCardBeliefHead,
    PublicBeliefSummaryBatch,
)
from ptcg_rl.model.simple_stateless.config import (
    SimpleStatelessModelConfig,
    uses_family_private_topology,
    uses_generalist_sequence,
)
from ptcg_rl.model.simple_stateless.family_private import (
    FamilyPrivateStrategyBank,
)
from ptcg_rl.model.simple_stateless.heads import SimpleStatelessPolicyValueHeads
from ptcg_rl.model.simple_stateless.layers import initialize_simple_stateless_module
from ptcg_rl.model.simple_stateless.packed import (
    PackedTokenBatch,
    unpack_after_prefix,
)
from ptcg_rl.model.simple_stateless.routing import SimpleExactRoutePlan
from ptcg_rl.model.state_encoder import StateBatch


class SimpleStatelessPolicyValueNet(nn.Module):
    """New-lineage model containing only the declared stateless components."""

    def __init__(
        self,
        config: SimpleStatelessModelConfig,
        *,
        static_features: npt.NDArray[np.float32] | Tensor | None = None,
        load_static_features: bool = True,
        initialize: bool = True,
    ) -> None:
        """Build the shared backbone and clean policy/value output heads."""
        super().__init__()
        self.config = config
        self.backbone = SimpleStatelessBackbone(
            config,
            static_features=static_features,
            load_static_features=load_static_features,
            initialize=initialize,
        )
        self.heads = SimpleStatelessPolicyValueHeads(config)
        self.belief_head = OpponentCardBeliefHead(d_model=config.d_model)
        self.sequence: GeneralistSequenceModule | None
        if uses_generalist_sequence(config):
            if config.sequence is None:
                raise ValueError("generalist sequence model is missing its config")
            self.sequence = GeneralistSequenceModule(config.sequence)
        else:
            self.sequence = None
        if initialize:
            initialize_simple_stateless_module(self.belief_head)

    @property
    def supports_bfloat16_rollout_inductor(self) -> bool:
        """Return whether this Torch build can enable rollout Inductor."""
        return self.backbone.trunk.supports_bfloat16_rollout_inductor

    @property
    def uses_bfloat16_rollout_inductor(self) -> bool:
        """Return whether every eligible BF16 rollout block uses Inductor."""
        family_private = self.backbone.family_private
        return self.backbone.trunk.uses_bfloat16_rollout_inductor and (
            family_private is None
            or family_private.uses_bfloat16_rollout_inductor
        )

    def enable_bfloat16_rollout_inductor(self) -> None:
        """Enable strict tensor-only Inductor segments on a rollout shadow."""
        adapters = self.backbone.v2_adapters
        self.backbone.trunk.enable_bfloat16_rollout_inductor(
            layer_boundary_callback_layers=(
                ()
                if adapters is None
                else tuple(
                    layer
                    for layer in adapters.stage_layers
                    if layer <= len(self.backbone.trunk.layers)
                )
            ),
        )
        family_private = self.backbone.family_private
        if family_private is not None:
            try:
                family_private.enable_bfloat16_rollout_inductor()
            except Exception:
                self.backbone.trunk.disable_bfloat16_rollout_inductor()
                raise

    def disable_bfloat16_rollout_inductor(self) -> None:
        """Explicitly return every compiled rollout block to eager execution."""
        self.backbone.trunk.disable_bfloat16_rollout_inductor()
        family_private = self.backbone.family_private
        if family_private is not None:
            family_private.disable_bfloat16_rollout_inductor()

    @property
    def uses_bfloat16_learner_inductor(self) -> bool:
        """Return whether every eligible learner block uses Inductor."""
        family_private = self.backbone.family_private
        return self.backbone.trunk.uses_bfloat16_learner_inductor and (
            family_private is None
            or family_private.uses_bfloat16_learner_inductor
        )

    def enable_bfloat16_learner_inductor(self) -> None:
        """Enable differentiable BF16 segments on the FP32 learner master."""
        adapters = self.backbone.v2_adapters
        self.backbone.trunk.enable_bfloat16_learner_inductor(
            layer_boundary_callback_layers=(
                ()
                if adapters is None
                else tuple(
                    layer
                    for layer in adapters.stage_layers
                    if layer <= len(self.backbone.trunk.layers)
                )
            ),
        )
        family_private = self.backbone.family_private
        if family_private is not None:
            try:
                family_private.enable_bfloat16_learner_inductor()
            except Exception:
                self.backbone.trunk.disable_bfloat16_learner_inductor()
                raise

    def disable_bfloat16_learner_inductor(self) -> None:
        """Explicitly return every compiled learner block to eager execution."""
        self.backbone.trunk.disable_bfloat16_learner_inductor()
        family_private = self.backbone.family_private
        if family_private is not None:
            family_private.disable_bfloat16_learner_inductor()

    def encode_public_state(
        self,
        *,
        public_context: Tensor,
        unique_deck_card_ids: Tensor,
        deck_counts: Tensor,
        deck_valid_mask: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        entity_tokens: PackedTokenBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Run the complete shared stateless Transformer."""
        return cast(
            SimpleStatelessBackboneOutput,
            self.backbone(
                public_context=public_context,
                unique_deck_card_ids=unique_deck_card_ids,
                deck_counts=deck_counts,
                deck_valid_mask=deck_valid_mask,
                belief_summary=belief_summary,
                entity_tokens=entity_tokens,
                route_plan=route_plan,
                allow_unrouted_rows=allow_unrouted_rows,
            ),
        )

    def encode_legal_options(
        self,
        state: SimpleStatelessBackboneOutput,
        options: OptionBatch,
        *,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> Tensor:
        """Encode options against contextualized public entity rows."""
        special_count = 5 + self.config.scratch_tokens
        entities, padding_mask = unpack_after_prefix(
            state.packed,
            prefix_tokens=special_count,
        )
        return self.heads.encode_options(
            entities,
            options,
            card_encoder=self.backbone.input_encoder.card_encoder,
            temporal_context=state.temporal_context,
            entity_valid_mask=~padding_mask,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def route_private_banks(self) -> tuple[tuple[str, nn.ModuleDict], ...]:
        """Enumerate every route-keyed bank in the complete model."""
        banks = [
            (f"heads.{name}", bank)
            for name, bank in self.heads.route_private_banks()
        ]
        adapters = self.backbone.v2_adapters
        if adapters is not None:
            banks.extend(
                (f"backbone.v2_adapters.{name}", bank)
                for name, bank in adapters.route_private_banks()
            )
        return tuple(banks)

    def family_private_banks(self) -> tuple[tuple[str, nn.ModuleDict], ...]:
        """Enumerate shareable family banks separately from exact strategies."""
        family_private = self.backbone.family_private
        if family_private is None:
            return ()
        return (("backbone.family_private.tails", family_private.tails),)

    def v2_inert_output_named_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return output gates that make all V2-only additions initially inert."""
        parameters: list[tuple[str, nn.Parameter]] = []
        adapters = self.backbone.v2_adapters
        if adapters is not None:
            parameters.extend(
                (f"backbone.v2_adapters.{name}", parameter)
                for name, parameter in adapters.inert_output_named_parameters()
            )
        family_private = self.backbone.family_private
        if family_private is not None:
            parameters.extend(
                (f"backbone.family_private.{name}", parameter)
                for name, parameter in (
                    family_private.inert_output_named_parameters()
                )
            )
        parameters.extend(
            (f"heads.{name}", parameter)
            for name, parameter in self.heads.inert_option_named_parameters()
        )
        if self.sequence is not None:
            parameters.extend(
                (f"sequence.{name}", parameter)
                for name, parameter in self.sequence.inert_output_named_parameters()
            )
        return tuple(sorted(parameters, key=lambda item: item[0]))

    def assert_v2_additions_output_inert(self) -> None:
        """Fail when a newly introduced V2 path changes migrated V1 outputs."""
        self.assert_named_outputs_inert(
            tuple(name for name, _ in self.v2_inert_output_named_parameters())
        )

    def assert_family_private_additions_output_inert(self) -> None:
        """Fail when a newly appended v3 block is not an exact identity."""
        if not uses_family_private_topology(self.config):
            raise ValueError("model does not use family-private topology")
        family_private = self.backbone.family_private
        if not isinstance(family_private, FamilyPrivateStrategyBank):
            raise RuntimeError("family-private model has no strategy bank")
        self.assert_named_outputs_inert(
            tuple(
                f"backbone.family_private.{name}"
                for name, _parameter in (
                    family_private.inert_output_named_parameters()
                )
            )
        )

    def assert_named_outputs_inert(self, names: tuple[str, ...]) -> None:
        """Validate an exact architecture-transition subset of output gates."""
        parameters = dict(self.named_parameters())
        if len(names) != len(set(names)):
            raise ValueError("inert output parameter names must be unique")
        missing = tuple(name for name in names if name not in parameters)
        if missing:
            raise ValueError(f"inert output parameters are missing: {missing}")
        for name in names:
            parameter = parameters[name]
            if parameter.is_meta:
                raise ValueError("cannot validate inert v2 parameters on meta device")
            if torch.count_nonzero(parameter).item() != 0:
                raise ValueError(f"v2 output parameter is not inert: {name}")

    def replay_sequence(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        actions: AcceptedActionBatch,
        *,
        sequence_offsets: tuple[int, ...],
        block_indices: Tensor,
    ) -> Tensor:
        """Rebuild temporal contexts from raw complete blocks."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.replay(
            snapshots,
            events,
            actions,
            sequence_offsets=sequence_offsets,
            block_indices=block_indices,
            card_encoder=self.backbone.input_encoder.card_encoder,
        )

    def condition_sequence(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        contexts: Tensor,
    ) -> SimpleStatelessBackboneOutput:
        """Apply temporal policy/value/belief residuals to snapshot rows."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.condition(snapshots, contexts)

    def prepare_sequence_incremental(
        self,
        snapshot: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        *,
        block_index: int,
        cache: TemporalKvCache | None,
    ) -> TemporalPreparedDecision:
        """Fork a provisional runtime cache for one current decision."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.prepare_incremental(
            snapshot,
            events,
            block_index=block_index,
            card_encoder=self.backbone.input_encoder.card_encoder,
            cache=cache,
        )

    def prepare_sequence_incremental_many(
        self,
        snapshots: SimpleStatelessBackboneOutput,
        events: PublicEventBatch,
        *,
        block_indices: Sequence[int],
        caches: Sequence[TemporalKvCache | None],
    ) -> tuple[TemporalPreparedDecision, ...]:
        """Fork batched provisional runtime caches for independent decisions."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.prepare_incremental_many(
            snapshots,
            events,
            block_indices=block_indices,
            card_encoder=self.backbone.input_encoder.card_encoder,
            caches=caches,
        )

    def commit_sequence_incremental(
        self,
        prepared: TemporalPreparedDecision,
        actions: AcceptedActionBatch,
    ) -> TemporalKvCache:
        """Commit only an accepted complete action to temporal history."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.commit_incremental(
            prepared,
            actions,
            card_encoder=self.backbone.input_encoder.card_encoder,
        )

    def commit_sequence_incremental_many(
        self,
        prepared: Sequence[TemporalPreparedDecision],
        actions: AcceptedActionBatch,
    ) -> tuple[TemporalKvCache, ...]:
        """Commit one accepted complete action for every prepared decision."""
        if self.sequence is None:
            raise ValueError("model has no generalist sequence branch")
        return self.sequence.commit_incremental_many(
            prepared,
            actions,
            card_encoder=self.backbone.input_encoder.card_encoder,
        )

    def encode_exact_deck_tokens(
        self,
        *,
        unique_deck_card_ids: Tensor,
        deck_counts: Tensor,
        deck_valid_mask: Tensor,
    ) -> Tensor:
        """Encode exact deck multisets for executor-local inference reuse."""
        return self.backbone.input_encoder.encode_decks(
            unique_deck_card_ids,
            deck_counts,
            deck_valid_mask,
        )

    def encode_observation_state(
        self,
        *,
        state: StateBatch,
        unique_deck_card_ids: Tensor,
        deck_counts: Tensor,
        deck_valid_mask: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Tensorize raw public-state arrays inside the clean model graph."""
        return self.backbone.forward_observation(
            state=state,
            unique_deck_card_ids=unique_deck_card_ids,
            deck_counts=deck_counts,
            deck_valid_mask=deck_valid_mask,
            belief_summary=belief_summary,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def encode_observation_state_with_deck_tokens(
        self,
        *,
        state: StateBatch,
        deck_tokens: Tensor,
        belief_summary: PublicBeliefSummaryBatch,
        route_plan: SimpleExactRoutePlan | None = None,
        allow_unrouted_rows: bool = False,
    ) -> SimpleStatelessBackboneOutput:
        """Tensorize dynamic observation inputs with cached exact-deck tokens."""
        return self.backbone.forward_observation_with_deck_tokens(
            state=state,
            deck_tokens=deck_tokens,
            belief_summary=belief_summary,
            route_plan=route_plan,
            allow_unrouted_rows=allow_unrouted_rows,
        )

    def belief_logits(self, state: SimpleStatelessBackboneOutput) -> Tensor:
        """Materialize diagnostic vocabulary logits only for learner/analysis."""
        return cast(
            Tensor,
            self.belief_head(
                state.opponent_belief,
                card_encoder=self.backbone.input_encoder.card_encoder,
            ),
        )


def materialize_simple_stateless_checkpoint_model(
    config: SimpleStatelessModelConfig,
    model_state: Mapping[str, Tensor],
) -> SimpleStatelessPolicyValueNet:
    """Build a complete checkpoint without allocating throwaway parameters."""
    # ``initialize=False`` skips the architecture's explicit initializer, but
    # PyTorch constructors still initialize every Linear and Embedding. Meta
    # placeholders plus assign preserve the validated checkpoint tensors while
    # avoiding both that CPU work and a second full-sized allocation.
    with torch.device("meta"):
        model = SimpleStatelessPolicyValueNet(
            config,
            load_static_features=False,
            initialize=False,
        )
    model.load_state_dict(model_state, strict=True, assign=True)
    if any(tensor.is_meta for tensor in model.state_dict().values()):
        raise RuntimeError("checkpoint left simple-stateless tensors on meta")
    return model
