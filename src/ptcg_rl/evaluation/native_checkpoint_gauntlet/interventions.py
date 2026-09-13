"""Evaluation-only input interventions on a frozen sequence actor.

The native engine still produces legal options and tactical facts. Masking is
applied after asynchronous feature transfer, immediately before option encoding.
History truncation preserves transaction coordinates and the current E/S block.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import PublicEventBatch
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_inference import PolicyInputBatch
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceActorSampleContinuation,
)
from ptcg_rl.rl.sequence_actor_transfer import SequenceActorHostTransfer
from ptcg_rl.rl.sequence_runtime import SequenceCacheFork

InputIntervention: TypeAlias = Literal[
    "baseline", "reset_temporal", "mask_engine_facts", "both"
]


def mask_option_facts(options: OptionBatch) -> None:
    """Mark only neural consequence channels unavailable; retain legal options."""
    with torch.inference_mode():
        options.dynamic_effect_features.zero_()
        options.dynamic_effect_masks.zero_()


class InterventionSequenceActor(GeneralistSequenceActorPolicy):
    """Keep intervention state private to one candidate actor."""

    intervention: InputIntervention = "baseline"
    intervention_rows: int = 0
    history_reset_rows: int = 0

    def _fork_with_preallocated_slot(
        self, fork: SequenceCacheFork
    ) -> SequenceCacheFork:
        if self.intervention in {"reset_temporal", "both"}:
            # Replace the read view, not the transaction's absolute block index.
            fork = replace(fork, committed_cache=None)
            self.history_reset_rows += 1
        return super()._fork_with_preallocated_slot(fork)

    def begin_preencoded_deferred(
        self,
        rows: Sequence[SimpleStatelessActorRow],
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        copy_stream: Any | None = None,
        public_event_batch: PublicEventBatch | None = None,
        host_semantic_batch: NativeSimpleStatelessPolicyBatch | None = None,
        materialize_host_actions: bool = True,
        evaluation_action_only: bool = False,
    ) -> SequenceActorSampleContinuation:
        continuation = super().begin_preencoded_deferred(
            rows,
            batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            copy_stream=copy_stream,
            public_event_batch=public_event_batch,
            host_semantic_batch=host_semantic_batch,
            materialize_host_actions=materialize_host_actions,
            evaluation_action_only=evaluation_action_only,
        )
        if self.intervention == "baseline":
            return continuation

        def resume(
            await_option_features: Callable[[], None] | None,
        ) -> SequenceActorHostTransfer:
            def finish_features() -> None:
                if await_option_features is not None:
                    await_option_features()
                if self.intervention in {"mask_engine_facts", "both"}:
                    mask_option_facts(batch.options)
                    if host_semantic_batch is not None:
                        mask_option_facts(host_semantic_batch.options)
                self.intervention_rows += len(rows)

            return continuation.resume(await_option_features=finish_features)

        return SequenceActorSampleContinuation(
            _resume_callback=resume, _cancel_callback=continuation.cancel
        )
