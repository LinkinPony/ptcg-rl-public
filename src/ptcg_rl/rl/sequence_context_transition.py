"""Fail-closed checkpoint transition for a longer temporal context window."""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.rl.model_compatibility import model_config_fingerprint


@dataclass(frozen=True, slots=True)
class StatelessSequenceContextTransitionPlan:
    """A parameter-inventory-preserving temporal context extension."""

    architecture: str
    source_max_context_blocks: int
    target_max_context_blocks: int
    source_model_config_fingerprint: str
    target_model_config_fingerprint: str

    @property
    def summary(self) -> dict[str, object]:
        """Return immutable evidence for the transition source report."""
        return {
            "format": "generalist-sequence-context-extension-v1",
            "architecture": self.architecture,
            "source_max_context_blocks": self.source_max_context_blocks,
            "target_max_context_blocks": self.target_max_context_blocks,
            "source_model_config_fingerprint": (self.source_model_config_fingerprint),
            "target_model_config_fingerprint": (self.target_model_config_fingerprint),
            "model_state_mode": "strict_identity_copy",
            "optimizer_state_mode": "exact_identity_copy",
        }


def build_stateless_sequence_context_transition_plan(
    source: SimpleStatelessModelConfig,
    target: SimpleStatelessModelConfig,
) -> StatelessSequenceContextTransitionPlan:
    """Authorize only an increase of ``sequence.max_context_blocks``.

    The context limit changes attention visibility and artifact identity, but it
    does not add, remove, or reshape parameters. Every other model field must
    remain byte-for-byte equivalent at the validated configuration level.
    """
    source_sequence = source.sequence
    target_sequence = target.sequence
    if source_sequence is None or target_sequence is None:
        raise ValueError("sequence context transition requires temporal models")
    if source.architecture != target.architecture:
        raise ValueError("sequence context transition changed model architecture")
    if target_sequence.max_context_blocks <= source_sequence.max_context_blocks:
        raise ValueError("sequence context transition must strictly extend context")
    expected_target = source.model_copy(
        update={
            "sequence": source_sequence.model_copy(
                update={
                    "max_context_blocks": target_sequence.max_context_blocks,
                }
            )
        }
    )
    if expected_target != target:
        raise ValueError(
            "sequence context transition changed fields other than max_context_blocks"
        )
    return StatelessSequenceContextTransitionPlan(
        architecture=source.architecture,
        source_max_context_blocks=source_sequence.max_context_blocks,
        target_max_context_blocks=target_sequence.max_context_blocks,
        source_model_config_fingerprint=model_config_fingerprint(source),
        target_model_config_fingerprint=model_config_fingerprint(target),
    )


__all__ = [
    "StatelessSequenceContextTransitionPlan",
    "build_stateless_sequence_context_transition_plan",
]
