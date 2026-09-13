"""Composite identity for determinization and public belief continuation."""

from __future__ import annotations

from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
)
from ptcg_rl.belief.identity import canonical_belief_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler


def belief_runtime_fingerprint(
    sampler: BeliefSampler,
    producer: PublicBeliefFeatureProducer | None,
) -> str:
    """Bind every belief component capable of changing planner evidence."""
    return canonical_belief_fingerprint(
        b"ptcg-rl/planner-belief-runtime/v1\x00",
        {
            "sampler_fingerprint": sampler.semantic_fingerprint,
            "producer_fingerprint": (
                "none" if producer is None else producer.producer_fingerprint
            ),
        },
    )


__all__ = ["belief_runtime_fingerprint"]
