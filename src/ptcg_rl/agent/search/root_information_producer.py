"""Forkable producer bridge for deployable root-information post-states."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import orjson

from ptcg_rl.agent.search.root_information import (
    canonical_float32_vector_bytes,
)
from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
    decode_root_information_observation,
    decode_root_observable_state,
    encode_root_information_producer_context,
)
from ptcg_rl.context import GameContext, GameContextFeatures, GameContextSnapshot

_SNAPSHOT_DOMAIN = b"ptcg-rl/root-information-context-snapshot/v1\x00"
_NO_BELIEF_PRODUCER_DOMAIN = b"ptcg-rl/public-belief-producer/none/v1\x00"
NO_PUBLIC_BELIEF_PRODUCER_FINGERPRINT = hashlib.sha256(
    _NO_BELIEF_PRODUCER_DOMAIN
).hexdigest()


@dataclass(frozen=True, slots=True)
class RootInformationProducerState:
    """Public tracker and model inputs reached by one exact engine branch."""

    context_snapshot: GameContextSnapshot
    producer_context: bytes
    belief_summary: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.producer_context:
            raise ValueError("producer_context must not be empty")
        if any(not math.isfinite(value) for value in self.belief_summary):
            raise ValueError("belief_summary must be finite")


@dataclass(frozen=True, slots=True)
class RootInformationProducerTransition:
    """Updated public producer state plus its deployable model observation."""

    state: RootInformationProducerState
    model_observation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RootInformationProducerBridge:
    """Advance a forked public tracker from native observations and logs.

    The bridge never infers game effects. Native supplies the exact post-state
    and transition-local logs; ``GameContext`` only accumulates those public
    facts using the same code path as deployed observations.
    """

    root_player: int
    belief_summary_width: int
    belief_feature_producer: PublicBeliefFeatureProducer | None = None

    def __post_init__(self) -> None:
        if self.root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        if self.belief_summary_width < 0:
            raise ValueError("belief_summary_width must be non-negative")
        public_belief_producer_fingerprint(self.belief_feature_producer)

    def validate_root(
        self,
        *,
        root_observable_state: bytes,
        context_snapshot: GameContextSnapshot,
        producer_context: bytes,
        belief_summary: tuple[float, ...],
    ) -> RootInformationProducerState:
        """Require the supplied root snapshot to reproduce its model inputs."""
        observation = decode_root_observable_state(
            root_observable_state,
            root_player=self.root_player,
        )
        tracker = self._tracker(context_snapshot)
        features = self._with_belief(
            observation,
            tracker.features(observation),
        )
        rebuilt = self._state(tracker, features)
        if rebuilt.producer_context != producer_context:
            raise ValueError(
                "root context snapshot does not reproduce producer_context"
            )
        if canonical_float32_vector_bytes(rebuilt.belief_summary) != (
            canonical_float32_vector_bytes(belief_summary)
        ):
            raise ValueError("root context snapshot does not reproduce belief_summary")
        return rebuilt

    def advance(
        self,
        *,
        parent_context_snapshot: GameContextSnapshot,
        transition_observable_state: bytes,
        model_observable_state: bytes,
    ) -> RootInformationProducerTransition:
        """Fork a branch and consume exactly one native transition log batch."""
        transition_observation = decode_root_observable_state(
            transition_observable_state,
            root_player=self.root_player,
        )
        tracker = self._tracker(parent_context_snapshot)
        features = tracker.update(
            transition_observation,
            deduplicate_log_batch=False,
        )
        features = self._with_belief(transition_observation, features)
        state = self._state(tracker, features)
        model_observation = dict(
            decode_root_information_observation(
                model_observable_state,
                state.producer_context,
            )
        )
        return RootInformationProducerTransition(
            state=state,
            model_observation=model_observation,
        )

    def _tracker(self, snapshot: GameContextSnapshot) -> GameContext:
        if snapshot.player_index != self.root_player:
            raise ValueError("context snapshot perspective differs from root player")
        return GameContext.from_snapshot(snapshot)

    def _with_belief(
        self,
        observation: dict[str, Any],
        features: GameContextFeatures,
    ) -> GameContextFeatures:
        if self.belief_feature_producer is None:
            return features
        return self.belief_feature_producer.augment(observation, features)

    def _state(
        self,
        tracker: GameContext,
        features: GameContextFeatures,
    ) -> RootInformationProducerState:
        return RootInformationProducerState(
            context_snapshot=tracker.snapshot(),
            producer_context=encode_root_information_producer_context(
                features,
                root_player=self.root_player,
            ),
            belief_summary=root_information_belief_summary(
                features,
                width=self.belief_summary_width,
            ),
        )


def root_information_belief_summary(
    features: GameContextFeatures,
    *,
    width: int,
) -> tuple[float, ...]:
    """Encode the canonical fixed-width public posterior summary."""
    if width < 0:
        raise ValueError("belief summary width must be non-negative")
    if width == 0:
        return ()
    values = [
        float(features.opponent_belief_entropy),
        float(features.opponent_belief_empty),
    ]
    for item in features.opponent_belief:
        values.extend((float(item.card_id), float(item.expected_count)))
    values = values[:width]
    values.extend(0.0 for _ in range(width - len(values)))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("belief summary must be finite")
    return tuple(values)


def context_snapshot_fingerprint(snapshot: GameContextSnapshot) -> str:
    """Return an exact public tracker-state identity for future branch updates."""
    payload = {
        item.name: _canonical_snapshot_value(getattr(snapshot, item.name))
        for item in fields(snapshot)
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(_SNAPSHOT_DOMAIN + encoded).hexdigest()


def public_belief_producer_fingerprint(
    producer: PublicBeliefFeatureProducer | None,
) -> str:
    """Return a validated identity for the exact public posterior function."""
    if producer is None:
        return NO_PUBLIC_BELIEF_PRODUCER_FINGERPRINT
    fingerprint = producer.producer_fingerprint
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("belief producer fingerprint must be lowercase SHA-256")
    return fingerprint


def _canonical_snapshot_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_snapshot_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, frozenset):
        return [_canonical_snapshot_value(item) for item in sorted(value)]
    if isinstance(value, tuple):
        return [_canonical_snapshot_value(item) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("GameContextSnapshot contains a non-canonical field")


__all__ = [
    "NO_PUBLIC_BELIEF_PRODUCER_FINGERPRINT",
    "RootInformationProducerBridge",
    "RootInformationProducerState",
    "RootInformationProducerTransition",
    "context_snapshot_fingerprint",
    "public_belief_producer_fingerprint",
    "root_information_belief_summary",
]
