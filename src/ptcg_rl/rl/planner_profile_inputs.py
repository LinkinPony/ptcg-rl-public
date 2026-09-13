"""Canonical tensor inputs for immutable planner-profile roots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from ptcg_rl.actions.encoding import (
    EncodedOptionInput,
    StateTokenLayout,
    encode_option_arrays,
)
from ptcg_rl.context import GameContextFeatures, OpponentBeliefFeatureProducer
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusRecord,
)
from ptcg_rl.model import (
    OptionBatch,
    StateTokenInput,
    collate_encoded_options,
    collate_state_tokens,
)
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.model.state_encoder import StateBatch, encode_observation_token_arrays


@dataclass(frozen=True, slots=True)
class PreparedProfileRoot:
    """One belief-augmented root before device collation."""

    state: StateTokenInput
    options: EncodedOptionInput
    min_count: int
    max_count: int
    root_input_fingerprint: str
    context_features: GameContextFeatures
    observation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CollatedProfileRoots:
    """One actor-local root wave ready for production inference."""

    prepared: tuple[PreparedProfileRoot, ...]
    states: StateBatch
    options: OptionBatch
    decks: DeckBatch


def prepare_profile_root(
    record: PlannerProfileCorpusRecord,
    *,
    producer: OpponentBeliefFeatureProducer,
) -> PreparedProfileRoot:
    """Rebuild the exact public posterior and canonical policy input."""
    context_features = producer.augment(record.observation, record.context_features)
    observation = dict(record.observation)
    observation["gameContext"] = context_features.as_observation_dict()
    layout = StateTokenLayout.from_observation(
        observation,
        context_features=context_features,
    )
    state = encode_observation_token_arrays(observation, layout=layout)
    options = encode_option_arrays(observation.get("select"), layout)
    select = cast(Mapping[str, Any], observation["select"])
    min_count = min(len(options), max(0, int(select.get("minCount", 0))))
    max_count = min(
        len(options),
        max(min_count, int(select.get("maxCount", len(options)))),
    )
    return PreparedProfileRoot(
        state=state,
        options=options,
        min_count=min_count,
        max_count=max_count,
        root_input_fingerprint=canonical_planner_root_input_fingerprint(
            state,
            options,
            min_count=min_count,
            max_count=max_count,
        ),
        context_features=context_features,
        observation=observation,
    )


def collate_profile_roots(
    records: Sequence[PlannerProfileCorpusRecord],
    *,
    producer: OpponentBeliefFeatureProducer,
    device: torch.device | str,
    model_deck: Sequence[int] | None = None,
) -> CollatedProfileRoots:
    """Collate one bounded root wave without retaining source corpus rows."""
    if not records:
        raise ValueError("planner profile root wave must not be empty")
    prepared = tuple(
        prepare_profile_root(record, producer=producer) for record in records
    )
    states = replace(
        collate_state_tokens(
            tuple(item.state for item in prepared),
            device=device,
        ),
        root_input_fingerprints=tuple(item.root_input_fingerprint for item in prepared),
    )
    options = collate_encoded_options(
        tuple(item.options for item in prepared),
        min_counts=tuple(item.min_count for item in prepared),
        max_counts=tuple(item.max_count for item in prepared),
        device=device,
    )
    decks = DeckBatch.from_decks(
        tuple(
            canonicalize_deck(record.own_deck if model_deck is None else model_deck)
            for record in records
        ),
        device=device,
    )
    return CollatedProfileRoots(
        prepared=prepared,
        states=states,
        options=options,
        decks=decks,
    )


__all__ = [
    "CollatedProfileRoots",
    "PreparedProfileRoot",
    "collate_profile_roots",
    "prepare_profile_root",
]
