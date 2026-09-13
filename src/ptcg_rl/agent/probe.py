"""Act-time 1-ply probe features and conservative verification helpers."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

from ptcg_rl.actions.encoding import EncodedOption
from ptcg_rl.agent.search.config import ActTimeSearchConfig as ActTimeSearchConfig
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContextFeatures,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_NAMES,
    DYNAMIC_EFFECT_FEATURE_SIZE,
)

CORE_PROBE_OPTION_TYPES = (int(OptionType.ATTACK), int(OptionType.ABILITY))
FEATURE_INDEX = {name: index for index, name in enumerate(DYNAMIC_EFFECT_FEATURE_NAMES)}


@dataclass(frozen=True)
class RuntimeProbeResult:
    """Act-time probe features and per-world verification vectors."""

    features: tuple[tuple[float, ...], ...]
    masks: tuple[bool, ...]
    world_vectors: Mapping[tuple[int, ...], tuple[tuple[float, ...], ...]]
    worlds_requested: int
    unresolved_options: int = 0
    unresolved_worlds: int = 0


def run_runtime_probe_features(
    observation: Any,
    context_features: GameContextFeatures,
    *,
    your_deck: Sequence[int],
    sampler: BeliefSampler,
    opponent_card_probs: Sequence[float] | None = None,
    opponent_hand_weights: Sequence[float] | None = None,
    rng: random.Random,
    config: ActTimeSearchConfig,
) -> RuntimeProbeResult | None:
    """Probe all root ATTACK/ABILITY single-option candidates."""
    core_candidates = core_option_candidates(_select_from_observation(observation))
    if not core_candidates:
        return None
    from ptcg_rl.belief.observation import extract_observation_evidence
    from ptcg_rl.engine.forward_model import (
        dynamic_effect_feature_from_probe_resolution,
        resolve_probe_action_from_session,
    )
    from ptcg_rl.engine.session import SearchSession

    evidence = extract_observation_evidence(observation)
    opponent_state = opponent_belief_state_from_evidence(evidence, context_features)
    vectors_by_action: dict[tuple[int, ...], list[tuple[float, ...]]] = {
        candidate: [] for candidate in core_candidates
    }
    unresolved_by_action = dict.fromkeys(core_candidates, 0)
    for _ in range(config.worlds):
        determinization = sampler.sample_from_evidence(
            evidence,
            your_deck=your_deck,
            opponent_state=opponent_state,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=rng,
        )
        with SearchSession.begin(
            observation,
            determinization.hidden,
            manual_coin=config.manual_coin,
        ) as session:
            for candidate in core_candidates:
                resolution = resolve_probe_action_from_session(
                    session,
                    session.root,
                    candidate,
                )
                try:
                    if resolution.resolved:
                        feature_row = dynamic_effect_feature_from_probe_resolution(
                            resolution,
                            session.root.observation,
                        )
                        vectors_by_action[candidate].append(feature_row.vector)
                    else:
                        unresolved_by_action[candidate] += 1
                finally:
                    session.release(resolution.search_id)

    return build_runtime_probe_result(
        option_count=len(_options(_select_from_observation(observation))),
        vectors_by_action=vectors_by_action,
        worlds_requested=config.worlds,
        unresolved_by_action=unresolved_by_action,
    )


def encoded_options_with_probe_features(
    options: Sequence[EncodedOption],
    observation: Any,
) -> tuple[EncodedOption, ...]:
    """Attach top-level probe feature rows to encoded options."""
    features = _sequence(_field(observation, "probeEffectFeatures", ()))
    masks = _sequence(_field(observation, "probeEffectMasks", ()))
    if not features or not masks:
        return tuple(options)
    updated: list[EncodedOption] = []
    for index, option in enumerate(options):
        has_feature = index < len(masks) and bool(masks[index])
        if not has_feature or index >= len(features):
            updated.append(option)
            continue
        vector = tuple(float(value) for value in _sequence(features[index]))
        if len(vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
            updated.append(option)
            continue
        updated.append(
            replace(
                option,
                dynamic_effect_features=vector,
                dynamic_effect_mask=True,
            )
        )
    return tuple(updated)


def observation_with_probe_features(
    observation: Any,
    probe_result: RuntimeProbeResult,
) -> Any:
    """Return an observation copy carrying probe feature rows."""
    feature_rows = [list(row) for row in probe_result.features]
    mask_rows = list(probe_result.masks)
    if isinstance(observation, Mapping):
        copied = dict(observation)
        copied["probeEffectFeatures"] = feature_rows
        copied["probeEffectMasks"] = mask_rows
        return copied
    return {
        "remainingOverageTime": _field(observation, "remainingOverageTime"),
        "current": _field(observation, "current"),
        "logs": _field(observation, "logs", ()),
        "search_begin_input": _field(observation, "search_begin_input"),
        "select": _field(observation, "select"),
        "gameContext": _field(observation, "gameContext"),
        "probeEffectFeatures": feature_rows,
        "probeEffectMasks": mask_rows,
    }


def core_option_candidates(select: Any) -> tuple[tuple[int, ...], ...]:
    """Return root single-option ATTACK/ABILITY candidates."""
    return tuple(
        (index,)
        for index, option in enumerate(_options(select))
        if _option_type(option) in CORE_PROBE_OPTION_TYPES
    )


def is_core_action(select: Any, action: Sequence[int]) -> bool:
    """Return whether ``action`` selects one ATTACK/ABILITY option."""
    if len(action) != 1:
        return False
    option_index = int(action[0])
    options = _options(select)
    if option_index < 0 or option_index >= len(options):
        return False
    return _option_type(options[option_index]) in CORE_PROBE_OPTION_TYPES


def all_worlds_verified_lethal(
    action: tuple[int, ...],
    probe_result: RuntimeProbeResult,
) -> bool:
    """Return whether every sampled world verifies an immediate terminal win."""
    return _all_worlds_match(
        action,
        probe_result,
        lambda vector: vector[FEATURE_INDEX["terminal_win"]] >= 0.5,
    )


def all_worlds_verified_self_loss(
    action: tuple[int, ...],
    probe_result: RuntimeProbeResult,
) -> bool:
    """Return whether every sampled world verifies an immediate terminal loss."""
    return _all_worlds_match(
        action,
        probe_result,
        lambda vector: vector[FEATURE_INDEX["terminal_loss"]] >= 0.5,
    )


def mean_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return an elementwise mean dynamic-effect vector."""
    if not vectors:
        return (0.0,) * DYNAMIC_EFFECT_FEATURE_SIZE
    totals = [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE
    for vector in vectors:
        if len(vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("probe vector has invalid width")
        for index, value in enumerate(vector):
            totals[index] += float(value)
    return tuple(value / float(len(vectors)) for value in totals)


def build_runtime_probe_result(
    *,
    option_count: int,
    vectors_by_action: Mapping[
        tuple[int, ...],
        Sequence[tuple[float, ...]],
    ],
    worlds_requested: int,
    unresolved_by_action: Mapping[tuple[int, ...], int] | None = None,
) -> RuntimeProbeResult:
    """Build probe tensors, masking candidates not resolved in every world."""
    if worlds_requested <= 0:
        raise ValueError("worlds_requested must be positive")
    unresolved_counts = unresolved_by_action or {}
    features: list[tuple[float, ...]] = [
        (0.0,) * DYNAMIC_EFFECT_FEATURE_SIZE for _ in range(option_count)
    ]
    masks = [False] * option_count
    for candidate, vectors in vectors_by_action.items():
        if len(candidate) != 1 or len(vectors) != worlds_requested:
            continue
        if int(unresolved_counts.get(candidate, 0)) > 0:
            continue
        option_index = int(candidate[0])
        if 0 <= option_index < option_count:
            features[option_index] = mean_vectors(vectors)
            masks[option_index] = True
    return RuntimeProbeResult(
        features=tuple(features),
        masks=tuple(masks),
        world_vectors={
            action: tuple(vectors) for action, vectors in vectors_by_action.items()
        },
        worlds_requested=worlds_requested,
        unresolved_options=sum(
            1 for count in unresolved_counts.values() if int(count) > 0
        ),
        unresolved_worlds=sum(int(count) for count in unresolved_counts.values()),
    )


def enumerate_select_actions(
    select: Any,
    *,
    max_actions: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate legal option sequences conservatively up to a fixed cap.

    Public prompts do not prove that a multi-select is order-independent, so
    this legacy helper must preserve the same ordered semantics as policy
    decoding and ``normalize_action_order``.
    """
    option_count = len(_options(select))
    min_count = min(option_count, max(0, _int_field(select, "minCount", 0)))
    max_count = min(
        option_count,
        max(min_count, _int_field(select, "maxCount", option_count)),
    )
    actions: list[tuple[int, ...]] = []
    for count in range(min_count, max_count + 1):
        for action in permutations(range(option_count), count):
            actions.append(tuple(int(index) for index in action))
            if len(actions) >= max_actions:
                return tuple(actions)
    return tuple(actions)


def _all_worlds_match(
    action: tuple[int, ...],
    probe_result: RuntimeProbeResult,
    predicate: Callable[[Sequence[float]], bool],
) -> bool:
    vectors = probe_result.world_vectors.get(tuple(action), ())
    if len(vectors) != probe_result.worlds_requested:
        return False
    return all(predicate(vector) for vector in vectors)


def _select_from_observation(observation: Any) -> Any:
    if isinstance(observation, Mapping):
        return observation.get("select")
    return getattr(observation, "select", None)


def _options(select: Any) -> Sequence[Any]:
    value = _field(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _option_type(option: Any) -> int:
    return _int_field(option, "type", -1)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
