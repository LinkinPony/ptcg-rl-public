"""Forward-model action resolution built on the engine Search API."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.effects import parse_effect_log_steps, parse_effect_logs
from ptcg_rl.engine.feature_vectors import (
    DynamicEffectFeatureRow,
    make_dynamic_effect_feature_row,
)
from ptcg_rl.engine.probe_resolution import ProbeTransition, resolve_probe_chain
from ptcg_rl.engine.protocols import (
    ObservationInput,
    ObservationLike,
    OptionLike,
    SearchStateLike,
    SelectDataLike,
)
from ptcg_rl.engine.session import HiddenInformation, SearchSession

DEFAULT_EFFECT_OPTION_TYPES = (
    int(OptionType.ATTACK),
    int(OptionType.ABILITY),
    int(OptionType.SKILL),
)


@dataclass(frozen=True)
class ActionResolution:
    """One Search API action and its immediate engine-resolved successor."""

    select: tuple[int, ...]
    search_id: int
    successor: SearchStateLike
    summary: EffectSummary
    option_types: tuple[int, ...] = ()
    attack_ids: tuple[int, ...] = ()
    card_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ProbeActionResolution:
    """One root action advanced through zero-cost forced continuations."""

    select: tuple[int, ...]
    search_id: int
    successor: SearchStateLike
    logs: tuple[Any, ...]
    resolved: bool
    forced_steps: int
    steps: int
    option_types: tuple[int, ...] = ()
    attack_ids: tuple[int, ...] = ()
    card_ids: tuple[int, ...] = ()
    transitions: tuple[ProbeTransition, ...] = ()


def enumerate_select_actions(
    select: SelectDataLike | None,
    *,
    max_actions: int = 64,
    rng: random.Random | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return legal option-index combinations for the current prompt.

    When the legal combination count exceeds ``max_actions``, actions are
    sampled uniformly by combination rank instead of taking the lexicographic
    prefix. This keeps the search fallback from always favoring low indices.
    """
    if select is None:
        return ()
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    option_count = len(select.option)
    min_count = min(option_count, max(0, int(select.minCount)))
    max_count = min(option_count, max(min_count, int(select.maxCount)))
    total_actions = _combination_count(option_count, min_count, max_count)
    if total_actions > max_actions:
        return _sample_select_actions(
            option_count,
            min_count,
            max_count,
            total_actions=total_actions,
            max_actions=max_actions,
            rng=rng or random.Random(),
        )

    actions: list[tuple[int, ...]] = []
    for count in range(min_count, max_count + 1):
        for combo in combinations(range(option_count), count):
            actions.append(tuple(combo))
    return tuple(actions)


def _combination_count(option_count: int, min_count: int, max_count: int) -> int:
    return sum(
        math.comb(option_count, count)
        for count in range(min_count, max_count + 1)
    )


def _sample_select_actions(
    option_count: int,
    min_count: int,
    max_count: int,
    *,
    total_actions: int,
    max_actions: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    ranks: set[int] = set()
    while len(ranks) < max_actions:
        ranks.add(rng.randrange(total_actions))
    return tuple(
        _unrank_select_action(option_count, min_count, max_count, rank)
        for rank in sorted(ranks)
    )


def _unrank_select_action(
    option_count: int,
    min_count: int,
    max_count: int,
    rank: int,
) -> tuple[int, ...]:
    remaining_rank = rank
    for count in range(min_count, max_count + 1):
        count_actions = math.comb(option_count, count)
        if remaining_rank < count_actions:
            return _unrank_combination(option_count, count, remaining_rank)
        remaining_rank -= count_actions
    raise ValueError("combination rank outside legal action range")


def _unrank_combination(
    option_count: int,
    count: int,
    rank: int,
) -> tuple[int, ...]:
    combo: list[int] = []
    next_value = 0
    remaining_rank = rank
    for remaining_slots in range(count, 0, -1):
        for value in range(next_value, option_count):
            suffix_count = math.comb(option_count - value - 1, remaining_slots - 1)
            if remaining_rank < suffix_count:
                combo.append(value)
                next_value = value + 1
                break
            remaining_rank -= suffix_count
    return tuple(combo)


def resolve_action_once(
    observation: ObservationInput,
    hidden: HiddenInformation,
    select: Sequence[int],
    *,
    manual_coin: bool = False,
) -> ActionResolution:
    """Start a short search, resolve one action, parse logs, then close it."""
    with SearchSession.begin(observation, hidden, manual_coin=manual_coin) as session:
        return resolve_action_from_session(session, session.root, select)


def resolve_action_from_session(
    session: SearchSession,
    parent: SearchStateLike,
    select: Sequence[int],
) -> ActionResolution:
    """Resolve one candidate action from an existing search state."""
    action = tuple(int(index) for index in select)
    successor = session.step(parent.searchId, action)
    parent_observation = parent.observation
    summary = parse_effect_logs(
        successor.observation.logs,
        before_state=parent_observation.current,
        after_state=successor.observation.current,
    )
    options = _selected_options(parent_observation.select, action)
    return ActionResolution(
        select=action,
        search_id=int(successor.searchId),
        successor=successor,
        summary=summary,
        option_types=tuple(int(option.type) for option in options),
        attack_ids=tuple(
            int(option.attackId) for option in options if option.attackId is not None
        ),
        card_ids=tuple(
            int(option.cardId) for option in options if option.cardId is not None
        ),
    )


def resolve_probe_action_from_session(
    session: SearchSession,
    parent: SearchStateLike,
    select: Sequence[int],
) -> ProbeActionResolution:
    """Resolve a probe candidate through forced prompts only.

    The final successor remains live and must be released by the caller.  A
    non-forced non-MAIN continuation is returned with ``resolved=False``.
    """
    action = tuple(int(index) for index in select)
    parent_observation = parent.observation
    selected = _selected_options(parent_observation.select, action)
    chain = resolve_probe_chain(
        parent,
        action,
        step=lambda state, continuation: session.step(
            state.searchId,
            continuation,
        ),
        release=lambda state: session.release(int(state.searchId)),
        observation=lambda state: state.observation,
    )
    return ProbeActionResolution(
        select=action,
        search_id=int(chain.state.searchId),
        successor=chain.state,
        logs=chain.logs,
        resolved=chain.resolved,
        forced_steps=chain.forced_steps,
        steps=chain.steps,
        option_types=tuple(int(option.type) for option in selected),
        attack_ids=tuple(
            int(option.attackId) for option in selected if option.attackId is not None
        ),
        card_ids=tuple(
            int(option.cardId) for option in selected if option.cardId is not None
        ),
        transitions=chain.transitions,
    )


def resolve_candidate_actions(
    observation: ObservationInput,
    hidden: HiddenInformation,
    *,
    candidates: Sequence[Sequence[int]] | None = None,
    manual_coin: bool = False,
    max_actions: int = 64,
    release_successors: bool = True,
) -> tuple[ActionResolution, ...]:
    """Resolve many root candidate actions under one Search API lifecycle."""
    with SearchSession.begin(observation, hidden, manual_coin=manual_coin) as session:
        root_observation = session.root.observation
        actions = (
            tuple(tuple(int(index) for index in candidate) for candidate in candidates)
            if candidates is not None
            else enumerate_select_actions(root_observation.select, max_actions=max_actions)
        )
        resolutions: list[ActionResolution] = []
        for action in actions:
            resolution = resolve_action_from_session(session, session.root, action)
            resolutions.append(resolution)
            if release_successors:
                session.release(resolution.search_id)
        return tuple(resolutions)


def extract_dynamic_effect_features(
    observation: ObservationInput,
    hidden: HiddenInformation,
    *,
    candidates: Sequence[Sequence[int]] | None = None,
    option_types: Sequence[int] = DEFAULT_EFFECT_OPTION_TYPES,
    max_actions: int = 64,
    manual_coin: bool = False,
) -> tuple[DynamicEffectFeatureRow, ...]:
    """Trial legal attack/ability/skill actions and return network features."""
    with SearchSession.begin(observation, hidden, manual_coin=manual_coin) as session:
        return extract_dynamic_effect_features_from_session(
            session,
            session.root,
            candidates=candidates,
            option_types=option_types,
            max_actions=max_actions,
            release_successors=True,
        )


def extract_dynamic_effect_features_from_session(
    session: SearchSession,
    parent: SearchStateLike,
    *,
    candidates: Sequence[Sequence[int]] | None = None,
    option_types: Sequence[int] = DEFAULT_EFFECT_OPTION_TYPES,
    max_actions: int = 64,
    release_successors: bool = True,
) -> tuple[DynamicEffectFeatureRow, ...]:
    """Extract feature rows from an existing search state."""
    parent_observation = parent.observation
    actions = (
        tuple(tuple(int(index) for index in candidate) for candidate in candidates)
        if candidates is not None
        else enumerate_select_actions(parent_observation.select, max_actions=max_actions)
    )
    allowed = {int(option_type) for option_type in option_types}
    rows: list[DynamicEffectFeatureRow] = []
    for action in actions:
        selected = _selected_options(parent_observation.select, action)
        if allowed and not any(int(option.type) in allowed for option in selected):
            continue
        resolution = resolve_probe_action_from_session(session, parent, action)
        if resolution.resolved:
            rows.append(
                dynamic_effect_feature_from_probe_resolution(
                    resolution,
                    parent_observation,
                )
            )
        if release_successors:
            session.release(resolution.search_id)
    return tuple(rows)


def dynamic_effect_feature_from_resolution(
    resolution: ActionResolution,
    parent_observation: ObservationLike,
    *,
    perspective_player: int | None = None,
) -> DynamicEffectFeatureRow:
    """Build one fixed-width feature row from a parsed action resolution."""
    return make_dynamic_effect_feature_row(
        select=resolution.select,
        option_types=resolution.option_types,
        attack_ids=resolution.attack_ids,
        card_ids=resolution.card_ids,
        summary=resolution.summary,
        before_state=parent_observation.current,
        after_state=resolution.successor.observation.current,
        perspective_player=perspective_player,
    )


def dynamic_effect_feature_from_probe_resolution(
    resolution: ProbeActionResolution,
    parent_observation: ObservationLike,
    *,
    perspective_player: int | None = None,
) -> DynamicEffectFeatureRow:
    """Build a feature row from a semantically complete probe chain."""
    if not resolution.resolved:
        raise ValueError("cannot build probe features from an unresolved continuation")
    summary = _parse_probe_transitions(
        resolution.transitions,
        fallback_logs=resolution.logs,
        before_state=parent_observation.current,
        after_state=resolution.successor.observation.current,
    )
    return make_dynamic_effect_feature_row(
        select=resolution.select,
        option_types=resolution.option_types,
        attack_ids=resolution.attack_ids,
        card_ids=resolution.card_ids,
        summary=summary,
        before_state=parent_observation.current,
        after_state=resolution.successor.observation.current,
        perspective_player=perspective_player,
    )


def dynamic_effect_feature_from_dict_resolution(
    *,
    select: tuple[int, ...],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    logs: Sequence[Any] | None = None,
    probe_transitions: Sequence[ProbeTransition] | None = None,
    perspective_player: int | None = None,
) -> DynamicEffectFeatureRow:
    """Build one dynamic-effect feature row from raw Search API dictionaries."""
    before_state = _mapping_field(before_observation, "current")
    after_state = _mapping_field(after_observation, "current")
    resolution_logs = (
        _sequence_field(after_observation, "logs") if logs is None else logs
    )
    summary = _parse_probe_transitions(
        probe_transitions,
        fallback_logs=resolution_logs,
        before_state=before_state,
        after_state=after_state,
    )
    selected = _selected_options_any(before_observation.get("select"), select)
    return make_dynamic_effect_feature_row(
        select=select,
        option_types=tuple(_int_field(option, "type", 0) for option in selected),
        attack_ids=tuple(
            _int_field(option, "attackId", 0)
            for option in selected
            if _field(option, "attackId") is not None
        ),
        card_ids=tuple(
            _int_field(option, "cardId", 0)
            for option in selected
            if _field(option, "cardId") is not None
        ),
        summary=summary,
        before_state=before_state,
        after_state=after_state,
        perspective_player=perspective_player,
    )


def _parse_probe_transitions(
    transitions: Sequence[ProbeTransition] | None,
    *,
    fallback_logs: Sequence[Any],
    before_state: Any | None,
    after_state: Any | None,
) -> EffectSummary:
    if transitions:
        return parse_effect_log_steps(
            tuple(
                (
                    transition.logs,
                    _field(transition.before_observation, "current"),
                    _field(transition.after_observation, "current"),
                )
                for transition in transitions
            )
        )
    # Native compact transitions and external callers expose only root/final
    # snapshots. HP_CHANGE log signs remain exact; intermediate state grounding
    # is intentionally unavailable rather than reconstructed from a full delta.
    return parse_effect_logs(
        fallback_logs,
        before_state=before_state,
        after_state=after_state,
    )


def _selected_options(
    select_data: SelectDataLike | None,
    select: Sequence[int],
) -> tuple[OptionLike, ...]:
    if select_data is None:
        return ()
    options: list[OptionLike] = []
    for index in select:
        if index < 0 or index >= len(select_data.option):
            raise IndexError(f"select index {index} outside option range")
        options.append(select_data.option[index])
    return tuple(options)


def _selected_options_any(
    select_data: Any,
    select: Sequence[int],
) -> tuple[Any, ...]:
    if select_data is None:
        return ()
    raw_options = _sequence_field(select_data, "option")
    options: list[Any] = []
    for index in select:
        if index < 0 or index >= len(raw_options):
            raise IndexError(f"select index {index} outside option range")
        options.append(raw_options[index])
    return tuple(options)


def _mapping_field(value: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    field = value.get(name)
    if field is None:
        return None
    if not isinstance(field, Mapping):
        raise TypeError(f"expected mapping field: {name}")
    return field


def _sequence_field(value: Any, name: str) -> Sequence[Any]:
    field = _field(value, name, ())
    return field if isinstance(field, Sequence) and not isinstance(field, str) else ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
