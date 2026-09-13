"""Manual-coin branch enumeration for the forward model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.effects import parse_effect_logs
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_NAMES,
    DYNAMIC_EFFECT_FEATURE_SIZE,
    DynamicEffectFeatureRow,
    make_dynamic_effect_feature_row,
)
from ptcg_rl.engine.protocols import (
    LogLike,
    ObservationInput,
    ObservationLike,
    OptionLike,
    SearchStateLike,
    SelectDataLike,
    StateLike,
)
from ptcg_rl.engine.session import HiddenInformation, SearchSession

_FEATURE_INDEX = {
    name: index for index, name in enumerate(DYNAMIC_EFFECT_FEATURE_NAMES)
}


@dataclass(frozen=True)
class CoinBranchOutcome:
    """One manually enumerated coin branch."""

    select_path: tuple[tuple[int, ...], ...]
    coin_heads: tuple[bool, ...]
    probability: float
    final_state: SearchStateLike
    summary: EffectSummary
    feature: DynamicEffectFeatureRow
    unresolved_prompt: bool = False


@dataclass(frozen=True)
class ManualCoinAnalysis:
    """Aggregated manual-coin branch analysis for one candidate action."""

    select: tuple[int, ...]
    outcomes: tuple[CoinBranchOutcome, ...]
    expected_vector: tuple[float, ...]
    worst_vector: tuple[float, ...]


def analyze_manual_coin_branches(
    observation: ObservationInput,
    hidden: HiddenInformation,
    select: Sequence[int],
    *,
    max_coin_flips: int = 8,
    max_auto_steps: int = 32,
) -> ManualCoinAnalysis:
    """Enumerate manually chosen coin branches and aggregate feature vectors."""
    with SearchSession.begin(observation, hidden, manual_coin=True) as session:
        return analyze_manual_coin_branches_from_session(
            session,
            session.root,
            select,
            max_coin_flips=max_coin_flips,
            max_auto_steps=max_auto_steps,
        )


def analyze_manual_coin_branches_from_session(
    session: SearchSession,
    parent: SearchStateLike,
    select: Sequence[int],
    *,
    max_coin_flips: int = 8,
    max_auto_steps: int = 32,
) -> ManualCoinAnalysis:
    """Enumerate coin branches from an existing manual-coin search root."""
    if max_coin_flips < 0:
        raise ValueError("max_coin_flips must be non-negative")
    action = tuple(int(index) for index in select)
    first_state = session.step(parent.searchId, action)
    parent_observation = parent.observation
    selected = _selected_options(parent_observation.select, action)
    try:
        outcomes = tuple(
            _expand_coin_branch(
                session=session,
                current=first_state,
                before_state=parent_observation.current,
                root_select=action,
                option_types=tuple(int(option.type) for option in selected),
                attack_ids=tuple(
                    int(option.attackId)
                    for option in selected
                    if option.attackId is not None
                ),
                card_ids=tuple(
                    int(option.cardId) for option in selected if option.cardId is not None
                ),
                logs=tuple(first_state.observation.logs),
                select_path=(action,),
                coin_heads=(),
                max_coin_flips=max_coin_flips,
                max_auto_steps=max_auto_steps,
                auto_step_count=0,
            )
        )
    finally:
        session.release(first_state.searchId)
    return ManualCoinAnalysis(
        select=action,
        outcomes=outcomes,
        expected_vector=_expected_vector(outcomes),
        worst_vector=_worst_vector(outcomes),
    )


def _expand_coin_branch(
    *,
    session: SearchSession,
    current: SearchStateLike,
    before_state: StateLike | None,
    root_select: tuple[int, ...],
    option_types: tuple[int, ...],
    attack_ids: tuple[int, ...],
    card_ids: tuple[int, ...],
    logs: tuple[LogLike, ...],
    select_path: tuple[tuple[int, ...], ...],
    coin_heads: tuple[bool, ...],
    max_coin_flips: int,
    max_auto_steps: int,
    auto_step_count: int,
) -> tuple[CoinBranchOutcome, ...]:
    observation = current.observation
    if (
        observation.current is None
        or observation.current.result >= 0
        or len(coin_heads) >= max_coin_flips
        or auto_step_count >= max_auto_steps
    ):
        return (
            _coin_outcome(
                root_select=root_select,
                option_types=option_types,
                attack_ids=attack_ids,
                card_ids=card_ids,
                before_state=before_state,
                final_state=current,
                logs=logs,
                select_path=select_path,
                coin_heads=coin_heads,
                unresolved_prompt=observation.select is not None,
            ),
        )

    coin_actions = _coin_choice_actions(observation)
    if coin_actions:
        outcomes: list[CoinBranchOutcome] = []
        for head, action in coin_actions:
            child = session.step(current.searchId, action)
            try:
                outcomes.extend(
                    _expand_coin_branch(
                        session=session,
                        current=child,
                        before_state=before_state,
                        root_select=root_select,
                        option_types=option_types,
                        attack_ids=attack_ids,
                        card_ids=card_ids,
                        logs=logs + tuple(child.observation.logs),
                        select_path=select_path + (action,),
                        coin_heads=coin_heads + (head,),
                        max_coin_flips=max_coin_flips,
                        max_auto_steps=max_auto_steps,
                        auto_step_count=auto_step_count + 1,
                    )
                )
            finally:
                session.release(child.searchId)
        return tuple(outcomes)

    forced_action = _forced_action(observation.select)
    if forced_action is None:
        return (
            _coin_outcome(
                root_select=root_select,
                option_types=option_types,
                attack_ids=attack_ids,
                card_ids=card_ids,
                before_state=before_state,
                final_state=current,
                logs=logs,
                select_path=select_path,
                coin_heads=coin_heads,
                unresolved_prompt=observation.select is not None,
            ),
        )
    child = session.step(current.searchId, forced_action)
    try:
        return _expand_coin_branch(
            session=session,
            current=child,
            before_state=before_state,
            root_select=root_select,
            option_types=option_types,
            attack_ids=attack_ids,
            card_ids=card_ids,
            logs=logs + tuple(child.observation.logs),
            select_path=select_path + (forced_action,),
            coin_heads=coin_heads,
            max_coin_flips=max_coin_flips,
            max_auto_steps=max_auto_steps,
            auto_step_count=auto_step_count + 1,
        )
    finally:
        session.release(child.searchId)


def _coin_outcome(
    *,
    root_select: tuple[int, ...],
    option_types: tuple[int, ...],
    attack_ids: tuple[int, ...],
    card_ids: tuple[int, ...],
    before_state: StateLike | None,
    final_state: SearchStateLike,
    logs: tuple[LogLike, ...],
    select_path: tuple[tuple[int, ...], ...],
    coin_heads: tuple[bool, ...],
    unresolved_prompt: bool,
) -> CoinBranchOutcome:
    summary = parse_effect_logs(
        logs,
        before_state=before_state,
        after_state=final_state.observation.current,
    )
    feature = make_dynamic_effect_feature_row(
        select=root_select,
        option_types=option_types,
        attack_ids=attack_ids,
        card_ids=card_ids,
        summary=summary,
        before_state=before_state,
        after_state=final_state.observation.current,
    )
    return CoinBranchOutcome(
        select_path=select_path,
        coin_heads=coin_heads,
        probability=0.5 ** len(coin_heads),
        final_state=final_state,
        summary=summary,
        feature=feature,
        unresolved_prompt=unresolved_prompt,
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


def _coin_choice_actions(
    observation: ObservationLike,
) -> tuple[tuple[bool, tuple[int, ...]], ...]:
    select = observation.select
    if select is None or int(select.context) != int(SelectContext.COIN_HEAD):
        return ()
    yes_index: int | None = None
    no_index: int | None = None
    for index, option in enumerate(select.option):
        if int(option.type) == int(OptionType.YES):
            yes_index = index
        elif int(option.type) == int(OptionType.NO):
            no_index = index
    actions: list[tuple[bool, tuple[int, ...]]] = []
    if yes_index is not None:
        actions.append((True, (yes_index,)))
    if no_index is not None:
        actions.append((False, (no_index,)))
    return tuple(actions)


def _forced_action(select: SelectDataLike | None) -> tuple[int, ...] | None:
    if select is None:
        return None
    min_count = int(select.minCount)
    max_count = int(select.maxCount)
    option_count = len(select.option)
    if min_count == 0 and max_count == 0:
        return ()
    if min_count == max_count == 1 and option_count == 1:
        return (0,)
    if min_count == max_count == option_count:
        return tuple(range(option_count))
    return None


def _expected_vector(outcomes: Sequence[CoinBranchOutcome]) -> tuple[float, ...]:
    if not outcomes:
        return tuple(0.0 for _ in range(DYNAMIC_EFFECT_FEATURE_SIZE))
    total_probability = sum(outcome.probability for outcome in outcomes)
    if total_probability <= 0.0:
        raise ValueError("coin outcomes have no probability mass")
    expected = [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE
    for outcome in outcomes:
        weight = outcome.probability / total_probability
        for index, value in enumerate(outcome.feature.vector):
            expected[index] += weight * value
    return tuple(expected)


def _worst_vector(outcomes: Sequence[CoinBranchOutcome]) -> tuple[float, ...]:
    if not outcomes:
        return tuple(0.0 for _ in range(DYNAMIC_EFFECT_FEATURE_SIZE))
    worst = max(outcomes, key=_branch_badness)
    return worst.feature.vector


def _branch_badness(outcome: CoinBranchOutcome) -> tuple[float, ...]:
    """Rank coin branches from the current player's perspective; higher is worse."""
    vector = outcome.feature.vector
    return (
        _feature(vector, "terminal_loss"),
        _feature(vector, "terminal_draw"),
        -_feature(vector, "terminal_win"),
        _feature(vector, "self_active_ko"),
        _feature(vector, "self_active_damage_norm"),
        _feature(vector, "self_bench_total_damage_norm"),
        -_feature(vector, "opponent_active_ko"),
        -_feature(vector, "opponent_active_damage_norm"),
        -_feature(vector, "opponent_bench_total_damage_norm"),
        -_feature(vector, "opponent_bench_max_damage_norm"),
        -_feature(vector, "opponent_bench_damaged_count_norm"),
        -_feature(vector, "prizes_taken_norm"),
        -_feature(vector, "cards_drawn_norm"),
        -_feature(vector, "energy_delta_norm"),
    )


def _feature(vector: tuple[float, ...], name: str) -> float:
    return vector[_FEATURE_INDEX[name]]
