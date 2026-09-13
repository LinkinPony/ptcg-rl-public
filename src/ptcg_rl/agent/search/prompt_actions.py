"""Bounded, order-aware legal action construction for engine prompts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from ptcg_rl.actions.selection import is_legal_action, is_unordered_set_selection
from ptcg_rl.engine.constants import SelectContext

DEFAULT_ORDERED_CONTEXTS = frozenset({int(SelectContext.SKILL_ORDER)})


@dataclass(frozen=True)
class PromptActionCandidates:
    """Legal complete selections retained for one engine prompt."""

    actions: tuple[tuple[int, ...], ...]
    sources: tuple[tuple[str, ...], ...]
    legal_action_count: int
    exhaustive: bool
    ordered: bool

    def contains(self, action: Sequence[int]) -> bool:
        """Return whether the exact option-index sequence was retained."""
        return tuple(int(index) for index in action) in self.actions


@dataclass(frozen=True)
class PromptActionSpace:
    """Cardinality and ordering semantics of one engine prompt."""

    option_count: int
    min_count: int
    max_count: int
    legal_action_count: int
    ordered: bool


def describe_prompt_action_space(
    select: Any,
    *,
    ordered_contexts: frozenset[int] = DEFAULT_ORDERED_CONTEXTS,
) -> PromptActionSpace:
    """Return exact legal-space size without materializing its actions."""
    option_count, min_count, max_count = _prompt_shape(select)
    context = _int_field(select, "context", -1)
    ordered = context in ordered_contexts or (
        max_count > 1
        and not is_unordered_set_selection(
            context=context,
            min_count=min_count,
            max_count=max_count,
        )
    )
    return PromptActionSpace(
        option_count=option_count,
        min_count=min_count,
        max_count=max_count,
        legal_action_count=_legal_action_count(
            option_count,
            min_count,
            max_count,
            ordered=ordered,
        ),
        ordered=ordered,
    )


def build_prompt_action_candidates(
    select: Any,
    *,
    greedy_action: Sequence[int] | None,
    ranked_actions: Sequence[Sequence[int]] = (),
    exhaustive_action_cap: int,
    beam_width: int,
    ordered_contexts: frozenset[int] = DEFAULT_ORDERED_CONTEXTS,
) -> PromptActionCandidates:
    """Return an exact action set when small, otherwise a policy-led beam.

    The public API exposes no generic flag proving that a multi-selection's
    sequence is irrelevant. Multi-select prompts retain permutations except
    for contexts registered after an engine-parity equivalence probe.
    The greedy policy action is always first and can never be displaced by the
    exhaustive cap or beam truncation.
    """
    if exhaustive_action_cap <= 0:
        raise ValueError("exhaustive_action_cap must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    space = describe_prompt_action_space(
        select,
        ordered_contexts=ordered_contexts,
    )
    option_count = space.option_count
    min_count = space.min_count
    max_count = space.max_count
    ordered = space.ordered
    legal_count = space.legal_action_count
    greedy = (
        _normalize(greedy_action, ordered=ordered)
        if greedy_action is not None
        else _unrank_action(
            option_count,
            min_count,
            max_count,
            0,
            ordered=ordered,
        )
    )
    if not is_legal_action(select, greedy):
        raise ValueError("greedy_action is not legal for the engine prompt")

    by_action: dict[tuple[int, ...], list[str]] = {}

    def add(action: Sequence[int], source: str) -> None:
        normalized = _normalize(action, ordered=ordered)
        if not is_legal_action(select, normalized):
            return
        labels = by_action.setdefault(normalized, [])
        if source not in labels:
            labels.append(source)

    add(greedy, "greedy" if greedy_action is not None else "deterministic_fallback")
    exhaustive = legal_count <= exhaustive_action_cap
    if exhaustive:
        for action in _enumerate_actions(
            option_count,
            min_count,
            max_count,
            ordered=ordered,
        ):
            add(action, "exhaustive")
    else:
        for ranked_action in ranked_actions:
            if len(by_action) >= beam_width:
                break
            add(ranked_action, "policy_beam")
        for rank in _stratified_ranks(legal_count, beam_width * 4):
            if len(by_action) >= beam_width:
                break
            add(
                _unrank_action(
                    option_count,
                    min_count,
                    max_count,
                    rank,
                    ordered=ordered,
                ),
                "stratified_fallback",
            )

    actions = tuple(by_action)
    return PromptActionCandidates(
        actions=actions,
        sources=tuple(tuple(by_action[action]) for action in actions),
        legal_action_count=legal_count,
        exhaustive=exhaustive,
        ordered=ordered,
    )


def _enumerate_actions(
    option_count: int,
    min_count: int,
    max_count: int,
    *,
    ordered: bool,
) -> tuple[tuple[int, ...], ...]:
    factory = permutations if ordered else combinations
    return tuple(
        tuple(action)
        for count in range(min_count, max_count + 1)
        for action in factory(range(option_count), count)
    )


def _legal_action_count(
    option_count: int,
    min_count: int,
    max_count: int,
    *,
    ordered: bool,
) -> int:
    counter = math.perm if ordered else math.comb
    return sum(counter(option_count, count) for count in range(min_count, max_count + 1))


def _unrank_action(
    option_count: int,
    min_count: int,
    max_count: int,
    rank: int,
    *,
    ordered: bool,
) -> tuple[int, ...]:
    remaining = int(rank)
    counter = math.perm if ordered else math.comb
    for count in range(min_count, max_count + 1):
        block_size = counter(option_count, count)
        if remaining < block_size:
            if ordered:
                return _unrank_permutation(option_count, count, remaining)
            return _unrank_combination(option_count, count, remaining)
        remaining -= block_size
    raise ValueError("action rank is outside the legal prompt space")


def _unrank_combination(
    option_count: int,
    count: int,
    rank: int,
) -> tuple[int, ...]:
    action: list[int] = []
    next_index = 0
    remaining = rank
    for remaining_slots in range(count, 0, -1):
        for index in range(next_index, option_count):
            suffixes = math.comb(option_count - index - 1, remaining_slots - 1)
            if remaining < suffixes:
                action.append(index)
                next_index = index + 1
                break
            remaining -= suffixes
    return tuple(action)


def _unrank_permutation(
    option_count: int,
    count: int,
    rank: int,
) -> tuple[int, ...]:
    available = list(range(option_count))
    action: list[int] = []
    remaining = rank
    for position in range(count):
        suffix_length = count - position - 1
        suffixes = math.perm(len(available) - 1, suffix_length)
        choice, remaining = divmod(remaining, suffixes)
        action.append(available.pop(choice))
    return tuple(action)


def _stratified_ranks(total: int, requested: int) -> tuple[int, ...]:
    if total <= 0 or requested <= 0:
        return ()
    count = min(total, requested)
    if count == 1:
        return (0,)
    return tuple(
        (index * (total - 1)) // (count - 1)
        for index in range(count)
    )


def _prompt_shape(select: Any) -> tuple[int, int, int]:
    options = _sequence(_field(select, "option", ()))
    option_count = len(options)
    min_count = min(option_count, max(0, _int_field(select, "minCount", 0)))
    max_count = min(
        option_count,
        max(min_count, _int_field(select, "maxCount", option_count)),
    )
    return option_count, min_count, max_count


def _normalize(action: Sequence[int], *, ordered: bool) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in action)
    return normalized if ordered else tuple(sorted(normalized))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


__all__ = [
    "DEFAULT_ORDERED_CONTEXTS",
    "PromptActionCandidates",
    "PromptActionSpace",
    "build_prompt_action_candidates",
    "describe_prompt_action_space",
]
