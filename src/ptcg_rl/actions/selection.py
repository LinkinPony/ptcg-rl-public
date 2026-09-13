"""Legal action helpers for engine ``SelectData`` prompts.

The bundled engine remains responsible for move legality. This module only
works with option indices that the engine has already exposed.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.protocols import SelectDataLike

SelectDataInput = SelectDataLike | Mapping[str, Any]

# The engine API exposes selections as sequences but does not provide a generic
# order-semantics flag.  Keep this registry deliberately narrow: DISCARD is
# backed by the native permutation-parity probe for Rocket Feathers.  Add a
# context only after an engine probe shows that permuting the same selected
# indices preserves the resulting state and consequence features.
ENGINE_PROVEN_UNORDERED_SET_CONTEXTS = frozenset(
    {
        int(SelectContext.DISCARD),
    }
)


def forced_action(select: SelectDataInput | None) -> tuple[int, ...] | None:
    """Return the zero-cost forced action for a prompt, if one exists."""
    if select is None:
        return None
    option_count = len(_options(select))
    min_count, max_count = _count_bounds(select, option_count)
    if max_count == 0:
        return ()
    if option_count == 1 and min_count == 1 and max_count == 1:
        return (0,)
    return None


def is_forced(select: SelectDataInput | None) -> bool:
    """Return whether the prompt has exactly one useful legal response."""
    return forced_action(select) is not None


def random_legal_action(
    select: SelectDataInput | None,
    *,
    rng: random.Random | None = None,
) -> tuple[int, ...]:
    """Sample a legal option-index action while respecting min/max counts."""
    if select is None:
        return ()
    active_rng = rng or random.Random()
    option_count = len(_options(select))
    min_count, max_count = _count_bounds(select, option_count)
    if option_count <= 0 or max_count <= 0:
        return ()

    count = active_rng.randint(min_count, max_count)
    if count <= 0:
        return ()
    if count >= option_count:
        action = tuple(range(option_count))
    else:
        action = tuple(active_rng.sample(range(option_count), count))
    return normalize_action_order(select, action)


def normalize_action_order(
    select: SelectDataInput | None,
    action: Sequence[int],
) -> tuple[int, ...]:
    """Canonicalize only selections whose order is proven irrelevant."""
    normalized = tuple(int(index) for index in action)
    if select is None or _is_order_sensitive(select):
        return normalized
    return tuple(sorted(normalized))


def is_unordered_set_selection(
    *,
    context: int,
    min_count: int,
    max_count: int,
) -> bool:
    """Return whether a prompt is an engine-proven unordered set."""
    return (
        int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        and 0 <= int(min_count) <= int(max_count)
        and int(max_count) > 1
    )


def is_legal_action(select: SelectDataInput | None, action: Sequence[int]) -> bool:
    """Return whether an index list obeys prompt cardinality and uniqueness."""
    if select is None:
        return False
    option_count = len(_options(select))
    min_count, max_count = _count_bounds(select, option_count)
    indices = tuple(int(index) for index in action)
    if len(indices) < min_count or len(indices) > max_count:
        return False
    if len(set(indices)) != len(indices):
        return False
    return all(0 <= index < option_count for index in indices)


def _is_order_sensitive(select: SelectDataInput) -> bool:
    if _int_field(select, "context", -1) == int(SelectContext.SKILL_ORDER):
        return True
    option_count = len(_options(select))
    _, max_count = _count_bounds(select, option_count)
    # The engine consumes ``selected`` as a sequence, and the public prompt has
    # no generic flag saying an effect ignores that sequence. Concrete subset
    # and all-select effects both preserve order. Runtime normalization stays
    # conservative so actions emitted by legacy frozen policies remain valid;
    # new set-aware policies opt into canonicalization inside their decoder.
    return max_count > 1


def _min_count(select: SelectDataInput) -> int:
    return max(0, _int_field(select, "minCount", 0))


def _count_bounds(select: SelectDataInput, option_count: int) -> tuple[int, int]:
    min_count = min(option_count, _min_count(select))
    raw_max = _int_field(select, "maxCount", option_count)
    max_count = min(option_count, max(min_count, raw_max))
    return min_count, max_count


def _options(select: SelectDataInput) -> Sequence[Any]:
    value = _field(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(select: SelectDataInput, name: str, default: int) -> int:
    value = _field(select, name, default)
    return int(value) if value is not None else default


def _field(select: SelectDataInput, name: str, default: Any) -> Any:
    if isinstance(select, Mapping):
        return select.get(name, default)
    return getattr(select, name, default)
