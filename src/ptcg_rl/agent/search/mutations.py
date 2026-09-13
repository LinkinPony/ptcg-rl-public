"""Generic, legality-filtered local mutations for complete selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ptcg_rl.actions.selection import is_legal_action

MutationKind = Literal["add", "remove", "swap", "reorder"]


@dataclass(frozen=True)
class LocalMutation:
    """One complete legal neighbor and all generic operations producing it."""

    action: tuple[int, ...]
    kinds: tuple[MutationKind, ...]


def legal_local_mutations(
    select: Any,
    action: Sequence[int],
    *,
    ordered: bool,
) -> tuple[LocalMutation, ...]:
    """Return deterministic add/remove/swap/reorder neighbors.

    The function consumes only option indices and prompt cardinality/legality.
    It never interprets card, deck, attack, or effect identity.
    """
    normalized = _normalize(action, ordered=ordered)
    if not is_legal_action(select, normalized):
        raise ValueError("mutation parent must be a legal complete action")
    option_count = len(_select_options(select))
    min_count = _int_field(select, "minCount", 0)
    max_count = _int_field(select, "maxCount", option_count)
    by_action: dict[tuple[int, ...], list[MutationKind]] = {}

    def retain(candidate: Sequence[int], kind: MutationKind) -> None:
        neighbor = _normalize(candidate, ordered=ordered)
        if neighbor == normalized or not is_legal_action(select, neighbor):
            return
        kinds = by_action.setdefault(neighbor, [])
        if kind not in kinds:
            kinds.append(kind)

    unselected = tuple(
        index for index in range(option_count) if index not in normalized
    )
    if len(normalized) < max_count:
        for option_index in unselected:
            if ordered:
                for position in range(len(normalized) + 1):
                    retain(
                        (*normalized[:position], option_index, *normalized[position:]),
                        "add",
                    )
            else:
                retain((*normalized, option_index), "add")

    if len(normalized) > min_count:
        for position in range(len(normalized)):
            retain((*normalized[:position], *normalized[position + 1 :]), "remove")

    for position in range(len(normalized)):
        for option_index in unselected:
            replaced = list(normalized)
            replaced[position] = option_index
            retain(replaced, "swap")

    if ordered and len(normalized) > 1:
        for position in range(len(normalized) - 1):
            reordered = list(normalized)
            reordered[position], reordered[position + 1] = (
                reordered[position + 1],
                reordered[position],
            )
            retain(reordered, "reorder")
        retain(tuple(reversed(normalized)), "reorder")

    return tuple(
        LocalMutation(action=candidate, kinds=tuple(by_action[candidate]))
        for candidate in sorted(by_action)
    )


def _normalize(action: Sequence[int], *, ordered: bool) -> tuple[int, ...]:
    result = tuple(int(index) for index in action)
    return result if ordered else tuple(sorted(result))


def _select_options(select: Any) -> Sequence[Any]:
    options = (
        select.get("option", ())
        if isinstance(select, Mapping)
        else getattr(select, "option", ())
    )
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        return options
    return ()


def _int_field(select: Any, name: str, default: int) -> int:
    value = (
        select.get(name, default)
        if isinstance(select, Mapping)
        else getattr(
            select,
            name,
            default,
        )
    )
    return int(value) if value is not None else default


__all__ = ["LocalMutation", "MutationKind", "legal_local_mutations"]
