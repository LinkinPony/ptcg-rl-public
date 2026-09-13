"""Ordered and unordered partial-selection search representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class OrderedSelectionPrefix:
    """Sequence-preserving prefix for an order-sensitive engine prompt."""

    option_count: int
    min_count: int
    max_count: int
    selected_sequence: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_prefix_shape(
            self.option_count,
            self.min_count,
            self.max_count,
            self.selected_sequence,
        )

    @property
    def selected(self) -> tuple[int, ...]:
        """Return the exact decoder sequence."""
        return self.selected_sequence

    @property
    def remaining_mask(self) -> tuple[bool, ...]:
        """Return options that remain legal to append by uniqueness alone."""
        selected = frozenset(self.selected_sequence)
        return tuple(index not in selected for index in range(self.option_count))

    @property
    def selected_count(self) -> int:
        """Return the current selection cardinality."""
        return len(self.selected_sequence)

    @property
    def stop_allowed(self) -> bool:
        """Return whether STOP forms a legal complete selection."""
        return self.selected_count >= self.min_count

    @property
    def complete_at_max(self) -> bool:
        """Return whether the cardinality ceiling completes the action."""
        return self.selected_count == self.max_count

    @property
    def decoder_prefix_key(self) -> tuple[int, ...]:
        """Key the shared pointer-decoder sequence latent."""
        return self.selected_sequence

    def extend(self, option_index: int) -> OrderedSelectionPrefix:
        """Append one unused option without canonicalizing its position."""
        _validate_extension(self, option_index)
        return OrderedSelectionPrefix(
            option_count=self.option_count,
            min_count=self.min_count,
            max_count=self.max_count,
            selected_sequence=(*self.selected_sequence, option_index),
        )


@dataclass(frozen=True)
class UnorderedSelectionPrefix:
    """Canonical pool prefix for an engine-proven order-invariant prompt."""

    option_count: int
    min_count: int
    max_count: int
    selected_pool: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        canonical = tuple(sorted(self.selected_pool))
        if canonical != self.selected_pool:
            raise ValueError("unordered selected_pool must be canonical")
        _validate_prefix_shape(
            self.option_count,
            self.min_count,
            self.max_count,
            self.selected_pool,
        )

    @property
    def selected(self) -> tuple[int, ...]:
        """Return the canonical selected-option pool."""
        return self.selected_pool

    @property
    def remaining_mask(self) -> tuple[bool, ...]:
        """Return options outside the canonical selected pool."""
        selected = frozenset(self.selected_pool)
        return tuple(index not in selected for index in range(self.option_count))

    @property
    def selected_count(self) -> int:
        """Return the current selection cardinality."""
        return len(self.selected_pool)

    @property
    def stop_allowed(self) -> bool:
        """Return whether STOP forms a legal complete selection."""
        return self.selected_count >= self.min_count

    @property
    def complete_at_max(self) -> bool:
        """Return whether the cardinality ceiling completes the action."""
        return self.selected_count == self.max_count

    @property
    def decoder_prefix_key(self) -> tuple[int, ...]:
        """Key permutation-invariant pooling and its shared decoder latent."""
        return self.selected_pool

    def extend(self, option_index: int) -> UnorderedSelectionPrefix:
        """Add one option and canonicalize the resulting selected pool."""
        _validate_extension(self, option_index)
        return UnorderedSelectionPrefix(
            option_count=self.option_count,
            min_count=self.min_count,
            max_count=self.max_count,
            selected_pool=tuple(sorted((*self.selected_pool, option_index))),
        )


SelectionPrefix: TypeAlias = OrderedSelectionPrefix | UnorderedSelectionPrefix


def initial_selection_prefix(
    *,
    option_count: int,
    min_count: int,
    max_count: int,
    ordered: bool,
) -> SelectionPrefix:
    """Create an explicit prefix representation for known order semantics."""
    prefix_type = OrderedSelectionPrefix if ordered else UnorderedSelectionPrefix
    return prefix_type(
        option_count=option_count,
        min_count=min_count,
        max_count=max_count,
    )


def _validate_prefix_shape(
    option_count: int,
    min_count: int,
    max_count: int,
    selected: tuple[int, ...],
) -> None:
    if option_count < 0 or min_count < 0:
        raise ValueError("option_count and min_count must be non-negative")
    if min_count > max_count or max_count > option_count:
        raise ValueError("selection cardinality bounds are invalid")
    if len(selected) > max_count:
        raise ValueError("selection prefix exceeds max_count")
    if len(selected) != len(set(selected)):
        raise ValueError("selection prefix cannot repeat an option")
    if any(index < 0 or index >= option_count for index in selected):
        raise ValueError("selection prefix contains an invalid option index")


def _validate_extension(prefix: SelectionPrefix, option_index: int) -> None:
    if prefix.complete_at_max:
        raise ValueError("cannot extend a complete selection prefix")
    if option_index < 0 or option_index >= prefix.option_count:
        raise ValueError("option index is outside the prompt")
    if not prefix.remaining_mask[option_index]:
        raise ValueError("option is already selected")


__all__ = [
    "OrderedSelectionPrefix",
    "SelectionPrefix",
    "UnorderedSelectionPrefix",
    "initial_selection_prefix",
]
