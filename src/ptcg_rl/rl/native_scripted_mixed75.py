"""Per-lane native-column runtime for immutable ``mixed75_71eb_v1``."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.rl.native_scripted_catalog import (
    MIXED75_71EB_DECK_DIGEST,
    NativeScriptedCatalog,
)
from ptcg_rl.rl.native_scripted_heuristic import NativeHeuristicChooser
from ptcg_rl.rl.native_scripted_state import NativeScriptedRow

NativeScriptedBranch = Literal["forced", "heuristic", "random", "public"]
_HEURISTIC_PROBABILITY = 0.75


@dataclass(frozen=True)
class NativeScriptedActionBatch:
    """Complete legal selections encoded as engine-ready CSR arrays."""

    offsets: npt.NDArray[np.uint32]
    choices: npt.NDArray[np.int32]
    branches: tuple[NativeScriptedBranch, ...]

    def __post_init__(self) -> None:
        if self.offsets.ndim != 1 or self.choices.ndim != 1:
            raise ValueError("native scripted actions must be rank-one CSR")
        if self.offsets.dtype != np.uint32 or self.choices.dtype != np.int32:
            raise TypeError("native scripted action CSR has an invalid dtype")
        if (
            self.offsets.shape != (len(self.branches) + 1,)
            or int(self.offsets[0]) != 0
            or int(self.offsets[-1]) != self.choices.size
            or bool(np.any(self.offsets[1:] < self.offsets[:-1]))
        ):
            raise ValueError("native scripted action CSR is malformed")

    def action(self, row: int) -> tuple[int, ...]:
        """Return one row's immutable selection."""
        if row < 0 or row >= len(self.branches):
            raise IndexError(f"native scripted action row is out of range: {row}")
        start = int(self.offsets[row])
        stop = int(self.offsets[row + 1])
        return tuple(int(index) for index in self.choices[start:stop])


class NativeMixed75Policy:
    """Exact mixed75 policy with one Python MT stream per native game slot."""

    def __init__(
        self,
        *,
        scripted_deck: Sequence[int],
        static_features: npt.NDArray[np.float32],
    ) -> None:
        deck = canonicalize_deck(scripted_deck)
        if deck.deck_digest != MIXED75_71EB_DECK_DIGEST:
            raise ValueError(
                "native mixed75 runtime only accepts immutable 71eb deck "
                f"{MIXED75_71EB_DECK_DIGEST}, got {deck.deck_digest}"
            )
        self.catalog = NativeScriptedCatalog(static_features)
        self._rng_by_slot: dict[int, random.Random] = {}

    def clone_empty(self) -> NativeMixed75Policy:
        """Fork one lane-local RNG runtime while sharing immutable features."""
        clone = object.__new__(NativeMixed75Policy)
        clone.catalog = self.catalog
        clone._rng_by_slot = {}
        return clone

    def reset(
        self,
        slots: npt.ArrayLike,
        seeds: npt.ArrayLike,
    ) -> None:
        """Bind fresh per-game random streams to reset native slots."""
        slot_rows = _integer_vector(slots, dtype=np.uint32, name="slots")
        seed_rows = _integer_vector(seeds, dtype=np.uint32, name="seeds")
        if slot_rows.shape != seed_rows.shape:
            raise ValueError("native scripted reset slots and seeds must align")
        if np.unique(slot_rows).size != slot_rows.size:
            raise ValueError("native scripted reset slots must be unique")
        for slot, seed in zip(slot_rows, seed_rows, strict=True):
            self._rng_by_slot[int(slot)] = random.Random(int(seed))

    def release(self, slots: npt.ArrayLike) -> None:
        """Forget completed game streams so slot reuse must reset explicitly."""
        slot_rows = _integer_vector(slots, dtype=np.uint32, name="slots")
        if np.unique(slot_rows).size != slot_rows.size:
            raise ValueError("native scripted released slots must be unique")
        missing = [
            int(slot) for slot in slot_rows if int(slot) not in self._rng_by_slot
        ]
        if missing:
            raise KeyError(f"native scripted slots were not live: {missing}")
        for slot in slot_rows:
            del self._rng_by_slot[int(slot)]

    def act_batch(
        self,
        view: NativeTrainingBatchView,
        rows: npt.ArrayLike | None = None,
        *,
        lane: object | None = None,
    ) -> NativeScriptedActionBatch:
        """Choose exact actions directly from aligned native SoA/CSR columns."""
        del lane
        selected = (
            np.arange(view.batch_size, dtype=np.int64)
            if rows is None
            else _integer_vector(rows, dtype=np.int64, name="rows")
        )
        if (
            selected.size <= 0
            or np.any(selected < 0)
            or np.any(selected >= view.batch_size)
            or np.unique(selected).size != selected.size
        ):
            raise ValueError(
                "native scripted action rows must be non-empty, unique, and in range"
            )
        slots = [int(view.slots[row]) for row in selected]
        missing = [slot for slot in slots if slot not in self._rng_by_slot]
        if missing:
            raise KeyError(
                f"native scripted slots must be reset before action: {missing}"
            )

        # A failed batch is never submitted to the engine.  Restore every
        # touched stream so retry/diagnostics cannot silently drift RNG state.
        rng_states = {slot: self._rng_by_slot[slot].getstate() for slot in slots}
        actions: list[tuple[int, ...]] = []
        branches: list[NativeScriptedBranch] = []
        try:
            for row_index, slot in zip(selected, slots, strict=True):
                state = NativeScriptedRow(view, int(row_index))
                action, branch = self._act_row(
                    state,
                    self._rng_by_slot[slot],
                )
                _require_legal(state, action)
                actions.append(action)
                branches.append(branch)
        except BaseException:
            for slot, rng_state in rng_states.items():
                self._rng_by_slot[slot].setstate(rng_state)
            raise

        offsets = np.zeros(len(actions) + 1, dtype=np.uint32)
        np.cumsum(
            np.fromiter(
                (len(action) for action in actions),
                dtype=np.uint32,
                count=len(actions),
            ),
            dtype=np.uint32,
            out=offsets[1:],
        )
        choices = np.fromiter(
            (index for action in actions for index in action),
            dtype=np.int32,
            count=int(offsets[-1]),
        )
        return NativeScriptedActionBatch(
            offsets=offsets,
            choices=choices,
            branches=tuple(branches),
        )

    def _act_row(
        self,
        row: NativeScriptedRow,
        rng: random.Random,
    ) -> tuple[tuple[int, ...], NativeScriptedBranch]:
        forced = _forced_action(row)
        if forced is not None:
            return forced, "forced"
        if rng.random() < _HEURISTIC_PROBABILITY:
            return NativeHeuristicChooser(row, self.catalog).choose(), "heuristic"
        return _random_legal_action(row, rng), "random"


def _forced_action(row: NativeScriptedRow) -> tuple[int, ...] | None:
    if row.maximum == 0:
        return ()
    if row.option_count == 1 and row.minimum == 1 and row.maximum == 1:
        return (0,)
    return None


def _random_legal_action(
    row: NativeScriptedRow,
    rng: random.Random,
) -> tuple[int, ...]:
    if row.option_count <= 0 or row.maximum <= 0:
        return ()
    count = rng.randint(row.minimum, row.maximum)
    if count <= 0:
        return ()
    if count >= row.option_count:
        return tuple(range(row.option_count))
    return tuple(rng.sample(range(row.option_count), count))


def _require_legal(
    row: NativeScriptedRow,
    action: tuple[int, ...],
) -> None:
    if not row.minimum <= len(action) <= row.maximum:
        raise ValueError("native scripted policy emitted invalid action count")
    if len(set(action)) != len(action):
        raise ValueError("native scripted policy emitted duplicate option index")
    if any(index < 0 or index >= row.option_count for index in action):
        raise ValueError("native scripted policy emitted out-of-range option")


def _integer_vector(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size <= 0:
        raise ValueError(f"native scripted {name} must be a non-empty vector")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"native scripted {name} must use an integer dtype")
    return np.ascontiguousarray(raw, dtype=dtype)


__all__ = [
    "NativeMixed75Policy",
    "NativeScriptedActionBatch",
    "NativeScriptedBranch",
]
