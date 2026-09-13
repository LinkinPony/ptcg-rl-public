"""Action assembly and legality checks for batched native rollout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
)
from ptcg_rl.rl.native_scripted_mixed75 import NativeScriptedActionBatch

Int64Array = npt.NDArray[np.int64]
UInt32Array = npt.NDArray[np.uint32]
Int32Array = npt.NDArray[np.int32]


class NativeActionAccumulator:
    """Fill one complete engine action per native output row exactly once."""

    def __init__(self, view: NativeTrainingBatchView) -> None:
        if view.batch_size <= 0:
            raise ValueError("native action batch must be non-empty")
        self.view = view
        option_counts = np.diff(view.option_offsets.astype(np.int64, copy=False))
        maximum = np.minimum(
            np.maximum(view.select_max.astype(np.int64, copy=False), 0),
            option_counts,
        )
        self._choice_capacity = max(int(maximum.max(initial=0)), 1)
        self._lengths = np.full(view.batch_size, -1, dtype=np.int32)
        self._choices = np.zeros(
            (view.batch_size, self._choice_capacity),
            dtype=np.int32,
        )

    def fill_forced(self) -> Int64Array:
        """Fill zero-cost forced prompts and return their absolute rows."""
        option_counts = cast(
            Int64Array,
            np.diff(self.view.option_offsets.astype(np.int64, copy=False)),
        )
        minimum = np.minimum(
            np.maximum(self.view.select_min.astype(np.int64, copy=False), 0),
            option_counts,
        )
        maximum = np.minimum(
            np.maximum(
                minimum,
                self.view.select_max.astype(np.int64, copy=False),
            ),
            option_counts,
        )
        empty = maximum == 0
        singleton = (option_counts == 1) & (minimum == 1) & (maximum == 1)
        rows = np.flatnonzero(empty | singleton).astype(np.int64, copy=False)
        self._lengths[rows] = singleton[rows].astype(np.int32, copy=False)
        return cast(Int64Array, rows)

    def fill_trace(
        self,
        rows: npt.ArrayLike,
        trace: NativePolicyNumpyTrace,
    ) -> None:
        """Fill sampled model actions aligned to the supplied absolute rows."""
        self._fill_policy_actions(
            rows,
            action_offsets=trace.action_offsets,
            action_choices=trace.action_choices,
            batch_size=trace.batch_size,
        )

    def fill_action_batch(
        self,
        rows: npt.ArrayLike,
        actions: NativePolicyNumpyActionBatch,
    ) -> None:
        """Fill an action-only opponent inference result."""
        self._fill_policy_actions(
            rows,
            action_offsets=actions.action_offsets,
            action_choices=actions.action_choices,
            batch_size=actions.batch_size,
        )

    def _fill_policy_actions(
        self,
        rows: npt.ArrayLike,
        *,
        action_offsets: Int64Array,
        action_choices: Int32Array,
        batch_size: int,
    ) -> None:
        """Scatter compact policy CSR without per-row Python actions."""
        selected = _validated_rows(rows, size=self.view.batch_size)
        if batch_size != selected.size:
            raise ValueError("native policy actions do not align with action rows")
        offsets = np.asarray(action_offsets)
        choices = np.asarray(action_choices)
        if (
            offsets.shape != (batch_size + 1,)
            or not np.issubdtype(offsets.dtype, np.integer)
            or choices.ndim != 1
            or not np.issubdtype(choices.dtype, np.integer)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != int(choices.size)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("native policy action CSR is malformed")
        if np.any(self._lengths[selected] >= 0):
            raise ValueError("native action row was filled more than once")
        lengths = np.diff(offsets).astype(np.int32, copy=False)
        if np.any(lengths > self._choice_capacity):
            raise ValueError("native policy action exceeds the prompt maximum")
        self._lengths[selected] = lengths
        if choices.size == 0:
            return
        target_rows = np.repeat(selected, lengths)
        positions = np.arange(choices.size, dtype=np.int64) - np.repeat(
            offsets[:-1],
            lengths,
        )
        self._choices[target_rows, positions] = choices.astype(np.int32, copy=False)

    def fill_scripted(
        self,
        rows: npt.ArrayLike,
        actions: NativeScriptedActionBatch,
    ) -> None:
        """Fill one immutable scripted action batch."""
        selected = _validated_rows(rows, size=self.view.batch_size)
        if len(actions.branches) != selected.size:
            raise ValueError("native scripted actions do not align with rows")
        self.fill_actions(
            selected,
            tuple(actions.action(row) for row in range(selected.size)),
        )

    def fill_actions(
        self,
        rows: npt.ArrayLike,
        actions: Sequence[Sequence[int]],
    ) -> None:
        """Fill explicitly decoded actions aligned to absolute rows."""
        selected = _validated_rows(rows, size=self.view.batch_size)
        if len(actions) != selected.size:
            raise ValueError("native actions do not align with selected rows")
        if np.any(self._lengths[selected] >= 0):
            raise ValueError("native action row was filled more than once")
        for raw_row, raw_action in zip(selected, actions, strict=True):
            row = int(raw_row)
            action = np.asarray(tuple(raw_action), dtype=np.int32)
            if action.size > self._choice_capacity:
                raise ValueError("native action exceeds the prompt maximum")
            self._lengths[row] = int(action.size)
            self._choices[row, : action.size] = action

    def finish(
        self,
        *,
        rows: npt.ArrayLike | None = None,
    ) -> tuple[UInt32Array, Int32Array]:
        """Validate selected rows and return engine-ready action CSR arrays.

        ``rows`` restricts the CSR output to an ordered acting subset; rows
        outside the subset are intentionally left unfilled (for example rows
        parked for a batched frozen-opponent release in a later wave).
        """
        selected = (
            np.arange(self.view.batch_size, dtype=np.int64)
            if rows is None
            else _validated_rows(rows, size=self.view.batch_size)
        )
        lengths = self._lengths[selected].astype(np.int64, copy=False)
        missing = selected[lengths < 0]
        if missing.size:
            raise RuntimeError(
                f"native action rows were not filled: {missing[:8].tolist()}"
            )
        choices = self._choices[selected]
        _require_legal_batch(
            self.view,
            rows=selected,
            lengths=lengths,
            choices=choices,
        )
        offsets: UInt32Array = np.zeros(
            selected.size + 1,
            dtype=np.uint32,
        )
        np.cumsum(
            lengths,
            dtype=np.uint32,
            out=offsets[1:],
        )
        valid = np.arange(self._choice_capacity, dtype=np.int64)[None, :] < (
            lengths[:, None]
        )
        return offsets, choices[valid].astype(np.int32, copy=False)


def _require_legal_batch(
    view: NativeTrainingBatchView,
    *,
    rows: Int64Array,
    lengths: Int64Array,
    choices: npt.NDArray[np.int32],
) -> None:
    """Validate one dense action buffer with vectorized prompt checks."""
    option_counts = np.diff(view.option_offsets.astype(np.int64, copy=False))[rows]
    minimum = np.minimum(
        np.maximum(view.select_min[rows].astype(np.int64, copy=False), 0),
        option_counts,
    )
    maximum = np.minimum(
        np.maximum(
            minimum,
            view.select_max[rows].astype(np.int64, copy=False),
        ),
        option_counts,
    )
    if np.any(lengths < minimum) or np.any(lengths > maximum):
        raise ValueError("native policy emitted an invalid action count")
    valid = np.arange(choices.shape[1], dtype=np.int64)[None, :] < lengths[:, None]
    if np.any(valid & ((choices < 0) | (choices >= option_counts[:, None]))):
        raise ValueError("native policy emitted an out-of-range option index")
    ordered = np.sort(
        np.where(valid, choices.astype(np.int64, copy=False), option_counts[:, None]),
        axis=1,
    )
    adjacent_valid = np.arange(max(choices.shape[1] - 1, 0))[None, :] < (
        lengths[:, None] - 1
    )
    if np.any(adjacent_valid & (ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError("native policy emitted duplicate option indices")


def _validated_rows(
    values: npt.ArrayLike,
    *,
    size: int,
) -> Int64Array:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise ValueError("native action rows must be a non-empty vector")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("native action rows must use an integer dtype")
    normalized: Int64Array = rows.astype(np.int64, copy=False)
    if (
        np.any(normalized < 0)
        or np.any(normalized >= size)
        or np.unique(normalized).size != normalized.size
    ):
        raise ValueError("native action rows must be unique and in range")
    return normalized


__all__ = ["NativeActionAccumulator"]
