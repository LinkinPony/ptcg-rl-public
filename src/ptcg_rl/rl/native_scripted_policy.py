"""Exact immutable scripted agents over cached native public observations."""

from __future__ import annotations

import json
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
)
from ptcg_rl.opponents.spec import BattleAgent
from ptcg_rl.rl.native_scripted_mixed75 import (
    NativeScriptedActionBatch,
    NativeScriptedBranch,
)
from ptcg_rl.rl.scripted_manifest import ResolvedScriptedOpponent

_PUBLIC_BRANCH: NativeScriptedBranch = "public"


class NativeScriptedPolicy(Protocol):
    """Game-local scripted policy lifecycle accepted by a native arena."""

    def clone_empty(self) -> Self:
        """Fork empty mutable per-slot state."""

    def reset(
        self,
        slots: npt.ArrayLike,
        seeds: npt.ArrayLike,
    ) -> None:
        """Bind fresh game state to selected native slots."""

    def release(self, slots: npt.ArrayLike) -> None:
        """Release selected terminal slots."""

    def act_batch(
        self,
        view: NativeTrainingBatchView,
        rows: npt.ArrayLike | None = None,
        *,
        lane: NativeTrainingLane | None = None,
    ) -> NativeScriptedActionBatch:
        """Return complete legal actions for selected native rows."""


class NativePublicScriptedPolicy:
    """Run one verified public agent instance per native game slot."""

    def __init__(self, resolved: ResolvedScriptedOpponent) -> None:
        """Retain the immutable factory and start with no live games."""
        self.resolved = resolved
        self._agents: dict[int, BattleAgent] = {}

    def clone_empty(self) -> NativePublicScriptedPolicy:
        """Fork lane-local agent state while sharing immutable code identity."""
        return NativePublicScriptedPolicy(self.resolved)

    def reset(
        self,
        slots: npt.ArrayLike,
        seeds: npt.ArrayLike,
    ) -> None:
        """Construct exact game-local agents from assignment-derived seeds."""
        slot_rows = _integer_vector(slots, dtype=np.uint32, name="slots")
        seed_rows = _integer_vector(seeds, dtype=np.uint32, name="seeds")
        if slot_rows.shape != seed_rows.shape:
            raise ValueError("native scripted reset slots and seeds must align")
        if np.unique(slot_rows).size != slot_rows.size:
            raise ValueError("native scripted reset slots must be unique")
        for slot, seed in zip(slot_rows, seed_rows, strict=True):
            self._agents[int(slot)] = self.resolved.build(seed=int(seed))

    def release(self, slots: npt.ArrayLike) -> None:
        """Forget completed public agents before their slots are reused."""
        slot_rows = _integer_vector(slots, dtype=np.uint32, name="slots")
        if np.unique(slot_rows).size != slot_rows.size:
            raise ValueError("native scripted released slots must be unique")
        missing = [int(slot) for slot in slot_rows if int(slot) not in self._agents]
        if missing:
            raise KeyError(f"native scripted slots were not live: {missing}")
        for slot in slot_rows:
            del self._agents[int(slot)]

    def act_batch(
        self,
        view: NativeTrainingBatchView,
        rows: npt.ArrayLike | None = None,
        *,
        lane: NativeTrainingLane | None = None,
    ) -> NativeScriptedActionBatch:
        """Invoke verified agents on exact engine-produced public observations."""
        if lane is None:
            raise ValueError("public scripted native inference requires its lane")
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
            raise ValueError("native scripted action rows are invalid")
        slots = view.slots[selected]
        missing = [int(slot) for slot in slots if int(slot) not in self._agents]
        if missing:
            raise KeyError(
                f"native scripted slots must be reset before action: {missing}"
            )
        observation_rows = lane.export_public_observations(slots)
        state_tokens = lane.export_public_state_tokens(slots)
        actions: list[tuple[int, ...]] = []
        for source_row, slot, encoded, token in zip(
            selected,
            slots,
            observation_rows,
            state_tokens,
            strict=True,
        ):
            observation = json.loads(encoded)
            if not isinstance(observation, dict):
                raise TypeError("native public observation must be a JSON object")
            observation["search_begin_input"] = token.decode("ascii")
            action = tuple(
                int(index) for index in self._agents[int(slot)].act(observation)
            )
            _require_legal(view, int(source_row), action)
            actions.append(action)
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
        branches = (_PUBLIC_BRANCH,) * len(actions)
        return NativeScriptedActionBatch(
            offsets=offsets,
            choices=np.fromiter(
                (choice for action in actions for choice in action),
                dtype=np.int32,
                count=int(offsets[-1]),
            ),
            branches=branches,
        )


def _require_legal(
    view: NativeTrainingBatchView,
    row: int,
    action: tuple[int, ...],
) -> None:
    minimum = int(view.select_min[row])
    maximum = int(view.select_max[row])
    option_count = int(view.option_offsets[row + 1] - view.option_offsets[row])
    if not minimum <= len(action) <= maximum:
        raise ValueError("native scripted policy emitted invalid action count")
    if len(set(action)) != len(action):
        raise ValueError("native scripted policy emitted duplicate option index")
    if any(index < 0 or index >= option_count for index in action):
        raise ValueError("native scripted policy emitted out-of-range option")


def _integer_vector(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
    name: str,
) -> npt.NDArray[np.integer]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size <= 0:
        raise ValueError(f"native scripted {name} must be a non-empty vector")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"native scripted {name} must use integers")
    return np.ascontiguousarray(raw, dtype=dtype)


__all__ = ["NativePublicScriptedPolicy", "NativeScriptedPolicy"]
