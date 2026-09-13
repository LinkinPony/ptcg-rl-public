"""Deterministic concatenated replay traces for S2 long-horizon stress."""

from __future__ import annotations

import glob
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps, replay_stub


@dataclass(frozen=True)
class FrozenTraceObservation:
    """One candidate callback located within a global concatenated trace."""

    global_step: int
    callback_index: int
    source_replay_index: int
    source_episode_id: int
    source_step: int
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenTraceCompletion:
    """Trace coverage counters returned after fully consuming an iterator."""

    global_steps: int
    callbacks: int
    source_replays: int


class ScaledVirtualClock:
    """Scale real elapsed time while preserving one monotonic virtual origin."""

    def __init__(
        self,
        slowdown_factor: float,
        *,
        real_clock: Any = time.perf_counter,
    ) -> None:
        self.slowdown_factor = float(slowdown_factor)
        self._real_clock = real_clock
        self._real_origin = float(real_clock())

    def __call__(self) -> float:
        """Return elapsed virtual seconds since construction."""
        return (float(self._real_clock()) - self._real_origin) * self.slowdown_factor


def resolve_stress_replays(
    replay_paths: Sequence[Path],
    replay_glob: str,
) -> tuple[Path, ...]:
    """Resolve a stable ordered replay source panel."""
    if replay_paths:
        paths = tuple(records.repo_path(path) for path in replay_paths)
    else:
        paths = tuple(
            Path(path)
            for path in sorted(glob.glob(str(records.repo_path(Path(replay_glob)))))
        )
    if not paths:
        raise ValueError("no replay paths matched trace stress")
    return paths


def iter_frozen_trace(
    replay_paths: Sequence[Path],
    *,
    team_name: str,
    seat: int,
    global_step_limit: int,
    chunk_size: int,
) -> Iterator[FrozenTraceObservation | FrozenTraceCompletion]:
    """Concatenate same-seat replay steps and yield only candidate callbacks.

    Global steps include inactive/opponent steps. Runtime state is intentionally
    not reset at source replay boundaries; this is a synthetic budget/deadline
    trace, not a claim that concatenated observations form one legal game.
    """
    global_step = 0
    callbacks = 0
    source_replays = 0
    for replay_path in replay_paths:
        metadata = replay_stub(replay_path, chunk_size=chunk_size)
        if _team_seat(metadata, team_name) != seat:
            continue
        source_replays += 1
        episode_id = int(
            _mapping(metadata.get("info")).get("EpisodeId", replay_path.stem)
        )
        for source_step, sides in iter_replay_steps(
            replay_path,
            chunk_size=chunk_size,
        ):
            if global_step >= global_step_limit:
                yield FrozenTraceCompletion(
                    global_steps=global_step,
                    callbacks=callbacks,
                    source_replays=source_replays,
                )
                return
            current_global_step = global_step
            global_step += 1
            if seat >= len(sides):
                continue
            side = sides[seat]
            observation = _mapping(side.get("observation"))
            if str(side.get("status", "")) != "ACTIVE":
                continue
            if not isinstance(observation.get("select"), Mapping):
                continue
            if _int_field(observation.get("current"), "yourIndex", -1) != seat:
                continue
            yield FrozenTraceObservation(
                global_step=current_global_step,
                callback_index=callbacks,
                source_replay_index=source_replays - 1,
                source_episode_id=episode_id,
                source_step=source_step,
                observation=observation,
            )
            callbacks += 1
    yield FrozenTraceCompletion(
        global_steps=global_step,
        callbacks=callbacks,
        source_replays=source_replays,
    )


def first_trace_observation(
    replay_paths: Sequence[Path],
    *,
    team_name: str,
    seat: int,
    chunk_size: int,
) -> FrozenTraceObservation:
    """Return the first actionable callback for one frozen seat trace."""
    for item in iter_frozen_trace(
        replay_paths,
        team_name=team_name,
        seat=seat,
        global_step_limit=1_000,
        chunk_size=chunk_size,
    ):
        if isinstance(item, FrozenTraceObservation):
            return item
    raise ValueError(f"trace contains no actionable callback for seat {seat}")


def _team_seat(metadata: Mapping[str, Any], team_name: str) -> int | None:
    team_names = _sequence(_mapping(metadata.get("info")).get("TeamNames"))
    matches = [
        index for index, value in enumerate(team_names) if str(value) == team_name
    ]
    if len(matches) > 1:
        raise ValueError(f"team {team_name!r} appears in multiple replay seats")
    return matches[0] if matches else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    if isinstance(value, Mapping):
        item = value.get(name, default)
    else:
        item = getattr(value, name, default)
    return int(item) if item is not None else default
