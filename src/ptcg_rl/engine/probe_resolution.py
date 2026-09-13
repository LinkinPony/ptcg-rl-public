"""Semantic resolution helpers for dynamic-effect engine probes.

A single engine step may stop at a follow-up prompt before the probed effect
has produced its consequences.  Probe features are valid only after the
effect reaches a terminal or MAIN boundary.  Prompts with exactly one useful
legal response may be advanced without adding policy semantics; every other
non-MAIN prompt remains unresolved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ptcg_rl.engine.constants import SelectContext

MAX_FORCED_PROBE_STEPS = 16

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ProbeTransition:
    """One Search API step with the states needed to ground its logs."""

    before_observation: Any
    after_observation: Any
    logs: tuple[Any, ...]


@dataclass(frozen=True)
class ProbeChainResult(Generic[StateT]):
    """Final state and logs from one root action plus forced continuations.

    The returned final state remains owned by the caller.  Intermediate states
    are released as soon as their child has been created.
    """

    state: StateT
    logs: tuple[Any, ...]
    resolved: bool
    forced_steps: int
    steps: int
    transitions: tuple[ProbeTransition, ...] = ()


def resolve_probe_chain(
    parent: StateT,
    root_action: Sequence[int],
    *,
    step: Callable[[StateT, tuple[int, ...]], StateT],
    release: Callable[[StateT], None],
    observation: Callable[[StateT], Any],
    max_forced_steps: int = MAX_FORCED_PROBE_STEPS,
) -> ProbeChainResult[StateT]:
    """Advance a root action through forced prompts to a semantic boundary.

    A terminal or MAIN successor is resolved.  A non-forced non-MAIN prompt,
    or a forced chain that reaches the safety cap, is unresolved and must not
    be exposed as a successful zero-effect probe.
    """
    if max_forced_steps < 0:
        raise ValueError("max_forced_steps must be non-negative")

    # Import lazily to keep the low-level engine package acyclic. Importing the
    # actions package while ``engine.forward_model`` is still initializing
    # reaches belief.search, which itself imports the forward model.
    from ptcg_rl.actions.selection import forced_action

    current_parent = parent
    action = tuple(int(index) for index in root_action)
    current: StateT | None = None
    logs: list[Any] = []
    transitions: list[ProbeTransition] = []
    forced_steps = 0
    steps = 0
    try:
        while True:
            before_observation = observation(current_parent)
            successor = step(current_parent, action)
            previous = current
            current = successor
            if previous is not None:
                release(previous)
            steps += 1

            successor_observation = observation(successor)
            step_logs = tuple(_sequence(_field(successor_observation, "logs", ())))
            logs.extend(step_logs)
            transitions.append(
                ProbeTransition(
                    before_observation=before_observation,
                    after_observation=successor_observation,
                    logs=step_logs,
                )
            )
            if is_resolved_probe_successor(successor_observation):
                return ProbeChainResult(
                    state=successor,
                    logs=tuple(logs),
                    resolved=True,
                    forced_steps=forced_steps,
                    steps=steps,
                    transitions=tuple(transitions),
                )

            continuation = forced_action(_field(successor_observation, "select"))
            if continuation is None or forced_steps >= max_forced_steps:
                return ProbeChainResult(
                    state=successor,
                    logs=tuple(logs),
                    resolved=False,
                    forced_steps=forced_steps,
                    steps=steps,
                    transitions=tuple(transitions),
                )
            current_parent = successor
            action = continuation
            forced_steps += 1
    except Exception:
        if current is not None:
            release(current)
        raise


def is_resolved_probe_successor(observation: Any) -> bool:
    """Return whether an engine successor has completed the probed effect."""
    current = _field(observation, "current")
    if _int_field(current, "result", -1) >= 0:
        return True
    select = _field(observation, "select")
    return _int_field(select, "context", -1) == int(SelectContext.MAIN)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field = _field(value, name, default)
    return int(field) if field is not None else default


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
