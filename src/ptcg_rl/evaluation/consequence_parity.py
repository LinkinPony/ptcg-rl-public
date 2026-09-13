"""Public-Search reference and native decision-transition parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.agent.search.context import public_search_observation
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.native_consequence_payload import (
    NativeConsequenceEndpoint,
)
from ptcg_rl.engine.native_macro_validation import (
    merge_search_logs,
)
from ptcg_rl.engine.search_api_probe import SearchApiProbeBackend
from ptcg_rl.engine.session import HiddenInformation

NATIVE_FORCED_STEP_CAP_ERROR = 90
_MAIN_SELECT_TYPE = 0


class SearchDecisionBackend(Protocol):
    """Minimal public Search lifecycle used by the parity reference."""

    def begin(
        self,
        state_token: str,
        hidden: HiddenInformation,
    ) -> Mapping[str, Any]:
        """Start one manual-coin Search root."""

    def step(
        self,
        search_id: int,
        action: Sequence[int],
    ) -> Mapping[str, Any]:
        """Advance one Search state."""

    def release(self, search_id: int) -> None:
        """Release one non-root Search state."""

    def end(self) -> None:
        """End the current Search lifecycle."""


class PublicSearchDecisionBackend:
    """Mapping-level adapter over the bundled public Search C API."""

    def __init__(self, *, manual_coin: bool = True) -> None:
        self._backend = SearchApiProbeBackend(manual_coin=manual_coin)

    def begin(
        self,
        state_token: str,
        hidden: HiddenInformation,
    ) -> Mapping[str, Any]:
        """Start a Search root with explicit manual-coin prompts enabled."""
        return self._backend._begin(  # pylint: disable=protected-access
            {"search_begin_input": state_token},
            hidden,
        )

    def step(
        self,
        search_id: int,
        action: Sequence[int],
    ) -> Mapping[str, Any]:
        """Advance one public Search state."""
        return self._backend._step(  # pylint: disable=protected-access
            search_id,
            action,
        )

    def release(self, search_id: int) -> None:
        """Release a public Search child state."""
        self._backend._release(search_id)  # pylint: disable=protected-access

    def end(self) -> None:
        """End a public Search root."""
        self._backend._end()  # pylint: disable=protected-access


@dataclass(frozen=True, slots=True)
class ReferenceDecisionTransition:
    """One root selection plus only forced prompts in the public Search API."""

    error_code: int
    endpoint: NativeConsequenceEndpoint
    root_observation: Mapping[str, Any]
    leaf_observation: Mapping[str, Any] | None
    leaf_actor_observation: Mapping[str, Any] | None
    leaf_player: int | None
    transition_steps: int
    forced_steps: int
    public_search_prize_defect_exposed: bool


def read_public_search_root(
    backend: SearchDecisionBackend,
    *,
    state_token: str,
    hidden: HiddenInformation,
    root_player: int,
) -> Mapping[str, Any]:
    """Read and privacy-project one public Search root, then close it."""
    root = backend.begin(state_token, hidden)
    try:
        observation = _observation(root)
        projected = root_visible_leaf_projection(
            observation,
            root_player=root_player,
        )
        current = _mapping(projected.get("current"))
        if int(current.get("yourIndex", -1)) != root_player:
            raise ValueError("public Search root player differs from replay root")
        return projected
    finally:
        backend.end()


def execute_public_decision(
    backend: SearchDecisionBackend,
    *,
    state_token: str,
    hidden: HiddenInformation,
    candidate_action: Sequence[int],
    root_player: int,
    max_forced_steps: int,
) -> ReferenceDecisionTransition:
    """Advance a complete root selection and only uniquely forced prompts."""
    if root_player not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    if max_forced_steps < 0:
        raise ValueError("max_forced_steps must be non-negative")
    root = backend.begin(state_token, hidden)
    root_search_id = _search_id(root)
    current = root
    current_search_id = root_search_id
    accumulated_logs: tuple[Mapping[str, Any], ...] = ()
    transition_steps = 0
    forced_steps = 0
    try:
        root_raw_observation = _observation(root)
        prize_defect_exposed = _has_visible_prize_identity(root_raw_observation)
        root_observation = root_visible_leaf_projection(
            root_raw_observation,
            root_player=root_player,
        )
        next_action = tuple(int(value) for value in candidate_action)
        while True:
            successor = backend.step(current_search_id, next_action)
            successor_id = _search_id(successor)
            if current_search_id != root_search_id:
                backend.release(current_search_id)
            current = successor
            current_search_id = successor_id
            transition_steps += 1
            raw_observation = _observation(current)
            accumulated_logs = merge_search_logs(
                accumulated_logs,
                mapping_logs(raw_observation),
            )
            endpoint = _semantic_endpoint(raw_observation, root_player)
            if endpoint is not None:
                leaf_player = _leaf_player(raw_observation)
                leaf = root_visible_leaf_projection(
                    raw_observation,
                    root_player=root_player,
                    logs=accumulated_logs,
                )
                return ReferenceDecisionTransition(
                    error_code=0,
                    endpoint=endpoint,
                    root_observation=root_observation,
                    leaf_observation=leaf,
                    leaf_actor_observation=(
                        None
                        if leaf_player is None
                        else root_visible_leaf_projection(
                            raw_observation,
                            root_player=leaf_player,
                            logs=accumulated_logs,
                        )
                    ),
                    leaf_player=leaf_player,
                    transition_steps=transition_steps,
                    forced_steps=forced_steps,
                    public_search_prize_defect_exposed=prize_defect_exposed,
                )

            forced = forced_action(_mapping(raw_observation.get("select")))
            if forced is None:
                leaf_player = _leaf_player(raw_observation)
                leaf = root_visible_leaf_projection(
                    raw_observation,
                    root_player=root_player,
                    logs=accumulated_logs,
                )
                return ReferenceDecisionTransition(
                    error_code=0,
                    endpoint=NativeConsequenceEndpoint.ROOT_STRATEGIC_PROMPT,
                    root_observation=root_observation,
                    leaf_observation=leaf,
                    leaf_actor_observation=(
                        None
                        if leaf_player is None
                        else root_visible_leaf_projection(
                            raw_observation,
                            root_player=leaf_player,
                            logs=accumulated_logs,
                        )
                    ),
                    leaf_player=leaf_player,
                    transition_steps=transition_steps,
                    forced_steps=forced_steps,
                    public_search_prize_defect_exposed=prize_defect_exposed,
                )
            if forced_steps >= max_forced_steps:
                return ReferenceDecisionTransition(
                    error_code=NATIVE_FORCED_STEP_CAP_ERROR,
                    endpoint=NativeConsequenceEndpoint.INVALID,
                    root_observation=root_observation,
                    leaf_observation=None,
                    leaf_actor_observation=None,
                    leaf_player=None,
                    transition_steps=transition_steps,
                    forced_steps=forced_steps,
                    public_search_prize_defect_exposed=prize_defect_exposed,
                )
            next_action = forced
            forced_steps += 1
    finally:
        if current_search_id != root_search_id:
            backend.release(current_search_id)
        backend.end()


def root_visible_leaf_projection(
    observation: Mapping[str, Any],
    *,
    root_player: int,
    logs: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Project a Search leaf to the immutable root information boundary."""
    source = dict(observation)
    if logs is not None:
        source["logs"] = list(logs)
    projected = public_search_observation(
        source,
        perspective_player_index=root_player,
    )
    return {
        "select": projected.get("select"),
        "logs": projected.get("logs", ()),
        "current": projected.get("current"),
    }


def _semantic_endpoint(
    observation: Mapping[str, Any],
    root_player: int,
) -> NativeConsequenceEndpoint | None:
    current = _mapping(observation.get("current"))
    result = int(current.get("result", -1))
    if result >= 0:
        return NativeConsequenceEndpoint.TERMINAL
    select = _mapping(observation.get("select"))
    context = int(select.get("context", -1))
    if context == int(SelectContext.COIN_HEAD):
        return NativeConsequenceEndpoint.CHANCE_PROMPT
    leaf_player = int(current.get("yourIndex", -1))
    if leaf_player != root_player:
        return NativeConsequenceEndpoint.TURN_HANDOFF
    select_type = int(select.get("type", -1))
    if context == int(SelectContext.MAIN) and select_type == _MAIN_SELECT_TYPE:
        return NativeConsequenceEndpoint.SAME_SEAT_MAIN
    return None


def _leaf_player(observation: Mapping[str, Any]) -> int | None:
    """Return the actual nonterminal actor selected by the engine state."""
    current = _mapping(observation.get("current"))
    if int(current.get("result", -1)) >= 0:
        return None
    player = int(current.get("yourIndex", -1))
    if player not in (0, 1):
        raise ValueError("Search leaf has no valid actor player")
    return player


def _observation(search_state: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(search_state.get("observation"))


def _search_id(search_state: Mapping[str, Any]) -> int:
    return int(search_state.get("searchId", 0))


def mapping_logs(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return only mapping log rows from a Search observation."""
    value = observation.get("logs", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Search payload is missing a required mapping")
    return cast(Mapping[str, Any], value)


def _has_visible_prize_identity(observation: Mapping[str, Any]) -> bool:
    current = _mapping(observation.get("current"))
    players = current.get("players", ())
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return False
    for player in players:
        if not isinstance(player, Mapping):
            continue
        prizes = player.get("prize", ())
        if not isinstance(prizes, Sequence) or isinstance(prizes, (str, bytes)):
            continue
        if any(isinstance(card, Mapping) and card.get("id") for card in prizes):
            return True
    return False


__all__ = [
    "NATIVE_FORCED_STEP_CAP_ERROR",
    "PublicSearchDecisionBackend",
    "ReferenceDecisionTransition",
    "SearchDecisionBackend",
    "execute_public_decision",
    "read_public_search_root",
    "mapping_logs",
    "root_visible_leaf_projection",
]
