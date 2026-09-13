"""Thin lifecycle-safe wrappers around the engine Search and Battle APIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self, cast

from ptcg_rl.engine.protocols import ObservationInput, ObservationLike, SearchStateLike
from ptcg_rl.engine.runtime import load_cg_api, load_cg_game, to_engine_observation


@dataclass(frozen=True)
class HiddenInformation:
    """One determinization of hidden zones required by ``search_begin``."""

    your_deck: tuple[int, ...] = ()
    your_prize: tuple[int, ...] = ()
    opponent_deck: tuple[int, ...] = ()
    opponent_prize: tuple[int, ...] = ()
    opponent_hand: tuple[int, ...] = ()
    opponent_active: tuple[int, ...] = ()

    @classmethod
    def from_sequences(
        cls,
        *,
        your_deck: Sequence[int] = (),
        your_prize: Sequence[int] = (),
        opponent_deck: Sequence[int] = (),
        opponent_prize: Sequence[int] = (),
        opponent_hand: Sequence[int] = (),
        opponent_active: Sequence[int] = (),
    ) -> HiddenInformation:
        """Store engine hidden-zone inputs as immutable integer tuples."""
        return cls(
            your_deck=tuple(int(card_id) for card_id in your_deck),
            your_prize=tuple(int(card_id) for card_id in your_prize),
            opponent_deck=tuple(int(card_id) for card_id in opponent_deck),
            opponent_prize=tuple(int(card_id) for card_id in opponent_prize),
            opponent_hand=tuple(int(card_id) for card_id in opponent_hand),
            opponent_active=tuple(int(card_id) for card_id in opponent_active),
        )


class SearchSession:
    """Context manager for one Search API root and its child states."""

    def __init__(self, root: SearchStateLike, cg_api: Any | None = None) -> None:
        """Create a session from an already-started root state."""
        self._cg_api = cg_api if cg_api is not None else load_cg_api()
        self.root = root
        self._released: set[int] = set()
        self._live_children: set[int] = set()
        self._peak_live_children = 0
        self._closed = False

    @classmethod
    def begin(
        cls,
        observation: ObservationInput,
        hidden: HiddenInformation,
        *,
        manual_coin: bool = False,
    ) -> SearchSession:
        """Call ``search_begin`` and return an exception-safe session."""
        cg_api = load_cg_api()
        engine_observation = to_engine_observation(observation)
        root = cast(
            SearchStateLike,
            cg_api.search_begin(
                engine_observation,
                list(hidden.your_deck),
                list(hidden.your_prize),
                list(hidden.opponent_deck),
                list(hidden.opponent_prize),
                list(hidden.opponent_hand),
                list(hidden.opponent_active),
                manual_coin=manual_coin,
            ),
        )
        return cls(root, cg_api=cg_api)

    @property
    def observation(self) -> ObservationLike:
        """Return the root observation."""
        return self.root.observation

    @property
    def closed(self) -> bool:
        """Whether ``search_end`` has already been called."""
        return self._closed

    @property
    def live_state_count(self) -> int:
        """Return child states not yet explicitly released."""
        return len(self._live_children)

    @property
    def peak_live_state_count(self) -> int:
        """Return peak root-plus-child states owned by this session."""
        return 1 + self._peak_live_children

    def __enter__(self) -> Self:
        """Return this session for ``with`` blocks."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always end the engine search lifecycle."""
        del exc_type, exc, traceback
        self.close()

    def step(self, search_id: int, select: Sequence[int]) -> SearchStateLike:
        """Advance one search state by selecting option indices."""
        if self._closed:
            raise RuntimeError("cannot step a closed SearchSession")
        selected = [int(index) for index in select]
        successor = cast(
            SearchStateLike,
            self._cg_api.search_step(search_id, selected),
        )
        successor_id = int(successor.searchId)
        if successor_id != int(self.root.searchId):
            self._live_children.add(successor_id)
            self._peak_live_children = max(
                self._peak_live_children,
                len(self._live_children),
            )
        return successor

    def release(self, search_id: int) -> None:
        """Release one engine search state if it has not been released yet."""
        if self._closed or search_id in self._released:
            return
        self._cg_api.search_release(int(search_id))
        self._released.add(int(search_id))
        self._live_children.discard(int(search_id))

    def close(self) -> None:
        """End the search and allow the engine to reuse its memory."""
        if self._closed:
            return
        first_error: Exception | None = None
        try:
            for search_id in tuple(self._live_children):
                try:
                    self.release(search_id)
                except Exception as exc:  # Continue to the mandatory search_end.
                    if first_error is None:
                        first_error = exc
        finally:
            try:
                self._cg_api.search_end()
            finally:
                self._closed = True
                self._live_children.clear()
        if first_error is not None:
            raise first_error


class BattleSession:
    """Context manager for the perfect-information local Battle API."""

    def __init__(
        self,
        deck0: Sequence[int],
        deck1: Sequence[int],
        cg_game: Any | None = None,
    ) -> None:
        """Start a local battle immediately."""
        self._cg_game = cg_game if cg_game is not None else load_cg_game()
        observation, start_data = self._cg_game.battle_start(
            [int(card_id) for card_id in deck0],
            [int(card_id) for card_id in deck1],
        )
        self.observation_dict = cast(dict[str, Any], observation)
        self.start_data = start_data
        self._closed = False

    def __enter__(self) -> Self:
        """Return this session for ``with`` blocks."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always finish the local battle."""
        del exc_type, exc, traceback
        self.close()

    def select(self, select: Sequence[int]) -> dict[str, Any]:
        """Advance the battle by selecting option indices."""
        if self._closed:
            raise RuntimeError("cannot select on a closed BattleSession")
        self.observation_dict = cast(
            dict[str, Any],
            self._cg_game.battle_select([int(index) for index in select]),
        )
        return self.observation_dict

    def close(self) -> None:
        """Finish the local battle if it is still active."""
        if self._closed:
            return
        self._cg_game.battle_finish()
        self._closed = True
