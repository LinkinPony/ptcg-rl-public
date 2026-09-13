"""Pointer-based local Battle API pool for vectorized rollouts."""

from __future__ import annotations

import ctypes
import importlib
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self, cast

import orjson

from ptcg_rl.profiling import StageTimer, time_stage

Deck = tuple[int, ...]
DeckPair = tuple[Deck, Deck]
DeckPairSampler = Callable[[], DeckPair]
GameIdFactory = Callable[[], str]


@dataclass
class VectorGame:
    """One live local battle addressed by explicit engine pointer."""

    game_id: str
    battle_ptr: int
    deck_pair: DeckPair
    observation: dict[str, Any]
    steps: int = 0


@dataclass(frozen=True)
class FinishedGame:
    """Terminal game snapshot emitted by ``VectorBattlePool.finished``."""

    game_id: str
    battle_ptr: int
    deck_pair: DeckPair
    observation: dict[str, Any]
    winner_index: int
    steps: int


class VectorBattlePool:
    """Single-threaded pool of concurrent local battles using explicit pointers."""

    def __init__(
        self,
        num_games: int,
        deck_pair_sampler: DeckPairSampler,
        *,
        cg_sim_lib: Any | None = None,
        game_id_factory: GameIdFactory | None = None,
        include_search_input: bool = False,
        timer: StageTimer | None = None,
    ) -> None:
        """Start ``num_games`` battles immediately."""
        if num_games <= 0:
            raise ValueError("num_games must be positive")
        self._target_games = int(num_games)
        self._deck_pair_sampler = deck_pair_sampler
        self._lib = cg_sim_lib if cg_sim_lib is not None else load_cg_sim_lib()
        self._game_id_factory = game_id_factory or _uuid_game_id
        self._include_search_input = bool(include_search_input)
        self._timer = timer
        self._games: dict[str, VectorGame] = {}
        self._closed = False
        try:
            for _ in range(self._target_games):
                self._start_next_game()
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "failed to close already-started vector battles after "
                    f"pool initialization failed: {cleanup_exc!r}"
                )
            raise

    @property
    def closed(self) -> bool:
        """Whether ``close`` has been called."""
        return self._closed

    @property
    def live_games(self) -> tuple[VectorGame, ...]:
        """Return all currently live games, terminal or pending."""
        return tuple(self._games.values())

    def __enter__(self) -> Self:
        """Return this pool for ``with`` blocks."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always finish live battles when leaving a context manager."""
        del exc_type, exc, traceback
        self.close()

    def pending(self) -> list[VectorGame]:
        """Return live games that have not reached a terminal result."""
        self._raise_if_closed()
        return [
            game for game in self._games.values() if result_index(game.observation) < 0
        ]

    def submit(self, game_id: str, action: Sequence[int]) -> None:
        """Submit one action to a pending game and refresh its observation."""
        self._raise_if_closed()
        game = self._game_by_id(game_id)
        if result_index(game.observation) >= 0:
            raise RuntimeError(f"cannot submit to finished game: {game_id}")
        game.observation = select_pointer_battle(
            self._lib,
            game.battle_ptr,
            action,
            include_search_input=self._include_search_input,
            timer=self._timer,
        )
        game.steps += 1

    def recycle(self, game_id: str) -> None:
        """Discard one non-terminal game, finish its pointer, and refill the pool."""
        self._raise_if_closed()
        game = self._game_by_id(game_id)
        if result_index(game.observation) >= 0:
            raise RuntimeError(f"cannot recycle finished game: {game_id}")
        finish_pointer_battle(self._lib, game.battle_ptr)
        del self._games[game.game_id]
        self._start_next_game()

    def finished(self) -> list[FinishedGame]:
        """Take terminal games, finish their pointers, and refill the pool."""
        self._raise_if_closed()
        output: list[FinishedGame] = []
        for game in list(self._games.values()):
            winner_index = result_index(game.observation)
            if winner_index < 0:
                continue
            output.append(
                FinishedGame(
                    game_id=game.game_id,
                    battle_ptr=game.battle_ptr,
                    deck_pair=game.deck_pair,
                    observation=dict(game.observation),
                    winner_index=winner_index,
                    steps=game.steps,
                )
            )
            finish_pointer_battle(self._lib, game.battle_ptr)
            del self._games[game.game_id]
            self._start_next_game()
        return output

    def close(self) -> None:
        """Finish all live battle pointers."""
        if self._closed:
            return
        for game in list(self._games.values()):
            finish_pointer_battle(self._lib, game.battle_ptr)
        self._games.clear()
        self._closed = True

    def _start_next_game(self) -> None:
        game_id = self._new_game_id()
        game = start_pointer_battle(
            self._lib,
            _normalize_deck_pair(self._deck_pair_sampler()),
            game_id=game_id,
            include_search_input=self._include_search_input,
            timer=self._timer,
        )
        try:
            self._games[game_id] = game
        except BaseException as exc:
            _finish_pointer_after_error(
                self._lib,
                game.battle_ptr,
                original=exc,
                context=f"registering vector game {game_id!r}",
            )
            raise

    def _new_game_id(self) -> str:
        for _ in range(100):
            game_id = str(self._game_id_factory())
            if game_id not in self._games:
                return game_id
        raise RuntimeError("game_id_factory produced repeated ids")

    def _game_by_id(self, game_id: str) -> VectorGame:
        try:
            return self._games[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown vector game id: {game_id}") from exc

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("VectorBattlePool is closed")


def load_cg_sim_lib() -> Any:
    """Return the bundled ``cg.sim.lib`` object, loading libcg on demand."""
    try:
        sim = importlib.import_module("cg.sim")
    except ImportError as exc:
        raise ImportError(
            "Could not import cg.sim. Run with PYTHONPATH=data/sample_submission:src."
        ) from exc
    return sim.lib


def start_pointer_battle(
    lib: Any,
    deck_pair: DeckPair,
    *,
    game_id: str,
    include_search_input: bool = False,
    timer: StageTimer | None = None,
) -> VectorGame:
    """Start one pointer-addressed local battle."""
    normalized = _normalize_deck_pair(deck_pair)
    cards = (*normalized[0], *normalized[1])
    start_data = lib.BattleStart(_int_array(cards))
    battle_ptr = _battle_ptr(start_data)
    try:
        _raise_deck_error_if_any(start_data)
        observation = get_pointer_battle_data(
            lib,
            battle_ptr,
            include_search_input=include_search_input,
            timer=timer,
        )
        return VectorGame(
            game_id=game_id,
            battle_ptr=battle_ptr,
            deck_pair=normalized,
            observation=observation,
        )
    except BaseException as exc:
        _finish_pointer_after_error(
            lib,
            battle_ptr,
            original=exc,
            context=f"starting vector game {game_id!r}",
        )
        raise


def select_pointer_battle(
    lib: Any,
    battle_ptr: int,
    action: Sequence[int],
    *,
    include_search_input: bool = False,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    """Advance one pointer-addressed battle with option indices."""
    selected = tuple(int(index) for index in action)
    with time_stage(timer, "engine_select"):
        err = int(lib.Select(battle_ptr, _int_array(selected), len(selected)))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken")
        raise IndexError()
    return get_pointer_battle_data(
        lib,
        battle_ptr,
        include_search_input=include_search_input,
        timer=timer,
    )


def get_pointer_battle_data(
    lib: Any,
    battle_ptr: int,
    *,
    include_search_input: bool = False,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    """Read one pointer-addressed battle observation."""
    with time_stage(timer, "engine_get_battle_data"):
        serial_data = lib.GetBattleData(battle_ptr)
    raw_json = cast(bytes, serial_data.json)
    with time_stage(timer, "json_decode"):
        observation = orjson.loads(raw_json)
    if not isinstance(observation, dict):
        raise TypeError("engine battle observation must decode to a dictionary")
    if include_search_input:
        with time_stage(timer, "search_input_decode"):
            observation["search_begin_input"] = ctypes.string_at(
                serial_data.data,
                int(serial_data.count),
            ).decode("ascii")
    else:
        observation["search_begin_input"] = None
    return cast(dict[str, Any], observation)


def finish_pointer_battle(lib: Any, battle_ptr: int) -> None:
    """Finish one pointer-addressed local battle."""
    lib.BattleFinish(battle_ptr)


def _finish_pointer_after_error(
    lib: Any,
    battle_ptr: int,
    *,
    original: BaseException,
    context: str,
) -> None:
    """Best-effort finish an unowned pointer without masking ``original``."""
    try:
        finish_pointer_battle(lib, battle_ptr)
    except BaseException as cleanup_exc:
        original.add_note(
            f"failed to finish battle pointer {battle_ptr} after {context} "
            f"failed: {cleanup_exc!r}"
        )


def result_index(observation: Mapping[str, Any]) -> int:
    """Return the engine terminal result index, or -1 while non-terminal."""
    return _int_field(_field_value(observation, "current"), "result", -1)


def _normalize_deck_pair(deck_pair: tuple[Sequence[int], Sequence[int]]) -> DeckPair:
    deck0 = tuple(int(card_id) for card_id in deck_pair[0])
    deck1 = tuple(int(card_id) for card_id in deck_pair[1])
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("both decks must contain 60 cards")
    return (deck0, deck1)


def _battle_ptr(start_data: Any) -> int:
    raw_ptr = getattr(start_data, "battlePtr", None)
    if raw_ptr is None or int(raw_ptr) == 0:
        raise ValueError("BattleStart returned a null battle pointer")
    return int(raw_ptr)


def _raise_deck_error_if_any(start_data: Any) -> None:
    error_player = int(getattr(start_data, "errorPlayer", -1))
    if error_player < 0:
        return
    raise ValueError(
        f"deck error: player={error_player} "
        f"type={getattr(start_data, 'errorType', None)}"
    )


def _int_array(values: Sequence[int]) -> Any:
    return (ctypes.c_int * len(values))(*[int(value) for value in values])


def _field_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    raw_value = _field_value(value, name, default)
    return int(raw_value) if raw_value is not None else default


def _uuid_game_id() -> str:
    return str(uuid.uuid4())
