"""Probe direct pointer-based Battle API behavior for vectorized rollouts."""

from __future__ import annotations

import json
import random
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.selection import random_legal_action
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.vector_battle import (
    finish_pointer_battle,
    get_pointer_battle_data,
    load_cg_sim_lib,
    result_index,
    select_pointer_battle,
    start_pointer_battle,
)


class VectorBattleProbeConfig(BaseModel):
    """Hydra-backed config for pointer Battle API probes."""

    model_config = ConfigDict(extra="forbid")

    deck_path: Path = Path("data/sample_submission/deck.csv")
    output_path: Path | None = Path("outputs/engine/vector_probe/summary.json")
    stability_concurrency: tuple[int, ...] = (64, 256)
    throughput_concurrency: tuple[int, ...] = (1, 256)
    throughput_selects: int = 10_000
    leak_games: int = 10_000
    memory_sample_every: int = 100
    max_steps_per_game: int = 10_000
    max_rss_growth_mb: float | None = 256.0
    seed: int = 0
    fail_on_error: bool = True

    @field_validator("stability_concurrency", "throughput_concurrency")
    @classmethod
    def valid_concurrency(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject empty or non-positive concurrency lists."""
        if not value:
            raise ValueError("concurrency lists must be non-empty")
        if any(item <= 0 for item in value):
            raise ValueError("concurrency values must be positive")
        return value

    @field_validator(
        "throughput_selects",
        "leak_games",
        "memory_sample_every",
        "max_steps_per_game",
    )
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive probe limits."""
        if value <= 0:
            raise ValueError("probe limits must be positive")
        return value

    @field_validator("max_rss_growth_mb")
    @classmethod
    def valid_rss_threshold(cls, value: float | None) -> float | None:
        """Reject negative memory growth thresholds."""
        if value is not None and value < 0.0:
            raise ValueError("max_rss_growth_mb must be non-negative when set")
        return value


@dataclass
class _PointerBattle:
    """One live engine battle addressed by explicit ``battlePtr``."""

    game_id: int
    battle_ptr: int
    observation: dict[str, Any]
    selects: int = 0
    closed: bool = False


def run_vector_battle_probe(config: VectorBattleProbeConfig) -> dict[str, Any]:
    """Run all configured direct-pointer Battle API probes."""
    lib = load_cg_sim_lib()
    deck = tuple(records.read_deck(records.repo_path(config.deck_path)))
    rng = random.Random(config.seed)

    stability_rows = [
        _run_stability_probe(
            lib=lib,
            deck=deck,
            concurrency=concurrency,
            max_steps=config.max_steps_per_game,
            rng=rng,
        )
        for concurrency in config.stability_concurrency
    ]
    throughput_rows = [
        _run_throughput_probe(
            lib=lib,
            deck=deck,
            concurrency=concurrency,
            target_selects=config.throughput_selects,
            max_steps=config.max_steps_per_game,
            rng=rng,
        )
        for concurrency in config.throughput_concurrency
    ]
    memory_row = _run_memory_probe(
        lib=lib,
        deck=deck,
        games=config.leak_games,
        sample_every=config.memory_sample_every,
        max_steps=config.max_steps_per_game,
        max_rss_growth_mb=config.max_rss_growth_mb,
        rng=rng,
    )
    illegal_row = _run_illegal_action_probe(lib=lib, deck=deck, rng=rng)
    checks = {
        "stability": all(bool(row["passed"]) for row in stability_rows),
        "throughput": all(bool(row["passed"]) for row in throughput_rows),
        "memory": bool(memory_row["passed"]),
        "illegal_action": bool(illegal_row["passed"]),
    }
    output = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "checks": checks,
        "passed": all(checks.values()),
        "stability": stability_rows,
        "throughput": throughput_rows,
        "memory": memory_row,
        "illegal_action": illegal_row,
    }
    _write_summary(config.output_path, output)
    if config.fail_on_error and not bool(output["passed"]):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("vector battle probe failed: " + ", ".join(failed))
    return output


def _run_stability_probe(
    *,
    lib: Any,
    deck: Sequence[int],
    concurrency: int,
    max_steps: int,
    rng: random.Random,
) -> dict[str, Any]:
    games = [_start_battle(lib, deck, deck, game_id=index) for index in range(concurrency)]
    start_time = time.perf_counter()
    try:
        unique_ptrs = len({game.battle_ptr for game in games}) == len(games)
        isolation = _check_pointer_isolation(lib=lib, games=games, rng=rng)
        terminal_reasons = {"finished": 0, "max_steps": 0, "error": 0}
        active = list(games)
        while active:
            next_active: list[_PointerBattle] = []
            for game in active:
                try:
                    if _result_index(game.observation) >= 0:
                        terminal_reasons["finished"] += 1
                        continue
                    if game.selects >= max_steps:
                        terminal_reasons["max_steps"] += 1
                        continue
                    _advance_randomly(lib=lib, game=game, rng=rng)
                except Exception:
                    terminal_reasons["error"] += 1
                    continue
                next_active.append(game)
            active = next_active
    finally:
        _finish_all(lib, games)

    elapsed_seconds = time.perf_counter() - start_time
    total_selects = sum(game.selects for game in games)
    passed = (
        unique_ptrs
        and bool(isolation["passed"])
        and terminal_reasons["finished"] == concurrency
        and terminal_reasons["error"] == 0
    )
    return {
        "concurrency": concurrency,
        "passed": passed,
        "unique_battle_ptrs": unique_ptrs,
        "isolation": isolation,
        "terminal_reasons": terminal_reasons,
        "finished_games": terminal_reasons["finished"],
        "selects": total_selects,
        "elapsed_seconds": elapsed_seconds,
        "selects_per_second": _safe_rate(float(total_selects), elapsed_seconds),
    }


def _check_pointer_isolation(
    *,
    lib: Any,
    games: Sequence[_PointerBattle],
    rng: random.Random,
) -> dict[str, Any]:
    if len(games) < 2:
        return {
            "checked": False,
            "passed": True,
            "other_games_checked": 0,
            "changed_game_ids": [],
        }

    subject = games[0]
    others = list(games[1:])
    before = {
        game.game_id: _observation_digest(_get_battle_data(lib, game.battle_ptr))
        for game in others
    }
    _advance_randomly(lib=lib, game=subject, rng=rng)
    changed_game_ids: list[int] = []
    for game in others:
        after = _observation_digest(_get_battle_data(lib, game.battle_ptr))
        if after != before[game.game_id]:
            changed_game_ids.append(game.game_id)
    return {
        "checked": True,
        "passed": not changed_game_ids,
        "other_games_checked": len(others),
        "changed_game_ids": changed_game_ids,
    }


def _run_throughput_probe(
    *,
    lib: Any,
    deck: Sequence[int],
    concurrency: int,
    target_selects: int,
    max_steps: int,
    rng: random.Random,
) -> dict[str, Any]:
    games = [_start_battle(lib, deck, deck, game_id=index) for index in range(concurrency)]
    started_games = len(games)
    finished_games = 0
    truncated_games = 0
    selects = 0
    start_time = time.perf_counter()
    try:
        while selects < target_selects:
            for index, game in enumerate(games):
                if selects >= target_selects:
                    break
                if _result_index(game.observation) >= 0:
                    finished_games += 1
                    games[index] = _restart_battle(
                        lib=lib,
                        old_game=game,
                        deck=deck,
                        game_id=started_games,
                    )
                    started_games += 1
                    game = games[index]
                elif game.selects >= max_steps:
                    truncated_games += 1
                    games[index] = _restart_battle(
                        lib=lib,
                        old_game=game,
                        deck=deck,
                        game_id=started_games,
                    )
                    started_games += 1
                    game = games[index]
                _advance_randomly(lib=lib, game=game, rng=rng)
                selects += 1
    finally:
        _finish_all(lib, games)

    elapsed_seconds = time.perf_counter() - start_time
    return {
        "concurrency": concurrency,
        "passed": selects == target_selects,
        "target_selects": target_selects,
        "selects": selects,
        "started_games": started_games,
        "finished_games": finished_games,
        "truncated_games": truncated_games,
        "elapsed_seconds": elapsed_seconds,
        "selects_per_second": _safe_rate(float(selects), elapsed_seconds),
    }


def _run_memory_probe(
    *,
    lib: Any,
    deck: Sequence[int],
    games: int,
    sample_every: int,
    max_steps: int,
    max_rss_growth_mb: float | None,
    rng: random.Random,
) -> dict[str, Any]:
    samples: list[dict[str, float | int]] = []
    start_rss = _rss_mb()
    samples.append({"games": 0, "rss_mb": start_rss})
    terminal_reasons = {"finished": 0, "max_steps": 0, "error": 0}
    total_selects = 0
    start_time = time.perf_counter()
    for game_index in range(1, games + 1):
        try:
            result = _play_random_game(
                lib=lib,
                deck=deck,
                game_id=game_index,
                max_steps=max_steps,
                rng=rng,
            )
            terminal_reasons[str(result["terminal_reason"])] += 1
            total_selects += int(result["selects"])
        except Exception:
            terminal_reasons["error"] += 1
        if game_index % sample_every == 0 or game_index == games:
            samples.append({"games": game_index, "rss_mb": _rss_mb()})

    elapsed_seconds = time.perf_counter() - start_time
    end_rss = samples[-1]["rss_mb"]
    max_rss = max(float(sample["rss_mb"]) for sample in samples)
    growth_mb = float(end_rss) - start_rss
    threshold_passed = (
        True if max_rss_growth_mb is None else growth_mb <= max_rss_growth_mb
    )
    passed = terminal_reasons["error"] == 0 and threshold_passed
    return {
        "passed": passed,
        "games": games,
        "finished_games": terminal_reasons["finished"],
        "terminal_reasons": terminal_reasons,
        "selects": total_selects,
        "elapsed_seconds": elapsed_seconds,
        "selects_per_second": _safe_rate(float(total_selects), elapsed_seconds),
        "start_rss_mb": start_rss,
        "end_rss_mb": end_rss,
        "max_sample_rss_mb": max_rss,
        "rss_growth_mb": growth_mb,
        "max_rss_growth_mb": max_rss_growth_mb,
        "samples": samples,
    }


def _run_illegal_action_probe(
    *,
    lib: Any,
    deck: Sequence[int],
    rng: random.Random,
) -> dict[str, Any]:
    first = _start_battle(lib, deck, deck, game_id=0)
    second = _start_battle(lib, deck, deck, game_id=1)
    caught_type = ""
    first_finish_ok = False
    second_unchanged = False
    second_advance_ok = False
    try:
        before = _observation_digest(_get_battle_data(lib, second.battle_ptr))
        try:
            _select_battle(lib, first.battle_ptr, (999_999,))
        except Exception as exc:  # noqa: BLE001 - record exact engine error type.
            caught_type = type(exc).__name__
        try:
            _finish_battle(lib, first)
            first_finish_ok = True
        finally:
            first.closed = True
        after = _observation_digest(_get_battle_data(lib, second.battle_ptr))
        second_unchanged = before == after
        _advance_randomly(lib=lib, game=second, rng=rng)
        second_advance_ok = True
    finally:
        _finish_battle(lib, first)
        _finish_battle(lib, second)

    passed = (
        caught_type == "IndexError"
        and first_finish_ok
        and second_unchanged
        and second_advance_ok
    )
    return {
        "passed": passed,
        "caught_type": caught_type,
        "first_finish_ok": first_finish_ok,
        "second_unchanged": second_unchanged,
        "second_advance_ok": second_advance_ok,
    }


def _play_random_game(
    *,
    lib: Any,
    deck: Sequence[int],
    game_id: int,
    max_steps: int,
    rng: random.Random,
) -> dict[str, Any]:
    game = _start_battle(lib, deck, deck, game_id=game_id)
    terminal_reason = "max_steps"
    try:
        for _step in range(max_steps):
            if _result_index(game.observation) >= 0:
                terminal_reason = "finished"
                break
            _advance_randomly(lib=lib, game=game, rng=rng)
        else:
            terminal_reason = (
                "finished" if _result_index(game.observation) >= 0 else "max_steps"
            )
        return {"terminal_reason": terminal_reason, "selects": game.selects}
    finally:
        _finish_battle(lib, game)


def _advance_randomly(
    *,
    lib: Any,
    game: _PointerBattle,
    rng: random.Random,
) -> None:
    select = game.observation.get("select")
    action = random_legal_action(select, rng=rng)
    game.observation = _select_battle(lib, game.battle_ptr, action)
    game.selects += 1


def _start_battle(
    lib: Any,
    deck0: Sequence[int],
    deck1: Sequence[int],
    *,
    game_id: int,
) -> _PointerBattle:
    game = start_pointer_battle(
        lib,
        (tuple(deck0), tuple(deck1)),
        game_id=str(game_id),
        include_search_input=True,
    )
    return _PointerBattle(
        game_id=game_id,
        battle_ptr=game.battle_ptr,
        observation=game.observation,
    )


def _restart_battle(
    *,
    lib: Any,
    old_game: _PointerBattle,
    deck: Sequence[int],
    game_id: int,
) -> _PointerBattle:
    _finish_battle(lib, old_game)
    return _start_battle(lib, deck, deck, game_id=game_id)


def _select_battle(
    lib: Any,
    battle_ptr: int,
    action: Sequence[int],
) -> dict[str, Any]:
    return select_pointer_battle(
        lib,
        battle_ptr,
        action,
        include_search_input=True,
    )


def _get_battle_data(lib: Any, battle_ptr: int) -> dict[str, Any]:
    return get_pointer_battle_data(lib, battle_ptr, include_search_input=True)


def _finish_all(lib: Any, games: Sequence[_PointerBattle]) -> None:
    for game in games:
        _finish_battle(lib, game)


def _finish_battle(lib: Any, game: _PointerBattle) -> None:
    if game.closed:
        return
    finish_pointer_battle(lib, game.battle_ptr)
    game.closed = True


def _result_index(observation: Mapping[str, Any]) -> int:
    return result_index(observation)


def _observation_digest(observation: Mapping[str, Any]) -> str:
    return json.dumps(observation, sort_keys=True, separators=(",", ":"))


def _rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        with status_path.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss / 1024.0


def _write_summary(path: Path | None, summary: Mapping[str, Any]) -> None:
    if path is None:
        return
    resolved = records.repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator
