"""Isolated archive-only callback runner for handoff adapter ActTime replay."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import queue
import resource
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def run_isolated_handoff_replay(
    *,
    agent_dir: Path,
    replay_path: Path,
    seat: int,
    expected_callbacks: int,
    initial_overage_seconds: float,
    seed: int,
    macro_payload: Mapping[str, Any],
    handoff_score_mode: str,
    planner_runtime_sha256: str,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Run one arm in a fresh process whose policy code comes from the archive."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_replay_child,
        kwargs={
            "agent_dir": agent_dir,
            "replay_path": replay_path,
            "seat": seat,
            "expected_callbacks": expected_callbacks,
            "initial_overage_seconds": initial_overage_seconds,
            "seed": seed,
            "macro_payload": dict(macro_payload),
            "handoff_score_mode": handoff_score_mode,
            "planner_runtime_sha256": planner_runtime_sha256,
            "result_queue": result_queue,
        },
        name=f"handoff-acttime-{replay_path.stem}-seat-{seat}",
    )
    try:
        process.start()
        deadline = time.monotonic() + timeout_seconds
        result: Any = None
        while result is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("packaged handoff ActTime replay did not finish")
            try:
                result = result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty as exc:
                if not process.is_alive():
                    process.join(timeout=5.0)
                    raise RuntimeError(
                        "packaged handoff ActTime replay exited without evidence"
                    ) from exc
        process.join(timeout=30.0)
        if process.is_alive():
            raise RuntimeError("packaged handoff ActTime replay child did not stop")
        if not isinstance(result, Mapping) or result.get("status") != "complete":
            detail = (
                result.get("traceback", result)
                if isinstance(result, Mapping)
                else result
            )
            raise RuntimeError(f"packaged handoff ActTime replay failed: {detail}")
        return cast(Mapping[str, Any], result)
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        if process.pid is not None and process.is_alive():
            process.kill()
            process.join(timeout=10.0)
        result_queue.close()
        result_queue.join_thread()


def _replay_child(
    *,
    agent_dir: Path,
    replay_path: Path,
    seat: int,
    expected_callbacks: int,
    initial_overage_seconds: float,
    seed: int,
    macro_payload: Mapping[str, Any],
    handoff_score_mode: str,
    planner_runtime_sha256: str,
    result_queue: Any,
) -> None:
    runtime_agent: Any | None = None
    outside_workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        steps = _load_steps(replay_path)
        recorded_deck = _recorded_deck(steps, seat=seat)
        callbacks = _active_observations(steps, seat=seat)
        if len(callbacks) != expected_callbacks:
            raise ValueError("ActTime callback count differs from preregistration")

        _require_read_only_package_tree(agent_dir)
        checkpoint_path = agent_dir / "agent_checkpoint.pt"
        deck_path = agent_dir / "deck.csv"
        belief_path = agent_dir / "belief_prior.csv"
        planner_runtime_path = agent_dir / "planner_runtime.json"
        packaged_deck = _load_deck(deck_path)
        _configure_package_environment(
            checkpoint_path=checkpoint_path,
            deck_path=deck_path,
            belief_path=belief_path,
            planner_runtime_path=planner_runtime_path,
            planner_runtime_sha256=planner_runtime_sha256,
            seed=seed,
        )
        outside_workspace = tempfile.TemporaryDirectory(
            prefix="ptcg-handoff-replay-cwd-"
        )
        os.chdir(outside_workspace.name)
        _remove_repository_paths()
        _purge_repository_modules()
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(agent_dir))
        try:
            packaged_main = importlib.import_module("main")
            importlib.import_module("cg")
            runtime_module = importlib.import_module("ptcg_rl.agent.runtime")
            config_module = importlib.import_module("ptcg_rl.agent.search.config")
        finally:
            _remove_exact_path(agent_dir)

        _validate_packaged_modules(agent_dir=agent_dir)
        original_agent = getattr(packaged_main, "_AGENT", None)
        if original_agent is None:
            raise TypeError("packaged main does not expose its runtime agent")
        original_agent.close()
        runtime_agent = _configured_runtime_agent(
            runtime_module=runtime_module,
            config_module=config_module,
            macro_payload=macro_payload,
            handoff_score_mode=handoff_score_mode,
        )
        cast(Any, packaged_main)._AGENT = runtime_agent
        runtime_agent.begin_game(player_index=seat, own_deck=recorded_deck)

        remaining = initial_overage_seconds
        elapsed_total = 0.0
        callback_rows: list[dict[str, Any]] = []
        for callback_index, raw_observation in enumerate(callbacks):
            observation = dict(raw_observation)
            observation["remainingOverageTime"] = remaining
            started = time.perf_counter()
            raw_action = packaged_main.agent(observation, None)
            elapsed = max(0.0, time.perf_counter() - started)
            elapsed_total += elapsed
            remaining_before = remaining
            remaining -= elapsed
            action = _validated_action(raw_action)
            select = observation.get("select")
            if select is None:
                legal = action == packaged_deck
                if not legal:
                    raise RuntimeError(
                        "packaged deck registration differs from archive deck"
                    )
                runtime_agent.begin_game(
                    player_index=seat,
                    own_deck=recorded_deck,
                )
            else:
                legal = _action_obeys_prompt(select, action)
                if not legal:
                    raise RuntimeError("packaged replay emitted an illegal action")
            telemetry = _runtime_telemetry(runtime_agent)
            status = _runtime_status(runtime_agent)
            _validate_runtime_status(status)
            callback_rows.append(
                {
                    "callback_index": callback_index,
                    "elapsed_seconds": elapsed,
                    "remaining_before_seconds": remaining_before,
                    "remaining_after_seconds": remaining,
                    "legal": legal,
                    "select_context": _int_field(select, "context", -1),
                    "option_count": len(_sequence(_field(select, "option", ()))),
                    "action_json": json.dumps(action, separators=(",", ":")),
                    "policy_error": status.get("policy_error"),
                    "search_error": status.get("search_error"),
                    "search_stop_reason": telemetry.get("stop_reason"),
                    "search_seconds": float(
                        telemetry.get("actual_search_seconds", 0.0)
                    ),
                    "whole_act_seconds": float(telemetry.get("whole_act_seconds", 0.0)),
                    "startup_seconds": float(telemetry.get("startup_seconds", 0.0)),
                    "candidates": int(telemetry.get("candidates", 0)),
                    "worlds_requested": int(telemetry.get("worlds_requested", 0)),
                    "worlds_completed": int(telemetry.get("worlds_completed", 0)),
                    "transitions": int(telemetry.get("transitions", 0)),
                    "same_seat_value_rows": int(
                        telemetry.get("same_seat_value_rows", 0)
                    ),
                    "handoff_value_rows": int(telemetry.get("handoff_value_rows", 0)),
                    "deadline_overshoot_seconds": float(
                        telemetry.get("deadline_overshoot_seconds", 0.0)
                    ),
                    "bank_spent_seconds": float(
                        telemetry.get("bank_spent_seconds", 0.0)
                    ),
                    "bank_left_seconds": float(telemetry.get("bank_left_seconds", 0.0)),
                }
            )
        final_status = _runtime_status(runtime_agent)
        result_queue.put(
            {
                "status": "complete",
                "decisions": len(callbacks),
                "elapsed_seconds": elapsed_total,
                "remaining_seconds": remaining,
                "exhausted": remaining <= 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "torch_num_threads": _torch_num_threads(),
                "checkpoint_sha256": final_status.get("checkpoint_registry_sha256"),
                "callbacks": callback_rows,
            }
        )
    except BaseException:
        result_queue.put({"status": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        if runtime_agent is not None:
            runtime_agent.close()
        if outside_workspace is not None:
            outside_workspace.cleanup()


def _configured_runtime_agent(
    *,
    runtime_module: ModuleType,
    config_module: ModuleType,
    macro_payload: Mapping[str, Any],
    handoff_score_mode: str,
) -> Any:
    runtime_api = cast(Any, runtime_module)
    config_api = cast(Any, config_module)
    act_time_type = runtime_api.ActTimeConfig
    runtime_type = runtime_api.PolicyRuntimeAgent
    macro_type = config_api.MacroSearchConfig
    base = act_time_type.from_env()
    macro = macro_type.model_validate(dict(macro_payload))
    rerank = macro.rerank.model_copy(update={"handoff_score_mode": handoff_score_mode})
    macro = macro.model_copy(update={"rerank": rerank})
    search = base.search.model_copy(
        update={
            "enabled": False,
            "conservative_override_enabled": False,
            "worlds": macro.worlds,
            "top_k": macro.top_k,
            "manual_coin": macro.manual_coin,
            "macro": macro,
        }
    )
    config = base.model_copy(update={"planner": None, "search": search})
    if config.search.macro.rerank.handoff_score_mode != handoff_score_mode:
        raise RuntimeError("packaged handoff scorer mode was not applied")
    return runtime_type(config=config)


def _configure_package_environment(
    *,
    checkpoint_path: Path,
    deck_path: Path,
    belief_path: Path,
    planner_runtime_path: Path,
    planner_runtime_sha256: str,
    seed: int,
) -> None:
    os.environ.pop("PYTHONPATH", None)
    os.environ.update(
        {
            "PTCG_RL_CHECKPOINT_PATH": str(checkpoint_path),
            "PTCG_RL_DECK_PATH": str(deck_path),
            "PTCG_RL_BELIEF_SUMMARY_PATH": str(belief_path),
            "PTCG_RL_PLANNER_RUNTIME_PATH": str(planner_runtime_path),
            "PTCG_RL_PLANNER_RUNTIME_SHA256": planner_runtime_sha256,
            "PTCG_RL_PLANNER_PROFILE_ENABLED": "0",
            "PTCG_RL_AGENT_SEED": str(seed),
        }
    )


def _validate_runtime_status(status: Mapping[str, Any]) -> None:
    expected = {
        "policy_loaded": True,
        "planner_configured": False,
        "planner_enabled": False,
        "used_random_fallback": False,
        "prewarm_error": None,
        "engine_prewarm_error": None,
    }
    for name, value in expected.items():
        if status.get(name) != value:
            raise RuntimeError(f"packaged runtime status differs at {name}")


def _runtime_status(runtime_agent: Any) -> Mapping[str, Any]:
    status = runtime_agent.runtime_status()
    if not isinstance(status, Mapping):
        raise TypeError("packaged runtime status is not a mapping")
    return cast(Mapping[str, Any], status)


def _runtime_telemetry(runtime_agent: Any) -> Mapping[str, Any]:
    telemetry = runtime_agent.last_act_telemetry()
    if not isinstance(telemetry, Mapping):
        raise TypeError("packaged callback telemetry is not a mapping")
    return cast(Mapping[str, Any], telemetry)


def _validate_packaged_modules(*, agent_dir: Path) -> None:
    required = {"main", "ptcg_rl.agent.runtime", "ptcg_rl.agent.search.config", "cg"}
    if not required.issubset(sys.modules):
        raise RuntimeError("ActTime replay did not pin required archive packages")
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        if name in {"main", "ptcg_rl", "cg"} or name.startswith(("ptcg_rl.", "cg.")):
            _require_module_origin(module, agent_dir=agent_dir)


def _require_module_origin(module: ModuleType, *, agent_dir: Path) -> None:
    archive_root = agent_dir.resolve()
    raw_file = getattr(module, "__file__", None)
    if isinstance(raw_file, str) and raw_file:
        if not Path(raw_file).resolve().is_relative_to(archive_root):
            raise RuntimeError(
                f"ActTime replay imported {module.__name__} outside the archive"
            )
        return
    raw_paths = getattr(module, "__path__", None)
    if raw_paths is None:
        raise RuntimeError(f"archive module {module.__name__} has no origin")
    paths = tuple(Path(str(path)).resolve() for path in raw_paths)
    if not paths or any(not path.is_relative_to(archive_root) for path in paths):
        raise RuntimeError(
            f"ActTime replay resolved {module.__name__} outside the archive"
        )


def _remove_repository_paths() -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo_root = source_root.parent
    retained: list[str] = []
    for item in sys.path:
        try:
            resolved = Path(item or os.getcwd()).resolve()
        except OSError:
            retained.append(item)
            continue
        if resolved == repo_root or resolved.is_relative_to(repo_root):
            continue
        retained.append(item)
    sys.path[:] = retained


def _purge_repository_modules() -> None:
    for name in tuple(sys.modules):
        if (
            name in {"main", "cg", "ptcg_rl"}
            or name.startswith("cg.")
            or name.startswith("ptcg_rl.")
        ):
            sys.modules.pop(name, None)


def _remove_exact_path(path: Path) -> None:
    resolved_path = path.resolve()
    sys.path[:] = [
        item
        for item in sys.path
        if Path(item or os.getcwd()).resolve() != resolved_path
    ]


def _require_read_only_package_tree(agent_dir: Path) -> None:
    if not agent_dir.is_dir():
        raise FileNotFoundError("shared package extraction is unavailable")
    writable = tuple(
        path
        for path in (agent_dir, *agent_dir.rglob("*"))
        if path.stat().st_mode & 0o222
    )
    if writable:
        raise RuntimeError("shared package extraction is writable")


def _load_steps(path: Path) -> Sequence[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload, Mapping) else None
    if not isinstance(steps, list):
        raise ValueError("ActTime replay has no steps")
    return steps


def _active_observations(
    steps: Sequence[Any],
    *,
    seat: int,
) -> tuple[Mapping[str, Any], ...]:
    observations: list[Mapping[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, Sequence) or len(raw_step) != 2:
            raise ValueError("ActTime replay step is not a two-seat row")
        raw = raw_step[seat]
        if not isinstance(raw, Mapping) or str(raw.get("status")) != "ACTIVE":
            continue
        observation = raw.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("active ActTime callback has no observation")
        observations.append(cast(Mapping[str, Any], observation))
    return tuple(observations)


def _recorded_deck(steps: Sequence[Any], *, seat: int) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    for raw_step in steps:
        if not isinstance(raw_step, Sequence) or len(raw_step) != 2:
            continue
        raw = raw_step[seat]
        if not isinstance(raw, Mapping):
            continue
        action = raw.get("action")
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            deck = tuple(int(value) for value in action)
            if len(deck) == 60:
                candidates.append(deck)
    if len(candidates) != 1:
        raise ValueError("ActTime replay must contain one recorded seat deck")
    return candidates[0]


def _load_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(line.strip()) for line in path.read_text().splitlines() if line.strip()
    )
    if len(deck) != 60:
        raise ValueError("packaged deck.csv does not contain 60 cards")
    return deck


def _validated_action(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise TypeError("packaged main.agent must return a list of ints")
    return tuple(value)


def _action_obeys_prompt(select: Any, action: Sequence[int]) -> bool:
    if not isinstance(select, Mapping):
        return False
    options = _sequence(select.get("option", ()))
    option_count = len(options)
    minimum = min(option_count, max(0, int(select.get("minCount", 0))))
    maximum = min(
        option_count,
        max(minimum, int(select.get("maxCount", option_count))),
    )
    indices = tuple(int(index) for index in action)
    return (
        minimum <= len(indices) <= maximum
        and len(indices) == len(set(indices))
        and all(0 <= index < option_count for index in indices)
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _torch_num_threads() -> int:
    torch_module = importlib.import_module("torch")
    return int(torch_module.get_num_threads())


__all__ = ["run_isolated_handoff_replay"]
