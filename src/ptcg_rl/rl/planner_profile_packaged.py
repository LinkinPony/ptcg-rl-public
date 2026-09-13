"""Ordered replay through the exact fingerprint-bound submission archive."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_package_models import (
    PlannerPackageAssetManifestEntry,
)
from ptcg_rl.rl.planner_profile_archive import (
    extract_validated_planner_package,
)


@dataclass(frozen=True, slots=True)
class PackagedReplayResult:
    """Aggregate exact callback work over isolated archive processes."""

    processes: int
    episodes: int
    decisions: int
    exhausted_episodes: int
    elapsed_seconds: float
    min_remaining_seconds: float
    planner_configured_callbacks: int
    planner_used_callbacks: int
    planner_fallback_callbacks: int
    policy_error_callbacks: int
    archive_sha256: str
    planner_runtime_sha256: str


def run_packaged_act_time_replay(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    planner_enabled: bool,
    asset: PlannerPackageAssetManifestEntry,
) -> PackagedReplayResult:
    """Replay every immutable episode/seat from one exact package archive."""
    packaged = runtime.packaged
    if packaged is None:
        raise ValueError("packaged replay requires a packaged runtime")
    runtime_id = _runtime_id(config, runtime)
    if asset.runtime_id != runtime_id or (asset.asset_id != packaged.package_asset_id):
        raise ValueError("packaged replay selected the wrong manifest asset")
    if asset.planner_enabled_by_default != packaged.planner_enabled_by_default:
        raise ValueError("packaged replay planner default differs from manifest")
    if planner_enabled != packaged.planner_enabled_by_default:
        raise ValueError("packaged replay differs from the immutable planner default")
    expected_act_time = _act_time_contract(runtime)
    results: list[Mapping[str, Any]] = []
    workspace = extract_validated_planner_package(asset)
    sealed = False
    try:
        workspace.seal_read_only()
        sealed = True
        for replay in config.act_time_replay.assets:
            if file_sha256(replay.path) != replay.sha256:
                raise ValueError("ActTime replay fingerprint differs from config")
            for seat in config.act_time_replay.seats:
                results.append(
                    _run_one_isolated_replay(
                        asset=asset,
                        agent_dir=workspace.agent_dir,
                        path=replay.path,
                        seat=seat,
                        expected_callbacks=replay.active_callbacks_by_seat[seat],
                        initial_overage_seconds=(
                            config.act_time_replay.initial_overage_seconds
                        ),
                        planner_enabled=planner_enabled,
                        expected_act_time=expected_act_time,
                    )
                )
        workspace.require_read_only()
    finally:
        try:
            if sealed:
                workspace.require_read_only()
        finally:
            workspace.close()
    result = PackagedReplayResult(
        processes=len(results),
        episodes=len(results),
        decisions=sum(int(item["decisions"]) for item in results),
        exhausted_episodes=sum(bool(item["exhausted"]) for item in results),
        elapsed_seconds=sum(float(item["elapsed_seconds"]) for item in results),
        min_remaining_seconds=min(
            (float(item["remaining_seconds"]) for item in results),
            default=0.0,
        ),
        planner_configured_callbacks=sum(
            int(item["planner_configured_callbacks"]) for item in results
        ),
        planner_used_callbacks=sum(
            int(item["planner_used_callbacks"]) for item in results
        ),
        planner_fallback_callbacks=sum(
            int(item["planner_fallback_callbacks"]) for item in results
        ),
        policy_error_callbacks=sum(
            int(item["policy_error_callbacks"]) for item in results
        ),
        archive_sha256=asset.submission_archive_sha256,
        planner_runtime_sha256=asset.planner_runtime_sha256,
    )
    expected_configured = result.decisions if planner_enabled else 0
    if result.planner_configured_callbacks != expected_configured:
        raise RuntimeError("packaged replay changed planner configuration by callback")
    if not planner_enabled and (
        result.planner_used_callbacks or result.planner_fallback_callbacks
    ):
        raise RuntimeError("packaged planner-off replay emitted planner evidence")
    if result.policy_error_callbacks:
        raise RuntimeError("packaged replay encountered policy callback errors")
    return result


def _run_one_isolated_replay(
    *,
    asset: PlannerPackageAssetManifestEntry,
    agent_dir: Path,
    path: Path,
    seat: int,
    expected_callbacks: int,
    initial_overage_seconds: float,
    planner_enabled: bool,
    expected_act_time: Mapping[str, Any],
) -> Mapping[str, Any]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_replay_child,
        kwargs={
            "asset": asset,
            "agent_dir": agent_dir,
            "path": path,
            "seat": seat,
            "expected_callbacks": expected_callbacks,
            "initial_overage_seconds": initial_overage_seconds,
            "planner_enabled": planner_enabled,
            "expected_act_time": dict(expected_act_time),
            "result_queue": result_queue,
        },
        name=f"planner-profile-acttime-{path.stem}-seat-{seat}",
    )
    try:
        process.start()
        try:
            result = result_queue.get(timeout=1800.0)
        except queue.Empty as exc:
            raise TimeoutError("packaged ActTime replay did not finish") from exc
        process.join(timeout=30.0)
        if process.is_alive():
            raise RuntimeError("packaged ActTime replay child did not stop")
        if not isinstance(result, Mapping) or result.get("status") != "complete":
            detail = (
                result.get("traceback", result)
                if isinstance(result, Mapping)
                else result
            )
            raise RuntimeError(f"packaged ActTime replay failed: {detail}")
        return result
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
    asset: PlannerPackageAssetManifestEntry,
    agent_dir: Path,
    path: Path,
    seat: int,
    expected_callbacks: int,
    initial_overage_seconds: float,
    planner_enabled: bool,
    expected_act_time: Mapping[str, Any],
    result_queue: Any,
) -> None:
    runtime_agent: Any | None = None
    outside_workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise ValueError("ActTime replay has no steps")
        recorded_deck = _recorded_deck(steps, seat=seat)
        callbacks = _active_observations(steps, seat=seat)
        if len(callbacks) != expected_callbacks:
            raise ValueError("ActTime callback count differs from preregistration")

        _require_read_only_package_tree(agent_dir)
        planner_runtime_path = agent_dir / "planner_runtime.json"
        checkpoint_path = agent_dir / "agent_checkpoint.pt"
        deck_path = agent_dir / "deck.csv"
        belief_path = agent_dir / "belief_prior.csv"
        packaged_deck = _load_deck(deck_path)
        _configure_package_environment(
            checkpoint_path=checkpoint_path,
            deck_path=deck_path,
            belief_path=belief_path,
            planner_runtime_path=planner_runtime_path,
            planner_runtime_sha256=asset.planner_runtime_sha256,
            planner_enabled=planner_enabled,
            seed=int(expected_act_time["seed"]),
        )
        outside_workspace = tempfile.TemporaryDirectory(
            prefix="ptcg-profile-replay-cwd-"
        )
        outside_cwd = Path(outside_workspace.name)
        os.chdir(outside_cwd)
        _remove_repository_paths()
        _purge_repository_modules()
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(agent_dir))
        try:
            packaged_main = importlib.import_module("main")
            # Pin the package root while the archive is the only candidate;
            # later lazy ``cg.*`` imports then follow this extracted path.
            importlib.import_module("cg")
        finally:
            _remove_exact_path(agent_dir)
        _validate_packaged_main(packaged_main, agent_dir=agent_dir)
        runtime_agent = getattr(packaged_main, "_AGENT", None)
        if runtime_agent is None:
            raise TypeError("packaged main does not expose its runtime agent")
        _validate_act_time_contract(runtime_agent, expected_act_time)
        begin_game = getattr(runtime_agent, "begin_game", None)
        if not callable(begin_game):
            raise TypeError("packaged runtime has no begin_game boundary")
        begin_game(player_index=seat, own_deck=recorded_deck)

        remaining = initial_overage_seconds
        elapsed_total = 0.0
        configured_callbacks = 0
        planner_used_callbacks = 0
        planner_fallback_callbacks = 0
        policy_error_callbacks = 0
        for raw_observation in callbacks:
            observation = dict(raw_observation)
            observation["remainingOverageTime"] = remaining
            started = time.perf_counter()
            raw_action = packaged_main.agent(observation, None)
            elapsed = max(0.0, time.perf_counter() - started)
            elapsed_total += elapsed
            remaining -= elapsed
            action = _validated_action(raw_action)
            select = observation.get("select")
            if select is None:
                if action != packaged_deck:
                    raise RuntimeError(
                        "packaged deck registration differs from archive deck"
                    )
                # The immutable replay state was generated with this recorded
                # seat deck; restore that binding after the package callback.
                begin_game(player_index=seat, own_deck=recorded_deck)
            elif not _action_obeys_prompt(select, action):
                raise RuntimeError("packaged replay emitted an illegal action")
            telemetry = _runtime_telemetry(runtime_agent)
            configured = bool(telemetry.get("planner_configured", False))
            if configured != planner_enabled:
                raise RuntimeError(
                    "packaged callback planner state differs from profile branch"
                )
            configured_callbacks += int(configured)
            planner_used_callbacks += int(bool(telemetry.get("planner_used", False)))
            planner_fallback_callbacks += int(
                telemetry.get("planner_fallback_reason") is not None
            )
            status = _runtime_status(runtime_agent)
            policy_error_callbacks += int(status.get("policy_error") is not None)
            _validate_runtime_status(
                status,
                planner_enabled=planner_enabled,
                planner_fingerprint=asset.planner_fingerprint,
                runtime_fingerprint=asset.runtime_fingerprint,
            )
        result_queue.put(
            {
                "status": "complete",
                "decisions": len(callbacks),
                "elapsed_seconds": elapsed_total,
                "remaining_seconds": remaining,
                "exhausted": remaining <= 0.0,
                "planner_configured_callbacks": configured_callbacks,
                "planner_used_callbacks": planner_used_callbacks,
                "planner_fallback_callbacks": planner_fallback_callbacks,
                "policy_error_callbacks": policy_error_callbacks,
            }
        )
    except BaseException:
        result_queue.put({"status": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        if runtime_agent is not None:
            close = getattr(runtime_agent, "close", None)
            if callable(close):
                close()
        if outside_workspace is not None:
            outside_workspace.cleanup()


def _runtime_id(
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
) -> str:
    matches = [
        runtime_id
        for runtime_id, candidate in config.runtime_profiles.items()
        if candidate is runtime or candidate == runtime
    ]
    if len(matches) != 1:
        raise ValueError("packaged replay runtime identity is ambiguous")
    return str(matches[0])


def _act_time_contract(runtime: PlannerProfileRuntimeConfig) -> dict[str, Any]:
    packaged = runtime.packaged
    if packaged is None:
        raise ValueError("ActTime contract requires a packaged runtime")
    config = packaged.act_time
    return {
        "seed": config.seed,
        "default_remaining_overage_time": config.default_remaining_overage_time,
        "min_budget_seconds": config.min_budget_seconds,
        "max_budget_seconds": config.max_budget_seconds,
        "low_overage_seconds": config.low_overage_seconds,
        "critical_overage_seconds": config.critical_overage_seconds,
        "min_remaining_decisions": config.min_remaining_decisions,
        "decision_history_size": config.decision_history_size,
        "prewarm_on_startup": config.prewarm_on_startup,
        "prewarm_engine": config.prewarm_engine,
        "prewarm_checkpoint": config.prewarm_checkpoint,
        "prewarm_policy_forward": getattr(
            config,
            "prewarm_policy_forward",
            False,
        ),
    }


def _configure_package_environment(
    *,
    checkpoint_path: Path,
    deck_path: Path,
    belief_path: Path,
    planner_runtime_path: Path,
    planner_runtime_sha256: str,
    planner_enabled: bool,
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
            "PTCG_RL_PLANNER_PROFILE_ENABLED": "1" if planner_enabled else "0",
            "PTCG_RL_AGENT_SEED": str(seed),
        }
    )


def _validate_act_time_contract(
    runtime_agent: Any,
    expected: Mapping[str, Any],
) -> None:
    actual = getattr(runtime_agent, "config", None)
    if actual is None:
        raise TypeError("packaged runtime does not expose ActTime config")
    changed = {
        name: (getattr(actual, name, None), value)
        for name, value in expected.items()
        if getattr(actual, name, None) != value
    }
    if changed:
        raise ValueError(f"packaged ActTime config differs from profile: {changed}")


def _validate_runtime_status(
    status: Mapping[str, Any],
    *,
    planner_enabled: bool,
    planner_fingerprint: str,
    runtime_fingerprint: str,
) -> None:
    expected = {
        "policy_loaded": True,
        "planner_configured": planner_enabled,
        "planner_enabled": planner_enabled,
        "planner_fingerprint": planner_fingerprint if planner_enabled else None,
        "planner_runtime_fingerprint": (
            runtime_fingerprint if planner_enabled else None
        ),
        "used_random_fallback": False,
    }
    for name, value in expected.items():
        if status.get(name) != value:
            raise RuntimeError(f"packaged runtime status differs at {name}")


def _runtime_status(runtime_agent: Any) -> Mapping[str, Any]:
    method = getattr(runtime_agent, "runtime_status", None)
    if not callable(method):
        raise TypeError("packaged runtime has no runtime_status boundary")
    status = method()
    if not isinstance(status, Mapping):
        raise TypeError("packaged runtime status is not a mapping")
    return cast(Mapping[str, Any], status)


def _runtime_telemetry(runtime_agent: Any) -> Mapping[str, Any]:
    method = getattr(runtime_agent, "last_act_telemetry", None)
    if not callable(method):
        raise TypeError("packaged runtime has no callback telemetry boundary")
    telemetry = method()
    if not isinstance(telemetry, Mapping):
        raise TypeError("packaged callback telemetry is not a mapping")
    return cast(Mapping[str, Any], telemetry)


def _validate_packaged_main(module: ModuleType, *, agent_dir: Path) -> None:
    _require_module_origin(module, agent_dir=agent_dir)
    required_modules = {"ptcg_rl.agent.runtime", "cg"}
    if not required_modules.issubset(sys.modules):
        raise RuntimeError("ActTime replay did not pin required archive packages")
    loaded_package_modules = {
        name: loaded
        for name, loaded in sys.modules.items()
        if name in {"ptcg_rl", "cg"}
        or name.startswith("ptcg_rl.")
        or name.startswith("cg.")
    }
    for loaded in loaded_package_modules.values():
        if loaded is not None:
            _require_module_origin(loaded, agent_dir=agent_dir)


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
    """Require a complete sealed tree before importing shared archive bytes."""
    if not agent_dir.is_dir():
        raise FileNotFoundError("shared planner package extraction is unavailable")
    writable = tuple(
        path
        for path in (agent_dir, *agent_dir.rglob("*"))
        if path.stat().st_mode & 0o222
    )
    if writable:
        raise RuntimeError("shared planner package extraction is writable")


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
    raw_options = select.get("option", ())
    options = (
        raw_options
        if isinstance(raw_options, Sequence) and not isinstance(raw_options, str)
        else ()
    )
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


__all__ = ["PackagedReplayResult", "run_packaged_act_time_replay"]
