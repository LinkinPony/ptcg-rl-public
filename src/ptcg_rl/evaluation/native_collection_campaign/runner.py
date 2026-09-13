"""Sequential per-device execution for immutable native collection tasks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_checkpoint_gauntlet import (
    run_native_checkpoint_gauntlet,
)
from ptcg_rl.evaluation.native_collection_campaign.artifact_io import (
    read_json,
    write_json_atomic,
)
from ptcg_rl.evaluation.native_collection_campaign.models import (
    CampaignTask,
    NativeCollectionCampaignPlan,
)
from ptcg_rl.evaluation.native_collection_campaign.planner import (
    load_campaign_plan,
)


def campaign_device_status(
    plan: NativeCollectionCampaignPlan,
    *,
    device_index: int,
    root: Path,
) -> dict[str, Any]:
    """Return a read-only status snapshot from durable task artifacts."""
    tasks = _device_tasks(plan, device_index=device_index)
    rows = [_task_status(task, root=root) for task in tasks]
    return {
        "format": "native_collection_campaign_device_status_v1",
        "plan_fingerprint": plan.plan_fingerprint,
        "stage_id": plan.stage_id,
        "device_index": device_index,
        "device_binding": plan.device_bindings[device_index],
        "tasks_total": len(rows),
        "tasks_complete": sum(row["state"] == "complete" for row in rows),
        "games_total": sum(int(row["games_total"]) for row in rows),
        "games_committed": sum(int(row["games_committed"]) for row in rows),
        "tasks": rows,
    }


def run_campaign_device(
    plan_path: Path,
    *,
    device_index: int,
    execute: bool = False,
    reconcile_stale_lock: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run or inspect one static device shard; failures stop the task queue."""
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    plan = load_campaign_plan(plan_path, root=repo_root)
    status = campaign_device_status(plan, device_index=device_index, root=repo_root)
    if not execute or status["tasks_complete"] == status["tasks_total"]:
        return status
    _validate_runner_source(plan, root=repo_root)
    binding = plan.device_bindings[device_index]
    _validate_device_binding(binding)
    output_dir = _resolve_path(plan.output_dir, root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f"device-{device_index}.lock.json"
    session = _acquire_lock(
        lock_path,
        plan=plan,
        device_index=device_index,
        reconcile_stale=reconcile_stale_lock,
        starting_games_committed=int(status["games_committed"]),
    )
    session_fields = {
        "session_started_at": session["created_at"],
        "session_start_games_committed": session["starting_games_committed"],
    }
    status_path = output_dir / f"status-device-{device_index}.json"
    try:
        for task in _device_tasks(plan, device_index=device_index):
            if _task_complete(task, root=repo_root):
                continue
            _validate_runner_source(plan, root=repo_root)
            running = campaign_device_status(
                plan,
                device_index=device_index,
                root=repo_root,
            )
            running.update(
                {
                    "state": "running",
                    "active_task_id": task.task_id,
                    "updated_at": datetime.now(UTC).isoformat(),
                    **session_fields,
                }
            )
            write_json_atomic(status_path, running)
            try:
                run_native_checkpoint_gauntlet(task.gauntlet)
                if not _task_complete(task, root=repo_root):
                    raise RuntimeError(
                        "native task returned without publishing complete artifacts"
                    )
            except BaseException as error:
                failed = campaign_device_status(
                    plan,
                    device_index=device_index,
                    root=repo_root,
                )
                failed.update(
                    {
                        "state": "failed",
                        "active_task_id": task.task_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "updated_at": datetime.now(UTC).isoformat(),
                        **session_fields,
                    }
                )
                write_json_atomic(status_path, failed)
                raise
        complete = campaign_device_status(
            plan,
            device_index=device_index,
            root=repo_root,
        )
        complete.update(
            {
                "state": "complete",
                "active_task_id": None,
                "updated_at": datetime.now(UTC).isoformat(),
                **session_fields,
            }
        )
        write_json_atomic(status_path, complete)
        return complete
    finally:
        lock_path.unlink(missing_ok=True)


def _device_tasks(
    plan: NativeCollectionCampaignPlan,
    *,
    device_index: int,
) -> tuple[CampaignTask, ...]:
    if not 0 <= device_index < len(plan.device_bindings):
        raise ValueError("campaign device index is out of range")
    return tuple(
        sorted(
            (task for task in plan.tasks if task.device_index == device_index),
            key=lambda task: task.task_id,
        )
    )


def _task_status(task: CampaignTask, *, root: Path) -> dict[str, Any]:
    output_dir = _resolve_path(task.gauntlet.output_dir, root=root)
    progress_path = output_dir / "progress.json"
    progress: dict[str, Any] = {}
    if progress_path.is_file():
        try:
            progress = read_json(progress_path)
        except (OSError, ValueError):
            progress = {}
    complete = _task_complete(task, root=root)
    committed = int(
        progress.get("completed_games", progress.get("games_committed", 0)) or 0
    )
    return {
        "task_id": task.task_id,
        "state": "complete" if complete else "pending",
        "games_total": task.gauntlet.total_games,
        "games_committed": min(committed, task.gauntlet.total_games),
        "progress_path": str(progress_path),
        "output_dir": str(output_dir),
    }


def _task_complete(task: CampaignTask, *, root: Path) -> bool:
    output_dir = _resolve_path(task.gauntlet.output_dir, root=root)
    progress_path = output_dir / "progress.json"
    if not progress_path.is_file():
        return False
    try:
        progress = read_json(progress_path)
    except (OSError, ValueError):
        return False
    return (
        progress.get("complete") is True
        and (output_dir / "games.parquet").is_file()
        and (output_dir / "summary.json").is_file()
    )


def _acquire_lock(
    path: Path,
    *,
    plan: NativeCollectionCampaignPlan,
    device_index: int,
    reconcile_stale: bool,
    starting_games_committed: int,
) -> dict[str, Any]:
    if starting_games_committed < 0:
        raise ValueError("starting committed games cannot be negative")
    if path.exists() and reconcile_stale:
        payload = read_json(path)
        if payload.get("hostname") != socket.gethostname():
            raise RuntimeError("cannot reconcile a lock owned by another host")
        pid = int(payload.get("pid", -1))
        if _process_exists(pid):
            raise RuntimeError("campaign device lock still has a live owner")
        path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise RuntimeError(
            "campaign device shard already has an owner; inspect its lock"
        ) from error
    payload = {
        "format": "native_collection_campaign_device_lock_v1",
        "plan_fingerprint": plan.plan_fingerprint,
        "device_index": device_index,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": datetime.now(UTC).isoformat(),
        "starting_games_committed": starting_games_committed,
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _validate_runner_source(
    plan: NativeCollectionCampaignPlan,
    *,
    root: Path,
) -> None:
    """Reject mutable source drift before this process spawns a new task."""
    expected_commits = {task.gauntlet.runner_source_commit for task in plan.tasks}
    if len(expected_commits) != 1:
        raise ValueError("campaign tasks bind different runner source commits")
    expected_commit = next(iter(expected_commits))
    resolved_commit = _git_stdout(
        root,
        "rev-parse",
        "--verify",
        f"{expected_commit}^{{commit}}",
    )
    if resolved_commit != expected_commit:
        raise RuntimeError(
            "campaign runner could not resolve the immutable plan source commit"
        )
    branch = _git_stdout(root, "branch", "--show-current")
    if branch != "main":
        raise RuntimeError("formal campaign execution requires the main branch")
    expected_source_tree = _git_stdout(
        root,
        "rev-parse",
        f"{expected_commit}:src",
    )
    observed_source_tree = _git_stdout(root, "rev-parse", "HEAD:src")
    if observed_source_tree != expected_source_tree:
        raise RuntimeError(
            "campaign runner source differs from the immutable plan source commit"
        )
    source_status = _git_stdout(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "src",
    )
    if source_status:
        raise RuntimeError("campaign source tree changed after plan publication")


def _git_stdout(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError("campaign runner could not inspect Git source") from error
    if result.returncode != 0:
        raise RuntimeError("campaign runner could not verify Git source identity")
    return result.stdout.strip()


def _validate_device_binding(expected_physical: str) -> None:
    """Validate direct CUDA or MPS-remapped physical device ownership."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    mps_physical = os.environ.get("PTCG_RL_CAMPAIGN_DEVICE_BINDING")
    if mps_physical is None:
        if visible != expected_physical:
            raise ValueError(
                "campaign device shard requires CUDA_VISIBLE_DEVICES="
                f"{expected_physical!r}, observed {visible!r}"
            )
        return
    if mps_physical != expected_physical:
        raise ValueError(
            "campaign MPS physical binding differs from its plan: "
            f"expected {expected_physical!r}, observed {mps_physical!r}"
        )
    if visible != "0":
        raise ValueError(
            "campaign MPS client must use remapped CUDA ordinal '0', "
            f"observed {visible!r}"
        )


def _resolve_path(path: Path, *, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


__all__ = ["campaign_device_status", "run_campaign_device"]
