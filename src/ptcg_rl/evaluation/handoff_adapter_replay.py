"""Direct packaged ActTime A/B for legacy handoff adapter scoring."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.handoff_adapter_replay_config import (
    HandoffAdapterActTimeReplayConfig,
    HandoffReplayArm,
)
from ptcg_rl.evaluation.handoff_adapter_replay_isolated import (
    run_isolated_handoff_replay,
)
from ptcg_rl.evaluation.handoff_adapter_replay_metrics import (
    publish_handoff_replay_artifacts,
)
from ptcg_rl.evaluation.planner_profile_package_support import (
    archive_contents_fingerprint,
    archive_member_sha256s,
)
from ptcg_rl.rl.planner_profile_archive import extract_verified_package_archive

_CHECKPOINT_MEMBER = PurePosixPath("agent_checkpoint.pt")
_DECK_MEMBER = PurePosixPath("deck.csv")
_PLANNER_RUNTIME_MEMBER = PurePosixPath("planner_runtime.json")


def run_handoff_adapter_act_time_replay(
    config: HandoffAdapterActTimeReplayConfig,
) -> dict[str, Any]:
    """Execute a paired scorer comparison through one exact package archive."""
    package_identity = _validate_package(config)
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"replay output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    run_results: list[Mapping[str, Any]] = []
    workspace = extract_verified_package_archive(
        config.package_archive_path,
        expected_sha256=config.expected_package_archive_sha256,
    )
    sealed = False
    try:
        workspace.seal_read_only()
        sealed = True
        for repetition in range(config.repetitions):
            for replay in config.replay_assets:
                if file_sha256(replay.path) != replay.sha256:
                    raise ValueError("ActTime replay fingerprint differs from config")
                for seat in config.seats:
                    for arm in _arm_order(config.arms, repetition=repetition):
                        result = run_isolated_handoff_replay(
                            agent_dir=workspace.agent_dir,
                            replay_path=replay.path.resolve(),
                            seat=seat,
                            expected_callbacks=replay.active_callbacks_by_seat[seat],
                            initial_overage_seconds=config.initial_overage_seconds,
                            seed=config.seed,
                            macro_payload=config.macro.model_dump(mode="json"),
                            handoff_score_mode=arm.handoff_score_mode,
                            planner_runtime_sha256=str(
                                package_identity["planner_runtime_sha256"]
                            ),
                            timeout_seconds=config.child_timeout_seconds,
                        )
                        run_results.append(
                            _with_run_identity(
                                result,
                                arm=arm,
                                repetition=repetition,
                                replay_path=replay.path,
                                replay_sha256=replay.sha256,
                                seat=seat,
                            )
                        )
        workspace.require_read_only()
    finally:
        try:
            if sealed:
                workspace.require_read_only()
        finally:
            workspace.close()

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        summary = publish_handoff_replay_artifacts(
            config=config,
            package_identity=package_identity,
            run_results=run_results,
            output_dir=staging,
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _validate_package(
    config: HandoffAdapterActTimeReplayConfig,
) -> dict[str, Any]:
    archive_path = config.package_archive_path.resolve()
    archive_sha256 = file_sha256(archive_path)
    if archive_sha256 != config.expected_package_archive_sha256:
        raise ValueError("package archive fingerprint differs from config")
    contents_fingerprint = archive_contents_fingerprint(archive_path)
    if contents_fingerprint != config.expected_archive_contents_fingerprint:
        raise ValueError("package logical contents differ from config")
    member_sha256s = archive_member_sha256s(
        archive_path,
        (_CHECKPOINT_MEMBER, _DECK_MEMBER, _PLANNER_RUNTIME_MEMBER),
    )
    if member_sha256s[_CHECKPOINT_MEMBER] != config.expected_checkpoint_sha256:
        raise ValueError("packaged checkpoint fingerprint differs from config")
    if member_sha256s[_DECK_MEMBER] != config.expected_deck_sha256:
        raise ValueError("packaged deck fingerprint differs from config")
    return {
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_contents_fingerprint": contents_fingerprint,
        "checkpoint_sha256": member_sha256s[_CHECKPOINT_MEMBER],
        "deck_sha256": member_sha256s[_DECK_MEMBER],
        "planner_runtime_sha256": member_sha256s[_PLANNER_RUNTIME_MEMBER],
    }


def _arm_order(
    arms: tuple[HandoffReplayArm, ...],
    *,
    repetition: int,
) -> tuple[HandoffReplayArm, ...]:
    return arms if repetition % 2 == 0 else tuple(reversed(arms))


def _with_run_identity(
    result: Mapping[str, Any],
    *,
    arm: HandoffReplayArm,
    repetition: int,
    replay_path: Path,
    replay_sha256: str,
    seat: int,
) -> dict[str, Any]:
    enriched = dict(result)
    enriched.update(
        {
            "arm_id": arm.arm_id,
            "handoff_score_mode": arm.handoff_score_mode,
            "repetition": repetition,
            "replay_path": str(replay_path.resolve()),
            "replay_sha256": replay_sha256,
            "seat": seat,
        }
    )
    callbacks = enriched.get("callbacks")
    if not isinstance(callbacks, list):
        raise TypeError("isolated replay returned no callback rows")
    for row in callbacks:
        if not isinstance(row, dict):
            raise TypeError("isolated replay callback row is not a mapping")
        row.update(
            {
                "arm_id": arm.arm_id,
                "handoff_score_mode": arm.handoff_score_mode,
                "repetition": repetition,
                "replay_path": str(replay_path.resolve()),
                "replay_sha256": replay_sha256,
                "seat": seat,
            }
        )
    return enriched


__all__ = ["run_handoff_adapter_act_time_replay"]
