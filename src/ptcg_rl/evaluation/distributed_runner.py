"""Coordinator for content-addressed multi-host release H2H evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.distributed_hosts import (
    DistributedHostConfig,
    HostAllocation,
    SSHHostProbe,
    allocate_hosts,
    host_probe_fingerprint,
    load_ssh_targets,
    probe_ssh_hosts_resilient,
)
from ptcg_rl.evaluation.distributed_release_h2h import (
    DistributedReleaseConfig,
    ShardAssignment,
    assign_balanced_game_blocks,
    build_distributed_campaign_identity,
    merge_release_h2h_shards,
)
from ptcg_rl.evaluation.distributed_remote import (
    PreparedRemoteShard,
    close_remote_shards,
    download_remote_result,
    prepare_remote_shard,
    start_remote_shard,
    wait_remote_shards,
)
from ptcg_rl.evaluation.distributed_snapshot import (
    EvaluationSnapshot,
    build_evaluation_snapshot,
    sync_snapshot_to_host,
    write_snapshot_manifest,
)
from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig
from ptcg_rl.evaluation.search_identity import write_identity_atomic
from ptcg_rl.training.run_config import resolve_training_output_dir

_SHARD_TOOL = Path("src/tools/eval_release_h2h_shard.py")


class DistributedReleaseLaunchConfig(BaseModel):
    """Hydra-facing distributed settings kept separate from game semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hosts: DistributedHostConfig = DistributedHostConfig()
    evaluation: DistributedReleaseConfig = DistributedReleaseConfig()

    @model_validator(mode="after")
    def authoritative_runtime_is_homogeneous(
        self,
    ) -> DistributedReleaseLaunchConfig:
        if self.evaluation.authoritative and not self.hosts.require_homogeneous_runtime:
            raise ValueError(
                "authoritative distributed evaluation requires homogeneous runtime"
            )
        return self


def run_distributed_release_h2h(
    release: ReleaseH2HConfig,
    launch: DistributedReleaseLaunchConfig,
) -> dict[str, Any]:
    """Probe, snapshot, shard, execute, recover, merge, and score one campaign."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="release_h2h",
            run=release.run,
            output_dir=release.output_dir,
        )
    ).resolve()
    coordinator_dir = output_dir / "coordinator"
    coordinator_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    _write_status(status_path, {"status": "probing_hosts"})

    targets = load_ssh_targets(
        records.repo_path(launch.hosts.hosts_env_path),
        key=launch.hosts.hosts_env_key,
    )
    probes, connection_failures = probe_ssh_hosts_resilient(targets, launch.hosts)
    if launch.hosts.require_all_hosts and connection_failures:
        raise RuntimeError(
            "required evaluation hosts could not be probed: "
            f"{_redacted_failures(connection_failures)}"
        )
    allocations, compatibility_skips = allocate_hosts(probes, launch.hosts)
    skipped = {
        **_redacted_failures(connection_failures),
        **{
            _target_id(target): reason
            for target, reason in compatibility_skips.items()
        },
    }
    write_identity_atomic(
        coordinator_dir / "host_probes.json",
        {
            "protocol": "DISTRIBUTED-HOST-PROBE-v1",
            "probe_fingerprint": host_probe_fingerprint(probes),
            "hosts": [_public_probe(probe) for probe in probes],
            "skipped": skipped,
        },
    )
    _write_status(
        status_path,
        {
            "status": "building_snapshot",
            "compatible_hosts": len(allocations),
            "skipped_hosts": skipped,
        },
    )

    snapshot = build_evaluation_snapshot(release, tool_paths=(_SHARD_TOOL,))
    snapshot_manifest_path = coordinator_dir / "snapshot_manifest.json"
    write_snapshot_manifest(snapshot_manifest_path, snapshot)
    assignments = assign_balanced_game_blocks(
        release.games,
        allocations,
        equal_host_weight=launch.evaluation.equal_host_weight,
    )
    campaign = build_distributed_campaign_identity(
        release,
        launch.evaluation,
        snapshot_fingerprint=snapshot.snapshot_fingerprint,
        allocations=allocations,
        assignments=assignments,
    )
    campaign_fingerprint = str(campaign["campaign_fingerprint"])
    campaign_payload = {
        **campaign,
        "release_config": release.model_dump(mode="json"),
        "distributed_config": launch.model_dump(mode="json"),
        "skipped_hosts": skipped,
    }
    _write_or_validate_identity(
        coordinator_dir / "campaign.json",
        campaign_payload,
    )

    _write_status(
        status_path,
        {
            "status": "syncing_snapshot",
            "campaign_fingerprint": campaign_fingerprint,
            "snapshot_fingerprint": snapshot.snapshot_fingerprint,
            "hosts": len(allocations),
            "shards": len(assignments),
            "games_total": release.games,
        },
    )
    snapshot_roots = _sync_snapshots(
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot=snapshot,
        allocations=allocations,
        hosts=launch.hosts,
    )
    prepared = _prepare_shards(
        release,
        campaign_fingerprint=campaign_fingerprint,
        snapshot_fingerprint=snapshot.snapshot_fingerprint,
        snapshot_roots=snapshot_roots,
        output_dir=output_dir,
        allocations=allocations,
        assignments=assignments,
        hosts=launch.hosts,
    )

    running = [
        start_remote_shard(
            item,
            connect_timeout_seconds=launch.hosts.connect_timeout_seconds,
        )
        for item in prepared
    ]
    completed = False
    try:
        wait_remote_shards(
            running,
            coordinator_status_path=status_path,
            campaign_fingerprint=campaign_fingerprint,
            poll_interval_seconds=launch.hosts.poll_interval_seconds,
            connect_timeout_seconds=launch.hosts.connect_timeout_seconds,
        )
        completed = True
    finally:
        close_remote_shards(
            running,
            terminate=not completed,
            connect_timeout_seconds=launch.hosts.connect_timeout_seconds,
        )

    _write_status(
        status_path,
        {
            "status": "collecting_results",
            "campaign_fingerprint": campaign_fingerprint,
            "games_total": release.games,
        },
    )
    downloaded = [download_remote_result(item) for item in prepared]
    summary = merge_release_h2h_shards(
        downloaded,
        release=release,
        campaign_fingerprint=campaign_fingerprint,
        output_dir=output_dir,
        authoritative=launch.evaluation.authoritative,
    )
    summary.update(
        {
            "snapshot_fingerprint": snapshot.snapshot_fingerprint,
            "host_probe_fingerprint": host_probe_fingerprint(probes),
            "host_count": len(allocations),
            "skipped_hosts": skipped,
        }
    )
    write_identity_atomic(output_dir / "summary.json", summary)
    _write_status(
        status_path,
        {
            "status": "completed",
            "campaign_fingerprint": campaign_fingerprint,
            "games_total": release.games,
            "games_committed": release.games,
            "games_finished": release.games,
            "games_remaining": 0,
            "progress_percent": 100.0,
            "summary_path": records.display_path(output_dir / "summary.json"),
        },
    )
    return summary


def _sync_snapshots(
    *,
    snapshot_manifest_path: Path,
    snapshot: EvaluationSnapshot,
    allocations: tuple[HostAllocation, ...],
    hosts: DistributedHostConfig,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=len(allocations)) as executor:
        futures = {
            allocation.host_label: executor.submit(
                sync_snapshot_to_host,
                snapshot,
                manifest_path=snapshot_manifest_path,
                allocation=allocation,
                remote_root=hosts.remote_root,
                connect_timeout_seconds=hosts.connect_timeout_seconds,
            )
            for allocation in allocations
        }
        for host_label, future in futures.items():
            roots[host_label] = future.result()
    return roots


def _prepare_shards(
    release: ReleaseH2HConfig,
    *,
    campaign_fingerprint: str,
    snapshot_fingerprint: str,
    snapshot_roots: Mapping[str, Path],
    output_dir: Path,
    allocations: tuple[HostAllocation, ...],
    assignments: tuple[ShardAssignment, ...],
    hosts: DistributedHostConfig,
) -> tuple[PreparedRemoteShard, ...]:
    allocation_by_host = {item.host_label: item for item in allocations}
    return tuple(
        prepare_remote_shard(
            release,
            campaign_fingerprint=campaign_fingerprint,
            snapshot_fingerprint=snapshot_fingerprint,
            snapshot_root=snapshot_roots[assignment.host_label],
            remote_root=hosts.remote_root,
            local_root=output_dir,
            allocation=allocation_by_host[assignment.host_label],
            assignment=assignment,
            connect_timeout_seconds=hosts.connect_timeout_seconds,
        )
        for assignment in assignments
    )


def _public_probe(probe: SSHHostProbe) -> dict[str, Any]:
    payload = probe.model_dump(mode="json")
    payload.pop("target", None)
    return payload


def _target_id(target: str) -> str:
    return hashlib.sha256(target.encode("utf-8")).hexdigest()[:12]


def _redacted_failures(failures: Mapping[str, str]) -> dict[str, str]:
    return {_target_id(target): message for target, message in failures.items()}


def _write_or_validate_identity(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"existing distributed campaign identity does not match: {path}"
            )
        return
    write_identity_atomic(path, payload)


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    write_identity_atomic(path, payload)


__all__ = [
    "DistributedReleaseLaunchConfig",
    "run_distributed_release_h2h",
]
