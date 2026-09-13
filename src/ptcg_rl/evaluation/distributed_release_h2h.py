"""Distributed shard protocol and merge logic for immutable release H2H."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.distributed_hosts import HostAllocation
from ptcg_rl.evaluation.release_h2h import (
    ReleaseH2HConfig,
    ReleaseH2HShardContext,
    run_release_h2h,
    score_release_h2h_games,
)
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)


class DistributedReleaseConfig(BaseModel):
    """Coordinator policy layered over one ordinary release H2H config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["DISTRIBUTED-RELEASE-H2H-v1"] = "DISTRIBUTED-RELEASE-H2H-v1"
    authoritative: bool = True
    equal_host_weight: bool = False


class ReleaseH2HShardConfig(BaseModel):
    """Immutable envelope consumed by exactly one remote worker shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["DISTRIBUTED-RELEASE-H2H-SHARD-v1"] = (
        "DISTRIBUTED-RELEASE-H2H-SHARD-v1"
    )
    campaign_fingerprint: str
    snapshot_fingerprint: str
    shard_id: str
    host_label: str
    runtime_fingerprint: str
    requested_resource: Literal["cpu", "cuda"]
    game_indices: tuple[int, ...]
    release: ReleaseH2HConfig

    @field_validator(
        "campaign_fingerprint",
        "snapshot_fingerprint",
        "shard_id",
        "host_label",
        "runtime_fingerprint",
    )
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("distributed release shard identities must be non-empty")
        return normalized

    @model_validator(mode="after")
    def valid_game_block_assignment(self) -> ReleaseH2HShardConfig:
        indices = self.game_indices
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("distributed shard game_indices must be non-empty and unique")
        if tuple(sorted(indices)) != indices:
            raise ValueError("distributed shard game_indices must be sorted")
        if any(not 0 <= index < self.release.games for index in indices):
            raise ValueError("distributed shard game_indices exceed campaign bounds")
        assigned = set(indices)
        for index in indices:
            paired = index + 1 if index % 2 == 0 else index - 1
            if paired not in assigned:
                raise ValueError("every distributed shard must own complete seat pairs")
        return self


class ReleaseH2HShardResult(BaseModel):
    """Durable evidence returned by one completed worker shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["DISTRIBUTED-RELEASE-H2H-SHARD-RESULT-v1"] = (
        "DISTRIBUTED-RELEASE-H2H-SHARD-RESULT-v1"
    )
    campaign_fingerprint: str
    snapshot_fingerprint: str
    shard_id: str
    host_label: str
    runtime_fingerprint: str
    requested_resource: Literal["cpu", "cuda"]
    game_indices: tuple[int, ...]
    games_path: Path
    games_sha256: str
    evaluation_fingerprint: str
    referee_fingerprint: str
    bridge_sha256: str
    candidate_capsule_fingerprint: str
    opponent_capsule_fingerprint: str
    candidate_policy_devices: tuple[str, ...]
    opponent_policy_devices: tuple[str, ...]
    score: dict[str, Any]


class ShardAssignment(BaseModel):
    """Coordinator-side mapping from one host resource slot to game blocks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shard_id: str
    host_label: str
    game_indices: tuple[int, ...]
    workers: int
    cpu_affinity: tuple[int, ...]
    gpu_index: int | None = None


def assign_balanced_game_blocks(
    total_games: int,
    allocations: Sequence[HostAllocation],
    *,
    equal_host_weight: bool,
) -> tuple[ShardAssignment, ...]:
    """Assign complete seat pairs with deterministic capacity balancing."""
    if total_games <= 0 or total_games % 2:
        raise ValueError("distributed release H2H requires a positive even game count")
    slots = _resource_slots(allocations)
    if total_games // 2 < len(slots):
        slots = slots[: total_games // 2]
    assigned_blocks: list[list[int]] = [[] for _ in slots]
    weights = [1 if equal_host_weight else slot[1] for slot in slots]
    for block_index in range(total_games // 2):
        slot_index = min(
            range(len(slots)),
            key=lambda index: (
                len(assigned_blocks[index]) / weights[index],
                index,
            ),
        )
        assigned_blocks[slot_index].append(block_index)

    assignments: list[ShardAssignment] = []
    for slot_index, ((allocation, workers, cpus, gpu_index), blocks) in enumerate(
        zip(slots, assigned_blocks, strict=True)
    ):
        indices = tuple(index for block in blocks for index in (2 * block, 2 * block + 1))
        suffix = f"gpu-{gpu_index}" if gpu_index is not None else "cpu"
        assignments.append(
            ShardAssignment(
                shard_id=f"{slot_index:02d}-{allocation.host_label}-{suffix}",
                host_label=allocation.host_label,
                game_indices=indices,
                workers=min(workers, len(indices)),
                cpu_affinity=cpus,
                gpu_index=gpu_index,
            )
        )
    return tuple(assignments)


def build_distributed_campaign_identity(
    release: ReleaseH2HConfig,
    distributed: DistributedReleaseConfig,
    *,
    snapshot_fingerprint: str,
    allocations: Sequence[HostAllocation],
    assignments: Sequence[ShardAssignment],
) -> dict[str, Any]:
    """Build the global identity shared by every heterogeneous execution shard."""
    semantic_release = release.model_dump(
        mode="json",
        exclude={
            "output_dir",
            "temp_root",
            "compression",
            "result_shard_size",
            "cleanup_runtime_cache",
            "num_workers",
            "partition_worker_cpus",
            "cpu_threads_per_worker",
        },
    )
    allocation_identity = [
        {
            "host_label": item.host_label,
            "runtime_fingerprint": item.runtime_fingerprint,
            "workers": item.workers,
            "cpu_threads_per_worker": item.cpu_threads_per_worker,
            "requested_resource": item.requested_resource,
            "runtime_backend": item.runtime_backend,
            "container_image": item.container_image,
            "container_archive_sha256": item.container_archive_sha256,
            "gpu_count": len(item.gpu_indices),
        }
        for item in sorted(allocations, key=lambda value: value.host_label)
    ]
    assignment_identity = [
        item.model_dump(mode="json")
        for item in sorted(assignments, key=lambda value: value.shard_id)
    ]
    payload = {
        "protocol": distributed.protocol,
        "authoritative": distributed.authoritative,
        "equal_host_weight": distributed.equal_host_weight,
        "release": semantic_release,
        "snapshot_fingerprint": snapshot_fingerprint,
        "allocations": allocation_identity,
        "assignments": assignment_identity,
    }
    return {**payload, "campaign_fingerprint": fingerprint_payload(payload)}


def build_shard_config(
    release: ReleaseH2HConfig,
    *,
    campaign_fingerprint: str,
    snapshot_fingerprint: str,
    allocation: HostAllocation,
    assignment: ShardAssignment,
    remote_output_dir: Path,
    remote_temp_root: Path,
) -> ReleaseH2HShardConfig:
    """Resolve host-specific execution fields without changing game semantics."""
    configured = release.model_copy(
        update={
            "output_dir": remote_output_dir,
            "temp_root": remote_temp_root,
            "num_workers": assignment.workers,
            "partition_worker_cpus": True,
            "cpu_threads_per_worker": allocation.cpu_threads_per_worker,
        }
    )
    return ReleaseH2HShardConfig(
        campaign_fingerprint=campaign_fingerprint,
        snapshot_fingerprint=snapshot_fingerprint,
        shard_id=assignment.shard_id,
        host_label=assignment.host_label,
        runtime_fingerprint=allocation.runtime_fingerprint,
        requested_resource=allocation.requested_resource,
        game_indices=assignment.game_indices,
        release=configured,
    )


def run_release_h2h_shard(
    config: ReleaseH2HShardConfig,
    *,
    status_path: Path,
    result_path: Path,
) -> ReleaseH2HShardResult:
    """Run one recoverable sparse shard and publish its verified result manifest."""
    _write_status(
        status_path,
        {
            "status": "running",
            "campaign_fingerprint": config.campaign_fingerprint,
            "shard_id": config.shard_id,
            "host_label": config.host_label,
            "games_total": len(config.game_indices),
        },
    )
    try:
        summary = run_release_h2h(
            config.release,
            game_indices=config.game_indices,
            shard_context=ReleaseH2HShardContext(
                campaign_fingerprint=config.campaign_fingerprint,
                shard_id=config.shard_id,
                host_label=config.host_label,
                runtime_fingerprint=config.runtime_fingerprint,
                requested_resource=config.requested_resource,
            ),
        )
        games_path = records.repo_path(Path(str(summary["games_path"]))).resolve()
        rows = cast(list[dict[str, Any]], pq.read_table(games_path).to_pylist())
        _validate_shard_rows(config, rows)
        fingerprints_path = records.repo_path(config.release.output_dir or Path()) / (
            "fingerprints.json"
        )
        fingerprints = json.loads(fingerprints_path.read_text(encoding="utf-8"))
        result = ReleaseH2HShardResult(
            campaign_fingerprint=config.campaign_fingerprint,
            snapshot_fingerprint=config.snapshot_fingerprint,
            shard_id=config.shard_id,
            host_label=config.host_label,
            runtime_fingerprint=config.runtime_fingerprint,
            requested_resource=config.requested_resource,
            game_indices=config.game_indices,
            games_path=games_path,
            games_sha256=file_sha256(games_path),
            evaluation_fingerprint=str(summary["evaluation_fingerprint"]),
            referee_fingerprint=str(fingerprints["referee_fingerprint"]),
            bridge_sha256=str(fingerprints["bridge_sha256"]),
            candidate_capsule_fingerprint=str(
                summary["candidate"]["capsule_fingerprint"]
            ),
            opponent_capsule_fingerprint=str(
                summary["opponent"]["capsule_fingerprint"]
            ),
            candidate_policy_devices=tuple(
                sorted({str(row["candidate_policy_device"]) for row in rows})
            ),
            opponent_policy_devices=tuple(
                sorted({str(row["opponent_policy_device"]) for row in rows})
            ),
            score=dict(summary["score"]),
        )
        write_identity_atomic(result_path, result.model_dump(mode="json"))
        _write_status(
            status_path,
            {
                "status": "completed",
                "campaign_fingerprint": config.campaign_fingerprint,
                "shard_id": config.shard_id,
                "host_label": config.host_label,
                "games_total": len(config.game_indices),
                "games_committed": len(rows),
                "games_finished": len(rows),
                "games_remaining": 0,
                "result_path": str(result_path),
            },
        )
        return result
    except BaseException as exc:
        _write_status(
            status_path,
            {
                "status": "failed",
                "campaign_fingerprint": config.campaign_fingerprint,
                "shard_id": config.shard_id,
                "host_label": config.host_label,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def merge_release_h2h_shards(
    shard_games: Sequence[tuple[ReleaseH2HShardResult, Path]],
    *,
    release: ReleaseH2HConfig,
    campaign_fingerprint: str,
    output_dir: Path,
    authoritative: bool,
) -> dict[str, Any]:
    """Validate, merge, and score a complete set of downloaded shard files."""
    if not shard_games:
        raise ValueError("distributed release merge requires shard results")
    expected_indices = set(range(release.games))
    observed_indices: set[int] = set()
    tables: list[pa.Table] = []
    invariant_values: dict[str, set[str]] = {
        "referee_fingerprint": set(),
        "bridge_sha256": set(),
        "candidate_capsule_fingerprint": set(),
        "opponent_capsule_fingerprint": set(),
    }
    runtime_fingerprints: set[str] = set()
    for result, path in shard_games:
        if result.campaign_fingerprint != campaign_fingerprint:
            raise ValueError(f"shard belongs to another campaign: {result.shard_id}")
        if file_sha256(path) != result.games_sha256:
            raise ValueError(f"downloaded shard hash mismatch: {result.shard_id}")
        duplicate = observed_indices & set(result.game_indices)
        if duplicate:
            raise ValueError(f"distributed shards overlap games: {sorted(duplicate)}")
        observed_indices.update(result.game_indices)
        runtime_fingerprints.add(result.runtime_fingerprint)
        for name in invariant_values:
            invariant_values[name].add(str(getattr(result, name)))
        table = pq.read_table(path)
        table_indices = {int(value) for value in table["game_index"].to_pylist()}
        if table_indices != set(result.game_indices):
            raise ValueError(f"shard game rows do not match assignment: {result.shard_id}")
        tables.append(table)
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        extra = sorted(observed_indices - expected_indices)
        raise ValueError(f"distributed result coverage mismatch: missing={missing}, extra={extra}")
    changed = {name: values for name, values in invariant_values.items() if len(values) != 1}
    if changed:
        raise ValueError(f"distributed immutable identities differ across hosts: {changed}")
    if authoritative and len(runtime_fingerprints) != 1:
        raise ValueError("authoritative merge requires one runtime fingerprint")

    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / "games.parquet"
    temporary = output_dir / ".games.parquet.tmp"
    combined = pa.concat_tables(tables, promote_options="default")
    order = pc.sort_indices(combined, sort_keys=[("game_index", "ascending")])
    combined = pc.take(combined, order)
    pq.write_table(combined, temporary, compression=release.compression)
    temporary.replace(games_path)
    score = score_release_h2h_games(games_path, decision=release.decision)
    score["authoritative"] = authoritative
    score["runtime_fingerprints"] = sorted(runtime_fingerprints)
    if not authoritative:
        score["statistical_interpretation"] = score["interpretation"]
        score["interpretation"] = "diagnostic_only_heterogeneous_runtime"
    write_identity_atomic(output_dir / "score.json", score)
    summary = {
        "protocol": "DISTRIBUTED-RELEASE-H2H-RESULT-v1",
        "campaign_fingerprint": campaign_fingerprint,
        "authoritative": authoritative,
        "games": release.games,
        "shards": len(shard_games),
        "runtime_fingerprints": sorted(runtime_fingerprints),
        "games_path": records.display_path(games_path),
        "games_sha256": file_sha256(games_path),
        "score_path": records.display_path(output_dir / "score.json"),
        "score": score,
    }
    write_identity_atomic(output_dir / "summary.json", summary)
    return summary


def _resource_slots(
    allocations: Sequence[HostAllocation],
) -> list[tuple[HostAllocation, int, tuple[int, ...], int | None]]:
    slots: list[tuple[HostAllocation, int, tuple[int, ...], int | None]] = []
    for allocation in sorted(allocations, key=lambda item: item.host_label):
        if allocation.requested_resource == "cpu":
            slots.append(
                (
                    allocation,
                    allocation.workers,
                    allocation.cpu_affinity,
                    None,
                )
            )
            continue
        if not allocation.gpu_indices:
            raise ValueError(f"CUDA allocation has no GPU: {allocation.host_label}")
        workers_per_gpu = max(1, allocation.workers // len(allocation.gpu_indices))
        cpu_stride = workers_per_gpu * allocation.cpu_threads_per_worker
        for gpu_slot, gpu_index in enumerate(allocation.gpu_indices):
            start = gpu_slot * cpu_stride
            cpus = allocation.cpu_affinity[start : start + cpu_stride]
            if cpus:
                slots.append((allocation, workers_per_gpu, cpus, gpu_index))
    if not slots:
        raise ValueError("distributed release schedule has no resource slots")
    return slots


def _validate_shard_rows(
    config: ReleaseH2HShardConfig,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    observed = {int(row["game_index"]) for row in rows}
    if observed != set(config.game_indices) or len(rows) != len(config.game_indices):
        raise ValueError("release shard did not produce its exact assigned game set")
    expected_device = config.requested_resource
    for row in rows:
        for role in ("candidate", "opponent"):
            if row.get(f"{role}_policy_loaded_observed") is not True:
                raise ValueError(f"{role} policy was not loaded in shard result")
            device = str(row.get(f"{role}_policy_device", ""))
            if not device.startswith(expected_device):
                raise ValueError(
                    f"{role} policy device {device!r} does not match "
                    f"requested {expected_device!r}"
                )
        if row.get("distributed_campaign_fingerprint") != config.campaign_fingerprint:
            raise ValueError("release shard row omitted the campaign fingerprint")
        if row.get("distributed_shard_id") != config.shard_id:
            raise ValueError("release shard row omitted the shard identity")


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    write_identity_atomic(
        path,
        {
            **payload,
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


__all__ = [
    "DistributedReleaseConfig",
    "ReleaseH2HShardConfig",
    "ReleaseH2HShardResult",
    "ShardAssignment",
    "assign_balanced_game_blocks",
    "build_distributed_campaign_identity",
    "build_shard_config",
    "merge_release_h2h_shards",
    "run_release_h2h_shard",
]
