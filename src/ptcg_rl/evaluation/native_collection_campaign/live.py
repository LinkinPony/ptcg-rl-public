"""Read-only live aggregation for an incremental native evaluation campaign."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ptcg_rl.evaluation.native_collection_campaign.artifact_io import read_json
from ptcg_rl.evaluation.native_collection_campaign.inventory import (
    load_historical_inventory,
)
from ptcg_rl.evaluation.native_collection_campaign.models import (
    CampaignTask,
    HistoricalCheckpointInventory,
    NativeCollectionCampaignPlan,
    PlannedBundle,
)
from ptcg_rl.evaluation.native_collection_campaign.planner import (
    load_campaign_plan,
)

_RESULT_COLUMNS = (
    "candidate_deck_id",
    "candidate_deck_hash",
    "baseline_deck_id",
    "baseline_deck_hash",
    "candidate_seat",
    "candidate_result",
    "candidate_score",
    "terminal_reason",
    "native_engine_steps",
    "native_candidate_policy_rows",
    "native_baseline_policy_rows",
)


@dataclass(slots=True)
class ScoreTally:
    """Sufficient statistics for provisional score and operational health."""

    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    unresolved: int = 0
    score_sum: float = 0.0
    seat_games: list[int] = field(default_factory=lambda: [0, 0])
    seat_resolved: list[int] = field(default_factory=lambda: [0, 0])
    seat_score_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @property
    def resolved(self) -> int:
        return self.wins + self.draws + self.losses

    def observe(self, *, result: str, score: float | None, seat: int) -> None:
        """Observe one immutable game row without inventing unresolved scores."""
        if seat not in (0, 1):
            raise ValueError("candidate seat must be 0 or 1")
        self.games += 1
        self.seat_games[seat] += 1
        if result == "unresolved":
            if score is not None:
                raise ValueError("unresolved game unexpectedly has a score")
            self.unresolved += 1
            return
        if score is None:
            raise ValueError("resolved game is missing its candidate score")
        expected_scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
        expected_score = expected_scores.get(result)
        if expected_score is None:
            raise ValueError(f"unknown candidate result: {result}")
        if score != expected_score:
            raise ValueError("candidate result and score disagree")
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        elif result == "loss":
            self.losses += 1
        self.score_sum += score
        self.seat_resolved[seat] += 1
        self.seat_score_sum[seat] += score

    def merge(self, other: ScoreTally) -> None:
        """Merge another independent tally into this aggregate."""
        self.games += other.games
        self.wins += other.wins
        self.draws += other.draws
        self.losses += other.losses
        self.unresolved += other.unresolved
        self.score_sum += other.score_sum
        for seat in (0, 1):
            self.seat_games[seat] += other.seat_games[seat]
            self.seat_resolved[seat] += other.seat_resolved[seat]
            self.seat_score_sum[seat] += other.seat_score_sum[seat]


@dataclass(slots=True)
class PartAggregate:
    """Compact cached statistics from one atomically published Parquet part."""

    overall: ScoreTally = field(default_factory=ScoreTally)
    bundles: dict[str, ScoreTally] = field(default_factory=dict)
    cells: dict[tuple[str, str], ScoreTally] = field(default_factory=dict)
    terminal_reasons: Counter[str] = field(default_factory=Counter)
    engine_steps: int = 0
    policy_rows: int = 0

    def merge(self, other: PartAggregate) -> None:
        """Merge a cached part into the current snapshot accumulator."""
        self.overall.merge(other.overall)
        _merge_tally_maps(self.bundles, other.bundles)
        _merge_tally_maps(self.cells, other.cells)
        self.terminal_reasons.update(other.terminal_reasons)
        self.engine_steps += other.engine_steps
        self.policy_rows += other.policy_rows


@dataclass(frozen=True, slots=True)
class _TaskBundles:
    candidate_by_digest: dict[str, PlannedBundle]
    opponent_by_digest: dict[str, PlannedBundle]


class CampaignLiveReader:
    """Incrementally cache committed parts and publish cheap UI snapshots."""

    def __init__(
        self,
        plan: NativeCollectionCampaignPlan,
        inventory: HistoricalCheckpointInventory,
        *,
        root: Path,
        gpu_poll_seconds: float = 5.0,
        standings_limit: int = 250,
        cells_limit: int = 6_000,
    ) -> None:
        if gpu_poll_seconds <= 0.0:
            raise ValueError("GPU polling interval must be positive")
        if standings_limit <= 0 or cells_limit <= 0:
            raise ValueError("live result limits must be positive")
        self.plan = plan
        self.inventory = inventory
        self.root = root.resolve()
        self._bundle_by_id = {
            bundle.bundle_id: bundle for bundle in (*plan.candidates, *plan.opponents)
        }
        self._checkpoint_by_id = {
            checkpoint.checkpoint_id: checkpoint for checkpoint in inventory.checkpoints
        }
        self._task_bundles = {
            task.task_id: self._resolve_task_bundles(task) for task in plan.tasks
        }
        self._part_cache: dict[Path, PartAggregate] = {}
        self._gpu_poll_seconds = gpu_poll_seconds
        self._gpu_polled_at = 0.0
        self._gpu_cache: list[dict[str, Any]] = []
        self._standings_limit = standings_limit
        self._cells_limit = cells_limit
        self._lock = threading.Lock()

    @classmethod
    def from_plan_path(
        cls,
        plan_path: Path,
        *,
        root: Path,
        gpu_poll_seconds: float = 5.0,
        standings_limit: int = 250,
        cells_limit: int = 6_000,
    ) -> CampaignLiveReader:
        """Load and verify a plan plus its immutable inventory binding."""
        plan = load_campaign_plan(plan_path, root=root)
        inventory = load_historical_inventory(plan.inventory_path, root=root)
        return cls(
            plan,
            inventory,
            root=root,
            gpu_poll_seconds=gpu_poll_seconds,
            standings_limit=standings_limit,
            cells_limit=cells_limit,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return one internally consistent provisional campaign snapshot."""
        with self._lock:
            generated_at = datetime.now(UTC)
            aggregate = self._aggregate_committed_parts()
            tasks, device_rows = self._task_and_device_rows(now=generated_at)
            active_rate = sum(
                float(row.get("active_games_per_second", 0.0) or 0.0)
                for row in device_rows
            )
            effective_rate = sum(
                float(row.get("effective_games_per_second", 0.0) or 0.0)
                for row in device_rows
            )
            games_total = sum(task.gauntlet.total_games for task in self.plan.tasks)
            games_committed = aggregate.overall.games
            games_remaining = max(0, games_total - games_committed)
            failed = any(row.get("state") == "failed" for row in device_rows)
            complete = games_committed == games_total and all(
                row["state"] == "complete" for row in tasks
            )
            state = (
                "failed"
                if failed
                else "complete"
                if complete
                else "running"
                if games_committed
                or any(row.get("active_task_id") for row in device_rows)
                else "pending"
            )
            all_standings = self._standings(aggregate)
            standings = all_standings[: self._standings_limit]
            visible_bundle_ids = {str(row["bundle_id"]) for row in standings}
            all_cells_count = len(aggregate.cells)
            cells = self._cells(
                aggregate,
                candidate_bundle_ids=visible_bundle_ids,
                limit=self._cells_limit,
            )
            device_etas = [
                float(row["eta_seconds"])
                for row in device_rows
                if int(row["games_remaining"]) > 0
                and row.get("eta_seconds") is not None
            ]
            devices_without_eta = any(
                int(row["games_remaining"]) > 0 and row.get("eta_seconds") is None
                for row in device_rows
            )
            eta_seconds = (
                None
                if failed
                else 0.0
                if not games_remaining
                else None
                if devices_without_eta
                else max(device_etas, default=None)
            )
            return {
                "format": "native_collection_campaign_live_snapshot_v1",
                "generated_at": generated_at.isoformat(),
                "state": state,
                "stage_id": self.plan.stage_id,
                "plan_fingerprint": self.plan.plan_fingerprint,
                "games_total": games_total,
                "games_committed": games_committed,
                "games_remaining": games_remaining,
                "progress_percent": (
                    100.0 * games_committed / games_total if games_total else 0.0
                ),
                "active_games_per_second": active_rate,
                "effective_games_per_second": effective_rate,
                "eta_seconds": eta_seconds,
                "estimated_completion_at": (
                    (generated_at + timedelta(seconds=eta_seconds)).isoformat()
                    if eta_seconds is not None
                    else None
                ),
                "eta_basis": "parallel-device-effective-wall-rate-v1",
                "tasks_total": len(tasks),
                "tasks_complete": sum(row["state"] == "complete" for row in tasks),
                "parts_cached": len(self._part_cache),
                "resolved_games": aggregate.overall.resolved,
                "unresolved_games": aggregate.overall.unresolved,
                "terminal_reasons": dict(sorted(aggregate.terminal_reasons.items())),
                "engine_steps": aggregate.engine_steps,
                "policy_rows": aggregate.policy_rows,
                "devices": device_rows,
                "gpus": self._gpu_rows(),
                "standings_total": len(all_standings),
                "standings_observed": sum(
                    int(row["games"]) > 0 for row in all_standings
                ),
                "standings_truncated": len(standings) < len(all_standings),
                "standings": standings,
                "cells_total": all_cells_count,
                "cells_truncated": len(cells) < all_cells_count,
                "cells": cells,
                "tasks": tasks,
            }

    def plan_summary(self) -> dict[str, Any]:
        """Return immutable, display-safe plan metadata."""
        return {
            "format": self.plan.format,
            "stage_id": self.plan.stage_id,
            "plan_fingerprint": self.plan.plan_fingerprint,
            "candidate_bundles": len(self.plan.candidates),
            "opponent_bundles": len(self.plan.opponents),
            "tasks": len(self.plan.tasks),
            "devices": list(self.plan.device_bindings),
            "games": sum(task.gauntlet.total_games for task in self.plan.tasks),
        }

    def _resolve_task_bundles(self, task: CampaignTask) -> _TaskBundles:
        candidates = {
            self._bundle_by_id[bundle_id].deck_digest: self._bundle_by_id[bundle_id]
            for bundle_id in task.candidate_bundle_ids
        }
        opponents = {
            self._bundle_by_id[bundle_id].deck_digest: self._bundle_by_id[bundle_id]
            for bundle_id in task.opponent_bundle_ids
        }
        if len(candidates) != len(task.candidate_bundle_ids) or len(opponents) != len(
            task.opponent_bundle_ids
        ):
            raise ValueError("task contains duplicate exact deck routes")
        return _TaskBundles(candidates, opponents)

    def _aggregate_committed_parts(self) -> PartAggregate:
        aggregate = PartAggregate()
        observed: set[Path] = set()
        for task in self.plan.tasks:
            parts_dir = self._resolve_path(task.gauntlet.output_dir) / "games_parts"
            if not parts_dir.is_dir():
                continue
            for part_path in parts_dir.glob("part-*.parquet"):
                observed.add(part_path)
                part = self._part_cache.get(part_path)
                if part is None:
                    part = self._read_part(part_path, task)
                    self._part_cache[part_path] = part
                aggregate.merge(part)
        stale = set(self._part_cache) - observed
        for path in stale:
            del self._part_cache[path]
        return aggregate

    def _read_part(self, path: Path, task: CampaignTask) -> PartAggregate:
        table = pq.read_table(path, columns=list(_RESULT_COLUMNS))
        bundles = self._task_bundles[task.task_id]
        output = PartAggregate()
        for row in table.to_pylist():
            candidate = bundles.candidate_by_digest.get(str(row["candidate_deck_id"]))
            opponent = bundles.opponent_by_digest.get(str(row["baseline_deck_id"]))
            if candidate is None or opponent is None:
                raise ValueError(f"part references a route outside its task: {path}")
            if (
                row["candidate_deck_hash"] != candidate.deck_hash
                or row["baseline_deck_hash"] != opponent.deck_hash
            ):
                raise ValueError(f"part changed an authoritative deck_hash: {path}")
            result = str(row["candidate_result"])
            score = _finite_score(row.get("candidate_score"))
            seat = int(row["candidate_seat"])
            output.overall.observe(result=result, score=score, seat=seat)
            output.bundles.setdefault(candidate.bundle_id, ScoreTally()).observe(
                result=result,
                score=score,
                seat=seat,
            )
            output.cells.setdefault(
                (candidate.bundle_id, opponent.bundle_id), ScoreTally()
            ).observe(result=result, score=score, seat=seat)
            output.terminal_reasons[str(row["terminal_reason"])] += 1
            output.engine_steps += int(row.get("native_engine_steps", 0) or 0)
            output.policy_rows += int(
                row.get("native_candidate_policy_rows", 0) or 0
            ) + int(row.get("native_baseline_policy_rows", 0) or 0)
        return output

    def _task_and_device_rows(
        self,
        *,
        now: datetime,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        device_status = {
            index: self._read_device_status(index)
            for index in range(len(self.plan.device_bindings))
        }
        task_rows: list[dict[str, Any]] = []
        for task in self.plan.tasks:
            output_dir = self._resolve_path(task.gauntlet.output_dir)
            progress = _read_json_optional(output_dir / "progress.json")
            complete = (
                progress.get("complete") is True
                and (output_dir / "summary.json").is_file()
                and (output_dir / "games.parquet").is_file()
            )
            active_task = device_status[task.device_index].get("active_task_id")
            failed = (
                device_status[task.device_index].get("state") == "failed"
                and active_task == task.task_id
            )
            state = (
                "complete"
                if complete
                else "failed"
                if failed
                else "running"
                if active_task == task.task_id
                else "pending"
            )
            committed = min(
                int(
                    progress.get(
                        "games_committed",
                        progress.get("completed_games", 0),
                    )
                    or 0
                ),
                task.gauntlet.total_games,
            )
            resumed_games = max(0, int(progress.get("resumed_games", 0) or 0))
            session_games = max(
                0,
                int(progress.get("session_games", committed - resumed_games) or 0),
            )
            task_rows.append(
                {
                    "task_id": task.task_id,
                    "task_short_id": task.task_id[:12],
                    "device_index": task.device_index,
                    "state": state,
                    "games_total": task.gauntlet.total_games,
                    "games_committed": committed,
                    "progress_percent": (100.0 * committed / task.gauntlet.total_games),
                    "games_per_second": float(
                        progress.get("session_games_per_second", 0.0) or 0.0
                    ),
                    "elapsed_seconds": max(
                        0.0,
                        float(progress.get("elapsed_seconds", 0.0) or 0.0),
                    ),
                    "session_games": session_games,
                    "updated_at": progress.get("updated_at"),
                    "candidate_checkpoint": self._checkpoint_label(
                        task.gauntlet.candidate.expected_checkpoint_sha256,
                        role="candidate",
                        task=task,
                    ),
                    "opponent_checkpoint": self._checkpoint_label(
                        task.gauntlet.baseline.expected_checkpoint_sha256,
                        role="opponent",
                        task=task,
                    ),
                }
            )
        task_rows.sort(key=lambda row: (row["device_index"], row["task_short_id"]))
        device_rows: list[dict[str, Any]] = []
        for index, binding in enumerate(self.plan.device_bindings):
            status = device_status[index]
            active_id = status.get("active_task_id")
            active_row = next(
                (row for row in task_rows if row["task_id"] == active_id),
                None,
            )
            device_tasks = [row for row in task_rows if row["device_index"] == index]
            games_total = sum(int(row["games_total"]) for row in device_tasks)
            games_committed = sum(int(row["games_committed"]) for row in device_tasks)
            active_games_per_second = (
                float(active_row["games_per_second"]) if active_row else 0.0
            )
            effective_games_per_second = _effective_device_rate(
                status=status,
                task_rows=device_tasks,
                games_committed=games_committed,
                now=now,
            )
            games_remaining = max(0, games_total - games_committed)
            device_failed = status.get("state") == "failed"
            device_rows.append(
                {
                    "device_index": index,
                    "device_binding": binding,
                    "state": status.get(
                        "state",
                        "complete"
                        if all(row["state"] == "complete" for row in device_tasks)
                        else "pending",
                    ),
                    "active_task_id": active_id,
                    "active_task_short_id": (
                        str(active_id)[:12] if isinstance(active_id, str) else None
                    ),
                    "active_games_per_second": active_games_per_second,
                    "effective_games_per_second": effective_games_per_second,
                    "eta_seconds": (
                        None
                        if device_failed
                        else games_remaining / effective_games_per_second
                        if games_remaining and effective_games_per_second > 0.0
                        else 0.0
                        if not games_remaining
                        else None
                    ),
                    "tasks_total": len(device_tasks),
                    "tasks_complete": sum(
                        row["state"] == "complete" for row in device_tasks
                    ),
                    "games_total": games_total,
                    "games_committed": games_committed,
                    "games_remaining": games_remaining,
                    "error_message": status.get("error_message"),
                    "updated_at": status.get("updated_at"),
                }
            )
        return task_rows, device_rows

    def _read_device_status(self, device_index: int) -> dict[str, Any]:
        output_dir = self._resolve_path(self.plan.output_dir)
        return _read_json_optional(output_dir / f"status-device-{device_index}.json")

    def _standings(self, aggregate: PartAggregate) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        opponent_cells: Counter[str] = Counter(
            candidate_id for candidate_id, _opponent_id in aggregate.cells
        )
        for bundle in self.plan.candidates:
            bundle_id = bundle.bundle_id
            tally = aggregate.bundles.get(bundle_id, ScoreTally())
            low, high = _wilson_interval(tally.score_sum, tally.resolved)
            rows.append(
                {
                    **self._bundle_display(bundle),
                    **_tally_payload(tally),
                    "ci95_low": low,
                    "ci95_high": high,
                    "opponent_cells": opponent_cells[bundle_id],
                }
            )
        rows.sort(
            key=lambda row: (
                row["score"] is not None,
                row["score"] if row["score"] is not None else -1.0,
                row["resolved"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _cells(
        self,
        aggregate: PartAggregate,
        *,
        candidate_bundle_ids: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (candidate_id, opponent_id), tally in aggregate.cells.items():
            if candidate_id not in candidate_bundle_ids:
                continue
            candidate = self._bundle_by_id[candidate_id]
            opponent = self._bundle_by_id[opponent_id]
            rows.append(
                {
                    "candidate_bundle_id": candidate_id,
                    "candidate_deck_hash": candidate.deck_hash,
                    "candidate_checkpoint": self._checkpoint_display(candidate),
                    "opponent_bundle_id": opponent_id,
                    "opponent_deck_hash": opponent.deck_hash,
                    "opponent_checkpoint": self._checkpoint_display(opponent),
                    **_tally_payload(tally),
                }
            )
        rows.sort(
            key=lambda row: (
                row["candidate_checkpoint"],
                row["candidate_deck_hash"],
                row["opponent_checkpoint"],
                row["opponent_deck_hash"],
            )
        )
        return rows[:limit]

    def _bundle_display(self, bundle: PlannedBundle) -> dict[str, Any]:
        checkpoint = self._checkpoint_by_id.get(bundle.checkpoint_id)
        return {
            "bundle_id": bundle.bundle_id,
            "checkpoint": self._checkpoint_display(bundle),
            "checkpoint_version": (
                checkpoint.checkpoint_version if checkpoint is not None else None
            ),
            "deck_hash": bundle.deck_hash,
            "deck_label": bundle.deck_label,
        }

    def _checkpoint_display(self, bundle: PlannedBundle) -> str:
        checkpoint = self._checkpoint_by_id.get(bundle.checkpoint_id)
        return checkpoint.label if checkpoint is not None else bundle.checkpoint_id[:12]

    def _checkpoint_label(
        self,
        checkpoint_sha256: str,
        *,
        role: str,
        task: CampaignTask,
    ) -> str:
        bundle_ids = (
            task.candidate_bundle_ids
            if role == "candidate"
            else task.opponent_bundle_ids
        )
        if bundle_ids:
            return self._checkpoint_display(self._bundle_by_id[bundle_ids[0]])
        return checkpoint_sha256[:12]

    def _resolve_path(self, path: Path) -> Path:
        expanded = path.expanduser()
        return (
            expanded.resolve()
            if expanded.is_absolute()
            else (self.root / expanded).resolve()
        )

    def _gpu_rows(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if now - self._gpu_polled_at < self._gpu_poll_seconds:
            return self._gpu_cache
        self._gpu_polled_at = now
        try:
            result = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._gpu_cache = []
            return self._gpu_cache
        if result.returncode != 0:
            self._gpu_cache = []
            return self._gpu_cache
        rows: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            columns = [column.strip() for column in line.split(",")]
            if len(columns) != 6:
                continue
            try:
                rows.append(
                    {
                        "index": int(columns[0]),
                        "name": columns[1],
                        "memory_used_mib": int(columns[2]),
                        "memory_total_mib": int(columns[3]),
                        "utilization_percent": int(columns[4]),
                        "power_watts": float(columns[5]),
                    }
                )
            except ValueError:
                continue
        self._gpu_cache = rows
        return self._gpu_cache


def _effective_device_rate(
    *,
    status: dict[str, Any],
    task_rows: list[dict[str, Any]],
    games_committed: int,
    now: datetime,
) -> float:
    """Estimate a stable device wall rate across model/task transitions."""
    observed_games = sum(max(0, int(row["session_games"])) for row in task_rows)
    observed_seconds = sum(max(0.0, float(row["elapsed_seconds"])) for row in task_rows)
    if observed_seconds > 0.0 and observed_games > 0:
        return observed_games / observed_seconds

    session_started_at = _parse_timestamp(status.get("session_started_at"))
    start_games_value = status.get("session_start_games_committed")
    if session_started_at is not None and isinstance(start_games_value, int):
        elapsed = max(0.0, (now - session_started_at).total_seconds())
        session_games = max(0, games_committed - start_games_value)
        if elapsed > 0.0 and session_games > 0:
            return session_games / elapsed

    rate_rows = [row for row in task_rows if float(row["games_per_second"]) > 0.0]
    weights = [max(1, int(row["session_games"])) for row in rate_rows]
    weight_total = sum(weights)
    return (
        sum(
            float(row["games_per_second"]) * weight
            for row, weight in zip(rate_rows, weights, strict=True)
        )
        / weight_total
        if weight_total
        else 0.0
    )


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _merge_tally_maps(
    destination: dict[Any, ScoreTally],
    source: dict[Any, ScoreTally],
) -> None:
    for key, tally in source.items():
        destination.setdefault(key, ScoreTally()).merge(tally)


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    if not 0.0 <= score <= 1.0:
        raise ValueError("candidate score must be within [0, 1]")
    return score


def _rate(score_sum: float, count: int) -> float | None:
    return score_sum / count if count else None


def _tally_payload(tally: ScoreTally) -> dict[str, Any]:
    return {
        "games": tally.games,
        "resolved": tally.resolved,
        "wins": tally.wins,
        "draws": tally.draws,
        "losses": tally.losses,
        "unresolved": tally.unresolved,
        "score": _rate(tally.score_sum, tally.resolved),
        "seat0_score": _rate(tally.seat_score_sum[0], tally.seat_resolved[0]),
        "seat1_score": _rate(tally.seat_score_sum[1], tally.seat_resolved[1]),
        "seat0_games": tally.seat_games[0],
        "seat1_games": tally.seat_games[1],
    }


def _wilson_interval(score_sum: float, count: int) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    probability = score_sum / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    return payload


__all__ = ["CampaignLiveReader", "PartAggregate", "ScoreTally"]
