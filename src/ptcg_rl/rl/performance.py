"""Low-overhead wall-clock performance diagnostics for distributed RL."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ptcg_rl.rl.experience import GameTrajectory
from ptcg_rl.rl.performance_history import PerformanceHistoryStore
from ptcg_rl.rl.performance_state import (
    KNOWN_OPPONENT_KINDS,
    KNOWN_OPPONENT_STRATA,
    SCHEMA_VERSION,
    MutablePerformanceWindow,
    OutcomeCellKey,
    OutcomeCounts,
    PerformanceOutcome,
    PerformanceReporterConfig,
    atomic_write_json,
    merge_window_payloads,
    normalize_legacy_performance_summary,
    normalize_performance_window,
    resolve_opponent_stratum,
    utc_timestamp,
)
from ptcg_rl.rl.tensorboard import TensorboardMetricWriter

_LOGGER = logging.getLogger(__name__)


class TrainingPerformanceReporter:
    """Aggregate decoded game outcomes and publish minute-level diagnostics."""

    def __init__(self, config: PerformanceReporterConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: list[dict[str, Any]] = []
        self._pending_history: list[dict[str, Any]] = []
        self._minute_index = 0
        self._started_at = time.time()
        self._cumulative_counts: dict[str, OutcomeCounts] = {}
        self._cumulative_decoded_games = 0
        self._cumulative_stale_excluded_games = 0
        self._cumulative_queued_games = 0
        self._cumulative_missing_metadata_games = 0
        self._last_partial_window: dict[str, Any] | None = None
        self._configured_slice_keys = self._build_configured_slice_keys()
        self._history_store = PerformanceHistoryStore(
            config.history_dir,
            config.history_manifest_path,
        )
        self._load_state()
        self._window = MutablePerformanceWindow(started_at=time.time())
        had_event_files = any(config.tensorboard_dir.glob("events.out.tfevents*"))
        self._writer = TensorboardMetricWriter.create(
            config.tensorboard_dir,
            enabled=config.tensorboard_enabled,
            flush_seconds=config.tensorboard_flush_seconds,
        )
        if self._history and not had_event_files:
            self._replay_history()

    def start(self) -> None:
        """Start the wall-clock reporting thread."""
        if self._thread is not None:
            raise RuntimeError("performance reporter is already started")
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist()
        self._thread = threading.Thread(
            target=self._run,
            name="training-performance-reporter",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop reporting and persist the final partial window without plotting it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._finalize_elapsed(time.time())
        with self._lock:
            self._last_partial_window = self._window_snapshot(
                self._window,
                ended_at=time.time(),
                complete=False,
                minute_index=None,
            )
        self._commit_pending_history(force=True)
        self._persist()
        self._writer.close()

    def observe_decoded(
        self,
        *,
        worker_id: str,
        trajectories: Sequence[GameTrajectory],
    ) -> None:
        """Record every successfully decoded trajectory before staleness filtering."""
        with self._lock:
            for trajectory in trajectories:
                self._observe_trajectory(worker_id, trajectory)

    def observe_outcomes(
        self,
        *,
        worker_id: str,
        outcomes: Sequence[PerformanceOutcome],
    ) -> None:
        """Record validated scored outcomes from a central single writer."""
        with self._lock:
            for outcome in outcomes:
                self._observe_result(
                    worker_id=worker_id,
                    deck_label=outcome.candidate_deck_label,
                    opponent_kind=outcome.opponent_kind,
                    opponent_stratum=outcome.opponent_stratum,
                    opponent_deck_label=outcome.opponent_deck_label,
                    opponent_id=outcome.opponent_id,
                    candidate_seat=outcome.candidate_seat,
                    reward=outcome.candidate_reward,
                    policy_version=outcome.policy_version,
                    missing_fields=(),
                )

    def record_delivery(self, *, stale_excluded_games: int, queued_games: int) -> None:
        """Record how many decoded games were excluded or entered the learner queue."""
        if stale_excluded_games < 0 or queued_games < 0:
            raise ValueError("delivery counts must be non-negative")
        with self._lock:
            self._window.stale_excluded_games += stale_excluded_games
            self._window.queued_games += queued_games

    def summary(self) -> dict[str, Any]:
        """Return compact live reporter status for coordinator diagnostics."""
        with self._lock:
            return {
                "enabled": True,
                "summary_path": str(self.config.summary_path),
                "minute_index": self._minute_index,
                "completed_windows": self._minute_index,
                "active_window_decoded_games": self._window.decoded_games,
                "cumulative_decoded_games": self._cumulative_decoded_games,
            }

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=min(1.0, self.config.interval_seconds)):
            self._finalize_elapsed(time.time())

    def _finalize_elapsed(self, now: float) -> None:
        completed: list[dict[str, Any]] = []
        with self._lock:
            while now - self._window.started_at >= self.config.interval_seconds:
                ended_at = self._window.started_at + self.config.interval_seconds
                self._minute_index += 1
                snapshot = self._window_snapshot(
                    self._window,
                    ended_at=ended_at,
                    complete=True,
                    minute_index=self._minute_index,
                )
                self._history.append(snapshot)
                self._pending_history.append(snapshot)
                self._history = self._history[-self.config.recent_window_limit :]
                self._merge_cumulative(snapshot)
                completed.append(snapshot)
                self._window = MutablePerformanceWindow(started_at=ended_at)
                self._last_partial_window = None
        if not completed:
            return
        for snapshot in completed:
            self._write_tensorboard_window(snapshot)
        self._commit_pending_history(force=False)
        self._persist()

    def _observe_trajectory(self, worker_id: str, trajectory: GameTrajectory) -> None:
        extra = trajectory.metadata.extra
        metadata = extra if isinstance(extra, Mapping) else {}
        missing_fields: list[str] = []
        deck_label = str(metadata.get("candidate_deck_label", "")).strip()
        if not deck_label:
            missing_fields.append("candidate_deck_label")
        opponent_kind = str(metadata.get("opponent_kind", "")).strip()
        if not opponent_kind:
            opponent_kind = "unknown"
            missing_fields.append("opponent_kind")
        opponent_deck_label = str(metadata.get("opponent_deck_label", "")).strip()
        opponent_id = str(metadata.get("opponent_id", "")).strip()
        opponent_stratum = str(metadata.get("opponent_stratum", "")).strip()
        if opponent_stratum not in KNOWN_OPPONENT_STRATA:
            if opponent_stratum:
                missing_fields.append("opponent_stratum_invalid")
            else:
                missing_fields.append("opponent_stratum")
            opponent_stratum = ""
        raw_candidate_seat = metadata.get("candidate_seat")
        cell_candidate_seat = -1
        try:
            candidate_seat = int(str(raw_candidate_seat))
        except (TypeError, ValueError):
            candidate_seat = 0
            missing_fields.append("candidate_seat")
        if candidate_seat in (0, 1):
            cell_candidate_seat = candidate_seat
        else:
            candidate_seat = 0
            missing_fields.append("candidate_seat")

        self._observe_result(
            worker_id=worker_id,
            deck_label=deck_label,
            opponent_kind=opponent_kind,
            opponent_stratum=opponent_stratum,
            opponent_deck_label=opponent_deck_label,
            opponent_id=opponent_id,
            candidate_seat=cell_candidate_seat,
            reward=trajectory.reward_for_seat(candidate_seat),
            policy_version=int(trajectory.metadata.policy_version),
            missing_fields=missing_fields,
        )

    def _observe_result(
        self,
        *,
        worker_id: str,
        deck_label: str,
        opponent_kind: str,
        opponent_stratum: str,
        opponent_deck_label: str,
        opponent_id: str,
        candidate_seat: int,
        reward: float,
        policy_version: int,
        missing_fields: Sequence[str],
    ) -> None:
        """Merge one normalized result into the active wall-clock window."""
        stratum = resolve_opponent_stratum(
            opponent_kind=opponent_kind,
            opponent_stratum=opponent_stratum,
        )
        slice_keys = {
            "overall/all",
            f"overall/{opponent_kind}",
            f"overall/stratum/{stratum}",
        }
        if opponent_kind in self.config.stationary_opponent_kinds:
            slice_keys.add("overall/stationary")
        if deck_label in self.config.target_deck_labels:
            slice_keys.update(
                {
                    f"deck/{deck_label}/all",
                    f"deck/{deck_label}/{opponent_kind}",
                    f"deck/{deck_label}/stratum/{stratum}",
                }
            )
            if opponent_kind in self.config.stationary_opponent_kinds:
                slice_keys.add(f"deck/{deck_label}/stationary")
        for key in slice_keys:
            self._window.counts.setdefault(key, OutcomeCounts()).observe(reward)
        cell_key = OutcomeCellKey(
            candidate_deck_label=deck_label,
            opponent_kind=opponent_kind,
            opponent_stratum=stratum,
            opponent_deck_label=opponent_deck_label,
            opponent_id=opponent_id,
            candidate_seat=candidate_seat,
        )
        self._window.cells.setdefault(cell_key, OutcomeCounts()).observe(reward)

        self._window.decoded_games += 1
        self._window.worker_games[worker_id or "unknown"] += 1
        if missing_fields:
            self._window.missing_metadata_games += 1
            self._window.missing_metadata_fields.update(set(missing_fields))
        if policy_version >= 0:
            self._window.policy_versions.append(policy_version)

    def _window_snapshot(
        self,
        window: MutablePerformanceWindow,
        *,
        ended_at: float,
        complete: bool,
        minute_index: int | None,
    ) -> dict[str, Any]:
        slice_keys = self._configured_slice_keys | set(window.counts)
        return {
            "minute_index": minute_index,
            "complete": complete,
            "started_at_utc": utc_timestamp(window.started_at),
            "ended_at_utc": utc_timestamp(ended_at),
            "ended_at_epoch_seconds": ended_at,
            "decoded_games": window.decoded_games,
            "stale_excluded_games": window.stale_excluded_games,
            "queued_games": window.queued_games,
            "missing_metadata_games": window.missing_metadata_games,
            "missing_metadata_fields": dict(
                sorted(window.missing_metadata_fields.items())
            ),
            "worker_games": dict(sorted(window.worker_games.items())),
            "policy_version_min": (
                min(window.policy_versions) if window.policy_versions else None
            ),
            "policy_version_max": (
                max(window.policy_versions) if window.policy_versions else None
            ),
            "slices": {
                key: window.counts.get(key, OutcomeCounts()).as_dict()
                for key in sorted(slice_keys)
            },
            "cells": [
                key.as_dict(counts) for key, counts in sorted(window.cells.items())
            ],
        }

    def _write_tensorboard_window(self, snapshot: Mapping[str, Any]) -> None:
        step = int(snapshot["minute_index"])
        walltime = float(snapshot["ended_at_epoch_seconds"])
        self._write_slice_scalars(
            snapshot["slices"],
            prefix="performance/1m",
            step=step,
            walltime=walltime,
        )
        self._write_system_scalars(snapshot, step=step, walltime=walltime)
        rolling = self._rolling_payload()
        if rolling is not None:
            self._write_slice_scalars(
                rolling["slices"],
                prefix=f"performance/{self.config.rolling_window_minutes}m",
                step=step,
                walltime=walltime,
            )
        self._writer.flush()

    def _write_system_scalars(
        self,
        snapshot: Mapping[str, Any],
        *,
        step: int,
        walltime: float,
    ) -> None:
        """Write sample-flow context for one completed reporting interval."""
        for name in (
            "decoded_games",
            "stale_excluded_games",
            "queued_games",
            "missing_metadata_games",
            "policy_version_min",
            "policy_version_max",
        ):
            self._writer.add_scalar(
                f"performance/system/{name}",
                snapshot.get(name),
                step,
                walltime=walltime,
            )

    def _write_slice_scalars(
        self,
        raw_slices: Any,
        *,
        prefix: str,
        step: int,
        walltime: float,
    ) -> None:
        if not isinstance(raw_slices, Mapping):
            return
        for key, raw_counts in raw_slices.items():
            if not isinstance(raw_counts, Mapping):
                continue
            self._writer.add_scalar(
                f"{prefix}/games/{key}",
                raw_counts.get("games", 0),
                step,
                walltime=walltime,
            )
            self._writer.add_scalar(
                f"{prefix}/score/{key}",
                raw_counts.get("score"),
                step,
                walltime=walltime,
            )

    def _rolling_payload(self) -> dict[str, Any] | None:
        with self._lock:
            required = self.config.rolling_window_intervals
            if len(self._history) < required:
                return None
            windows = list(self._history[-required:])
        return merge_window_payloads(windows, self._configured_slice_keys)

    def _merge_cumulative(self, snapshot: Mapping[str, Any]) -> None:
        raw_slices = snapshot.get("slices")
        if isinstance(raw_slices, Mapping):
            for key, raw_counts in raw_slices.items():
                if isinstance(raw_counts, Mapping):
                    self._cumulative_counts.setdefault(str(key), OutcomeCounts()).merge(
                        OutcomeCounts.from_mapping(raw_counts)
                    )
        self._cumulative_decoded_games += int(snapshot.get("decoded_games", 0))
        self._cumulative_stale_excluded_games += int(
            snapshot.get("stale_excluded_games", 0)
        )
        self._cumulative_queued_games += int(snapshot.get("queued_games", 0))
        self._cumulative_missing_metadata_games += int(
            snapshot.get("missing_metadata_games", 0)
        )

    def _persist(self) -> None:
        with self._lock:
            rolling = self._rolling_payload_unlocked()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "started_at_utc": utc_timestamp(self._started_at),
                "updated_at_utc": utc_timestamp(time.time()),
                "semantics": self._semantics(),
                "minute_index": self._minute_index,
                "completed_windows": list(self._history),
                "history_manifest": str(self.config.history_manifest_path.name),
                "latest_rolling_window": rolling,
                "last_partial_window": self._last_partial_window,
                "cumulative": {
                    "decoded_games": self._cumulative_decoded_games,
                    "stale_excluded_games": self._cumulative_stale_excluded_games,
                    "queued_games": self._cumulative_queued_games,
                    "missing_metadata_games": self._cumulative_missing_metadata_games,
                    "slices": {
                        key: counts.as_dict()
                        for key, counts in sorted(self._cumulative_counts.items())
                    },
                },
            }
        atomic_write_json(self.config.summary_path, payload)

    def _rolling_payload_unlocked(self) -> dict[str, Any] | None:
        required = self.config.rolling_window_intervals
        if len(self._history) < required:
            return None
        return merge_window_payloads(
            self._history[-required:],
            self._configured_slice_keys,
        )

    def _load_state(self) -> None:
        path = self.config.summary_path
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("performance summary must be a JSON object")
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in (1, 2, SCHEMA_VERSION):
            raise ValueError("unsupported performance summary schema")
        expected_semantics = {
            1: self._schema_v1_semantics(),
            2: self._schema_v2_semantics(),
            SCHEMA_VERSION: self._semantics(),
        }[schema_version]
        if payload.get("semantics") != expected_semantics:
            self._archive_incompatible_state(
                observed_semantics=payload.get("semantics"),
                expected_semantics=expected_semantics,
            )
            return
        raw_history = payload.get("completed_windows", [])
        if not isinstance(raw_history, list):
            raise ValueError("completed_windows must be a list")
        raw_windows = [
            self._normalize_window(
                dict(item),
                rebuild_stratum_slices=schema_version < SCHEMA_VERSION,
            )
            for item in raw_history
            if isinstance(item, Mapping)
        ]
        self._history = raw_windows[-self.config.recent_window_limit :]
        self._minute_index = int(payload.get("minute_index", len(self._history)))
        raw_started_at = payload.get("started_at_utc")
        if isinstance(raw_started_at, str):
            self._started_at = datetime.fromisoformat(
                raw_started_at.replace("Z", "+00:00")
            ).timestamp()
        if schema_version == 1:
            for snapshot in raw_windows:
                self._merge_cumulative(snapshot)
            return
        self._load_cumulative(
            payload.get("cumulative"),
            rebuild_stratum_slices=schema_version < SCHEMA_VERSION,
        )
        last_committed = self._history_store.last_committed_minute()
        self._pending_history = [
            snapshot
            for snapshot in self._history
            if int(snapshot.get("minute_index") or 0) > last_committed
        ]

    def _archive_incompatible_state(
        self,
        *,
        observed_semantics: Any,
        expected_semantics: Mapping[str, Any],
    ) -> None:
        """Atomically preserve derived metrics that cannot be resumed safely."""
        state_dir = self.config.summary_path.parent
        identity = hashlib.sha256(
            json.dumps(
                {
                    "observed": observed_semantics,
                    "expected": expected_semantics,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:12]
        archive_dir = state_dir.with_name(
            f"{state_dir.name}.incompatible-{time.time_ns()}-{identity}"
        )
        state_dir.replace(archive_dir)
        atomic_write_json(
            archive_dir / "incompatible_state.json",
            {
                "reason": "performance summary semantics do not match config",
                "observed_semantics": observed_semantics,
                "expected_semantics": dict(expected_semantics),
                "archived_at_utc": utc_timestamp(time.time()),
            },
        )
        _LOGGER.warning(
            "archived incompatible derived performance state: source=%s archive=%s",
            state_dir,
            archive_dir,
        )

    def _replay_history(self) -> None:
        for index, snapshot in enumerate(self._history):
            step = int(snapshot["minute_index"])
            walltime = float(snapshot["ended_at_epoch_seconds"])
            self._write_slice_scalars(
                snapshot.get("slices"),
                prefix="performance/1m",
                step=step,
                walltime=walltime,
            )
            self._write_system_scalars(snapshot, step=step, walltime=walltime)
            required = self.config.rolling_window_intervals
            if index + 1 >= required:
                rolling = merge_window_payloads(
                    self._history[index + 1 - required : index + 1],
                    self._configured_slice_keys,
                )
                self._write_slice_scalars(
                    rolling["slices"],
                    prefix=f"performance/{self.config.rolling_window_minutes}m",
                    step=step,
                    walltime=walltime,
                )
        self._writer.flush()

    def _build_configured_slice_keys(self) -> set[str]:
        kinds = set(KNOWN_OPPONENT_KINDS)
        keys = {"overall/all", "overall/stationary"}
        keys.update(f"overall/{kind}" for kind in kinds)
        keys.update(f"overall/stratum/{item}" for item in KNOWN_OPPONENT_STRATA)
        for label in self.config.target_deck_labels:
            keys.update(
                {
                    f"deck/{label}/all",
                    f"deck/{label}/stationary",
                }
            )
            keys.update(f"deck/{label}/{kind}" for kind in kinds)
            keys.update(
                f"deck/{label}/stratum/{item}"
                for item in KNOWN_OPPONENT_STRATA
            )
        return keys

    def _semantics(self) -> dict[str, Any]:
        return {
            "count_stage": self.config.count_stage,
            "score": {"win": 1.0, "draw": 0.5, "loss": 0.0},
            "interval_seconds": self.config.interval_seconds,
            "rolling_window_minutes": self.config.rolling_window_minutes,
            "rolling_window_intervals": self.config.rolling_window_intervals,
            "stationary_opponent_kinds": list(self.config.stationary_opponent_kinds),
            "opponent_strata": list(KNOWN_OPPONENT_STRATA),
            "target_deck_labels": list(self.config.target_deck_labels),
            "exact_cell_dimensions": [
                "candidate_deck_label",
                "opponent_kind",
                "opponent_stratum",
                "opponent_deck_label",
                "opponent_id",
                "candidate_seat",
            ],
            "recent_window_limit": self.config.recent_window_limit,
            "parquet_shard_windows": self.config.parquet_shard_windows,
        }

    def _schema_v2_semantics(self) -> dict[str, Any]:
        """Return the exact schema-v2 contract for resume compatibility."""
        return {
            "count_stage": self.config.count_stage,
            "score": {"win": 1.0, "draw": 0.5, "loss": 0.0},
            "interval_seconds": self.config.interval_seconds,
            "rolling_window_minutes": self.config.rolling_window_minutes,
            "rolling_window_intervals": self.config.rolling_window_intervals,
            "stationary_opponent_kinds": list(self.config.stationary_opponent_kinds),
            "target_deck_labels": list(self.config.target_deck_labels),
            "exact_cell_dimensions": [
                "candidate_deck_label",
                "opponent_kind",
                "opponent_deck_label",
                "opponent_id",
                "candidate_seat",
            ],
            "recent_window_limit": self.config.recent_window_limit,
            "parquet_shard_windows": self.config.parquet_shard_windows,
        }

    def _schema_v1_semantics(self) -> dict[str, Any]:
        """Return the exact schema-v1 contract for resume compatibility."""
        semantics = self._schema_v2_semantics()
        semantics.pop("exact_cell_dimensions")
        semantics.pop("recent_window_limit")
        semantics.pop("parquet_shard_windows")
        return semantics

    def _normalize_window(
        self,
        window: dict[str, Any],
        *,
        rebuild_stratum_slices: bool,
    ) -> dict[str, Any]:
        """Populate canonical strata when resuming legacy summaries."""
        return normalize_performance_window(
            window,
            rebuild_stratum_slices=rebuild_stratum_slices,
        )

    def _load_cumulative(
        self,
        raw_cumulative: Any,
        *,
        rebuild_stratum_slices: bool,
    ) -> None:
        if not isinstance(raw_cumulative, Mapping):
            raise ValueError("performance cumulative summary must be an object")
        if rebuild_stratum_slices:
            raw_cumulative = normalize_legacy_performance_summary(
                {"cumulative": raw_cumulative}
            )["cumulative"]
        raw_slices = raw_cumulative.get("slices")
        if not isinstance(raw_slices, Mapping):
            raise ValueError("performance cumulative slices must be an object")
        self._cumulative_counts = {
            str(key): OutcomeCounts.from_mapping(value)
            for key, value in raw_slices.items()
            if isinstance(value, Mapping)
        }
        self._cumulative_decoded_games = int(raw_cumulative.get("decoded_games", 0))
        self._cumulative_stale_excluded_games = int(
            raw_cumulative.get("stale_excluded_games", 0)
        )
        self._cumulative_queued_games = int(raw_cumulative.get("queued_games", 0))
        self._cumulative_missing_metadata_games = int(
            raw_cumulative.get("missing_metadata_games", 0)
        )

    def _commit_pending_history(self, *, force: bool) -> None:
        with self._lock:
            if not self._pending_history:
                return
            if (
                not force
                and len(self._pending_history) < self.config.parquet_shard_windows
            ):
                return
            pending = list(self._pending_history)
        self._history_store.commit(pending)
        committed_minute = self._history_store.last_committed_minute()
        with self._lock:
            self._pending_history = [
                window
                for window in self._pending_history
                if int(window.get("minute_index") or 0) > committed_minute
            ]
