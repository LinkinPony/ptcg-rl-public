"""Local, read-only repository for RL performance summaries and history."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from omegaconf import OmegaConf

from ptcg_rl.dashboard.deck_routes import (
    DeckRouteIdentity,
    DeckRouteMetadata,
    label_hash,
    resolve_deck_route_metadata,
)
from ptcg_rl.dashboard.models import (
    DashboardConfig,
    DeckRow,
    DeckSeries,
    HealthPayload,
    MatchupRow,
    OutcomeStats,
    PerformanceTable,
    RunInfo,
    ScopeName,
    SeriesPoint,
    WindowName,
)
from ptcg_rl.rl.performance_history import PerformanceHistoryStore
from ptcg_rl.rl.performance_state import (
    KNOWN_OPPONENT_STRATA,
    OutcomeCounts,
    normalize_legacy_performance_summary,
)

_SLICES = (
    "all",
    "self_play",
    "sentinel",
    "adaptive_history",
    "scripted",
    "legacy_unknown",
    "frozen",
    "stationary",
)

_PathSignature = tuple[int, int]
_HistoryPartSignature = tuple[str, int, str]


@dataclass
class _HistoryCacheEntry:
    """Validated immutable history, split by append-only shard identity."""

    manifest_signature: _PathSignature | None
    part_windows: dict[_HistoryPartSignature, tuple[dict[str, Any], ...]]
    windows: list[dict[str, Any]]
    warnings: list[str]


class DashboardRepository:
    """Load current and historical mirrored runs without mutating them."""

    def __init__(self, config: DashboardConfig, *, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root.resolve()
        raw_root = Path(config.run_root)
        self.run_root = (
            raw_root if raw_root.is_absolute() else self.repo_root / raw_root
        ).resolve()
        self._summary_cache: dict[str, tuple[_PathSignature, dict[str, Any]]] = {}
        self._history_cache: dict[str, _HistoryCacheEntry] = {}
        self._deck_route_cache: dict[
            tuple[str, tuple[str, ...]],
            tuple[tuple[int, int] | None, DeckRouteMetadata],
        ] = {}
        self._deck_catalog = self._load_deck_catalogs()

    @classmethod
    def from_config(cls, path: Path, *, repo_root: Path) -> DashboardRepository:
        """Load one Hydra-compatible YAML dashboard profile."""
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(payload, dict):
            raise ValueError("dashboard config must be an object")
        return cls(DashboardConfig.model_validate(payload), repo_root=repo_root)

    def list_runs(self) -> tuple[RunInfo, ...]:
        """Return every local run containing performance or live learner status."""
        run_ids = {
            path.parent.parent.name
            for path in self.run_root.glob("*/performance/training_performance.json")
            if path.is_file()
        }
        run_ids.update(
            path.parent.name
            for path in self.run_root.glob("*/learner_status.json")
            if path.is_file()
        )
        run_ids.update(
            path.parent.parent.name
            for path in self.run_root.glob("*/weights/latest.json")
            if path.is_file()
        )
        run_ids.update(
            segment
            for lineage in self.config.lineages.values()
            for segment in lineage.segments
            if self._summary_path(segment).is_file()
        )
        rows = [self._run_info(run_id) for run_id in sorted(run_ids)]
        rows.sort(key=lambda row: (not row.current, row.run_id), reverse=False)
        return tuple(rows)

    def run_info(self, run: str) -> RunInfo:
        """Return one run row without scanning or decoding every other run."""
        return self._run_info(self.resolve_run_id(run))

    def training_history_revision(
        self,
        run: str,
    ) -> tuple[str, _PathSignature | None, _PathSignature | None, _PathSignature | None]:
        """Return cheap artifact identities for cache invalidation."""
        run_id = self.resolve_run_id(run)
        performance_dir = self._summary_path(run_id).parent
        return (
            run_id,
            _path_signature(self._summary_path(run_id)),
            _path_signature(performance_dir / "history_manifest.json"),
            _path_signature(self._run_dir(run_id) / "resolved_config.json"),
        )

    def resolve_run_id(self, value: str) -> str:
        """Resolve the `current` alias and reject undiscovered paths."""
        run_id = self.config.current_run if value == "current" else value
        run_dir = self._run_dir(run_id)
        discovered = (
            (run_dir / "performance" / "training_performance.json").is_file()
            or (run_dir / "learner_status.json").is_file()
            or (run_dir / "weights" / "latest.json").is_file()
        )
        if discovered:
            return run_id
        raise KeyError(f"unknown local training run: {value}")

    def run_directory(self, run: str) -> Path:
        """Return the validated local directory for one discovered run."""
        return self._run_dir(self.resolve_run_id(run))

    def performance_table(
        self,
        run: str,
        *,
        window: WindowName = "cumulative",
        scope: ScopeName = "segment",
    ) -> PerformanceTable:
        """Build controller-sliced deck standings for one segment or lineage."""
        run_id = self.resolve_run_id(run)
        if not self._summary_path(run_id).is_file():
            return _status_only_performance_table(
                run_id,
                window=window,
                scope=scope,
            )
        segments = self._scope_segments(run_id, scope, window)
        payloads = [(segment, self._read_summary(segment)) for segment in segments]
        merged = _empty_slice_counts()
        started: str | None = None
        ended: str | None = None
        target_labels: list[str] = []
        exact_available = True
        for _segment, payload in payloads:
            selected = _selected_payload(payload, window)
            if selected is None:
                continue
            started = started or _optional_text(selected.get("started_at_utc"))
            ended = _optional_text(selected.get("ended_at_utc")) or ended
            _merge_slices(merged, selected.get("slices"))
            target_labels.extend(_target_labels(payload))
            exact_available = (
                exact_available and int(payload.get("schema_version", 1)) >= 2
            )
        unique_targets = tuple(dict.fromkeys(target_labels))
        overall = {
            name: _outcome_stats(
                merged.get(_slice_path("overall", name), OutcomeCounts())
            )
            for name in _SLICES
        }
        decks = tuple(
            DeckRow(
                deck_label=label,
                deck_hash=_deck_hash(label),
                display_name=self._deck_name(label),
                slices={
                    name: _outcome_stats(
                        merged.get(_slice_path(f"deck/{label}", name), OutcomeCounts())
                    )
                    for name in _SLICES
                },
            )
            for label in unique_targets
        )
        return PerformanceTable(
            run_id=run_id,
            scope=scope,
            window=window,
            started_at_utc=started,
            ended_at_utc=ended,
            source_segments=segments,
            overall=overall,
            decks=decks,
            exact_dimensions_available=exact_available,
        )

    def series(
        self,
        run: str,
        *,
        deck_label: str | None = None,
        opponent_slice: str = "all",
        rolling_minutes: int = 15,
    ) -> tuple[SeriesPoint, ...]:
        """Return a sample-count paired rolling score series for one segment."""
        if opponent_slice not in _SLICES:
            raise ValueError(f"unsupported opponent slice: {opponent_slice}")
        if rolling_minutes <= 0:
            raise ValueError("rolling_minutes must be positive")
        run_id = self.resolve_run_id(run)
        if not self._summary_path(run_id).is_file():
            return ()
        payload = self._read_summary(run_id)
        windows, _warnings = self._all_windows(run_id, payload)
        interval = float(_semantics(payload).get("interval_seconds", 60.0))
        required = max(1, round(rolling_minutes * 60.0 / interval))
        key = _slice_path(
            f"deck/{deck_label}" if deck_label else "overall",
            opponent_slice,
        )
        return _rolling_series(windows, key=key, required=required)

    def deck_series(
        self,
        run: str,
        *,
        opponent_slice: str = "all",
        rolling_minutes: int = 15,
        scope: ScopeName = "segment",
    ) -> tuple[DeckSeries, ...]:
        """Return aligned rolling win-rate series for every target deck."""
        if opponent_slice not in _SLICES:
            raise ValueError(f"unsupported opponent slice: {opponent_slice}")
        if rolling_minutes <= 0:
            raise ValueError("rolling_minutes must be positive")
        run_id = self.resolve_run_id(run)
        if not self._summary_path(run_id).is_file():
            return ()
        segments = self._scope_segments(run_id, scope, "cumulative")
        windows: list[dict[str, Any]] = []
        labels: list[str] = []
        interval = 60.0
        for segment in segments:
            payload = self._read_summary(segment)
            segment_windows, _warnings = self._all_windows(segment, payload)
            windows.extend(segment_windows)
            labels.extend(_target_labels(payload))
            interval = float(_semantics(payload).get("interval_seconds", interval))
        required = max(1, round(rolling_minutes * 60.0 / interval))
        return tuple(
            DeckSeries(
                deck_label=label,
                deck_hash=_deck_hash(label),
                display_name=self._deck_name(label),
                points=_rolling_series(
                    windows,
                    key=_slice_path(f"deck/{label}", opponent_slice),
                    required=required,
                ),
            )
            for label in dict.fromkeys(labels)
        )

    def matchups(
        self,
        run: str,
        *,
        window: WindowName = "cumulative",
        opponent_kind: str | None = None,
        candidate_seat: int | None = None,
    ) -> tuple[MatchupRow, ...]:
        """Aggregate exact v2 cells by candidate, opponent, pilot and seat."""
        if candidate_seat not in (None, 0, 1):
            raise ValueError("candidate_seat must be 0, 1, or omitted")
        run_id = self.resolve_run_id(run)
        if not self._summary_path(run_id).is_file():
            return ()
        payload = self._read_summary(run_id)
        if int(payload.get("schema_version", 1)) < 2:
            return ()
        if window == "cumulative":
            all_windows, _warnings = self._all_windows(run_id, payload)
            windows = _select_windows(all_windows, payload, window)
        else:
            windows = _select_windows(_recent_windows(payload), payload, window)
        totals: dict[tuple[str, str, str, str, int], OutcomeCounts] = {}
        for minute in windows:
            raw_cells = minute.get("cells")
            if not isinstance(raw_cells, list):
                continue
            for raw_cell in raw_cells:
                cell = _mapping(raw_cell)
                kind = str(cell.get("opponent_kind", ""))
                seat = int(cell.get("candidate_seat", -1))
                if opponent_kind is not None and kind != opponent_kind:
                    continue
                if candidate_seat is not None and seat != candidate_seat:
                    continue
                key = (
                    str(cell.get("candidate_deck_label", "")),
                    kind,
                    str(cell.get("opponent_deck_label", "")),
                    str(cell.get("opponent_id", "")),
                    seat,
                )
                totals.setdefault(key, OutcomeCounts()).merge(
                    OutcomeCounts.from_mapping(cell)
                )
        rows = [
            MatchupRow(
                candidate_deck_label=key[0],
                candidate_deck_hash=_exact_deck_hash(key[0]),
                candidate_display_name=self._deck_name(key[0]),
                opponent_kind=key[1],
                opponent_deck_label=key[2],
                opponent_deck_hash=_exact_deck_hash(key[2]),
                opponent_display_name=self._opponent_name(
                    deck_label=key[2],
                    opponent_id=key[3],
                ),
                opponent_id=key[3],
                candidate_seat=key[4],
                outcomes=_outcome_stats(counts),
            )
            for key, counts in totals.items()
        ]
        rows.sort(
            key=lambda row: (
                row.candidate_deck_label,
                row.opponent_kind,
                row.opponent_deck_label,
                row.opponent_id,
                row.candidate_seat,
            )
        )
        return tuple(rows)

    def health(self, run: str) -> HealthPayload:
        """Return normalized run status and non-gating data warnings."""
        from tools.rl_monitor import build_status

        run_id = self.resolve_run_id(run)
        summary_path = self._summary_path(run_id)
        path = (
            summary_path if summary_path.is_file() else self._status_marker_path(run_id)
        )
        age = max(0.0, time.time() - path.stat().st_mtime) if path.exists() else None
        state = _data_state(age)
        warnings: list[str] = []
        if summary_path.is_file():
            payload = self._read_summary(run_id)
        else:
            payload = {}
            warnings.append(
                "该 run 尚未发布 performance outcome 镜像；"
                "当前仅显示 learner 与 checkpoint 运行状态"
            )
        if int(payload.get("schema_version", 0)) >= 2:
            warnings.extend(self._history_warnings(run_id))
        return HealthPayload(
            run_id=run_id,
            data_age_seconds=age,
            data_state=state,
            status=cast(dict[str, object], build_status(self._run_dir(run_id))),
            warnings=tuple(warnings),
        )

    def training_history(
        self,
        run: str,
    ) -> tuple[
        str,
        Mapping[str, Any],
        tuple[dict[str, Any], ...],
        tuple[str, ...],
    ]:
        """Return immutable-complete observed windows and their source semantics."""
        run_id = self.resolve_run_id(run)
        if not self._summary_path(run_id).is_file():
            return run_id, {}, (), ("training performance history is unavailable",)
        payload = self._read_summary(run_id)
        windows, warnings = self._all_windows(run_id, payload)
        return run_id, payload, tuple(windows), tuple(warnings)

    def deck_display_name(self, label: str) -> str:
        """Resolve one deck label through the configured aliases and catalogs."""
        return self._deck_name(label)

    def deck_hash(self, label: str) -> str:
        """Return the compact exact digest when a label carries one."""
        return _deck_hash(label)

    def exact_deck_hash(self, label: str) -> str | None:
        """Return the compact exact digest, or None for non-exact identities."""
        return _exact_deck_hash(label)

    def deck_route_metadata(
        self,
        run: str,
        *,
        deck_labels: Sequence[str],
    ) -> DeckRouteMetadata:
        """Resolve performance labels to immutable exact and family routes."""
        run_id = self.resolve_run_id(run)
        labels = tuple(deck_labels)
        cache_key = (run_id, labels)
        path = self._run_dir(run_id) / "resolved_config.json"
        signature = (
            (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
        )
        cached = self._deck_route_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        payload = _read_optional_json(path)
        metadata = resolve_deck_route_metadata(
            payload,
            repo_root=self.repo_root,
            deck_labels=labels,
        )
        self._deck_route_cache[cache_key] = (signature, metadata)
        return metadata

    def deck_family_display_name(
        self,
        family_id: str,
        *,
        deck_labels: Sequence[str],
    ) -> str:
        """Resolve a family presentation alias without changing its identity."""
        configured = self.config.family_aliases.get(family_id)
        if configured is not None:
            return configured
        names = [self._deck_name(label) for label in deck_labels]
        if names:
            return min(names, key=lambda name: (len(name), name))
        return f"Family {family_id[:8]}"

    def opponent_display_name(self, *, deck_label: str, opponent_id: str) -> str:
        """Resolve one opponent identity without discarding its pilot label."""
        return self._opponent_name(
            deck_label=deck_label,
            opponent_id=opponent_id,
        )

    def checkpoint_route_compatibility(
        self,
        run: str,
        *,
        deck_labels: Sequence[str],
        active_deck_digests: Sequence[str],
        exact_registry_fingerprint: str | None,
        route_identities: Mapping[str, DeckRouteIdentity] | None = None,
    ) -> dict[str, bool | None]:
        """Bind training labels to a checkpoint's exact route registry."""
        unknown: dict[str, bool | None] = dict.fromkeys(deck_labels)
        if exact_registry_fingerprint is None:
            return unknown
        run_id = self.resolve_run_id(run)
        payload = _read_optional_json(self._run_dir(run_id) / "resolved_config.json")
        if (
            _optional_text(payload.get("exact_registry_fingerprint"))
            != exact_registry_fingerprint
        ):
            return unknown
        active = set(active_deck_digests)
        identities = (
            route_identities
            or self.deck_route_metadata(
                run_id,
                deck_labels=deck_labels,
            ).identities
        )
        return {
            label: (
                identity.deck_digest in active
                if (identity := identities.get(label)) is not None
                else None
                if _exact_deck_hash(label) is None
                else False
            )
            for label in deck_labels
        }

    def _run_info(self, run_id: str) -> RunInfo:
        if not self._summary_path(run_id).is_file():
            marker = self._status_marker_path(run_id)
            try:
                age = max(0.0, time.time() - marker.stat().st_mtime)
                payload = _read_optional_json(
                    self._run_dir(run_id) / "learner_status.json"
                )
                return RunInfo(
                    run_id=run_id,
                    display_name=run_id,
                    current=run_id == self.config.current_run,
                    lineage_id=self._lineage_id(run_id),
                    schema_version=0,
                    updated_at_utc=_mtime_utc(marker),
                    minute_index=int(payload.get("update_index", 0)),
                    data_state=_data_state(age),
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                return RunInfo(
                    run_id=run_id,
                    display_name=run_id,
                    current=run_id == self.config.current_run,
                    lineage_id=self._lineage_id(run_id),
                    schema_version=0,
                    updated_at_utc=None,
                    minute_index=0,
                    data_state="invalid",
                    detail=str(error),
                )
        try:
            payload = self._read_summary(run_id)
            age = max(0.0, time.time() - self._summary_path(run_id).stat().st_mtime)
            return RunInfo(
                run_id=run_id,
                display_name=run_id,
                current=run_id == self.config.current_run,
                lineage_id=self._lineage_id(run_id),
                schema_version=int(payload.get("schema_version", 1)),
                updated_at_utc=_optional_text(payload.get("updated_at_utc")),
                minute_index=int(payload.get("minute_index", 0)),
                data_state=_data_state(age),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return RunInfo(
                run_id=run_id,
                display_name=run_id,
                current=run_id == self.config.current_run,
                lineage_id=self._lineage_id(run_id),
                schema_version=-1,
                updated_at_utc=None,
                minute_index=0,
                data_state="invalid",
                detail=str(error),
            )

    def _read_summary(self, run_id: str) -> dict[str, Any]:
        path = self._summary_path(run_id)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._summary_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        payload = orjson.loads(path.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"performance summary must be an object: {path}")
        schema = int(payload.get("schema_version", -1))
        if schema not in (1, 2, 3):
            raise ValueError(f"unsupported performance schema {schema}: {path}")
        validated = (
            normalize_legacy_performance_summary(payload)
            if schema < 3
            else cast(dict[str, Any], payload)
        )
        self._summary_cache[run_id] = (signature, validated)
        return validated

    def _all_windows(
        self, run_id: str, payload: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        recent = [
            dict(window)
            for window in _sequence(payload.get("completed_windows"))
            if isinstance(window, Mapping)
        ]
        if int(payload.get("schema_version", 1)) < 2:
            return recent, []
        performance_dir = self._summary_path(run_id).parent
        manifest_path = performance_dir / "history_manifest.json"
        manifest_signature = (
            (manifest_path.stat().st_mtime_ns, manifest_path.stat().st_size)
            if manifest_path.is_file()
            else None
        )
        cached = self._history_cache.get(run_id)
        if cached is not None and cached.manifest_signature == manifest_signature:
            immutable = cached.windows
            warnings = list(cached.warnings)
        else:
            entry = self._load_immutable_history(
                performance_dir,
                manifest_signature=manifest_signature,
                cached=cached,
            )
            self._history_cache[run_id] = entry
            immutable = entry.windows
            warnings = list(entry.warnings)
        by_index = {
            int(window.get("minute_index") or 0): window
            for window in (*immutable, *recent)
            if int(window.get("minute_index") or 0) > 0
        }
        return [by_index[index] for index in sorted(by_index)], warnings

    def _load_immutable_history(
        self,
        performance_dir: Path,
        *,
        manifest_signature: _PathSignature | None,
        cached: _HistoryCacheEntry | None,
    ) -> _HistoryCacheEntry:
        store = PerformanceHistoryStore(
            performance_dir / "history",
            performance_dir / "history_manifest.json",
        )
        warnings: list[str] = []
        part_windows: dict[
            _HistoryPartSignature,
            tuple[dict[str, Any], ...],
        ] = {}
        try:
            manifest = store.read_manifest()
            raw_parts = manifest.get("files", [])
            if not isinstance(raw_parts, list):
                raise ValueError("performance history manifest files must be a list")
            previous = {} if cached is None else cached.part_windows
            for raw_part in raw_parts:
                if not isinstance(raw_part, Mapping):
                    raise ValueError(
                        "performance history manifest entries must be objects"
                    )
                signature = _history_part_signature(raw_part)
                part_windows[signature] = previous.get(signature) or tuple(
                    store.load_windows_from_parts((raw_part,))
                )
            by_index = {
                int(window.get("minute_index") or 0): window
                for windows in part_windows.values()
                for window in windows
                if int(window.get("minute_index") or 0) > 0
            }
            immutable = [by_index[index] for index in sorted(by_index)]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            immutable = []
            part_windows = {}
            warnings.append(str(error))
        return _HistoryCacheEntry(
            manifest_signature=manifest_signature,
            part_windows=part_windows,
            windows=immutable,
            warnings=warnings,
        )

    def _history_warnings(self, run_id: str) -> list[str]:
        """Validate cheap manifest metadata without decoding full history."""
        performance_dir = self._summary_path(run_id).parent
        manifest_path = performance_dir / "history_manifest.json"
        signature = _path_signature(manifest_path)
        cached = self._history_cache.get(run_id)
        if cached is not None and cached.manifest_signature == signature:
            return list(cached.warnings)
        store = PerformanceHistoryStore(
            performance_dir / "history",
            manifest_path,
        )
        try:
            manifest = store.read_manifest()
            raw_parts = manifest.get("files", [])
            if not isinstance(raw_parts, list):
                raise ValueError("performance history manifest files must be a list")
            for raw_part in raw_parts:
                if not isinstance(raw_part, Mapping):
                    raise ValueError(
                        "performance history manifest entries must be objects"
                    )
                relative_path, size_bytes, _digest = _history_part_signature(raw_part)
                path = manifest_path.parent.parent / relative_path
                if not path.is_file():
                    raise FileNotFoundError(f"missing performance shard: {path}")
                if path.stat().st_size != size_bytes:
                    raise ValueError(f"performance shard size mismatch: {path}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return [str(error)]
        return []

    def _scope_segments(
        self, run_id: str, scope: ScopeName, window: WindowName
    ) -> tuple[str, ...]:
        if scope == "segment" or window != "cumulative":
            return (run_id,)
        lineage_id = self._lineage_id(run_id)
        if lineage_id is None:
            return (run_id,)
        return tuple(
            segment
            for segment in self.config.lineages[lineage_id].segments
            if self._summary_path(segment).is_file()
        )

    def _lineage_id(self, run_id: str) -> str | None:
        for name, lineage in self.config.lineages.items():
            if run_id in lineage.segments:
                return name
        return None

    def _deck_name(self, label: str) -> str:
        exact_alias = self.config.deck_aliases.get(label)
        if exact_alias is not None:
            return exact_alias
        deck_hash = _exact_deck_hash(label)
        if deck_hash is not None:
            for alias_key in (f"main_{deck_hash}", deck_hash):
                alias = self.config.deck_aliases.get(alias_key)
                if alias is not None:
                    return alias
            catalog_name = self._deck_catalog.get(deck_hash)
            if catalog_name is not None:
                return catalog_name
        return _fallback_deck_name(label)

    def _opponent_name(self, *, deck_label: str, opponent_id: str) -> str:
        """Prefer a pilot alias when multiple public agents share one deck."""
        pilot_alias = self.config.deck_aliases.get(opponent_id)
        if pilot_alias is not None:
            return pilot_alias
        return self._deck_name(deck_label or opponent_id)

    def _load_deck_catalogs(self) -> dict[str, str]:
        catalog: dict[str, str] = {}
        for configured_path in self.config.deck_catalogs:
            path = (self.repo_root / configured_path).resolve()
            if not path.is_relative_to(self.repo_root):
                raise ValueError(f"deck catalog escaped repository root: {path}")
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    deck_hash = str(row.get("deck_hash", "")).strip()
                    raw_name = str(row.get("archetype", "")).strip()
                    if _exact_deck_hash(deck_hash) is None or not raw_name:
                        continue
                    catalog.setdefault(deck_hash, raw_name)
        return catalog

    def _run_dir(self, run_id: str) -> Path:
        path = (self.run_root / run_id).resolve()
        if path.parent != self.run_root:
            raise ValueError("run path escaped configured root")
        return path

    def _summary_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "performance" / "training_performance.json"

    def _status_marker_path(self, run_id: str) -> Path:
        run_dir = self._run_dir(run_id)
        learner_status = run_dir / "learner_status.json"
        if learner_status.is_file():
            return learner_status
        return run_dir / "weights" / "latest.json"


def _slice_path(prefix: str, opponent_slice: str) -> str:
    if opponent_slice in KNOWN_OPPONENT_STRATA:
        return f"{prefix}/stratum/{opponent_slice}"
    return f"{prefix}/{opponent_slice}"


def _status_only_performance_table(
    run_id: str,
    *,
    window: WindowName,
    scope: ScopeName,
) -> PerformanceTable:
    empty = {name: _outcome_stats(OutcomeCounts()) for name in _SLICES}
    return PerformanceTable(
        run_id=run_id,
        scope=scope,
        window=window,
        started_at_utc=None,
        ended_at_utc=None,
        source_segments=(run_id,),
        overall=empty,
        decks=(),
        exact_dimensions_available=False,
    )


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mtime_utc(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _selected_payload(
    payload: Mapping[str, Any], window: WindowName
) -> Mapping[str, Any] | None:
    if window == "cumulative":
        cumulative = _mapping(payload.get("cumulative"))
        return {
            **cumulative,
            "started_at_utc": payload.get("started_at_utc"),
            "ended_at_utc": payload.get("updated_at_utc"),
        }
    windows = [
        item
        for item in _sequence(payload.get("completed_windows"))
        if isinstance(item, Mapping)
    ]
    selected = _select_windows(windows, payload, window)
    if not selected:
        return None
    counts = _empty_slice_counts()
    for minute in selected:
        _merge_slices(counts, minute.get("slices"))
    return {
        "started_at_utc": selected[0].get("started_at_utc"),
        "ended_at_utc": selected[-1].get("ended_at_utc"),
        "slices": {key: value.as_dict() for key, value in counts.items()},
    }


def _select_windows(
    windows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    window: WindowName,
) -> list[Mapping[str, Any]]:
    if window == "cumulative":
        return list(windows)
    interval = float(_semantics(payload).get("interval_seconds", 60.0))
    minutes = 15 if window == "15m" else 60
    count = max(1, round(minutes * 60.0 / interval))
    return list(windows[-count:])


def _recent_windows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the summary-resident bounded history without immutable I/O."""
    return [
        item
        for item in _sequence(payload.get("completed_windows"))
        if isinstance(item, Mapping)
    ]


def _merge_slices(destination: dict[str, OutcomeCounts], raw_slices: Any) -> None:
    if not isinstance(raw_slices, Mapping):
        return
    for key, raw_counts in raw_slices.items():
        if isinstance(raw_counts, Mapping):
            destination.setdefault(str(key), OutcomeCounts()).merge(
                OutcomeCounts.from_mapping(raw_counts)
            )


def _empty_slice_counts() -> dict[str, OutcomeCounts]:
    return {}


def _outcome_stats(counts: OutcomeCounts) -> OutcomeStats:
    return OutcomeStats(
        games=counts.games,
        wins=counts.wins,
        draws=counts.draws,
        losses=counts.losses,
        win_rate=(counts.wins / counts.games if counts.games else None),
        score_rate=counts.score,
    )


def _rolling_series(
    windows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    required: int,
) -> tuple[SeriesPoint, ...]:
    points: list[SeriesPoint] = []
    for index in range(len(windows)):
        selected = windows[max(0, index + 1 - required) : index + 1]
        counts = OutcomeCounts()
        for window in selected:
            raw = _mapping(_mapping(window.get("slices")).get(key))
            counts.merge(OutcomeCounts.from_mapping(raw))
        points.append(
            SeriesPoint(
                minute_index=int(windows[index].get("minute_index", index + 1)),
                ended_at_utc=str(windows[index].get("ended_at_utc", "")),
                games=counts.games,
                win_rate=(counts.wins / counts.games if counts.games else None),
                score_rate=counts.score,
            )
        )
    return tuple(points)


def _target_labels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _semantics(payload).get("target_deck_labels", [])
    return tuple(str(item) for item in _sequence(raw))


def _deck_hash(label: str) -> str:
    return _exact_deck_hash(label) or label


def _exact_deck_hash(label: str) -> str | None:
    return label_hash(label)


def _fallback_deck_name(label: str) -> str:
    prefixes = ("opponent_", "main_", "aux_", "tail_", "sparring_")
    cleaned = label
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.replace("_", " ").strip().title() or "Unknown deck"


def _semantics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("semantics"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _path_signature(path: Path) -> _PathSignature | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _history_part_signature(part: Mapping[str, Any]) -> _HistoryPartSignature:
    relative = str(part.get("relative_path", ""))
    size = int(part.get("size_bytes", -1))
    digest = str(part.get("sha256", ""))
    if not relative or size < 0 or not digest:
        raise ValueError("invalid performance history manifest identity")
    return relative, size, digest


def _data_state(
    age_seconds: float | None,
) -> Literal["ready", "stale", "invalid"]:
    if age_seconds is None:
        return "invalid"
    return "ready" if age_seconds <= 120.0 else "stale"
