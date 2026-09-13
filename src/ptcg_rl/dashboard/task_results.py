"""Progress projection and bounded result access for dashboard tasks."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.dashboard.deck_selection import build_deck_selection_payload
from ptcg_rl.dashboard.deck_selection_models import DeckSelectionPayload
from ptcg_rl.dashboard.task_models import (
    TaskArtifact,
    TaskArtifactContent,
    TaskArtifactList,
    TaskLogPayload,
    TaskProgress,
    TaskReceipt,
    TaskResultSummary,
    TaskTablePayload,
)

_ARTIFACT_SUFFIXES = (
    ".json",
    ".md",
    ".parquet",
    ".tar.gz",
)
_PREVIEW_SUFFIXES = (".json", ".md")
_PUBLIC_ARTIFACT_NAMES = frozenset(
    {
        "bundle_standings.parquet",
        "raw_bundle_standings.parquet",
        "raw_pilot_standings.parquet",
        "raw_deck_standings.parquet",
        "deck_standings.parquet",
        "cells.parquet",
        "meta.parquet",
        "contrasts.parquet",
        "games.parquet",
        "matchups.parquet",
        "standings_uniform.parquet",
        "standings_recent_weighted.parquet",
        "summary.json",
        "report.md",
        "score.json",
        "status.json",
        "progress.json",
        "fingerprints.json",
        "environment.json",
        "resolved_config.json",
    }
)
_TABLES = {
    "bundle_strength": {
        "raw_bundle_standings": "strength/raw_bundle_standings.parquet",
        "raw_pilot_standings": "strength/raw_pilot_standings.parquet",
        "raw_deck_standings": "strength/raw_deck_standings.parquet",
        "bundle_standings": "strength/bundle_standings.parquet",
        "deck_standings": "strength/deck_standings.parquet",
        "cells": "strength/cells.parquet",
        "meta": "strength/meta.parquet",
        "contrasts": "strength/contrasts.parquet",
        "games": "bundle/games.parquet",
    },
    "release_h2h": {"games": "games.parquet"},
    "runtime_elo": {
        "standings_uniform": "standings_uniform.parquet",
        "standings_recent_weighted": "standings_recent_weighted.parquet",
        "matchups": "matchups.parquet",
        "games": "games.parquet",
    },
}


class TaskResultService:
    """Expose task-owned logs, summaries, tables, and artifacts."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self._hash_cache: dict[tuple[str, int, int], str] = {}

    def progress(self, receipt: TaskReceipt) -> TaskProgress:
        """Normalize tool-specific durable status fields."""
        if receipt.state == "queued":
            return TaskProgress(
                phase="queued",
                detail=receipt.queue_reason or "等待资源门禁放行",
            )
        if receipt.state == "starting":
            return TaskProgress(phase="starting", detail="任务 worker 正在启动")
        payload = self._status_payload(receipt)
        if not payload:
            return TaskProgress(
                phase=receipt.state,
                detail=receipt.detail,
            )
        phase = str(payload.get("status") or payload.get("phase") or receipt.state)
        total = _optional_int(payload.get("games_total", payload.get("total_games")))
        committed = _optional_int(
            payload.get("games_committed", payload.get("completed_games"))
        )
        finished = _optional_int(
            payload.get("games_finished", payload.get("completed_games"))
        )
        percent = _optional_float(
            payload.get("progress_percent", payload.get("percent"))
        )
        if percent is None and total and committed is not None:
            percent = 100.0 * committed / total
        rate = _optional_float(
            payload.get(
                "session_games_per_second",
                payload.get("games_per_second"),
            )
        )
        eta = _optional_float(payload.get("eta_seconds"))
        if eta is None and total and committed is not None and rate and rate > 0:
            eta = max(0, total - committed) / rate
        warnings = payload.get("quality_warnings", ())
        return TaskProgress(
            phase=phase,
            games_total=total,
            games_committed=committed,
            games_finished=finished,
            percent=None if percent is None else max(0.0, min(100.0, percent)),
            rate_per_second=rate,
            eta_seconds=eta,
            quality_warnings=(
                tuple(str(item) for item in warnings)
                if isinstance(warnings, list)
                else ()
            ),
            detail=str(payload.get("error_message") or receipt.detail or "") or None,
        )

    def summary(self, receipt: TaskReceipt) -> TaskResultSummary:
        """Build a workflow-aware compact result header."""
        output = self._output_path(receipt)
        warnings: tuple[str, ...] = ()
        metrics: dict[str, Any] = {}
        headline = _state_headline(receipt)
        semantics = {
            "bundle_strength": (
                "deployment_bundle_posterior_v1"
                if receipt.mode == "formal"
                else "diagnostic_checkpoint_bundle_posterior_v1"
            ),
            "release_h2h": "deployment_native_release_h2h_v1",
            "runtime_elo": "diagnostic_order_dependent_runtime_elo_v1",
            "package_validation": "local_package_validation_no_upload_v1",
            "config_dry_run": "training_config_resolution_only_v1",
        }[receipt.kind]
        summary_path = self._summary_path(receipt, output)
        summary = _read_json_object(summary_path)
        if summary:
            raw_warnings = summary.get("quality_warnings", ())
            if isinstance(raw_warnings, list):
                warnings = tuple(str(item) for item in raw_warnings)
            metrics = _small_metrics(receipt.kind, summary)
            headline = _result_headline(receipt.kind, summary, fallback=headline)
        tables = tuple(
            name
            for name, relative in _TABLES.get(receipt.kind, {}).items()
            if output is not None and (output / relative).is_file()
        )
        report_path = (
            None
            if output is None or output.is_file()
            else next(
                (
                    path
                    for path in (
                        output / "strength" / "report.md",
                        output / "report.md",
                    )
                    if path.is_file()
                ),
                None,
            )
        )
        report = (
            None
            if report_path is None
            else self._artifact(receipt, report_path).artifact_id
        )
        return TaskResultSummary(
            task_id=receipt.task_id,
            kind=receipt.kind,
            state=receipt.state,
            semantics=semantics,
            headline=headline,
            metrics=metrics,
            quality_warnings=warnings,
            tables=tables,
            report_artifact_id=report,
        )

    def deck_selection(self, receipt: TaskReceipt) -> DeckSelectionPayload:
        """Project one runtime ladder into equal-opponent decision evidence."""
        if receipt.kind != "runtime_elo":
            raise ValueError("deck selection requires a runtime Elo task")
        spec_path = (self.repo_root / receipt.spec_path).resolve()
        expected_spec_path = (
            self.repo_root
            / "outputs"
            / "dashboard"
            / "tasks"
            / receipt.task_id
            / "spec.json"
        ).resolve()
        if spec_path != expected_spec_path:
            raise ValueError("task spec is not owned by its receipt")
        spec = _read_json_object(spec_path)
        if not spec:
            raise KeyError("task spec is not available")
        output = self._output_path(receipt)
        summary_path = self._summary_path(receipt, output)
        summary = _read_json_object(summary_path)
        matchup_rows: list[dict[str, Any]] = []
        result_hashes: list[str] = []
        if summary_path is not None and summary_path.is_file():
            self._require_owned(receipt, summary_path.resolve())
            result_hashes.append(_file_sha256(summary_path))
        if output is not None:
            matchups_path = (output / "matchups.parquet").resolve()
            self._require_owned(receipt, matchups_path)
            if matchups_path.is_file():
                result_hashes.append(_file_sha256(matchups_path))
                parquet = pq.ParquetFile(matchups_path)
                for batch in parquet.iter_batches(batch_size=1024):
                    matchup_rows.extend(
                        {str(key): _json_value(value) for key, value in raw.items()}
                        for raw in batch.to_pylist()
                    )
        result_fingerprint = (
            None
            if not result_hashes
            else hashlib.sha256("\n".join(result_hashes).encode("utf-8")).hexdigest()
        )
        return build_deck_selection_payload(
            receipt,
            spec=spec,
            summary=summary,
            matchup_rows=matchup_rows,
            result_fingerprint=result_fingerprint,
        )

    def table(
        self,
        receipt: TaskReceipt,
        table: str,
        *,
        offset: int,
        limit: int,
        filters: Mapping[str, str] | None = None,
    ) -> TaskTablePayload:
        """Read at most one bounded Parquet page."""
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("table page requires offset >= 0 and 1 <= limit <= 200")
        relative = _TABLES.get(receipt.kind, {}).get(table)
        output = self._output_path(receipt)
        if relative is None or output is None:
            raise KeyError("unknown task result table")
        path = (output / relative).resolve()
        self._require_owned(receipt, path)
        if not path.is_file():
            raise KeyError("task result table is not available")
        parquet = pq.ParquetFile(path)
        columns = tuple(parquet.schema_arrow.names)
        active_filters = dict(filters or {})
        unknown_filters = set(active_filters) - set(columns)
        if unknown_filters:
            raise ValueError(f"unknown table filter columns: {sorted(unknown_filters)}")
        selected: list[dict[str, Any]] = []
        matched = 0
        for batch in parquet.iter_batches(batch_size=max(1024, limit)):
            for raw in batch.to_pylist():
                row = {str(key): _json_value(value) for key, value in raw.items()}
                if any(
                    str(row.get(key, "")) != expected
                    for key, expected in active_filters.items()
                ):
                    continue
                if matched < offset:
                    matched += 1
                    continue
                selected.append(row)
                matched += 1
                if len(selected) > limit:
                    break
            if len(selected) > limit:
                break
        returned = min(limit, len(selected))
        return TaskTablePayload(
            task_id=receipt.task_id,
            table=table,
            columns=columns,
            rows=tuple(selected[:limit]),
            offset=offset,
            limit=limit,
            returned=returned,
            has_more=len(selected) > limit,
        )

    def artifacts(self, receipt: TaskReceipt) -> TaskArtifactList:
        """Index only receipt-owned result files."""
        paths = self._artifact_paths(receipt)
        artifacts = tuple(self._artifact(receipt, path) for path in paths)
        return TaskArtifactList(task_id=receipt.task_id, artifacts=artifacts)

    def artifact_content(
        self,
        receipt: TaskReceipt,
        artifact_id: str,
        *,
        max_bytes: int,
    ) -> TaskArtifactContent:
        """Read a bounded textual artifact preview."""
        if not 1 <= max_bytes <= 1_000_000:
            raise ValueError("artifact preview must be between 1 and 1000000 bytes")
        path = self.artifact_path(receipt, artifact_id)
        if not path.name.endswith(_PREVIEW_SUFFIXES):
            raise ValueError("artifact is not text-previewable")
        size = path.stat().st_size
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        artifact = self._artifact(receipt, path)
        return TaskArtifactContent(
            task_id=receipt.task_id,
            artifact_id=artifact.artifact_id,
            media_type=artifact.media_type,
            text=payload[:max_bytes].decode("utf-8", errors="replace"),
            truncated=size > max_bytes,
        )

    def artifact_path(self, receipt: TaskReceipt, artifact_id: str) -> Path:
        """Resolve one artifact ID without accepting a filesystem path."""
        for path in self._artifact_paths(receipt):
            artifact = self._artifact(receipt, path)
            if artifact.artifact_id == artifact_id:
                return path
        raise KeyError("unknown task artifact")

    def log(self, receipt: TaskReceipt, *, tail_bytes: int) -> TaskLogPayload:
        """Read a bounded task log tail."""
        if not 1 <= tail_bytes <= 1_000_000:
            raise ValueError("task log tail must be between 1 and 1000000 bytes")
        path = (self.repo_root / receipt.log_path).resolve()
        self._require_repo(path)
        if not path.is_file():
            return TaskLogPayload(
                task_id=receipt.task_id,
                text="",
                truncated=False,
            )
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > tail_bytes:
                stream.seek(-tail_bytes, os.SEEK_END)
            payload = stream.read()
        return TaskLogPayload(
            task_id=receipt.task_id,
            text=payload.decode("utf-8", errors="replace"),
            truncated=size > tail_bytes,
        )

    def _status_payload(self, receipt: TaskReceipt) -> dict[str, Any]:
        if receipt.status_path is None:
            return {}
        path = (self.repo_root / receipt.status_path).resolve()
        self._require_owned(receipt, path)
        return _read_json_object(path)

    def _summary_path(
        self,
        receipt: TaskReceipt,
        output: Path | None,
    ) -> Path | None:
        if output is None or output.is_file():
            return None
        if receipt.kind == "bundle_strength":
            return output / "strength" / "summary.json"
        if receipt.kind in {"release_h2h", "runtime_elo"}:
            return output / "summary.json"
        return None

    def _output_path(self, receipt: TaskReceipt) -> Path | None:
        if receipt.output_dir is None:
            return None
        path = (self.repo_root / receipt.output_dir).resolve()
        self._require_repo(path)
        return path

    def _artifact_paths(self, receipt: TaskReceipt) -> tuple[Path, ...]:
        output = self._output_path(receipt)
        paths: set[Path] = set()
        if output is not None and output.is_file():
            paths.add(output)
        elif output is not None and output.is_dir():
            for path in output.rglob("*"):
                if (
                    path.is_file()
                    and path.name in _PUBLIC_ARTIFACT_NAMES
                    and path.name.endswith(_ARTIFACT_SUFFIXES)
                ):
                    paths.add(path.resolve())
        return tuple(sorted(paths, key=lambda path: path.as_posix()))

    def _artifact(self, receipt: TaskReceipt, path: Path) -> TaskArtifact:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        sha = self._hash_cache.get(key)
        if sha is None:
            sha = _file_sha256(path)
            self._hash_cache[key] = sha
        relative = path.relative_to(self.repo_root).as_posix()
        media_type = (
            "application/x-parquet"
            if path.suffix == ".parquet"
            else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        return TaskArtifact(
            artifact_id=hashlib.sha256(relative.encode("utf-8")).hexdigest(),
            label=relative,
            media_type=media_type,
            size_bytes=stat.st_size,
            sha256=sha,
            previewable=path.name.endswith(_PREVIEW_SUFFIXES),
            downloadable=True,
        )

    def _require_owned(self, receipt: TaskReceipt, path: Path) -> None:
        self._require_repo(path)
        output = self._output_path(receipt)
        if output is None:
            raise ValueError("task has no owned output path")
        if output.is_file() and path != output:
            raise ValueError("artifact is outside task-owned output")
        if not output.is_file() and not path.is_relative_to(output):
            raise ValueError("artifact is outside task-owned output")

    def _require_repo(self, path: Path) -> None:
        if not path.is_relative_to(self.repo_root):
            raise ValueError("task result path escapes repository")


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file() or path.stat().st_size > 20_000_000:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _state_headline(receipt: TaskReceipt) -> str:
    if receipt.state == "queued":
        return receipt.queue_reason or "任务正在等待资源"
    if receipt.state in {"starting", "running", "cancelling"}:
        return "任务执行中，结果将在 artifact 原子发布后出现"
    if receipt.state == "succeeded":
        return "任务已成功完成"
    return receipt.detail or f"任务状态：{receipt.state}"


def _small_metrics(kind: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "bundle_strength":
        games = summary.get("games")
        return {
            "games": games,
            "effective_meta_coverage": summary.get("effective_meta_coverage"),
            "unknown_meta_mass": summary.get("unknown_meta_mass"),
            "bundle_count": summary.get("bundle_count"),
        }
    if kind == "release_h2h":
        score = summary.get("score")
        return {"games": summary.get("games"), "score": score}
    if kind == "runtime_elo":
        return {
            "games": summary.get("games"),
            "decks": summary.get("decks"),
            "matchups": summary.get("matchups"),
            "agent_error_games": summary.get("agent_error_games"),
        }
    return {}


def _result_headline(
    kind: str,
    summary: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    if kind == "runtime_elo":
        standings = summary.get("standings_uniform")
        if isinstance(standings, list) and standings and isinstance(standings[0], dict):
            row = standings[0]
            return f"诊断 Elo 第一名：{row.get('deck_label', 'unknown')}"
    if kind == "release_h2h":
        score = summary.get("score")
        if isinstance(score, dict):
            decision = score.get("decision")
            if decision:
                return f"Release H2H 结论：{decision}"
    if kind == "bundle_strength":
        return "Bundle 后验评分已完成；请以 standings 与质量门禁共同决策"
    return fallback


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if output >= 0.0 else None


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, pa.Scalar):
        return _json_value(value.as_py())
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
