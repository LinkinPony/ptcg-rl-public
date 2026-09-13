"""Immutable identity and report helpers for teacher sibling campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingConfig

_RUNTIME_SOURCE_PATHS = (
    Path("src/ptcg_rl/evaluation/teacher_sibling.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_config.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_engine.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_io.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_metrics.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_report.py"),
    Path("src/ptcg_rl/evaluation/teacher_sibling_support.py"),
    Path("src/ptcg_rl/agent/runtime.py"),
    Path("src/ptcg_rl/agent/search/context.py"),
    Path("src/ptcg_rl/agent/search/macro.py"),
    Path("src/ptcg_rl/agent/search/scoring.py"),
    Path("src/ptcg_rl/belief/sampling.py"),
    Path("src/ptcg_rl/training/bc_dataset.py"),
)
_ENGINE_ASSET_PATHS = (
    Path("data/sample_submission/cg/api.py"),
    Path("data/sample_submission/cg/libcg.so"),
    Path("src/ptcg_rl/engine/effects.py"),
    Path("src/ptcg_rl/engine/forward_model.py"),
    Path("src/ptcg_rl/engine/session.py"),
)


def teacher_sibling_campaign_identity(
    config: TeacherSiblingConfig,
    *,
    sampling: Mapping[str, Any],
    initial_path: Path,
    trained_path: Path,
) -> dict[str, Any]:
    """Return a hash-ready identity covering data, weights, code, and engine."""
    return {
        "protocol": "PUBLIC-TEACHER-ENGINE-SIBLING-v1",
        "experiment_id": config.experiment_id,
        "teacher_manifest_sha256": sampling["manifest_sha256"],
        "initial_checkpoint": {
            "tag": config.initial_checkpoint.tag,
            "path": records.display_path(initial_path),
            "sha256": file_sha256(initial_path),
        },
        "trained_checkpoint": {
            "tag": config.trained_checkpoint.tag,
            "path": records.display_path(trained_path),
            "sha256": file_sha256(trained_path),
        },
        "resolved_config": config.model_dump(mode="json"),
        "runtime_sources": _path_hashes(_RUNTIME_SOURCE_PATHS),
        "engine_assets": _path_hashes(_ENGINE_ASSET_PATHS),
    }


def verified_checkpoint(path: Path, expected_sha256: str | None) -> Path:
    """Resolve a checkpoint and enforce its optional frozen digest."""
    resolved = records.repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"teacher sibling checkpoint not found: {resolved}")
    actual = file_sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {resolved}")
    return resolved


def require_absent_output(path: Path) -> None:
    """Refuse to mix a campaign with any existing output directory."""
    if path.exists():
        raise FileExistsError(f"teacher sibling output must be immutable: {path}")


def write_teacher_sibling_report(path: Path, summary: Mapping[str, Any]) -> None:
    """Write the compact human-readable sibling diagnostic report."""
    initial = summary["initial"]
    trained = summary["trained"]
    deltas = summary["deltas"]
    lines = [
        f"# Teacher engine siblings: {summary['experiment_id']}",
        "",
        f"- Campaign: `{summary['campaign_fp']}`",
        f"- Roots: {summary['processed_roots']}",
        f"- Engine-supervised pairs: {summary['comparable_pairs']}",
        "- Decision role: diagnostic only",
        f"- Warnings: {', '.join(summary['diagnostic_warnings']) or 'none'}",
        f"- Behavior kind: `{summary['behavior_kind']}`",
        "- PPO ratio eligible: false",
        "",
        "| Metric | Initial | BC-trained | Delta |",
        "|---|---:|---:|---:|",
        _metric_line(
            "engine pairwise accuracy",
            initial["engine_pairwise_accuracy"],
            trained["engine_pairwise_accuracy"],
            deltas["engine_pairwise_accuracy"],
        ),
        _metric_line(
            "engine top-1 regret",
            initial["engine_top1_regret"],
            trained["engine_top1_regret"],
            deltas["engine_top1_regret"],
        ),
        _metric_line(
            "teacher MRR",
            initial["teacher_mrr"],
            trained["teacher_mrr"],
            deltas["teacher_mrr"],
        ),
        _metric_line(
            "teacher pairwise accuracy",
            initial["teacher_pairwise_accuracy"],
            trained["teacher_pairwise_accuracy"],
            deltas["teacher_pairwise_accuracy"],
        ),
        "",
        "All effects, legal transitions, terminal results, and tactical labels are "
        "read from the bundled engine Search API. The persisted rows intentionally "
        "contain no base-policy behavior log-probability and are supervised-only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _path_hashes(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {
            "path": records.display_path(records.repo_path(path)),
            "sha256": file_sha256(records.repo_path(path)),
        }
        for path in paths
    ]


def _metric_line(name: str, initial: Any, trained: Any, delta: Any) -> str:
    return f"| {name} | {_format_metric(initial)} | {_format_metric(trained)} | {_format_metric(delta)} |"


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


__all__ = [
    "require_absent_output",
    "teacher_sibling_campaign_identity",
    "verified_checkpoint",
    "write_teacher_sibling_report",
]
