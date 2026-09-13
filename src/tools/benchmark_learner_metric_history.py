"""Measure the conservative checkpoint-history overhead bound.

This isolated benchmark does not replace a paired training throughput run. It
does verify that even charging serialization synchronously to the learner would
stay below the configured fraction of one observed training window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_rl.rl.learner_metric_history import (
    LearnerMetricHistory,
    LearnerMetricRecord,
    NonBlockingLearnerMetricWriter,
)
from ptcg_rl.rl.performance_state import atomic_write_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def main() -> None:
    """Run the non-mutating-to-training overhead probe and publish evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--maximum-overhead-fraction", type=float, default=0.01)
    args = parser.parse_args()
    if args.iterations < 8:
        raise ValueError("iterations must be at least 8")
    if not 0.0 < args.maximum_overhead_fraction < 1.0:
        raise ValueError("maximum overhead fraction must be in (0, 1)")
    run_dir = _repo_path(args.run_dir)
    output = _repo_path(args.output)
    status_path = run_dir / "learner_status.json"
    status = _read_json(status_path)
    timing = _mapping(status.get("latest_timing"))
    reference_seconds = float(timing.get("total_seconds", 0.0))
    if reference_seconds <= 0.0:
        raise ValueError("run has no positive latest_timing.total_seconds")

    scratch_root = _REPO_ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=scratch_root,
        prefix="learner-metric-benchmark-",
    ) as raw_scratch:
        scratch = Path(raw_scratch)
        enqueue_seconds = _benchmark_enqueue(
            scratch / "async",
            iterations=args.iterations,
        )
        commit_seconds = _benchmark_commit(
            scratch / "direct",
            iterations=args.iterations,
        )

    enqueue_p99 = float(np.quantile(enqueue_seconds, 0.99))
    commit_p95 = float(np.quantile(commit_seconds, 0.95))
    conservative_fraction = (enqueue_p99 + commit_p95) / reference_seconds
    latest_path = run_dir / "weights" / "latest.json"
    report = {
        "format": "learner-metric-history-overhead-probe-v1",
        "observed_at_utc": _utc_now(),
        "reference": {
            "run_dir": str(run_dir.relative_to(_REPO_ROOT)),
            "learner_status_sha256": _sha256(status_path),
            "latest_manifest_sha256": (
                _sha256(latest_path) if latest_path.is_file() else None
            ),
            "window_total_seconds": reference_seconds,
        },
        "iterations": args.iterations,
        "enqueue_seconds": _distribution(enqueue_seconds),
        "commit_seconds": _distribution(commit_seconds),
        "conservative_accounting": (
            "enqueue p99 + direct Parquet commit p95 charged synchronously "
            "against one observed learner window"
        ),
        "conservative_overhead_fraction": conservative_fraction,
        "maximum_overhead_fraction": args.maximum_overhead_fraction,
        "isolated_gate_passed": conservative_fraction <= args.maximum_overhead_fraction,
        "paired_training_gate_status": "not_run",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _benchmark_enqueue(run_dir: Path, *, iterations: int) -> list[float]:
    writer = NonBlockingLearnerMetricWriter(
        run_dir,
        queue_size=iterations + 1,
    )
    measurements: list[float] = []
    for version in range(1, iterations + 1):
        started = time.perf_counter()
        accepted = writer.publish(_record(version))
        measurements.append(time.perf_counter() - started)
        if not accepted:
            raise RuntimeError(
                "oversized benchmark queue unexpectedly dropped a record"
            )
    writer.close()
    if writer.status.dropped_records or writer.status.last_error:
        raise RuntimeError(f"asynchronous writer failed: {writer.status}")
    return measurements


def _benchmark_commit(run_dir: Path, *, iterations: int) -> list[float]:
    history = LearnerMetricHistory(run_dir)
    measurements: list[float] = []
    for version in range(1, iterations + 1):
        started = time.perf_counter()
        history.commit(_record(version))
        measurements.append(time.perf_counter() - started)
    return measurements


def _record(version: int) -> LearnerMetricRecord:
    return LearnerMetricRecord(
        recorded_at_utc="2026-01-01T00:00:00Z",
        run_version="benchmark",
        update_index=version - 1,
        optimizer_step_index=version,
        checkpoint_version=version,
        pair_manifest_sha256=_SHA_A,
        policy_sha256=_SHA_B,
        learner_state_sha256=_SHA_C,
        decisions=1,
        fragments_seen=1,
        fragments_stale=0,
        loss=0.1,
        policy_loss=0.01,
        value_loss=0.2,
        belief_loss=0.03,
        entropy=0.7,
        ratio_mean=1.0,
        approximate_kl=0.01,
        clip_fraction=0.1,
        gradient_norm=0.5,
        learning_rate=3e-4,
        kept_decisions_per_second=1.0,
        collection_seconds=1.0,
        learner_seconds=1.0,
        checkpoint_seconds=1.0,
        total_seconds=3.0,
    )


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": max(values),
    }


def _repo_path(value: Path) -> Path:
    path = value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()
    if not path.is_relative_to(_REPO_ROOT):
        raise ValueError("benchmark paths must stay inside the repository")
    return path


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
