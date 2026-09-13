"""Final checkpoint export and CPU runtime diagnostics for verification."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.deck_conditioning_audit_probes import latency_summary
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.submission.checkpoint_assets import (
    RuntimeCheckpointExportConfig,
    export_runtime_checkpoint,
)


def export_and_benchmark_runtime(
    raw_target: Path,
    *,
    release_deck: Path,
    output_dir: Path,
    iterations: int,
) -> dict[str, Any]:
    """Export FP16 full registry bytes and measure deployment CPU paths."""
    runtime_checkpoint = output_dir / "migrated_v1_full_fp16.pt"
    export_summary = export_runtime_checkpoint(
        RuntimeCheckpointExportConfig(
            source_checkpoint=raw_target,
            output_checkpoint=runtime_checkpoint,
            precision="fp16",
            deck_path=release_deck,
        )
    )
    return {
        "raw_path": records.display_path(raw_target),
        "raw_sha256": file_sha256(raw_target),
        "runtime_path": records.display_path(runtime_checkpoint),
        "runtime_sha256": file_sha256(runtime_checkpoint),
        "export": dict(export_summary),
        "cpu_benchmark": _runtime_benchmark(
            runtime_checkpoint,
            deck_path=release_deck,
            iterations=iterations,
        ),
    }


def _runtime_benchmark(
    checkpoint_path: Path,
    *,
    deck_path: Path,
    iterations: int,
) -> dict[str, Any]:
    deck = records.read_deck(deck_path)
    load_started = time.perf_counter()
    policy = CheckpointPolicy(checkpoint_path, device="cpu", own_deck=deck)
    load_seconds = time.perf_counter() - load_started
    prewarm_started = time.perf_counter()
    policy.prewarm()
    prewarm_seconds = time.perf_counter() - prewarm_started
    observation = _runtime_observation()
    policy.configure_inference_cache(enabled=False)
    uncached = _policy_latencies(policy, observation, iterations=iterations)
    policy.configure_inference_cache(enabled=True)
    policy.select_action(observation)
    cached = _policy_latencies(policy, observation, iterations=iterations)
    return {
        "load_seconds": load_seconds,
        "prewarm_seconds": prewarm_seconds,
        "uncached": latency_summary(uncached),
        "cached": latency_summary(cached),
        "selected_private_profile_module_key": (
            policy.selected_private_profile_module_key
        ),
        "packaged_private_profile_count": policy.packaged_private_profile_count,
    }


def _policy_latencies(
    policy: CheckpointPolicy,
    observation: Mapping[str, Any],
    *,
    iterations: int,
) -> list[float]:
    latencies = []
    for _ in range(iterations):
        started = time.perf_counter()
        policy.select_action(observation)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def _runtime_observation() -> dict[str, Any]:
    return {
        "current": {
            "yourIndex": 0,
            "result": -1,
            "players": [{"prize": [None] * 6}, {"prize": [None] * 6}],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}, {"type": 2}],
        },
    }


__all__ = ["export_and_benchmark_runtime"]
