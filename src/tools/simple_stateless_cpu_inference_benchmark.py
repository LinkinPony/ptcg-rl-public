"""Benchmark current simple-stateless rollout inference on CPU.

The ``prepare`` command converts learner-side compact fragments into a small
public-model-input bundle.  Only that bundle and the immutable policy
checkpoint need to be copied to a benchmark host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.model.simple_stateless import (
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
)
from ptcg_rl.rl.native_policy_batch import NativeSimpleStatelessPolicyBatch
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.stateless_array_collation import (
    collate_stateless_array_microbatch,
)
from ptcg_rl.rl.stateless_array_replay import (
    StatelessArrayOptimizerWindow,
    prepare_stateless_array_optimizer_window,
)
from ptcg_rl.rl.stateless_checkpoint import StatelessPolicyIdentity
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import load_compact_fragment_part

_INPUT_FORMAT = "simple_stateless_cpu_inference_input_v1"
_REPORT_FORMAT = "simple_stateless_cpu_inference_benchmark_v1"


def main() -> None:
    """Prepare safe model inputs or benchmark an already prepared bundle."""
    arguments = _arguments()
    if arguments.command == "prepare":
        _prepare(arguments)
    elif arguments.command == "run":
        _run(arguments)
    else:  # pragma: no cover - argparse enforces the command.
        raise ValueError(f"unsupported command: {arguments.command}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument(
        "--fragment-part",
        action="append",
        type=Path,
        required=True,
    )
    prepare.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=(1, 4, 16, 32, 64, 82),
    )
    prepare.add_argument(
        "--selection",
        choices=("mixed", "homogeneous"),
        default="mixed",
    )
    prepare.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--input-bundle", type=Path, required=True)
    run.add_argument("--batch-sizes", nargs="+", type=int)
    run.add_argument("--torch-threads", type=int, required=True)
    run.add_argument("--warmup-iterations", type=int, default=2)
    run.add_argument("--minimum-measure-seconds", type=float, default=5.0)
    run.add_argument("--maximum-iterations", type=int, default=100)
    run.add_argument("--seed", type=int, default=20260724)
    run.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "prepare":
        if any(size <= 0 for size in arguments.batch_sizes):
            parser.error("batch sizes must be positive")
    else:
        if arguments.torch_threads <= 0:
            parser.error("--torch-threads must be positive")
        if arguments.batch_sizes is not None and any(
            size <= 0 for size in arguments.batch_sizes
        ):
            parser.error("batch sizes must be positive")
        if arguments.warmup_iterations < 0:
            parser.error("--warmup-iterations cannot be negative")
        if arguments.minimum_measure_seconds <= 0.0:
            parser.error("--minimum-measure-seconds must be positive")
        if arguments.maximum_iterations <= 0:
            parser.error("--maximum-iterations must be positive")
    return arguments


def _prepare(arguments: argparse.Namespace) -> None:
    checkpoint = _checkpoint_payload(arguments.checkpoint)
    model_config = SimpleStatelessModelConfig.model_validate(
        checkpoint["model_config"]
    )
    model = SimpleStatelessPolicyValueNet(
        model_config,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(_model_state(checkpoint), strict=True)
    card_vocab_size = (
        model.backbone.input_encoder.card_encoder.num_card_ids
    )
    parts = tuple(
        load_compact_fragment_part(path) for path in arguments.fragment_part
    )
    version = int(checkpoint["version"])
    window = prepare_stateless_array_optimizer_window(
        parts,
        current_policy_version=version,
        maximum_version_age=0,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_epsilon=1.0e-8,
    )
    maximum_size = max(arguments.batch_sizes)
    indices = _selected_indices(
        window,
        size=maximum_size,
        selection=arguments.selection,
    )
    batches: dict[int, NativeSimpleStatelessPolicyBatch] = {}
    collation_seconds: dict[int, float] = {}
    for size in sorted(set(arguments.batch_sizes)):
        started_at = time.perf_counter()
        microbatch = collate_stateless_array_microbatch(
            window,
            indices[:size],
            card_vocab_size=card_vocab_size,
            device="cpu",
        )
        batches[size] = _policy_batch(microbatch)
        collation_seconds[size] = time.perf_counter() - started_at

    payload = {
        "format": _INPUT_FORMAT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "checkpoint_model_fingerprint": str(
            checkpoint["model_fingerprint"]
        ),
        "checkpoint_version": version,
        "selection": str(arguments.selection),
        "fragment_parts": tuple(
            {
                "name": path.name,
                "sha256": _sha256(path),
            }
            for path in arguments.fragment_part
        ),
        "decision_pool_size": window.decision_count,
        "decision_pool_routes": len(set(window.route_deck_digests)),
        "collation_seconds": collation_seconds,
        "batches": batches,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "batch_sizes": sorted(batches),
                "decision_pool_size": window.decision_count,
                "decision_pool_routes": len(set(window.route_deck_digests)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run(arguments: argparse.Namespace) -> None:
    torch.set_num_threads(arguments.torch_threads)
    torch.set_num_interop_threads(1)
    checkpoint_started_at = time.perf_counter()
    checkpoint = _checkpoint_payload(arguments.checkpoint)
    model_config = SimpleStatelessModelConfig.model_validate(
        checkpoint["model_config"]
    )
    model = SimpleStatelessPolicyValueNet(
        model_config,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(_model_state(checkpoint), strict=True)
    model.eval()
    policy_identity = StatelessPolicyIdentity.model_validate(
        checkpoint["identity"]
    )
    identity = _fragment_identity(
        checkpoint,
        policy_identity=policy_identity,
    )
    executor = NativePolicyInferenceExecutor(
        model,
        identity=identity,
        device="cpu",
        verify_model_state=False,
    )
    checkpoint_load_seconds = time.perf_counter() - checkpoint_started_at

    input_payload = torch.load(
        arguments.input_bundle,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(input_payload, Mapping):
        raise TypeError("CPU inference input bundle must be a mapping")
    if input_payload.get("format") != _INPUT_FORMAT:
        raise ValueError("CPU inference input bundle format is unsupported")
    if (
        input_payload.get("checkpoint_model_fingerprint")
        != checkpoint["model_fingerprint"]
    ):
        raise ValueError("CPU inference input and checkpoint fingerprints differ")
    raw_batches = input_payload.get("batches")
    if not isinstance(raw_batches, Mapping):
        raise TypeError("CPU inference input bundle has no batches")
    requested_sizes = (
        None
        if arguments.batch_sizes is None
        else set(arguments.batch_sizes)
    )

    results = []
    for raw_size, raw_batch in sorted(
        raw_batches.items(),
        key=lambda item: int(item[0]),
    ):
        if requested_sizes is not None and int(raw_size) not in requested_sizes:
            continue
        if not isinstance(raw_batch, NativeSimpleStatelessPolicyBatch):
            raise TypeError("CPU inference input contains an invalid policy batch")
        size = int(raw_size)
        if raw_batch.batch_size != size:
            raise ValueError("CPU inference input batch size is inconsistent")
        results.append(
            _benchmark_batch(
                executor,
                raw_batch,
                warmup_iterations=arguments.warmup_iterations,
                minimum_seconds=arguments.minimum_measure_seconds,
                maximum_iterations=arguments.maximum_iterations,
                seed=arguments.seed + size,
            )
        )
    if not results:
        raise ValueError("no requested CPU inference batch is in the input bundle")

    report = {
        "format": _REPORT_FORMAT,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "runtime_id": os.environ.get("PTCG_BENCHMARK_RUNTIME_ID"),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "checkpoint_model_fingerprint": str(
            checkpoint["model_fingerprint"]
        ),
        "checkpoint_version": int(checkpoint["version"]),
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_dtype": str(next(model.parameters()).dtype),
        "input_bundle": str(arguments.input_bundle),
        "input_bundle_sha256": _sha256(arguments.input_bundle),
        "selection": input_payload.get("selection"),
        "decision_pool_size": input_payload.get("decision_pool_size"),
        "decision_pool_routes": input_payload.get("decision_pool_routes"),
        "peak_rss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _benchmark_batch(
    executor: NativePolicyInferenceExecutor,
    batch: NativeSimpleStatelessPolicyBatch,
    *,
    warmup_iterations: int,
    minimum_seconds: float,
    maximum_iterations: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _ in range(warmup_iterations):
        executor.sample(batch, generator=generator)

    inference_seconds: list[float] = []
    compaction_seconds: list[float] = []
    actions = 0
    measured_started_at = time.perf_counter()
    while len(inference_seconds) < maximum_iterations:
        started_at = time.perf_counter()
        trace = executor.sample_device(batch, generator=generator)
        inference_seconds.append(time.perf_counter() - started_at)
        started_at = time.perf_counter()
        host_trace = trace.to_host()
        compaction_seconds.append(time.perf_counter() - started_at)
        actions = int(host_trace.action_choices.shape[0])
        if time.perf_counter() - measured_started_at >= minimum_seconds:
            break
    total_seconds = [
        inference + compaction
        for inference, compaction in zip(
            inference_seconds,
            compaction_seconds,
            strict=True,
        )
    ]
    _validate_trace(batch, host_trace)
    measured_seconds = sum(total_seconds)
    return {
        "batch_size": batch.batch_size,
        "route_count": len(set(batch.deck_signatures)),
        "mean_state_tokens": statistics.fmean(
            batch.states.sequence_lengths
        ),
        "mean_options": statistics.fmean(batch.options.option_lengths),
        "iterations": len(total_seconds),
        "measured_seconds": measured_seconds,
        "rows_per_second": (
            batch.batch_size * len(total_seconds) / measured_seconds
        ),
        "inference_seconds": _summary(inference_seconds),
        "host_compaction_seconds": _summary(compaction_seconds),
        "total_seconds": _summary(total_seconds),
        "last_action_values": actions,
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "maximum": ordered[-1],
    }


def _selected_indices(
    window: StatelessArrayOptimizerWindow,
    *,
    size: int,
    selection: str,
) -> tuple[int, ...]:
    by_route: dict[str, deque[int]] = defaultdict(deque)
    for index, digest in enumerate(window.route_deck_digests):
        by_route[digest].append(index)
    if selection == "homogeneous":
        route = max(by_route, key=lambda item: len(by_route[item]))
        indices = tuple(by_route[route])
        if len(indices) < size:
            raise ValueError("largest exact route cannot fill requested batch")
        return indices[:size]

    selected: list[int] = []
    routes = sorted(by_route)
    while len(selected) < size:
        progressed = False
        for route in routes:
            if by_route[route]:
                selected.append(by_route[route].popleft())
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise ValueError("decision pool cannot fill requested batch")
    return tuple(selected)


def _policy_batch(microbatch: Any) -> NativeSimpleStatelessPolicyBatch:
    return NativeSimpleStatelessPolicyBatch(
        states=microbatch.states,
        options=microbatch.options,
        unique_deck_card_ids=microbatch.unique_deck_card_ids,
        deck_counts=microbatch.deck_counts,
        deck_valid_mask=microbatch.deck_valid_mask,
        deck_signatures=microbatch.deck_signatures,
        belief_summary=microbatch.belief_summary,
        min_counts=tuple(
            int(value) for value in microbatch.options.min_counts.tolist()
        ),
        max_counts=tuple(
            int(value) for value in microbatch.options.max_counts.tolist()
        ),
        public_deck_catalog_fingerprint=(
            microbatch.public_deck_catalog_fingerprint
        ),
        input_contract_fingerprint=microbatch.input_contract_fingerprint,
    )


def _validate_trace(
    batch: NativeSimpleStatelessPolicyBatch,
    trace: Any,
) -> None:
    if trace.batch_size != batch.batch_size:
        raise RuntimeError("CPU inference trace batch size changed")
    lengths = trace.action_offsets[1:] - trace.action_offsets[:-1]
    for row, length in enumerate(lengths):
        if length < batch.min_counts[row] or length > batch.max_counts[row]:
            raise RuntimeError("CPU inference emitted an invalid action length")


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("format") != "simple_stateless_policy_v1":
        raise ValueError("unsupported simple-stateless checkpoint format")
    return payload


def _model_state(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise TypeError("checkpoint model state is not a tensor mapping")
    return state


def _fragment_identity(
    payload: Mapping[str, Any],
    *,
    policy_identity: StatelessPolicyIdentity,
) -> StatelessFragmentIdentity:
    return StatelessFragmentIdentity(
        horizon=64,
        behavior_policy_version=int(payload["version"]),
        behavior_policy_fingerprint=str(payload["model_fingerprint"]),
        model_config_fingerprint=policy_identity.model_config_fingerprint,
        action_schema_fingerprint=policy_identity.action_schema_fingerprint,
        public_context_fingerprint=policy_identity.public_context_fingerprint,
        card_catalog_fingerprint=policy_identity.card_catalog_fingerprint,
        public_deck_catalog_fingerprint=(
            policy_identity.public_deck_catalog_fingerprint
        ),
        exact_registry_fingerprint=policy_identity.exact_registry_fingerprint,
        belief_target_semantics_fingerprint=(
            policy_identity.belief_target_semantics_fingerprint
        ),
        input_contract_fingerprint=policy_identity.input_contract_fingerprint,
        resolved_config_fingerprint=policy_identity.resolved_config_fingerprint,
    )


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
