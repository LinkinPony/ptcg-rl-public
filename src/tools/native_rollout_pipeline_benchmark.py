"""Benchmark native arena through current-policy actions with real artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.engine.native_public_context import NativePublicContextTracker
from ptcg_rl.engine.native_rollout import NativeRolloutEncoder
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.engine.native_training_view import select_native_training_rows
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
)
from ptcg_rl.rl.native_policy_batch import (
    encode_native_simple_stateless_batch,
    move_native_simple_stateless_batch,
)
from ptcg_rl.rl.native_policy_context import select_native_public_context_rows
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_policy_trace import NativePolicyNumpyTrace
from ptcg_rl.rl.native_rollout_batch import (
    NativeRolloutSource,
    encode_native_rollout_sources,
)
from ptcg_rl.rl.stateless_checkpoint import StatelessPolicyIdentity
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity

_READY = 1


def main() -> None:
    """Run a real native mirror-policy loop and emit stage-bound evidence."""
    arguments = _arguments()
    device = torch.device(arguments.device)
    if device.type != "cuda":
        raise ValueError("this H200 rollout benchmark requires a CUDA device")
    catalog, _catalog_manifest = load_public_deck_catalog(
        arguments.catalog_manifest
    )
    payload = _checkpoint_payload(arguments.checkpoint)
    model_config = SimpleStatelessModelConfig.model_validate(
        payload["model_config"]
    )
    policy_identity = StatelessPolicyIdentity.model_validate(payload["identity"])
    if catalog.fingerprint != policy_identity.public_deck_catalog_fingerprint:
        raise ValueError("checkpoint and public catalog fingerprints differ")
    model = SimpleStatelessPolicyValueNet(
        model_config,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(_model_state(payload), strict=True)
    identity = _fragment_identity(
        payload,
        policy_identity=policy_identity,
        horizon=arguments.fragment_horizon,
    )
    executor = NativePolicyInferenceExecutor(
        model,
        identity=identity,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(arguments.policy_seed)

    deck_pair = _deck_pair(model_config, tuple(arguments.deck_indices))
    decks = np.broadcast_to(
        deck_pair,
        (arguments.lanes, 2, 60),
    ).copy()
    slots = np.arange(arguments.lanes, dtype=np.uint32)
    candidate_seats = (slots % 2).astype(np.int32)
    buffers = tuple(
        NativeTrainingOutputBuffer(
            slot_capacity=arguments.lanes,
            option_capacity=arguments.lanes * arguments.options_per_lane,
        )
        for _index in range(2)
    )
    stage_seconds = {
        "row_selection": 0.0,
        "feature_encoding_cpu": 0.0,
        "tensor_h2d": 0.0,
        "policy_inference_and_d2h": 0.0,
        "action_assembly": 0.0,
        "engine_step": 0.0,
        "public_context": 0.0,
        "cohort_reset": 0.0,
    }
    policy_rows = 0
    forced_rows = 0
    candidate_decisions = 0
    terminal_rows = 0
    cohort_resets = 0
    option_rows = 0
    log_rows = 0
    seed_cursor = arguments.engine_seed

    with ExitStack() as stack:
        lane = stack.enter_context(
            NativeTrainingLane(
                arguments.lanes,
                library_path=arguments.library,
            )
        )
        native_encoder: NativeRolloutEncoder | None = None
        python_tracker: NativePublicContextTracker | None = None
        if arguments.encoder == "native":
            native_encoder = stack.enter_context(
                NativeRolloutEncoder(
                    slot_capacity=arguments.lanes,
                    library=lane.library,
                    catalog=catalog,
                    input_contract_fingerprint=(
                        identity.input_contract_fingerprint
                    ),
                )
            )
        else:
            python_tracker = NativePublicContextTracker(
                slot_capacity=arguments.lanes,
                catalog=catalog,
                input_contract_fingerprint=(
                    identity.input_contract_fingerprint
                ),
            )
        view = lane.reset(
            decks,
            _seeds(seed_cursor, arguments.lanes),
            slots=slots,
            output=buffers[0],
        )
        seed_cursor += arguments.lanes
        _require_ready(view)
        if native_encoder is not None:
            native_encoder.consume_reset(view, decks)
            context = None
        else:
            if python_tracker is None:
                raise AssertionError("Python tracker was not initialized")
            context = python_tracker.consume_reset(view, decks)
        torch.cuda.synchronize(device)
        measured_at = 0.0
        for batch_index in range(arguments.warmup_batches + arguments.batches):
            measuring = batch_index >= arguments.warmup_batches
            if measuring and measured_at == 0.0:
                torch.cuda.synchronize(device)
                measured_at = time.perf_counter()

            started = time.perf_counter()
            option_counts = np.diff(
                view.option_offsets.astype(np.int64, copy=False)
            )
            forced_mask = (
                (option_counts == 1)
                & (view.select_min == 1)
                & (view.select_max == 1)
            )
            sampled_rows = np.flatnonzero(~forced_mask)
            if sampled_rows.size <= 0:
                raise RuntimeError("benchmark batch has no sampled policy rows")
            if native_encoder is None:
                if context is None:
                    raise AssertionError("Python public context is absent")
                selected_view = select_native_training_rows(
                    view,
                    sampled_rows,
                )
                selected_context = select_native_public_context_rows(
                    context,
                    sampled_rows,
                )
            else:
                selected_view = None
                selected_context = None
            _add_time(stage_seconds, "row_selection", started, measuring)

            started = time.perf_counter()
            if native_encoder is None:
                if selected_view is None or selected_context is None:
                    raise AssertionError("Python encoder inputs are absent")
                cpu_batch = encode_native_simple_stateless_batch(
                    selected_view,
                    selected_context,
                    device="cpu",
                )
            else:
                cpu_batch = encode_native_rollout_sources(
                    (
                        NativeRolloutSource.from_arrays(
                            native_encoder,
                            view.slots[sampled_rows],
                            view.select_player[sampled_rows],
                        ),
                    ),
                    pin_memory=True,
                )
            _add_time(stage_seconds, "feature_encoding_cpu", started, measuring)

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            device_batch = move_native_simple_stateless_batch(
                cpu_batch,
                device=device,
                non_blocking=native_encoder is not None,
            )
            torch.cuda.synchronize(device)
            _add_time(stage_seconds, "tensor_h2d", started, measuring)

            started = time.perf_counter()
            trace = executor.sample(
                device_batch,
                generator=generator,
            )
            _add_time(
                stage_seconds,
                "policy_inference_and_d2h",
                started,
                measuring,
            )

            started = time.perf_counter()
            action_offsets, action_choices = _merge_actions(
                view,
                sampled_rows=sampled_rows,
                forced_mask=forced_mask,
                trace=trace,
            )
            _add_time(stage_seconds, "action_assembly", started, measuring)

            next_buffer = buffers[(batch_index + 1) % 2]
            started = time.perf_counter()
            view = lane.step(
                slots,
                action_offsets,
                action_choices,
                output=next_buffer,
            )
            _require_clean(view)
            _add_time(stage_seconds, "engine_step", started, measuring)

            started = time.perf_counter()
            if native_encoder is not None:
                native_encoder.consume_step(view)
            else:
                if python_tracker is None:
                    raise AssertionError("Python tracker was not initialized")
                context = python_tracker.consume_step(view)
            _add_time(stage_seconds, "public_context", started, measuring)

            if measuring:
                policy_rows += int(sampled_rows.size)
                forced_rows += int(np.count_nonzero(forced_mask))
                option_rows += int(
                    option_counts[sampled_rows].sum(dtype=np.int64)
                )
                log_rows += view.log_count
                candidate_decisions += int(
                    np.count_nonzero(
                        view.select_player[sampled_rows]
                        == candidate_seats[view.slots[sampled_rows]]
                    )
                )
            finished = view.result >= 0
            if not bool(np.any(finished)):
                continue
            if measuring:
                terminal_rows += int(np.count_nonzero(finished))
                cohort_resets += 1
            started = time.perf_counter()
            view = lane.reset(
                decks,
                _seeds(seed_cursor, arguments.lanes),
                slots=slots,
                output=buffers[batch_index % 2],
            )
            seed_cursor += arguments.lanes
            _require_ready(view)
            if native_encoder is not None:
                native_encoder.consume_reset(view, decks)
            else:
                if python_tracker is None:
                    raise AssertionError("Python tracker was not initialized")
                context = python_tracker.consume_reset(view, decks)
            _add_time(stage_seconds, "cohort_reset", started, measuring)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - measured_at
        library_path = lane.library_path

    arena_rows = arguments.lanes * arguments.batches
    cpu_rollout_chain_stages = (
        "row_selection",
        "feature_encoding_cpu",
        "action_assembly",
        "engine_step",
        "public_context",
        "cohort_reset",
    )
    cpu_rollout_chain_seconds = sum(
        stage_seconds[name] for name in cpu_rollout_chain_stages
    )
    report = {
        "format": "native-rollout-pipeline-throughput-v2",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "encoder": arguments.encoder,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "checkpoint_model_fingerprint": identity.behavior_policy_fingerprint,
        "catalog_manifest": str(arguments.catalog_manifest.resolve()),
        "catalog_fingerprint": catalog.fingerprint,
        "library": str(library_path),
        "library_sha256": _sha256(library_path),
        "gpu": torch.cuda.get_device_name(device),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "lanes": arguments.lanes,
        "measured_batches": arguments.batches,
        "warmup_batches": arguments.warmup_batches,
        "arena_rows": arena_rows,
        "sampled_policy_rows": policy_rows,
        "forced_rows": forced_rows,
        "mirror_candidate_decisions": candidate_decisions,
        "terminal_rows_observed": terminal_rows,
        "cohort_resets": cohort_resets,
        "elapsed_seconds": elapsed,
        "arena_rows_per_second": arena_rows / elapsed,
        "cpu_rollout_chain_seconds": cpu_rollout_chain_seconds,
        "arena_rows_per_cpu_rollout_chain_second": (
            arena_rows / cpu_rollout_chain_seconds
        ),
        "sampled_policy_rows_per_second": policy_rows / elapsed,
        "mirror_candidate_decisions_per_second": (
            candidate_decisions / elapsed
        ),
        "mean_sampled_prompt_options": option_rows / float(policy_rows),
        "mean_next_public_log_rows": log_rows / float(arena_rows),
        "stage_seconds": stage_seconds,
        "measurement_scope": (
            "native source-engine step/forced-zero chain/public-state/log "
            "projection, selected rollout encoder public-history/catalog "
            "tracking and model-ready tensor packing, CPU-to-H200 "
            "transfer, BF16 one-pass current-policy sampling/root value, one "
            "packed D2H, action CSR assembly, and cohort reset; mirror candidate "
            "decisions are real non-forced rows for alternating assigned seats. "
            "The native encoder mode uses the production pinned-host "
            "encode_native_rollout_sources path and avoids Python CSR selection. "
            "Excludes trajectory publication, replay preparation, learner, "
            "historical/scripted opponents, and per-game terminal replacement"
        ),
    }
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--catalog-manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoder",
        choices=("native", "python"),
        default="native",
    )
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--options-per-lane", type=int, default=256)
    parser.add_argument("--fragment-horizon", type=int, default=64)
    parser.add_argument("--engine-seed", type=int, default=20260723)
    parser.add_argument("--policy-seed", type=int, default=20260723)
    parser.add_argument("--deck-indices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    for name in ("lanes", "batches", "options_per_lane", "fragment_horizon"):
        if int(getattr(arguments, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")
    return arguments


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("format") != "simple_stateless_policy_v1":
        raise ValueError("unsupported simple-stateless checkpoint format")
    return payload


def _model_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise TypeError("checkpoint model state is not a tensor mapping")
    return state


def _fragment_identity(
    payload: Mapping[str, Any],
    *,
    policy_identity: StatelessPolicyIdentity,
    horizon: int,
) -> StatelessFragmentIdentity:
    return StatelessFragmentIdentity(
        horizon=horizon,
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


def _deck_pair(
    config: SimpleStatelessModelConfig,
    indices: tuple[int, int],
) -> np.ndarray:
    selected: list[tuple[int, ...]] = []
    for index in indices:
        try:
            route = config.exact_routes[index]
        except IndexError as error:
            raise ValueError("deck index is absent from exact routes") from error
        selected.append(route.canonical_card_ids)
    return np.asarray(selected, dtype=np.int32)


def _merge_actions(
    view: NativeTrainingBatchView,
    *,
    sampled_rows: np.ndarray,
    forced_mask: np.ndarray,
    trace: NativePolicyNumpyTrace,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_lengths = np.diff(trace.action_offsets)
    if trace.batch_size != sampled_rows.size:
        raise ValueError("policy trace rows do not align with native rows")
    lengths = np.ones(view.batch_size, dtype=np.int64)
    lengths[sampled_rows] = sampled_lengths
    if np.any(lengths < 0):
        raise ValueError("policy emitted a negative action length")
    offsets = np.zeros(view.batch_size + 1, dtype=np.uint32)
    np.cumsum(lengths, dtype=np.uint32, out=offsets[1:])
    choices = np.empty(int(offsets[-1]), dtype=np.int32)
    forced_rows = np.flatnonzero(forced_mask)
    choices[offsets[forced_rows]] = 0
    source_rows = np.repeat(
        np.arange(sampled_rows.size, dtype=np.int64),
        sampled_lengths,
    )
    source_local = np.arange(trace.action_choices.size, dtype=np.int64)
    source_local -= np.repeat(trace.action_offsets[:-1], sampled_lengths)
    destinations = offsets[sampled_rows[source_rows]] + source_local
    choices[destinations] = trace.action_choices
    return offsets, choices


def _require_clean(view: NativeTrainingBatchView) -> None:
    bad = np.flatnonzero(view.error)
    if bad.size:
        rows = bad[:8]
        raise RuntimeError(
            "native arena reported row errors: "
            f"rows={rows.tolist()} errors={view.error[rows].tolist()}"
        )


def _require_ready(view: NativeTrainingBatchView) -> None:
    _require_clean(view)
    bad = np.flatnonzero(view.status != _READY)
    if bad.size:
        raise RuntimeError(
            f"native arena reset did not yield ready rows: {bad[:8].tolist()}"
        )


def _add_time(
    stages: dict[str, float],
    name: str,
    started: float,
    measuring: bool,
) -> None:
    if measuring:
        stages[name] += time.perf_counter() - started


def _seeds(start: int, count: int) -> np.ndarray:
    maximum = int(np.iinfo(np.uint32).max) + 1
    return (
        (np.arange(count, dtype=np.uint64) + int(start)) % maximum
    ).astype(np.uint32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
