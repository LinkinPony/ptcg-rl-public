"""Benchmark immutable legacy historical-policy inference on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.context import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.data.kaggle_deck.records import read_deck
from ptcg_rl.decks import DeckBatch, canonicalize_deck
from ptcg_rl.engine.native_public_context import NativePublicContextTracker
from ptcg_rl.engine.native_training import (
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.rl.native_legacy_belief import (
    append_legacy_belief_tokens,
    legacy_known_counts,
)
from ptcg_rl.rl.native_policy_options import encode_native_option_batch
from ptcg_rl.rl.native_policy_state import encode_native_state_batch

_INPUT_FORMAT = "legacy_cpu_inference_input_v1"
_REPORT_FORMAT = "legacy_cpu_inference_benchmark_v1"


def main() -> None:
    """Prepare public reset inputs or run the historical-policy benchmark."""
    arguments = _arguments()
    if arguments.command == "prepare":
        _prepare(arguments)
    elif arguments.command == "run":
        _run(arguments)
    else:  # pragma: no cover - argparse enforces this.
        raise ValueError(f"unsupported command: {arguments.command}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--deck", type=Path, required=True)
    prepare.add_argument("--belief-summary", type=Path, required=True)
    prepare.add_argument("--public-catalog-manifest", type=Path, required=True)
    prepare.add_argument("--public-catalog-fingerprint", required=True)
    prepare.add_argument("--input-contract-fingerprint", required=True)
    prepare.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8, 16, 32),
    )
    prepare.add_argument("--seed", type=int, default=20260724)
    prepare.add_argument("--library", type=Path)
    prepare.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--input-bundle", type=Path, required=True)
    run.add_argument("--batch-sizes", nargs="+", type=int)
    run.add_argument("--torch-threads", type=int, required=True)
    run.add_argument("--warmup-iterations", type=int, default=2)
    run.add_argument("--minimum-measure-seconds", type=float, default=5.0)
    run.add_argument("--maximum-iterations", type=int, default=100)
    run.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    sizes = getattr(arguments, "batch_sizes", None)
    if sizes is not None and any(size <= 0 for size in sizes):
        parser.error("batch sizes must be positive")
    if arguments.command == "run":
        if arguments.torch_threads <= 0:
            parser.error("--torch-threads must be positive")
        if arguments.warmup_iterations < 0:
            parser.error("--warmup-iterations cannot be negative")
        if arguments.minimum_measure_seconds <= 0.0:
            parser.error("--minimum-measure-seconds must be positive")
        if arguments.maximum_iterations <= 0:
            parser.error("--maximum-iterations must be positive")
    return arguments


def _prepare(arguments: argparse.Namespace) -> None:
    deck = canonicalize_deck(read_deck(arguments.deck))
    belief_config = OpponentBeliefFeatureConfig(
        enabled=True,
        deck_signature_summary_path=arguments.belief_summary,
        deck_signature_summary_sha256=_sha256(arguments.belief_summary),
    )
    belief = OpponentBeliefFeatureProducer.from_config(belief_config)
    catalog, _manifest = load_public_deck_catalog(
        arguments.public_catalog_manifest
    )
    if catalog.fingerprint != arguments.public_catalog_fingerprint:
        raise ValueError("public catalog fingerprint differs from the request")

    batches: dict[int, dict[str, Any]] = {}
    for size in sorted(set(arguments.batch_sizes)):
        deck_pair = np.asarray((deck.card_ids, deck.card_ids), dtype=np.int32)
        raw_decks = np.broadcast_to(deck_pair, (size, 2, 60)).copy()
        tracker = NativePublicContextTracker(
            slot_capacity=size,
            catalog=catalog,
            input_contract_fingerprint=arguments.input_contract_fingerprint,
        )
        buffer = NativeTrainingOutputBuffer(
            slot_capacity=size,
            option_capacity=size * 256,
        )
        with NativeTrainingLane(size, library_path=arguments.library) as lane:
            view = lane.reset(
                raw_decks,
                np.arange(
                    arguments.seed,
                    arguments.seed + size,
                    dtype=np.uint32,
                ),
                output=buffer,
            )
            context = tracker.consume_reset(view, raw_decks)
            known = tracker.known_opponent_batch(
                view.slots,
                view.select_player,
            )
            states, lookup = encode_native_state_batch(
                view,
                context,
                device="cpu",
            )
            options = encode_native_option_batch(
                view,
                lookup,
                device="cpu",
            )
            belief_rows = tuple(
                belief.features_from_known_counts(
                    legacy_known_counts(view, known, row=row)
                )
                for row in range(size)
            )
            states = append_legacy_belief_tokens(states, belief_rows)
        batches[size] = {
            "states": states,
            "options": options,
            "decks": DeckBatch.from_decks(
                tuple(deck for _ in range(size)),
                device="cpu",
            ),
        }

    payload = {
        "format": _INPUT_FORMAT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "deck_digest": deck.deck_digest,
        "own_deck_card_ids": deck.card_ids,
        "belief_summary_sha256": _sha256(arguments.belief_summary),
        "public_catalog_fingerprint": catalog.fingerprint,
        "input_contract_fingerprint": arguments.input_contract_fingerprint,
        "scope": "public reset-state model inputs only",
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
                "deck_digest": deck.deck_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run(arguments: argparse.Namespace) -> None:
    torch.set_num_threads(arguments.torch_threads)
    torch.set_num_interop_threads(1)
    payload = torch.load(
        arguments.input_bundle,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or payload.get("format") != _INPUT_FORMAT:
        raise ValueError("legacy CPU inference input bundle is invalid")
    if payload.get("checkpoint_sha256") != _sha256(arguments.checkpoint):
        raise ValueError("legacy CPU inference input and checkpoint differ")
    own_deck = payload.get("own_deck_card_ids")
    if not isinstance(own_deck, Sequence):
        raise TypeError("legacy CPU inference input has no own deck")

    load_started_at = time.perf_counter()
    policy = CheckpointPolicy(
        arguments.checkpoint,
        device="cpu",
        own_deck=tuple(int(card_id) for card_id in own_deck),
    )
    checkpoint_load_seconds = time.perf_counter() - load_started_at
    raw_batches = payload.get("batches")
    if not isinstance(raw_batches, Mapping):
        raise TypeError("legacy CPU inference input has no batches")
    requested = (
        None
        if arguments.batch_sizes is None
        else set(arguments.batch_sizes)
    )
    results = []
    for raw_size, raw_batch in sorted(
        raw_batches.items(),
        key=lambda item: int(item[0]),
    ):
        size = int(raw_size)
        if requested is not None and size not in requested:
            continue
        if not isinstance(raw_batch, Mapping):
            raise TypeError("legacy CPU inference batch must be a mapping")
        results.append(
            _benchmark_batch(
                policy,
                raw_batch,
                size=size,
                warmup_iterations=arguments.warmup_iterations,
                minimum_seconds=arguments.minimum_measure_seconds,
                maximum_iterations=arguments.maximum_iterations,
            )
        )
    if not results:
        raise ValueError("no requested legacy CPU inference batch is available")

    model = policy.planner_model
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
        "runtime_id": os.environ.get("PTCG_BENCHMARK_RUNTIME_ID"),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_dtype": str(next(model.parameters()).dtype),
        "input_bundle_sha256": _sha256(arguments.input_bundle),
        "input_scope": payload.get("scope"),
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
    policy: CheckpointPolicy,
    batch: Mapping[str, Any],
    *,
    size: int,
    warmup_iterations: int,
    minimum_seconds: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    states = batch["states"]
    options = batch["options"]
    decks = batch["decks"]
    for _ in range(warmup_iterations):
        policy.select_preencoded_actions(states, options, decks)
    durations: list[float] = []
    actions: tuple[tuple[int, ...], ...] = ()
    measured_at = time.perf_counter()
    while len(durations) < maximum_iterations:
        started_at = time.perf_counter()
        actions = policy.select_preencoded_actions(states, options, decks)
        durations.append(time.perf_counter() - started_at)
        if time.perf_counter() - measured_at >= minimum_seconds:
            break
    if len(actions) != size:
        raise RuntimeError("legacy CPU inference returned the wrong batch size")
    measured_seconds = sum(durations)
    ordered = sorted(durations)
    return {
        "batch_size": size,
        "mean_state_tokens": _mean_rows(
            states.sequence_lengths,
            fallback=states.padding_mask.logical_not().sum(dim=1),
        ),
        "mean_options": _mean_rows(
            options.option_lengths,
            fallback=options.valid_options.sum(dim=1),
        ),
        "iterations": len(durations),
        "measured_seconds": measured_seconds,
        "rows_per_second": size * len(durations) / measured_seconds,
        "seconds": {
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "p95": ordered[
                min(len(ordered) - 1, int(0.95 * len(ordered)))
            ],
            "maximum": ordered[-1],
        },
    }


def _mean_rows(values: Sequence[int], *, fallback: Any) -> float:
    if values:
        return statistics.fmean(values)
    return float(fallback.to(dtype=torch.float64).mean().item())


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
