"""Benchmark the zero-JSON native live-game arena with real exact decks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)


def main() -> None:
    """Run deterministic complete selections and emit a bound evidence snapshot."""
    arguments = _arguments()
    config = json.loads(arguments.resolved_config.read_text(encoding="utf-8"))
    deck_pair = _deck_pair(config, tuple(arguments.deck_indices))
    lane_count = arguments.lanes
    decks = np.broadcast_to(deck_pair, (lane_count, 2, 60)).copy()
    slots = np.arange(lane_count, dtype=np.uint32)
    first = NativeTrainingOutputBuffer(
        slot_capacity=lane_count,
        option_capacity=lane_count * arguments.options_per_lane,
    )
    second = NativeTrainingOutputBuffer(
        slot_capacity=lane_count,
        option_capacity=lane_count * arguments.options_per_lane,
    )
    seed_cursor = arguments.seed
    selections = 0
    resets = 0
    terminal_rows = 0
    option_rows = 0
    public_log_rows = 0
    cohort_restarts = 0
    reset_forced_selection_advances = 0
    step_selection_advances = 0

    with NativeTrainingLane(
        lane_count,
        library_path=arguments.library,
    ) as lane:
        started_at = time.perf_counter()
        view = lane.reset(
            decks,
            _seeds(seed_cursor, lane_count),
            slots=slots,
            output=first,
        )
        seed_cursor += lane_count
        resets += lane_count
        reset_forced_selection_advances += int(
            view.selection_advance_count.sum(dtype=np.uint64)
        )
        _require_clean(view)
        for batch_index in range(arguments.batches):
            offsets, choices = _minimum_legal_actions(view)
            view = lane.step(
                slots,
                offsets,
                choices,
                output=second if batch_index % 2 == 0 else first,
            )
            _require_clean(view)
            selections += lane_count
            step_selection_advances += int(
                view.selection_advance_count.sum(dtype=np.uint64)
            )
            option_rows += view.option_count
            public_log_rows += view.log_count
            finished = view.result >= 0
            if not bool(np.any(finished)):
                continue
            terminal_rows += int(np.count_nonzero(finished))
            cohort_restarts += 1
            view = lane.reset(
                decks,
                _seeds(seed_cursor, lane_count),
                slots=slots,
                output=first if batch_index % 2 == 0 else second,
            )
            seed_cursor += lane_count
            resets += lane_count
            reset_forced_selection_advances += int(
                view.selection_advance_count.sum(dtype=np.uint64)
            )
            _require_clean(view)
        elapsed = time.perf_counter() - started_at
        library_path = lane.library_path

    report = {
        "format": "native-training-arena-throughput-v4",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "resolved_config": str(arguments.resolved_config.resolve()),
        "resolved_config_fingerprint": config.get("resolved_config_fingerprint"),
        "library": str(library_path),
        "library_sha256": _sha256(library_path),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "lanes": lane_count,
        "batches": arguments.batches,
        "strategic_selections": selections,
        "slot_resets": resets,
        "terminal_rows_observed": terminal_rows,
        "cohort_restarts": cohort_restarts,
        "reset_forced_selection_advances": reset_forced_selection_advances,
        "step_selection_advances": step_selection_advances,
        "selection_advances": (
            reset_forced_selection_advances + step_selection_advances
        ),
        "elapsed_seconds": elapsed,
        "strategic_selections_per_second": selections / elapsed,
        "arena_operations_per_second": (selections + resets) / elapsed,
        "selection_advances_per_second": (
            reset_forced_selection_advances + step_selection_advances
        )
        / elapsed,
        "mean_selection_advances_per_strategic_selection": (
            step_selection_advances / float(selections)
        ),
        "mean_next_prompt_options": option_rows / float(selections),
        "mean_public_log_delta_rows": public_log_rows / float(selections),
        "measurement_scope": (
            "engine reset/complete-selection/forced-chain with exact callback "
            "advance accounting, public current-state "
            "and perspective-projected public-log-delta projection, and Python "
            "ctypes/CSR; excludes history aggregation/model feature encoding, "
            "policy inference, replay, and learner"
        ),
    }
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--options-per-lane", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--deck-indices",
        type=int,
        nargs=2,
        default=(0, 1),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.lanes <= 0 or arguments.batches <= 0:
        parser.error("--lanes and --batches must be positive")
    if arguments.options_per_lane <= 0:
        parser.error("--options-per-lane must be positive")
    if not 0 <= arguments.seed <= np.iinfo(np.uint32).max:
        parser.error("--seed must fit uint32")
    return arguments


def _deck_pair(
    config: dict[str, Any],
    indices: tuple[int, int],
) -> np.ndarray:
    model = config.get("resolved_model_config")
    routes = model.get("exact_routes") if isinstance(model, dict) else None
    if not isinstance(routes, list):
        raise ValueError("resolved config has no exact route deck table")
    selected: list[list[int]] = []
    for index in indices:
        try:
            raw = routes[index]["canonical_card_ids"]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                "deck index is absent from resolved exact routes"
            ) from error
        cards = [int(card_id) for card_id in raw]
        if len(cards) != 60:
            raise ValueError("resolved exact route does not contain 60 cards")
        selected.append(cards)
    return np.asarray(selected, dtype=np.int32)


def _minimum_legal_actions(
    view: NativeTrainingBatchView,
) -> tuple[np.ndarray, np.ndarray]:
    option_counts = np.diff(view.option_offsets).astype(np.int32, copy=False)
    lengths = np.minimum(view.select_min, option_counts)
    if bool(np.any(lengths < 0)):
        raise RuntimeError("native prompt exposed a negative selection minimum")
    offsets = np.empty(view.batch_size + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint32, out=offsets[1:])
    if int(offsets[-1]) == 0:
        return (offsets, np.empty(0, dtype=np.int32))
    choices = np.concatenate(
        tuple(np.arange(int(length), dtype=np.int32) for length in lengths)
    )
    return (offsets, choices)


def _require_clean(view: NativeTrainingBatchView) -> None:
    bad = np.flatnonzero(view.error)
    if bad.size:
        rows = bad[:8]
        raise RuntimeError(
            "native arena reported row errors: "
            f"rows={rows.tolist()} errors={view.error[rows].tolist()}"
        )


def _seeds(start: int, count: int) -> np.ndarray:
    maximum = int(np.iinfo(np.uint32).max) + 1
    return ((np.arange(count, dtype=np.uint64) + int(start)) % maximum).astype(
        np.uint32
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
