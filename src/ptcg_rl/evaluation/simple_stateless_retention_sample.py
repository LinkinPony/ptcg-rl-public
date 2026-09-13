"""Streaming deterministic row sampling for routed retention audits."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_rl.training.simple_stateless_pretrain_data import (
    file_sha256,
    load_pretraining_part,
)

_GENERIC_STRATUM = "generic"
_SAMPLE_DOMAIN = b"ptcg-rl/simple-stateless-retention-sample/v1\x00"


@dataclass(frozen=True)
class RetentionSampleRef:
    """One immutable compact-corpus row selected for retention evaluation."""

    part_index: int
    row_index: int
    episode_id: int
    stratum: str
    priority: int


def sample_retention_rows(
    *,
    dataset_manifest_path: Path,
    sample_rows: int,
    seed: int,
    dataset: Any,
    exact_routes: Sequence[Any],
) -> tuple[tuple[RetentionSampleRef, ...], dict[str, Any]]:
    """Verify each part and retain an equal-waterfill routed sample."""
    strata = tuple(
        [f"exact/{route.deck_digest}" for route in exact_routes] + [_GENERIC_STRATUM]
    )
    by_cards = {
        tuple(int(card_id) for card_id in route.canonical_card_ids): (
            f"exact/{route.deck_digest}"
        )
        for route in exact_routes
    }
    counts = dict.fromkeys(strata, 0)
    reservoirs: dict[
        str,
        list[tuple[int, int, int, RetentionSampleRef]],
    ] = {stratum: [] for stratum in strata}
    parts_dir = dataset_manifest_path.parent / "parts"
    for part_index, record in enumerate(dataset.parts):
        path = parts_dir / record.filename
        if (
            not path.is_file()
            or path.stat().st_size != record.size_bytes
            or file_sha256(path) != record.sha256
        ):
            raise ValueError(f"retention corpus part failed identity checks: {path}")
        part = load_pretraining_part(path)
        own_decks = np.asarray(part.arrays["own_decks"], dtype=np.int64)
        episode_ids = np.asarray(part.arrays["episode_ids"], dtype=np.int64)
        unique_decks, inverse = np.unique(
            own_decks,
            axis=0,
            return_inverse=True,
        )
        unique_strata = tuple(
            by_cards.get(
                tuple(int(card_id) for card_id in deck.tolist()),
                _GENERIC_STRATUM,
            )
            for deck in unique_decks
        )
        priorities = _part_priorities(
            seed=seed,
            dataset_fingerprint=dataset.fingerprint,
            filename=record.filename,
            rows=part.example_count,
        )
        for row_index in range(part.example_count):
            stratum = unique_strata[int(inverse[row_index])]
            counts[stratum] += 1
            reference = RetentionSampleRef(
                part_index=part_index,
                row_index=row_index,
                episode_id=int(episode_ids[row_index]),
                stratum=stratum,
                priority=int(priorities[row_index]),
            )
            consider_sample(
                reservoirs[stratum],
                reference,
                capacity=sample_rows,
            )
    selected_total = min(sample_rows, sum(counts.values()))
    allocations = allocate_stratum_samples(counts, selected_total)
    selected = tuple(
        sorted(
            (
                reference
                for stratum in strata
                for _negative_priority, _part, _row, reference in sorted(
                    reservoirs[stratum],
                    key=lambda item: (-item[0], -item[1], -item[2]),
                )[: allocations[stratum]]
            ),
            key=lambda reference: (
                reference.part_index,
                reference.row_index,
            ),
        )
    )
    if len(selected) != selected_total:
        raise RuntimeError("stratified retention sampler produced the wrong size")
    sample_fingerprint = _sample_fingerprint(
        dataset_fingerprint=dataset.fingerprint,
        seed=seed,
        requested_rows=sample_rows,
        selected=selected,
    )
    return (
        selected,
        {
            "seed": seed,
            "requested_rows": sample_rows,
            "selected_rows": len(selected),
            "sample_fingerprint": sample_fingerprint,
            "allocation": "equal-stratum-waterfill",
            "strata": {
                stratum: {
                    "available_rows": counts[stratum],
                    "sampled_rows": allocations[stratum],
                }
                for stratum in strata
            },
        },
    )


def consider_sample(
    heap: list[tuple[int, int, int, RetentionSampleRef]],
    reference: RetentionSampleRef,
    *,
    capacity: int,
) -> None:
    """Retain the deterministic lowest-priority rows for one stratum."""
    item = (
        -reference.priority,
        -reference.part_index,
        -reference.row_index,
        reference,
    )
    if len(heap) < capacity:
        heapq.heappush(heap, item)
        return
    worst_rank = (-heap[0][0], -heap[0][1], -heap[0][2])
    candidate_rank = (
        reference.priority,
        reference.part_index,
        reference.row_index,
    )
    if candidate_rank < worst_rank:
        heapq.heapreplace(heap, item)


def allocate_stratum_samples(
    counts: Mapping[str, int],
    sample_rows: int,
) -> dict[str, int]:
    """Equally allocate rows, redistributing capacity from sparse strata."""
    if (
        sample_rows < 0
        or any(count < 0 for count in counts.values())
        or sample_rows > sum(counts.values())
    ):
        raise ValueError("stratified sample size or capacity is invalid")
    allocations = dict.fromkeys(counts, 0)
    remaining = sample_rows
    while remaining:
        eligible = tuple(
            stratum
            for stratum in sorted(counts)
            if allocations[stratum] < counts[stratum]
        )
        if not eligible:
            raise RuntimeError("stratified sample allocation exhausted capacity")
        base, extra = divmod(remaining, len(eligible))
        proposed = max(base, 1)
        assigned = 0
        for index, stratum in enumerate(eligible):
            request = proposed + int(base > 0 and index < extra)
            capacity = counts[stratum] - allocations[stratum]
            delta = min(request, capacity, remaining - assigned)
            allocations[stratum] += delta
            assigned += delta
            if assigned == remaining:
                break
        if assigned <= 0:
            raise RuntimeError("stratified sample allocation did not progress")
        remaining -= assigned
    return allocations


def _part_priorities(
    *,
    seed: int,
    dataset_fingerprint: str,
    filename: str,
    rows: int,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    digest = hashlib.sha256(
        f"{seed}:{dataset_fingerprint}:{filename}".encode()
    ).digest()
    generator = np.random.default_rng(
        int.from_bytes(digest[:8], byteorder="big", signed=False)
    )
    return generator.integers(
        0,
        np.iinfo(np.uint64).max,
        size=rows,
        dtype=np.uint64,
    )


def _sample_fingerprint(
    *,
    dataset_fingerprint: str,
    seed: int,
    requested_rows: int,
    selected: Sequence[RetentionSampleRef],
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "seed": seed,
        "requested_rows": requested_rows,
        "selected": [
            (
                reference.part_index,
                reference.row_index,
                reference.episode_id,
                reference.stratum,
            )
            for reference in selected
        ],
    }
    return hashlib.sha256(
        _SAMPLE_DOMAIN
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


__all__ = [
    "RetentionSampleRef",
    "allocate_stratum_samples",
    "consider_sample",
    "sample_retention_rows",
]
