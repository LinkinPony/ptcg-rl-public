"""Deterministic, bounded replay-root sampling for consequence audits.

The first pass reads only small scalar columns.  State tokens and hidden-zone
material are loaded for the retained roots in a second streaming pass, so a
full replay corpus is never materialized in Python memory.
"""

from __future__ import annotations

import hashlib
import heapq
import struct
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import pyarrow.parquet as pq

from ptcg_rl.engine.constants import SelectContext

COVERAGE_LABELS = (
    "direct",
    "subset",
    "ordered",
    "multi_prompt",
    "manual_coin",
    "handoff",
)

_SCAN_COLUMNS = (
    "date",
    "episode_id",
    "step_index",
    "player_index",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
)
_CASE_ID_DOMAIN = b"ptcg-rl/decision-transition-audit/case/v1\x00"
_RESERVOIR_DOMAIN = b"ptcg-rl/decision-transition-audit/reservoir/v1\x00"
_CORPUS_DOMAIN = b"ptcg-rl/decision-transition-audit/corpus/v1\x00"

_ItemT = TypeVar("_ItemT")


@dataclass(frozen=True, slots=True)
class RootShape:
    """Small replay row used to classify a root before loading private data."""

    date: str
    episode_id: int
    step_index: int
    player_index: int
    select_context: int
    select_min_count: int
    select_max_count: int
    select_option_count: int

    @property
    def key(self) -> str:
        """Return a stable source locator key without state-token content."""
        return f"{self.date}/{self.episode_id}/{self.step_index}/{self.player_index}"


@dataclass(frozen=True, slots=True)
class CaseLocator:
    """Physical location and structural labels for one retained root."""

    path: Path
    row_index: int
    shape: RootShape
    labels: frozenset[str]

    @property
    def case_id(self) -> str:
        """Return a privacy-safe identifier independent of the state token."""
        return hashlib.sha256(
            _CASE_ID_DOMAIN + self.shape.key.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplingReport:
    """Deterministic scan counts and identities for an audit sample."""

    scanned_rows: int
    scanned_files: int
    corpus_label_counts: Mapping[str, int]
    retained_label_counts: Mapping[str, int]
    retained_roots: int
    corpus_fingerprint: str


class MultiLabelHashReservoir(Generic[_ItemT]):
    """Keep the lowest deterministic hashes independently for each label."""

    def __init__(
        self,
        *,
        labels: Sequence[str],
        capacity_per_label: int,
        seed: str,
    ) -> None:
        if capacity_per_label <= 0:
            raise ValueError("capacity_per_label must be positive")
        canonical_labels = tuple(str(label) for label in labels)
        if not canonical_labels or len(set(canonical_labels)) != len(canonical_labels):
            raise ValueError("labels must be a non-empty unique sequence")
        self._labels = canonical_labels
        self._capacity = int(capacity_per_label)
        self._seed = seed.encode("utf-8")
        self._heaps: dict[str, list[tuple[int, str, _ItemT]]] = {
            label: [] for label in canonical_labels
        }
        self._keys: dict[str, set[str]] = {label: set() for label in canonical_labels}

    def add(self, key: str, item: _ItemT, labels: Iterable[str]) -> None:
        """Offer an item to every named label reservoir."""
        for label in sorted(set(labels)):
            if label not in self._heaps:
                raise ValueError(f"unknown reservoir label: {label}")
            keys = self._keys[label]
            if key in keys:
                continue
            rank = self._rank(label, key)
            entry = (-rank, key, item)
            heap = self._heaps[label]
            if len(heap) < self._capacity:
                heapq.heappush(heap, entry)
                keys.add(key)
                continue
            worst_rank = -heap[0][0]
            worst_key = heap[0][1]
            if (rank, key) >= (worst_rank, worst_key):
                continue
            removed = heapq.heapreplace(heap, entry)
            keys.remove(removed[1])
            keys.add(key)

    def retained_by_label(self) -> Mapping[str, tuple[_ItemT, ...]]:
        """Return each deterministic label sample in rank order."""
        return {
            label: tuple(
                entry[2]
                for entry in sorted(
                    heap,
                    key=lambda entry: (-entry[0], entry[1]),
                )
            )
            for label, heap in self._heaps.items()
        }

    def retained_union(self, *, key: Any) -> tuple[_ItemT, ...]:
        """Return the deduplicated union of all label samples."""
        by_identity: dict[str, _ItemT] = {}
        for items in self.retained_by_label().values():
            for item in items:
                identity = str(key(item))
                by_identity.setdefault(identity, item)
        return tuple(by_identity[identity] for identity in sorted(by_identity))

    def _rank(self, label: str, key: str) -> int:
        digest = hashlib.sha256()
        digest.update(_RESERVOIR_DOMAIN)
        _update_framed(digest, self._seed)
        _update_framed(digest, label.encode("utf-8"))
        _update_framed(digest, key.encode("utf-8"))
        return int.from_bytes(digest.digest()[:16], "big")


def structural_coverage_labels(
    root: RootShape,
    successor: RootShape | None,
) -> frozenset[str]:
    """Classify a root with overlapping prompt and observed-boundary labels."""
    labels: set[str] = set()
    if root.select_max_count <= 1:
        labels.add("direct")
    else:
        labels.add("subset")
        # The public prompt has no general unordered flag.  Until an engine
        # equivalence probe narrows that contract, every multi-select is kept
        # order-sensitive by the production candidate grammar.
        labels.add("ordered")
    if root.select_context == int(SelectContext.SKILL_ORDER):
        labels.add("ordered")
    if root.select_context == int(SelectContext.COIN_HEAD):
        labels.add("manual_coin")
    if successor is not None:
        if successor.player_index != root.player_index:
            labels.add("handoff")
        elif successor.select_context != int(SelectContext.MAIN):
            labels.add("multi_prompt")
        if successor.select_context == int(SelectContext.COIN_HEAD):
            labels.add("manual_coin")
    return frozenset(labels)


def sample_replay_roots(
    paths: Sequence[Path],
    *,
    capacity_per_label: int,
    seed: str,
    batch_size: int,
) -> tuple[tuple[CaseLocator, ...], SamplingReport]:
    """Stream scalar replay columns into deterministic per-label reservoirs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered_paths = tuple(sorted(Path(path) for path in paths))
    if not ordered_paths:
        raise ValueError("at least one replay Parquet path is required")
    reservoir: MultiLabelHashReservoir[CaseLocator] = MultiLabelHashReservoir(
        labels=COVERAGE_LABELS,
        capacity_per_label=capacity_per_label,
        seed=seed,
    )
    label_counts: Counter[str] = Counter()
    corpus_digest = hashlib.sha256(_CORPUS_DOMAIN)
    previous: tuple[Path, int, RootShape] | None = None
    closed_episodes: set[tuple[str, int]] = set()
    active_episode: tuple[str, int] | None = None
    scanned_rows = 0

    def retain(
        pending: tuple[Path, int, RootShape],
        successor: RootShape | None,
    ) -> None:
        path, row_index, shape = pending
        labels = structural_coverage_labels(shape, successor)
        label_counts.update(labels)
        locator = CaseLocator(
            path=path,
            row_index=row_index,
            shape=shape,
            labels=labels,
        )
        reservoir.add(shape.key, locator, labels)
        _update_framed(corpus_digest, shape.key.encode("utf-8"))
        _update_framed(
            corpus_digest,
            ",".join(sorted(labels)).encode("ascii"),
        )

    for path in ordered_paths:
        parquet_file = pq.ParquetFile(path)
        _require_columns(parquet_file.schema_arrow, _SCAN_COLUMNS, path)
        absolute_row = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(_SCAN_COLUMNS),
            use_threads=True,
        ):
            for raw in batch.to_pylist():
                shape = _root_shape(cast(Mapping[str, Any], raw))
                episode = (shape.date, shape.episode_id)
                if active_episode is None:
                    active_episode = episode
                elif episode != active_episode:
                    if previous is not None:
                        retain(previous, None)
                        previous = None
                    closed_episodes.add(active_episode)
                    if episode in closed_episodes:
                        raise ValueError(
                            "replay rows are not contiguous by date and episode"
                        )
                    active_episode = episode
                elif previous is not None and (
                    shape.step_index <= previous[2].step_index
                ):
                    raise ValueError(
                        "replay step indices must increase within an episode"
                    )

                if previous is not None:
                    retain(previous, shape)
                previous = (path, absolute_row, shape)
                absolute_row += 1
                scanned_rows += 1
    if previous is not None:
        retain(previous, None)

    by_label = reservoir.retained_by_label()
    retained = reservoir.retained_union(key=lambda item: item.shape.key)
    report = SamplingReport(
        scanned_rows=scanned_rows,
        scanned_files=len(ordered_paths),
        corpus_label_counts={
            label: int(label_counts[label]) for label in COVERAGE_LABELS
        },
        retained_label_counts={
            label: len(by_label[label]) for label in COVERAGE_LABELS
        },
        retained_roots=len(retained),
        corpus_fingerprint=corpus_digest.hexdigest(),
    )
    return retained, report


def _root_shape(row: Mapping[str, Any]) -> RootShape:
    return RootShape(
        date=str(row["date"]),
        episode_id=int(row["episode_id"]),
        step_index=int(row["step_index"]),
        player_index=int(row["player_index"]),
        select_context=int(row["select_context"]),
        select_min_count=int(row["select_min_count"]),
        select_max_count=int(row["select_max_count"]),
        select_option_count=int(row["select_option_count"]),
    )


def _require_columns(
    schema: Any,
    required: Sequence[str],
    path: Path,
) -> None:
    missing = sorted(set(required).difference(schema.names))
    if missing:
        raise ValueError(f"Parquet input {path} is missing columns: {missing}")


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


__all__ = [
    "COVERAGE_LABELS",
    "CaseLocator",
    "MultiLabelHashReservoir",
    "RootShape",
    "SamplingReport",
    "sample_replay_roots",
    "structural_coverage_labels",
]
