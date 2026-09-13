"""Streaming source selection for the fixed planner decision corpus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.context import context_features_from_row
from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.evaluation.consequence_audit_sampling import (
    CaseLocator,
    MultiLabelHashReservoir,
    RootShape,
)
from ptcg_rl.evaluation.planner_profile_config import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
)
from ptcg_rl.evaluation.planner_profile_corpus_types import (
    SCAN_COLUMNS,
    SelectedSourceRoot,
)


def load_engine_chance_case_ids(paths: Sequence[Path]) -> frozenset[str]:
    """Load roots where exact engine evidence observed RNG consumption."""
    result: set[str] = set()
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        if "case_id" not in names:
            raise ValueError(f"chance evidence has no case_id column: {path}")
        if "native_rng_unsupported" in names:
            columns = ["case_id", "native_rng_unsupported"]
            for batch in parquet_file.iter_batches(columns=columns, use_threads=True):
                for case_id, consumed in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                    strict=True,
                ):
                    if bool(consumed):
                        result.add(str(case_id))
            continue
        if "status" in names:
            columns = ["case_id", "status"]
            for batch in parquet_file.iter_batches(columns=columns, use_threads=True):
                for case_id, status in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                    strict=True,
                ):
                    if str(status) == "unsupported_rng_or_chance":
                        result.add(str(case_id))
            continue
        raise ValueError(f"chance evidence has no RNG-consumption signal: {path}")
    return frozenset(result)


def sample_locators(
    paths: Sequence[Path],
    *,
    chance_case_ids: frozenset[str],
    rows_per_shape: int,
    scan_batch_size: int,
    seed: str,
) -> tuple[tuple[CaseLocator, ...], int]:
    """Select a deterministic bounded multi-label union of structural roots."""
    reservoir: MultiLabelHashReservoir[CaseLocator] = MultiLabelHashReservoir(
        labels=tuple(sorted(REQUIRED_PLANNER_PROFILE_SHAPES)),
        capacity_per_label=rows_per_shape,
        seed=seed,
    )
    previous: tuple[Path, int, RootShape] | None = None
    active_episode: tuple[str, int] | None = None
    closed_episodes: set[tuple[str, int]] = set()
    scanned_rows = 0

    def retain(
        pending: tuple[Path, int, RootShape],
        successor: RootShape | None,
    ) -> None:
        path, row_index, shape = pending
        provisional = CaseLocator(path, row_index, shape, frozenset())
        labels = _decision_shapes(
            shape,
            successor,
            engine_chance=provisional.case_id in chance_case_ids,
        )
        locator = CaseLocator(path, row_index, shape, labels)
        reservoir.add(shape.key, locator, labels)

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(SCAN_COLUMNS).difference(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"planner corpus input {path} is missing {missing}")
        absolute_row = 0
        for batch in parquet_file.iter_batches(
            batch_size=scan_batch_size,
            columns=list(SCAN_COLUMNS),
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
                        raise ValueError("planner replay episodes are not contiguous")
                    active_episode = episode
                elif (
                    previous is not None and shape.step_index <= previous[2].step_index
                ):
                    raise ValueError("planner replay steps do not increase in episode")
                if previous is not None:
                    retain(previous, shape)
                previous = (path, absolute_row, shape)
                absolute_row += 1
                scanned_rows += 1
    if previous is not None:
        retain(previous, None)
    retained = reservoir.retained_union(key=lambda item: item.case_id)
    return tuple(sorted(retained, key=lambda item: item.case_id)), scanned_rows


def selected_source_roots(
    locators: Sequence[CaseLocator],
) -> Mapping[str, SelectedSourceRoot]:
    """Read only compact fields needed to verify exact replay roots."""
    by_path: dict[Path, dict[int, CaseLocator]] = defaultdict(dict)
    for locator in locators:
        by_path[locator.path][locator.row_index] = locator
    selected: dict[str, SelectedSourceRoot] = {}
    for path in sorted(by_path):
        parquet_file = pq.ParquetFile(path)
        targets = by_path[path]
        absolute_start = 0
        remaining = set(targets)
        for row_group_index in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group_index).num_rows
            retained = sorted(
                index
                for index in remaining
                if absolute_start <= index < absolute_start + row_count
            )
            if retained:
                table = parquet_file.read_row_group(row_group_index, use_threads=True)
                local_indices = pa.array(
                    [index - absolute_start for index in retained],
                    type=pa.int64(),
                )
                for index, raw in zip(
                    retained,
                    table.take(local_indices).to_pylist(),
                    strict=True,
                ):
                    locator = targets[index]
                    row = cast(Mapping[str, Any], raw)
                    _validate_source_locator(row, locator)
                    state_token = row.get("search_begin_input")
                    if not isinstance(state_token, (str, bytes)) or not state_token:
                        raise ValueError("selected profile root lacks an engine token")
                    own = parse_canonical_signature(
                        str(row.get("deck_signature") or "")
                    )
                    root = SelectedSourceRoot(
                        locator=locator,
                        own_deck=own.card_ids,
                        context_features=context_features_from_row(row),
                        search_begin_input=state_token,
                    )
                    if selected.setdefault(locator.case_id, root) is not root:
                        raise ValueError("profile corpus selected a duplicate root")
                remaining.difference_update(retained)
            absolute_start += row_count
        if remaining:
            raise ValueError(f"planner source row exceeds Parquet file: {path}")
    if len(selected) != len(locators):
        raise RuntimeError("profile source lookup omitted selected roots")
    return selected


def _decision_shapes(
    root: RootShape,
    successor: RootShape | None,
    *,
    engine_chance: bool,
) -> frozenset[str]:
    labels: set[str] = set()
    if root.select_max_count <= 1:
        labels.add("direct")
    else:
        labels.update(("subset", "ordered"))
    if root.select_context == int(SelectContext.SKILL_ORDER):
        labels.add("ordered")
    if successor is not None:
        if successor.player_index != root.player_index:
            labels.add("handoff")
        elif successor.select_context != int(SelectContext.MAIN):
            labels.add("multi_prompt")
    if engine_chance:
        labels.add("engine_chance")
    return frozenset(labels)


def _validate_source_locator(
    row: Mapping[str, Any],
    locator: CaseLocator,
) -> None:
    shape = locator.shape
    actual = (
        str(row.get("date")),
        int(row.get("episode_id", -1)),
        int(row.get("step_index", -1)),
        int(row.get("player_index", -1)),
        int(row.get("select_context", -1)),
        int(row.get("select_min_count", -1)),
        int(row.get("select_max_count", -1)),
        int(row.get("select_option_count", -1)),
    )
    expected = (
        shape.date,
        shape.episode_id,
        shape.step_index,
        shape.player_index,
        shape.select_context,
        shape.select_min_count,
        shape.select_max_count,
        shape.select_option_count,
    )
    if actual != expected:
        raise ValueError("profile source row differs from its sampled locator")


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


__all__ = [
    "load_engine_chance_case_ids",
    "sample_locators",
    "selected_source_roots",
]
