"""Explicit builder and atomic writer for planner profile corpora."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.evaluation.consequence_audit_sampling import CaseLocator
from ptcg_rl.evaluation.consequence_parity_artifact import (
    file_sha256,
    resolve_parquet_paths,
    write_json_atomic,
)
from ptcg_rl.evaluation.planner_profile_config import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
    PlannerProfileCorpusBuildConfig,
)
from ptcg_rl.evaluation.planner_profile_context import (
    PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION,
    PROFILE_OBSERVATION_CODEC_VERSION,
)
from ptcg_rl.evaluation.planner_profile_corpus_replay import (
    reconstruct_exact_roots,
)
from ptcg_rl.evaluation.planner_profile_corpus_sampling import (
    load_engine_chance_case_ids,
    sample_locators,
    selected_source_roots,
)
from ptcg_rl.evaluation.planner_profile_corpus_types import (
    CORPUS_SCHEMA_VERSION,
    PROFILE_CASE_ID,
    PROFILE_CONTEXT_CODEC,
    PROFILE_CONTEXT_FINGERPRINT,
    PROFILE_CONTEXT_SNAPSHOT,
    PROFILE_OBSERVATION,
    PROFILE_OBSERVATION_CODEC,
    PROFILE_OBSERVATION_FINGERPRINT,
    PROFILE_REPLAY_SHA256,
    PROFILE_SCHEMA_VERSION,
    PROFILE_SHAPES,
    PROFILE_SOURCE_CONTEXT_MATCH,
    PROFILE_SOURCE_ROW,
    ExactProfileRoot,
    PlannerProfileCorpusManifest,
)


def build_planner_profile_corpus(
    config: PlannerProfileCorpusBuildConfig,
) -> PlannerProfileCorpusManifest:
    """Sample six shapes and publish exact replay/context-bound root rows."""
    if config.output_path.exists() or config.manifest_path.exists():
        raise FileExistsError("planner corpus outputs are immutable; use fresh paths")
    source_paths = resolve_parquet_paths(config.source_globs)
    chance_paths = resolve_parquet_paths(config.chance_evidence_globs)
    chance_ids = load_engine_chance_case_ids(chance_paths)
    if not chance_ids:
        raise ValueError("chance evidence contains no engine RNG-consumption roots")
    locators, scanned_rows = sample_locators(
        source_paths,
        chance_case_ids=chance_ids,
        rows_per_shape=config.rows_per_shape,
        scan_batch_size=config.scan_batch_size,
        seed=config.seed,
    )
    retained_shapes = Counter(label for locator in locators for label in locator.labels)
    missing = sorted(REQUIRED_PLANNER_PROFILE_SHAPES.difference(retained_shapes))
    if missing:
        raise ValueError(f"planner corpus is missing required shapes: {missing}")
    selected_roots = selected_source_roots(locators)
    exact_roots, replay_assets, replay_archives = reconstruct_exact_roots(
        selected_roots,
        replay_root=config.replay_root,
        replay_archive_root=config.replay_archive_root,
    )
    _write_selected_rows(
        locators,
        exact_roots=exact_roots,
        output_path=config.output_path,
        compression=config.compression,
    )
    corpus_sha256 = file_sha256(config.output_path)
    manifest = PlannerProfileCorpusManifest(
        corpus_path=str(config.output_path),
        corpus_sha256=corpus_sha256,
        corpus_schema_version=CORPUS_SCHEMA_VERSION,
        rows=len(locators),
        shape_counts=dict(sorted(retained_shapes.items())),
        source_files=len(source_paths),
        source_rows_scanned=scanned_rows,
        chance_evidence_files=len(chance_paths),
        chance_case_ids=len(chance_ids),
        rows_per_shape=config.rows_per_shape,
        seed=config.seed,
        observation_codec_version=PROFILE_OBSERVATION_CODEC_VERSION,
        context_snapshot_codec_version=PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION,
        replay_assets=replay_assets,
        replay_archives=replay_archives,
        source_context_match_rows=sum(
            int(root.source_context_match) for root in exact_roots.values()
        ),
        source_context_mismatch_rows=sum(
            int(not root.source_context_match) for root in exact_roots.values()
        ),
        source_context_drift_field_counts=dict(
            Counter(
                field_name
                for root in exact_roots.values()
                for field_name in root.source_context_drift_fields
            )
        ),
    )
    write_json_atomic(config.manifest_path, manifest.as_dict())
    return manifest


def _write_selected_rows(
    locators: Sequence[CaseLocator],
    *,
    exact_roots: Mapping[str, ExactProfileRoot],
    output_path: Path,
    compression: str,
) -> None:
    by_path: dict[Path, dict[int, CaseLocator]] = defaultdict(dict)
    for locator in locators:
        if locator.row_index in by_path[locator.path]:
            raise ValueError("planner corpus retained a duplicate source row")
        by_path[locator.path][locator.row_index] = locator
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for path in sorted(by_path):
            parquet_file = pq.ParquetFile(path)
            targets = by_path[path]
            absolute_start = 0
            remaining = set(targets)
            for row_group_index in range(parquet_file.num_row_groups):
                row_count = parquet_file.metadata.row_group(row_group_index).num_rows
                selected = sorted(
                    index
                    for index in remaining
                    if absolute_start <= index < absolute_start + row_count
                )
                if selected:
                    source = parquet_file.read_row_group(
                        row_group_index,
                        use_threads=True,
                    )
                    local = pa.array(
                        [index - absolute_start for index in selected],
                        type=pa.int64(),
                    )
                    table = source.take(local)
                    table = table.append_column(
                        PROFILE_CASE_ID,
                        pa.array(
                            [targets[index].case_id for index in selected],
                            type=pa.string(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_SHAPES,
                        pa.array(
                            [sorted(targets[index].labels) for index in selected],
                            type=pa.list_(pa.string()),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_SOURCE_ROW,
                        pa.array(selected, type=pa.int64()),
                    )
                    exact = [exact_roots[targets[index].case_id] for index in selected]
                    action_index = table.column_names.index("action")
                    action_field = table.schema.field(action_index)
                    table = table.set_column(
                        action_index,
                        action_field,
                        pa.array(
                            [list(item.executed_action) for item in exact],
                            type=action_field.type,
                        ),
                    )
                    table = table.append_column(
                        PROFILE_SCHEMA_VERSION,
                        pa.array(
                            [CORPUS_SCHEMA_VERSION] * len(selected),
                            type=pa.int16(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_OBSERVATION,
                        pa.array(
                            [item.observation_payload for item in exact],
                            type=pa.binary(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_OBSERVATION_FINGERPRINT,
                        pa.array(
                            [item.observation_fingerprint for item in exact],
                            type=pa.string(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_OBSERVATION_CODEC,
                        pa.array(
                            [PROFILE_OBSERVATION_CODEC_VERSION] * len(selected),
                            type=pa.int16(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_CONTEXT_SNAPSHOT,
                        pa.array(
                            [item.context_snapshot_payload for item in exact],
                            type=pa.binary(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_CONTEXT_FINGERPRINT,
                        pa.array(
                            [item.context_snapshot_fingerprint for item in exact],
                            type=pa.string(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_CONTEXT_CODEC,
                        pa.array(
                            [PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION] * len(selected),
                            type=pa.int16(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_REPLAY_SHA256,
                        pa.array(
                            [item.replay_sha256 for item in exact],
                            type=pa.string(),
                        ),
                    )
                    table = table.append_column(
                        PROFILE_SOURCE_CONTEXT_MATCH,
                        pa.array(
                            [item.source_context_match for item in exact],
                            type=pa.bool_(),
                        ),
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            table.schema,
                            compression=compression,
                            write_statistics=True,
                        )
                    elif table.schema != writer.schema:
                        raise ValueError("planner replay shard schemas differ")
                    writer.write_table(table)
                    rows_written += table.num_rows
                    remaining.difference_update(selected)
                absolute_start += row_count
            if remaining:
                raise ValueError(f"planner source row exceeds Parquet file: {path}")
        if writer is None or rows_written != len(locators):
            raise RuntimeError("planner corpus did not publish every retained row")
        writer.close()
        writer = None
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


__all__ = ["PlannerProfileCorpusManifest", "build_planner_profile_corpus"]
