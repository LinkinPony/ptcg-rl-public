"""Streaming public-replay extraction for simple-stateless pretraining."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch

from ptcg_rl.actions.selection import (
    ENGINE_PROVEN_UNORDERED_SET_CONTEXTS,
    is_forced,
    is_legal_action,
    is_unordered_set_selection,
    normalize_action_order,
)
from ptcg_rl.agent.probe import core_option_candidates
from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContextFeatures,
    PublicCatalogContext,
    PublicEventDecisionToken,
)
from ptcg_rl.data.kaggle_deck.records import fast_episode_side_rows
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.prospective_facts import (
    ProspectiveEngineFactConfig,
    ProspectiveEngineFactProducer,
    ProspectiveEngineFactResult,
)
from ptcg_rl.model.sequence.action import build_accepted_action_record
from ptcg_rl.rl.policy_inputs import (
    PolicyInputContract,
    SimpleStatelessActorRow,
    SimpleStatelessPublicInputAdapter,
)
from ptcg_rl.training.simple_stateless_pretrain_config import (
    ReplayPretrainingDataConfig,
)
from ptcg_rl.training.simple_stateless_pretrain_data import (
    REPLAY_SPLITS,
    TEMPORAL_PRETRAINING_SHARD_SCHEMA,
    PretrainingPartRecord,
    RejectedReplayRecord,
    ReplayPretrainingDatasetManifest,
    ReplayPretrainingExample,
    ReplayPretrainingShardWriter,
    ReplaySplit,
    file_sha256,
    write_pretraining_part,
)


@dataclass(frozen=True)
class ReplaySourceRecord:
    """One immutable raw replay selected by the source manifest."""

    source_index: int
    date: str
    episode_id: int
    split: ReplaySplit
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True)
class ReplayExtractionResult:
    """All demonstrated rows from one complete source replay."""

    source_index: int
    examples: tuple[ReplayPretrainingExample, ...]
    counters: dict[str, int]
    resident_bytes: int = 0


@dataclass(frozen=True)
class ReplayExtractionFailure:
    """One replay-level data failure isolated from the remaining corpus."""

    source_index: int
    rejection: RejectedReplayRecord
    counters: dict[str, int]


ReplayExtractionOutcome = ReplayExtractionResult | ReplayExtractionFailure


@dataclass(frozen=True)
class ReplayExtractionBatchResult:
    """One worker-built compact part and its small ordered commit metadata."""

    start_source_index: int
    source_replays: int
    part: PretrainingPartRecord | None
    counters: dict[str, int]
    rejections: tuple[RejectedReplayRecord, ...]


@dataclass(frozen=True)
class _PendingDecision:
    observation: Mapping[str, Any]
    context: PublicCatalogContext
    select: Mapping[str, Any]
    step_index: int
    known_opponent_counts: tuple[tuple[int, int], ...]
    forced: bool
    actor_row: SimpleStatelessActorRow | None = None
    event_token: PublicEventDecisionToken | None = None
    decision_index: int | None = None


@dataclass(frozen=True)
class _WorkerSettings:
    catalog_manifest_path: Path
    input_contract: dict[str, Any]
    source_selection: Literal[
        "team_allowlist", "episode_team_bindings", "all_sides"
    ]
    team_indices: dict[str, int]
    episode_team_indices: dict[int, dict[str, int]]
    date_indices: dict[str, int]
    prefix_bytes: int
    json_chunk_bytes: int
    require_done_status: bool
    drop_forced_actions: bool
    target_deck_digest: str | None
    temporal_sequence: bool
    route_expert_ids: dict[str, str]
    engine_fact_config: dict[str, Any] | None = None
    engine_fact_producer_fingerprint: str | None = None


_WORKER_SETTINGS: _WorkerSettings | None = None
_WORKER_CATALOG: Any = None
_WORKER_CONTRACT: PolicyInputContract | None = None
_WORKER_DATASET_IDENTITY: ReplayPretrainingDatasetManifest | None = None
_WORKER_PARTS_DIR: Path | None = None
_WORKER_ENGINE_FACT_PRODUCER: _EligibleEngineFactProducer | None = None


class _EligibleEngineFactProducer:
    """Skip producer work unless the public root has a supported option."""

    def __init__(self, producer: ProspectiveEngineFactProducer) -> None:
        self._producer = producer
        self.fingerprint = producer.fingerprint

    def run(
        self,
        observation: dict[str, Any],
        context_features: GameContextFeatures,
        *,
        your_deck: tuple[int, ...],
    ) -> ProspectiveEngineFactResult | None:
        """Probe only root ATTACK/ABILITY options, preserving unavailable rows."""
        if not core_option_candidates(observation.get("select")):
            return None
        result = self._producer.run(
            observation,
            context_features,
            your_deck=your_deck,
        )
        if result is not None and result.producer_fingerprint != self.fingerprint:
            raise ValueError("engine-fact result changed producer identity")
        return result


def engine_fact_producer_fingerprint(
    config: ProspectiveEngineFactConfig | None,
) -> str | None:
    """Resolve the concrete producer identity, including its backend ABI."""
    if config is None or not config.enabled:
        return None
    return _build_engine_fact_producer(config).fingerprint


def _build_engine_fact_producer(
    config: ProspectiveEngineFactConfig,
) -> _EligibleEngineFactProducer:
    """Build one reusable production-equivalent engine-fact producer."""
    producer = ProspectiveEngineFactProducer(
        sampler=BeliefSampler(config=config.sampler),
        config=config,
    )
    return _EligibleEngineFactProducer(producer)


def build_replay_source_index(
    config: ReplayPretrainingDataConfig,
    *,
    repo_root: Path,
) -> tuple[
    tuple[ReplaySourceRecord, ...],
    ReplayPretrainingDatasetManifest,
]:
    """Verify small source manifests and resolve every local replay path."""
    source_root = _path(config.source_root, repo_root)
    source_manifest_path = _path(config.source_manifest_path, repo_root)
    replay_manifest_path = _path(config.replay_manifest_path, repo_root)
    with source_manifest_path.open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    outputs = _mapping(source_manifest.get("outputs"))
    expected_replays_sha = str(outputs.get("replays_manifest_sha256", ""))
    replay_manifest_sha = file_sha256(replay_manifest_path)
    if replay_manifest_sha != expected_replays_sha:
        raise ValueError("raw replay manifest differs from source declaration")
    declared_selection = str(
        _mapping(source_manifest.get("selection")).get(
            "mode",
            "team_allowlist",
        )
    )
    if declared_selection != config.source_selection:
        raise ValueError("source manifest selection differs from training config")
    top_teams_sha: str | None = None
    episode_teams_sha: str | None = None
    episode_team_indices: dict[int, dict[str, int]] = {}
    if config.source_selection in {"team_allowlist", "episode_team_bindings"}:
        if config.top_teams_path is None:
            raise ValueError("explicit-selection source has no cohort path")
        top_teams_path = _path(config.top_teams_path, repo_root)
        expected_teams_sha = str(outputs.get("top30_sha256", ""))
        top_teams_sha = file_sha256(top_teams_path)
        if top_teams_sha != expected_teams_sha:
            raise ValueError("team cohort snapshot differs from source declaration")
        teams = _read_top_teams(top_teams_path)
        if config.source_selection == "episode_team_bindings":
            if config.episode_teams_path is None:
                raise ValueError("episode-team source has no binding path")
            episode_teams_path = _path(config.episode_teams_path, repo_root)
            episode_teams_sha = file_sha256(episode_teams_path)
            if episode_teams_sha != str(
                outputs.get("episode_teams_sha256", "")
            ):
                raise ValueError("episode-team bindings differ from source declaration")
            episode_team_indices = _read_episode_team_bindings(
                episode_teams_path,
                team_indices={
                    team.casefold(): index for index, team in enumerate(teams)
                },
            )
    else:
        if outputs.get("top30_sha256") is not None:
            raise ValueError("all-sides source unexpectedly declares a team cohort")
        teams = ("all_sides",)
    per_date = source_manifest.get("per_date")
    if not isinstance(per_date, Sequence):
        raise ValueError("source manifest has no per-date inventory")
    dates = tuple(str(_mapping(row).get("date", "")) for row in per_date)
    if not dates or any(not date for date in dates):
        raise ValueError("source manifest dates are invalid")

    records: list[ReplaySourceRecord] = []
    episode_splits: dict[int, ReplaySplit] = {}
    with replay_manifest_path.open(encoding="utf-8", newline="") as handle:
        for source_index, row in enumerate(csv.DictReader(handle)):
            episode_id = int(row["episode_id"])
            raw_split = str(row.get("split", "") or "train").strip().lower()
            if raw_split not in REPLAY_SPLITS:
                raise ValueError("source replay split must be train/validation/test")
            split: ReplaySplit = raw_split
            previous_split = episode_splits.get(episode_id)
            if previous_split is not None and previous_split != split:
                raise ValueError("source replay episode appears in multiple splits")
            if previous_split is not None:
                raise ValueError("source replay episode IDs are not unique")
            episode_splits[episode_id] = split
            relative_path = Path(row["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("source replay path escapes the corpus root")
            path = source_root / relative_path
            size_bytes = int(row["size_bytes"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if config.verify_replay_sizes and path.stat().st_size != size_bytes:
                raise ValueError(f"source replay size changed: {path}")
            records.append(
                ReplaySourceRecord(
                    source_index=source_index,
                    date=str(row["date"]),
                    episode_id=episode_id,
                    split=split,
                    relative_path=relative_path.as_posix(),
                    path=path,
                    size_bytes=size_bytes,
                    sha256=_optional_sha256(row.get("sha256")),
                )
            )
            if (
                config.maximum_replays is not None
                and len(records) >= config.maximum_replays
            ):
                break
    if not records:
        raise ValueError("source replay manifest selected no files")
    if config.source_selection == "episode_team_bindings" and any(
        record.episode_id not in episode_team_indices for record in records
    ):
        raise ValueError("episode-team bindings do not cover every source replay")
    identity = ReplayPretrainingDatasetManifest(
        source_manifest_sha256=file_sha256(source_manifest_path),
        replay_manifest_sha256=replay_manifest_sha,
        top_teams_sha256=top_teams_sha,
        episode_teams_sha256=episode_teams_sha,
        source_selection=config.source_selection,
        public_catalog_fingerprint="0" * 64,
        input_contract_fingerprint="0" * 64,
        source_replays=len(records),
        split_assignment_fingerprint=_split_assignment_fingerprint(records),
        split_source_replays={
            split: sum(record.split == split for record in records)
            for split in REPLAY_SPLITS
        },
        split_examples_committed=dict.fromkeys(REPLAY_SPLITS, 0),
        dates=dates,
        teams=teams,
    )
    return (tuple(records), identity)


def extract_replay_dataset(
    config: ReplayPretrainingDataConfig,
    *,
    repo_root: Path,
    catalog_manifest_path: Path,
    input_contract: PolicyInputContract,
    public_catalog_fingerprint: str,
    output_dir: Path,
    target_deck_digest: str | None = None,
    model_config_fingerprint: str | None = None,
    exact_registry_fingerprint: str | None = None,
    event_contract_fingerprint: str | None = None,
    sequence_contract_fingerprint: str | None = None,
    engine_fact_config: ProspectiveEngineFactConfig | None = None,
    engine_fact_fingerprint: str | None = None,
    route_expert_ids: Mapping[str, str] | None = None,
) -> ReplayPretrainingDatasetManifest:
    """Extract all source replays with ordered, bounded parallel workers."""
    records, base_identity = build_replay_source_index(config, repo_root=repo_root)
    temporal_updates: dict[str, object] = {}
    if config.temporal_sequence:
        temporal_contracts = (
            model_config_fingerprint,
            exact_registry_fingerprint,
            event_contract_fingerprint,
            sequence_contract_fingerprint,
        )
        if any(value is None for value in temporal_contracts):
            raise ValueError("temporal extraction requires complete model contracts")
        resolved_engine_fact_fingerprint = engine_fact_producer_fingerprint(
            engine_fact_config
        )
        if resolved_engine_fact_fingerprint != engine_fact_fingerprint:
            raise ValueError("temporal engine-fact producer identity changed")
        temporal_updates = {
            "format": TEMPORAL_PRETRAINING_SHARD_SCHEMA,
            "model_config_fingerprint": model_config_fingerprint,
            "exact_registry_fingerprint": exact_registry_fingerprint,
            "event_contract_fingerprint": event_contract_fingerprint,
            "sequence_contract_fingerprint": sequence_contract_fingerprint,
            "engine_fact_producer_fingerprint": engine_fact_fingerprint,
        }
    elif engine_fact_config is not None or engine_fact_fingerprint is not None:
        raise ValueError("non-temporal extraction cannot configure engine facts")
    identity = base_identity.model_copy(
        update={
            "public_catalog_fingerprint": public_catalog_fingerprint,
            "input_contract_fingerprint": input_contract.fingerprint,
            "target_deck_digest": target_deck_digest,
            **temporal_updates,
        }
    )
    writer = ReplayPretrainingShardWriter(
        output_dir,
        identity=identity,
        shard_rows=config.shard_rows,
        resume=config.resume,
        shard_uncompressed_bytes=config.shard_uncompressed_bytes,
        compression_workers=config.compression_workers,
    )
    if writer.manifest.complete:
        return writer.manifest
    start_cursor = writer.manifest.source_cursor
    if start_cursor > len(records):
        raise ValueError("dataset source cursor exceeds configured replay inventory")
    remaining = records[start_cursor:]
    if not remaining:
        return writer.close()

    settings = _WorkerSettings(
        catalog_manifest_path=catalog_manifest_path,
        input_contract=input_contract.model_dump(mode="python"),
        source_selection=identity.source_selection,
        team_indices={
            team.casefold(): index for index, team in enumerate(identity.teams)
        },
        episode_team_indices=(
            {}
            if config.episode_teams_path is None
            else _read_episode_team_bindings(
                _path(config.episode_teams_path, repo_root),
                team_indices={
                    team.casefold(): index
                    for index, team in enumerate(identity.teams)
                },
            )
        ),
        date_indices={date: index for index, date in enumerate(identity.dates)},
        prefix_bytes=config.prefix_bytes,
        json_chunk_bytes=config.json_chunk_bytes,
        require_done_status=config.require_done_status,
        drop_forced_actions=config.drop_forced_actions,
        target_deck_digest=target_deck_digest,
        temporal_sequence=config.temporal_sequence,
        route_expert_ids=dict(route_expert_ids or {}),
        engine_fact_config=(
            None
            if engine_fact_config is None or not engine_fact_config.enabled
            else engine_fact_config.model_dump(mode="python")
        ),
        engine_fact_producer_fingerprint=engine_fact_fingerprint,
    )
    if config.extraction_batch_replays > 1:
        return _extract_replay_dataset_in_batches(
            config,
            records=records,
            remaining=remaining,
            writer=writer,
            settings=settings,
        )
    started = time.perf_counter()
    completed = start_cursor
    rejected = len(writer.manifest.rejections)
    maximum_ready_results = config.pending_replays * 8
    pending_source_bytes = 0
    ready_result_bytes = 0
    print(
        "extraction_start "
        f"cursor={start_cursor}/{len(records)} "
        f"workers={config.extraction_workers} "
        f"max_inflight={config.pending_replays} "
        f"max_inflight_bytes={config.maximum_inflight_source_bytes} "
        f"max_reorder_buffer={maximum_ready_results} "
        f"max_ready_bytes={config.maximum_ready_result_bytes}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=config.extraction_workers,
        initializer=_initialize_worker,
        initargs=(settings,),
    ) as executor:
        pending: dict[Future[ReplayExtractionOutcome], ReplaySourceRecord] = {}
        ready: dict[int, ReplayExtractionOutcome] = {}
        submit_cursor = 0
        source_exhausted = not remaining
        while pending or ready or not source_exhausted:
            while completed in ready:
                result = ready.pop(completed)
                ready_result_bytes -= _outcome_resident_bytes(result)
                rejected = _commit_extraction_outcome(
                    result,
                    writer=writer,
                    config=config,
                    completed=completed,
                    rejected=rejected,
                    source_replays=len(records),
                    start_cursor=start_cursor,
                    started=started,
                )
                completed += 1
            source_exhausted = submit_cursor >= len(remaining)
            while (
                not source_exhausted
                and len(pending) < config.pending_replays
                and len(ready) < maximum_ready_results
                and ready_result_bytes < config.maximum_ready_result_bytes
            ):
                record = remaining[submit_cursor]
                proposed_bytes = pending_source_bytes + record.size_bytes
                if pending and proposed_bytes > config.maximum_inflight_source_bytes:
                    break
                future = executor.submit(
                    _extract_replay_with_retries,
                    record,
                    config.replay_attempts,
                )
                pending[future] = record
                pending_source_bytes = proposed_bytes
                submit_cursor += 1
                source_exhausted = submit_cursor >= len(remaining)
            if not pending:
                if ready:
                    raise RuntimeError(
                        "parallel extraction has a gap in source replay order"
                    )
                if source_exhausted:
                    break
                continue
            done, _not_done = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            ordered_done = sorted(
                done,
                key=lambda future: pending[future].source_index,
            )
            for future in ordered_done:
                submitted = pending.pop(future)
                pending_source_bytes -= submitted.size_bytes
                result = future.result()
                if result.source_index != submitted.source_index:
                    raise RuntimeError(
                        "parallel extraction worker changed source identity"
                    )
                if result.source_index in ready:
                    raise RuntimeError(
                        "parallel extraction returned a duplicate replay"
                    )
                ready[result.source_index] = result
                ready_result_bytes += _outcome_resident_bytes(result)
    return writer.close()


def _extract_replay_dataset_in_batches(
    config: ReplayPretrainingDataConfig,
    *,
    records: Sequence[ReplaySourceRecord],
    remaining: Sequence[ReplaySourceRecord],
    writer: ReplayPretrainingShardWriter,
    settings: _WorkerSettings,
) -> ReplayPretrainingDatasetManifest:
    """Build compact parts inside worker processes and commit them in order."""
    batch_size = config.extraction_batch_replays
    batches = tuple(
        tuple(remaining[start : start + batch_size])
        for start in range(0, len(remaining), batch_size)
    )
    maximum_pending_batches = max(
        1,
        (config.pending_replays + batch_size - 1) // batch_size,
    )
    start_cursor = writer.manifest.source_cursor
    completed = start_cursor
    rejected = len(writer.manifest.rejections)
    pending_source_bytes = 0
    submit_batch_index = 0
    ready: dict[int, ReplayExtractionBatchResult] = {}
    started = time.perf_counter()
    next_report = ((start_cursor // 100) + 1) * 100
    print(
        "parallel_batch_extraction_start "
        f"cursor={start_cursor}/{len(records)} "
        f"workers={config.extraction_workers} "
        f"batch_replays={batch_size} "
        f"max_inflight_batches={maximum_pending_batches} "
        f"max_inflight_replays={config.pending_replays} "
        f"max_inflight_bytes={config.maximum_inflight_source_bytes}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=config.extraction_workers,
        initializer=_initialize_worker,
        initargs=(settings, writer.manifest, writer.parts_dir),
    ) as executor:
        pending: dict[
            Future[ReplayExtractionBatchResult],
            tuple[ReplaySourceRecord, ...],
        ] = {}
        while pending or ready or submit_batch_index < len(batches):
            while completed in ready:
                result = ready.pop(completed)
                writer.commit_prebuilt_batch(
                    part=result.part,
                    source_replays=result.source_replays,
                    counters=result.counters,
                    rejections=result.rejections,
                )
                rejected += len(result.rejections)
                for rejection in result.rejections:
                    print(
                        "extraction_rejection "
                        f"source_index={rejection.source_index} "
                        f"episode_id={rejection.episode_id} "
                        f"error={rejection.error_kind} "
                        f"attempts={rejection.attempts}",
                        flush=True,
                    )
                completed += result.source_replays
                _enforce_replay_error_budget(
                    config,
                    rejected=rejected,
                    completed=completed,
                )
                if completed >= next_report or completed == len(records):
                    elapsed = max(time.perf_counter() - started, 1.0e-6)
                    print(
                        "parallel_batch_extraction "
                        f"replays={completed}/{len(records)} "
                        f"examples={writer.manifest.examples_committed} "
                        f"parts={len(writer.manifest.parts)} "
                        f"rejected={rejected} "
                        "replays_per_second="
                        f"{(completed - start_cursor) / elapsed:.2f}",
                        flush=True,
                    )
                    next_report = ((completed // 100) + 1) * 100
            while (
                submit_batch_index < len(batches)
                and len(pending) < maximum_pending_batches
            ):
                batch = batches[submit_batch_index]
                batch_bytes = sum(record.size_bytes for record in batch)
                proposed_bytes = pending_source_bytes + batch_bytes
                if (
                    pending
                    and proposed_bytes > config.maximum_inflight_source_bytes
                ):
                    break
                future = executor.submit(
                    _extract_replay_batch_worker,
                    batch,
                    config.replay_attempts,
                )
                pending[future] = batch
                pending_source_bytes = proposed_bytes
                submit_batch_index += 1
            if not pending:
                if ready:
                    raise RuntimeError(
                        "parallel batch extraction has a source-order gap"
                    )
                if submit_batch_index >= len(batches):
                    break
                continue
            done, _not_done = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            for future in sorted(
                done,
                key=lambda item: pending[item][0].source_index,
            ):
                batch = pending.pop(future)
                pending_source_bytes -= sum(record.size_bytes for record in batch)
                result = future.result()
                if (
                    result.start_source_index != batch[0].source_index
                    or result.source_replays != len(batch)
                    or result.start_source_index in ready
                ):
                    raise RuntimeError(
                        "parallel batch worker changed source ownership"
                    )
                ready[result.start_source_index] = result
    return writer.close()


def _commit_extraction_outcome(
    result: ReplayExtractionOutcome,
    *,
    writer: ReplayPretrainingShardWriter,
    config: ReplayPretrainingDataConfig,
    completed: int,
    rejected: int,
    source_replays: int,
    start_cursor: int,
    started: float,
) -> int:
    """Commit one contiguous result while preserving deterministic source order."""
    if result.source_index != completed:
        raise RuntimeError("parallel extraction changed source replay order")
    if isinstance(result, ReplayExtractionFailure):
        writer.add_replay(
            (),
            counters=result.counters,
            rejection=result.rejection,
        )
        writer.flush()
        rejected += 1
        print(
            "extraction_rejection "
            f"source_index={result.source_index} "
            f"episode_id={result.rejection.episode_id} "
            f"error={result.rejection.error_kind} "
            f"attempts={result.rejection.attempts}",
            flush=True,
        )
    else:
        writer.add_replay(result.examples, counters=result.counters)
    next_cursor = completed + 1
    _enforce_replay_error_budget(
        config,
        rejected=rejected,
        completed=next_cursor,
    )
    if next_cursor % 100 == 0 or next_cursor == source_replays:
        elapsed = max(time.perf_counter() - started, 1.0e-6)
        print(
            "extraction "
            f"replays={next_cursor}/{source_replays} "
            "examples="
            f"{writer.manifest.examples_committed + writer.buffered_examples} "
            f"rejected={rejected} "
            f"part_bytes={writer.buffered_uncompressed_bytes} "
            f"replays_per_second={(next_cursor - start_cursor) / elapsed:.2f}",
            flush=True,
        )
    return rejected


def _initialize_worker(
    settings: _WorkerSettings,
    dataset_identity: ReplayPretrainingDatasetManifest | None = None,
    parts_dir: Path | None = None,
) -> None:
    global _WORKER_CATALOG
    global _WORKER_CONTRACT
    global _WORKER_DATASET_IDENTITY
    global _WORKER_ENGINE_FACT_PRODUCER
    global _WORKER_PARTS_DIR
    global _WORKER_SETTINGS
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    _WORKER_SETTINGS = settings
    _WORKER_DATASET_IDENTITY = dataset_identity
    _WORKER_PARTS_DIR = parts_dir
    _WORKER_CATALOG, _manifest = load_public_deck_catalog(
        settings.catalog_manifest_path
    )
    _WORKER_CONTRACT = PolicyInputContract.model_validate(settings.input_contract)
    _WORKER_ENGINE_FACT_PRODUCER = None
    if settings.engine_fact_config is not None:
        fact_config = ProspectiveEngineFactConfig.model_validate(
            settings.engine_fact_config
        )
        _WORKER_ENGINE_FACT_PRODUCER = _build_engine_fact_producer(fact_config)
        if (
            _WORKER_ENGINE_FACT_PRODUCER.fingerprint
            != settings.engine_fact_producer_fingerprint
        ):
            raise ValueError("extraction worker engine-fact producer identity changed")


def _extract_replay_batch_worker(
    records: tuple[ReplaySourceRecord, ...],
    attempts: int,
) -> ReplayExtractionBatchResult:
    """Extract and compress a contiguous replay batch inside one worker."""
    if (
        not records
        or _WORKER_DATASET_IDENTITY is None
        or _WORKER_PARTS_DIR is None
    ):
        raise RuntimeError("parallel batch worker is not initialized")
    expected_indices = tuple(
        range(records[0].source_index, records[0].source_index + len(records))
    )
    if tuple(record.source_index for record in records) != expected_indices:
        raise ValueError("parallel extraction batch is not source-contiguous")
    examples: list[ReplayPretrainingExample] = []
    counters: Counter[str] = Counter()
    rejections: list[RejectedReplayRecord] = []
    accepted_source_replays = 0
    for record in records:
        outcome = _extract_replay_with_retries(record, attempts)
        counters.update(outcome.counters)
        if isinstance(outcome, ReplayExtractionFailure):
            rejections.append(outcome.rejection)
            continue
        accepted_source_replays += 1
        examples.extend(outcome.examples)
    part = (
        None
        if not examples
        else write_pretraining_part(
            tuple(examples),
            identity=_WORKER_DATASET_IDENTITY,
            part_index=records[0].source_index,
            parts_dir=_WORKER_PARTS_DIR,
            source_replays=accepted_source_replays,
        )
    )
    return ReplayExtractionBatchResult(
        start_source_index=records[0].source_index,
        source_replays=len(records),
        part=part,
        counters=dict(counters),
        rejections=tuple(rejections),
    )


def _extract_replay_with_retries(
    record: ReplaySourceRecord,
    attempts: int,
) -> ReplayExtractionOutcome:
    """Retry replay-level data failures without masking process/code failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = _extract_replay_worker(record)
        except (
            EOFError,
            IndexError,
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as error:
            last_error = error
            continue
        counters = Counter(result.counters)
        if attempt > 1:
            counters["replay_retries"] += attempt - 1
            counters["replays_recovered_after_retry"] += 1
        return replace(result, counters=dict(counters))
    if last_error is None:
        raise RuntimeError("replay extraction attempts were not executed")
    observed_sha256: str | None
    try:
        observed_sha256 = file_sha256(record.path)
    except OSError:
        observed_sha256 = None
    kind = _exception_kind(last_error)
    rejection = RejectedReplayRecord(
        source_index=record.source_index,
        episode_id=record.episode_id,
        date=record.date,
        split=record.split,
        relative_path=record.relative_path,
        size_bytes=record.size_bytes,
        expected_sha256=record.sha256,
        observed_sha256=observed_sha256,
        attempts=attempts,
        error_kind=kind,
        error_message=_clean_error_message(last_error, record=record),
    )
    return ReplayExtractionFailure(
        source_index=record.source_index,
        rejection=rejection,
        counters={
            "replays_rejected": 1,
            "replay_retries": max(attempts - 1, 0),
            f"replay_error_{kind}": 1,
        },
    )


def _extract_replay_worker(record: ReplaySourceRecord) -> ReplayExtractionResult:
    settings = _require_worker_settings()
    if _WORKER_CONTRACT is None or _WORKER_CATALOG is None:
        raise RuntimeError("pretraining extraction worker is not initialized")
    source_sha256 = file_sha256(record.path)
    if record.sha256 is not None and source_sha256 != record.sha256:
        raise ValueError(f"source replay SHA-256 changed: {record.path}")
    side_rows = fast_episode_side_rows(
        replay_path=record.path,
        card_meta={},
        known_decks={},
        prefix_bytes=settings.prefix_bytes,
        include_step_count=False,
    )
    if side_rows is None or len(side_rows) != 2:
        raise ValueError(f"cannot resolve exact replay decks: {record.path}")
    metadata = {int(row["player_index"]): row for row in side_rows}
    selected: dict[int, Mapping[str, Any]] = {}
    counters: Counter[str] = Counter(
        {
            "replays_seen": 1,
            f"replays_{record.split}": 1,
        }
    )
    for player_index, row in metadata.items():
        team_key = str(row.get("team_name", "")).casefold()
        team_index = _source_team_index(
            team_key,
            settings,
            episode_id=record.episode_id,
        )
        if team_index is None:
            continue
        counters["selected_source_seats_seen"] += 1
        if settings.require_done_status and str(row.get("status", "")) != "DONE":
            counters["non_done_seats_dropped"] += 1
            continue
        reward = row.get("reward")
        if reward is None or float(reward) not in (-1.0, 0.0, 1.0):
            raise ValueError(f"replay has invalid terminal reward: {record.path}")
        if (
            len(cast(Sequence[int], row.get("deck_ids", ()))) != 60
            or len(cast(Sequence[int], row.get("opponent_deck_ids", ()))) != 60
        ):
            raise ValueError(f"replay is missing a registered deck: {record.path}")
        if not _matches_target_deck(
            cast(Sequence[int], row["deck_ids"]),
            settings.target_deck_digest,
        ):
            counters["eligible_foreign_deck_seats_dropped"] += 1
            continue
        selected[player_index] = row
    if not selected:
        counters["replays_without_trainable_seat"] += 1
        return ReplayExtractionResult(
            source_index=record.source_index,
            examples=(),
            counters=dict(counters),
        )

    adapters = {
        player_index: SimpleStatelessPublicInputAdapter(
            _WORKER_CATALOG,
            contract=_WORKER_CONTRACT,
            player_index=player_index,
            own_deck=tuple(int(value) for value in row["deck_ids"]),
            engine_fact_producer=cast(
                ProspectiveEngineFactProducer | None,
                _WORKER_ENGINE_FACT_PRODUCER,
            ),
        )
        for player_index, row in selected.items()
    }
    pending: dict[int, _PendingDecision] = {}
    examples_by_seat: dict[int, list[ReplayPretrainingExample]] = {
        player_index: [] for player_index in selected
    }
    for step_index, sides in iter_replay_steps(
        record.path,
        chunk_size=settings.json_chunk_bytes,
    ):
        counters["steps_seen"] += 1
        for player_index in selected:
            if player_index >= len(sides):
                raise ValueError(f"replay step is missing a seat: {record.path}")
            side = sides[player_index]
            waiting = pending.pop(player_index, None)
            if waiting is not None:
                action = _integer_action(side.get("action"))
                if action is None or not is_legal_action(waiting.select, action):
                    raise ValueError(
                        "replay action is not legal for its pending prompt: "
                        f"{record.path}:{waiting.step_index}:{player_index}"
                    )
                action = _canonical_action(waiting.select, action)
                if waiting.forced and settings.drop_forced_actions:
                    counters["forced_actions_dropped"] += 1
                else:
                    actor_row = waiting.actor_row
                    if actor_row is None:
                        actor_row = adapters[player_index].tensorize_observed(
                            waiting.observation,
                            context=waiting.context,
                        )
                    accepted_action = None
                    if settings.temporal_sequence:
                        if (
                            waiting.event_token is None
                            or waiting.decision_index is None
                        ):
                            raise RuntimeError(
                                "temporal pending decision has no transaction"
                            )
                        accepted_action = build_accepted_action_record(
                            state=actor_row.state,
                            options=actor_row.options,
                            action=action,
                            min_count=actor_row.min_count,
                            max_count=actor_row.max_count,
                            stop_sampled=_stop_sampled(actor_row, action),
                        )
                        committed_delta = adapters[player_index].commit_decision(
                            waiting.event_token
                        )
                        if committed_delta != actor_row.public_event_delta:
                            raise RuntimeError(
                                "temporal event delta changed before commit"
                            )
                        eligible_options = len(
                            core_option_candidates(waiting.select)
                        )
                        if eligible_options:
                            counters["engine_fact_eligible_roots"] += 1
                            counters["engine_fact_eligible_options"] += (
                                eligible_options
                            )
                        if actor_row.engine_fact_producer_fingerprint is not None:
                            counters["engine_fact_bound_rows"] += 1
                        counters["engine_fact_resolved_options"] += sum(
                            bool(value)
                            for value in actor_row.options.dynamic_effect_masks
                        )
                    metadata_row = selected[player_index]
                    opponent_deck = tuple(
                        int(value)
                        for value in cast(
                            Sequence[int],
                            metadata_row["opponent_deck_ids"],
                        )
                    )
                    _validate_public_target(
                        opponent_deck,
                        waiting.known_opponent_counts,
                    )
                    examples_by_seat[player_index].append(
                        ReplayPretrainingExample(
                            episode_id=record.episode_id,
                            player_index=player_index,
                            step_index=waiting.step_index,
                            team_index=_required_source_team_index(
                                str(metadata_row["team_name"]).casefold(),
                                settings,
                                episode_id=record.episode_id,
                            ),
                            date_index=settings.date_indices[record.date],
                            replay_index=record.source_index,
                            split=record.split,
                            actor_row=actor_row,
                            action=action,
                            opponent_deck=opponent_deck,
                            known_opponent_counts=(waiting.known_opponent_counts),
                            outcome=float(metadata_row["reward"]),
                            example_weight=1.0,
                            decision_index=waiting.decision_index,
                            source_replay_sha256=(
                                source_sha256 if settings.temporal_sequence else None
                            ),
                            route_expert_id=(
                                settings.route_expert_ids.get(
                                    actor_row.own_deck.deck_digest
                                )
                                if settings.temporal_sequence
                                else None
                            ),
                            accepted_action=accepted_action,
                        )
                    )
                    counters["examples"] += 1
                    counters[f"examples_{record.split}"] += 1

            if str(side.get("status", "")) != "ACTIVE":
                continue
            observation = _mapping(side.get("observation"))
            select = _mapping(observation.get("select"))
            if not select:
                continue
            context = adapters[player_index].observe(observation)
            forced = is_forced(select)
            actor_row = None
            event_token = None
            decision_index = None
            if settings.temporal_sequence and not forced:
                event_token = adapters[player_index].prepare_decision()
                actor_row = adapters[player_index].tensorize_observed(
                    observation,
                    context=context,
                )
                if actor_row.public_event_delta != event_token.delta:
                    raise RuntimeError(
                        "tensorized temporal events differ from prepared events"
                    )
                decision_index = len(examples_by_seat[player_index])
            pending[player_index] = _PendingDecision(
                observation=observation,
                context=context,
                select=select,
                step_index=step_index,
                known_opponent_counts=(adapters[player_index].known_opponent_counts),
                forced=forced,
                actor_row=actor_row,
                event_token=event_token,
                decision_index=decision_index,
            )
            counters["active_select_observations"] += 1
    if pending:
        raise ValueError(f"replay ended with pending select prompts: {record.path}")

    normalized: list[ReplayPretrainingExample] = []
    for seat_examples in examples_by_seat.values():
        if not seat_examples:
            continue
        weight = 1.0 / float(len(seat_examples))
        normalized.extend(
            replace(example, example_weight=weight) for example in seat_examples
        )
    counters["trainable_seats"] += sum(
        bool(seat_examples) for seat_examples in examples_by_seat.values()
    )
    return ReplayExtractionResult(
        source_index=record.source_index,
        examples=tuple(normalized),
        counters=dict(counters),
        resident_bytes=_resident_size(normalized),
    )


def _enforce_replay_error_budget(
    config: ReplayPretrainingDataConfig,
    *,
    rejected: int,
    completed: int,
) -> None:
    """Fail closed when isolated replay errors indicate a systemic problem."""
    if rejected > config.maximum_replay_errors:
        raise RuntimeError(
            "pretraining replay errors exceeded the absolute budget: "
            f"{rejected}>{config.maximum_replay_errors}"
        )
    if (
        completed >= config.replay_error_fraction_minimum_cursor
        and rejected / completed > config.maximum_replay_error_fraction
    ):
        raise RuntimeError(
            "pretraining replay errors exceeded the fractional budget: "
            f"{rejected}/{completed}>"
            f"{config.maximum_replay_error_fraction:.6f}"
        )


def _split_assignment_fingerprint(
    records: Sequence[ReplaySourceRecord],
) -> str:
    """Bind every source episode to exactly one durable split."""
    payload = tuple(
        (record.episode_id, record.split)
        for record in sorted(records, key=lambda item: item.episode_id)
    )
    return hashlib.sha256(
        b"ptcg-rl/simple-stateless-pretraining-splits/v1\x00"
        + json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _matches_target_deck(
    deck_ids: Sequence[int],
    target_deck_digest: str | None,
) -> bool:
    """Match a seat by canonical exact-deck identity when target-only."""
    return (
        target_deck_digest is None
        or canonicalize_deck(deck_ids).deck_digest == target_deck_digest
    )


def _exception_kind(error: Exception) -> str:
    """Return a stable snake-case exception category for the manifest."""
    name = type(error).__name__
    characters: list[str] = []
    for index, character in enumerate(name):
        if character.isupper() and index > 0:
            characters.append("_")
        characters.append(character.lower())
    return "".join(characters)


def _clean_error_message(
    error: Exception,
    *,
    record: ReplaySourceRecord,
) -> str:
    """Bound messages and replace host-specific source paths."""
    message = str(error).replace(str(record.path), record.relative_path).strip()
    if not message:
        message = type(error).__name__
    return message[:2048]


def _canonical_action(
    select: Mapping[str, Any],
    action: tuple[int, ...],
) -> tuple[int, ...]:
    options = select.get("option")
    option_count = (
        len(options)
        if isinstance(options, Sequence) and not isinstance(options, str)
        else 0
    )
    minimum = min(option_count, max(0, int(select.get("minCount", 0))))
    maximum = min(
        option_count,
        max(minimum, int(select.get("maxCount", option_count))),
    )
    if is_unordered_set_selection(
        context=int(select.get("context", -1)),
        min_count=minimum,
        max_count=maximum,
    ):
        return tuple(sorted(action))
    return normalize_action_order(select, action)


def _stop_sampled(
    row: SimpleStatelessActorRow,
    action: tuple[int, ...],
) -> bool:
    """Mirror the production complete-action STOP semantics."""
    unordered = row.min_count < row.max_count and any(
        int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        for context in row.options.contexts
    )
    return not unordered and len(action) < row.max_count


def _validate_public_target(
    opponent_deck: tuple[int, ...],
    known_counts: tuple[tuple[int, int], ...],
) -> None:
    exact = Counter(opponent_deck)
    if any(exact[card_id] < count for card_id, count in known_counts):
        raise ValueError("public evidence exceeds the exact opponent deck")


def _read_top_teams(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda row: int(row["Rank"]),
        )
    teams = tuple(str(row["TeamName"]).strip() for row in rows)
    if not teams or any(not team for team in teams):
        raise ValueError("Top team snapshot contains an empty team name")
    if len({team.casefold() for team in teams}) != len(teams):
        raise ValueError("Top team snapshot names are not unique")
    return teams


def _read_episode_team_bindings(
    path: Path,
    *,
    team_indices: Mapping[str, int],
) -> dict[int, dict[str, int]]:
    """Bind each source episode to only its audited submission-side names."""
    bindings: dict[int, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if tuple(rows.fieldnames or ()) != (
            "episode_id",
            "submission_id",
            "team_name",
        ):
            raise ValueError("episode-team binding columns are invalid")
        for row in rows:
            episode_id = int(row["episode_id"])
            team_key = str(row["team_name"]).strip().casefold()
            if not team_key or team_key not in team_indices:
                raise ValueError("episode-team binding names an unknown team")
            episode = bindings.setdefault(episode_id, {})
            previous = episode.setdefault(team_key, team_indices[team_key])
            if previous != team_indices[team_key]:
                raise ValueError("episode-team binding changed a team index")
    if not bindings:
        raise ValueError("episode-team binding file is empty")
    return bindings


def _source_team_index(
    team_key: str,
    settings: _WorkerSettings,
    *,
    episode_id: int,
) -> int | None:
    """Resolve one seat to its durable source-selection stratum."""
    if settings.source_selection == "all_sides":
        return 0
    if settings.source_selection == "episode_team_bindings":
        return settings.episode_team_indices.get(episode_id, {}).get(team_key)
    return settings.team_indices.get(team_key)


def _required_source_team_index(
    team_key: str,
    settings: _WorkerSettings,
    *,
    episode_id: int,
) -> int:
    """Return the already-validated source stratum for a selected seat."""
    index = _source_team_index(team_key, settings, episode_id=episode_id)
    if index is None:
        raise RuntimeError("selected replay seat no longer matches source selection")
    return index


def _optional_sha256(value: object) -> str | None:
    """Normalize an optional replay hash from a source inventory."""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source replay SHA-256 is malformed")
    return normalized


def _outcome_resident_bytes(result: ReplayExtractionOutcome) -> int:
    """Return the bounded queue accounting size for one worker result."""
    if isinstance(result, ReplayExtractionResult):
        return result.resident_bytes
    return _resident_size(result)


def _resident_size(value: object, seen: set[int] | None = None) -> int:
    """Estimate one worker result's resident bytes for backpressure."""
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return size + sum(
            _resident_size(key, visited) + _resident_size(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_resident_size(item, visited) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return size + _resident_size(attributes, visited)
    return size


def _require_worker_settings() -> _WorkerSettings:
    if _WORKER_SETTINGS is None:
        raise RuntimeError("pretraining extraction worker is not initialized")
    return _WORKER_SETTINGS


def _integer_action(value: Any) -> tuple[int, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(type(item) is int for item in value)
    ):
        return None
    return tuple(int(item) for item in value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path(value: Path, repo_root: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


__all__ = [
    "ReplayExtractionResult",
    "ReplaySourceRecord",
    "build_replay_source_index",
    "engine_fact_producer_fingerprint",
    "extract_replay_dataset",
]
