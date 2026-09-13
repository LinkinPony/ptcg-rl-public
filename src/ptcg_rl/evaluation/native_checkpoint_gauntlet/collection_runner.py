"""Training-path native collection backend for cross-checkpoint evaluation."""

from __future__ import annotations

import gc
import hashlib
import multiprocessing as mp
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.evaluation.native_checkpoint_gauntlet.interventions import (
    InterventionSequenceActor,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    RESULTS_FORMAT,
    NativeCheckpointGauntletConfig,
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.replica_partition import (
    partition_replica_chunks,
    partition_replica_items,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.runner import (
    _campaign_payload,
    _fingerprint,
    _load_decks,
    _load_parts,
    _verify_participant,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.schedule import (
    schedule_cross_checkpoint_games,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.scoring import (
    write_final_artifacts,
)
from ptcg_rl.evaluation.native_deck_elo.storage import (
    publish_or_validate_manifest,
    resolve_path,
    verified_file,
    write_part,
    write_progress,
)
from ptcg_rl.model.simple_stateless import (
    materialize_simple_stateless_checkpoint_model,
)
from ptcg_rl.rl.native_banked_route_collection import (
    collect_native_banked_assigned,
)
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_stateless_collection import NativeStatelessCollector
from ptcg_rl.rl.stateless_checkpoint import (
    LoadedStatelessPolicyCheckpoint,
    StatelessPolicyIdentity,
    load_stateless_policy_checkpoint,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_curriculum import CurriculumAssignment, PfspMember
from ptcg_rl.rl.stateless_deck_balance import DeckSeatAssignment
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_opponents import PastSelfPolicyPool


def run_native_collection_checkpoint_gauntlet(
    config: NativeCheckpointGauntletConfig,
) -> dict[str, Any]:
    """Run a durable campaign through the formal training collection core."""
    root = records.repo_path(Path(".")).resolve()
    checkpoint_paths = {
        "candidate": _verify_participant(config.candidate, root=root),
        "baseline": _verify_participant(config.baseline, root=root),
    }
    library_path = verified_file(
        config.native_library_path,
        root=root,
        expected_sha256=config.expected_native_library_sha256,
    )
    candidate_decks = _load_decks(config.candidate, root=root)
    baseline_decks = _load_decks(config.baseline, root=root)
    campaign = _campaign_payload(
        config,
        candidate_decks=candidate_decks,
        baseline_decks=baseline_decks,
        runtime_fingerprint=_execution_fingerprint(config),
        belief_fingerprint=_catalog_pair_fingerprint(config),
    )
    campaign_fingerprint = _fingerprint(campaign)
    games = schedule_cross_checkpoint_games(
        candidate_decks,
        baseline_decks,
        total_games=config.total_games,
        seed=config.seed,
        campaign_fingerprint=campaign_fingerprint,
        match_seed_namespace=config.match_seed_namespace,
    )
    output_dir = resolve_path(config.output_dir, root=root)
    parts_dir = output_dir / "games_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    publish_or_validate_manifest(
        output_dir / "manifest.json",
        {**campaign, "campaign_fingerprint": campaign_fingerprint},
    )
    rows = _load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    completed = {int(row["game_index"]) for row in rows}
    remaining = tuple(game for game in games if game.game_index not in completed)
    started_clock = time.perf_counter()
    resumed_games = len(rows)
    progress_path = output_dir / "progress.json"
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=not remaining,
    )

    if remaining:
        _run_native_worker_group(
            config,
            root=root,
            library_path=library_path,
            checkpoint_paths=checkpoint_paths,
            candidate_decks=candidate_decks,
            baseline_decks=baseline_decks,
            games=remaining,
            parts_dir=parts_dir,
            campaign_fingerprint=campaign_fingerprint,
            progress_path=progress_path,
            total_games=len(games),
            resumed_games=resumed_games,
            started_clock=started_clock,
        )

    rows = _load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    if len(rows) != len(games):
        raise RuntimeError(
            f"checkpoint gauntlet retained {len(rows)} games, expected {len(games)}"
        )
    summary = write_final_artifacts(
        config,
        rows=rows,
        output_dir=output_dir,
        campaign_fingerprint=campaign_fingerprint,
        elapsed_seconds=time.perf_counter() - started_clock,
        resumed_games=resumed_games,
        execution_telemetry=_execution_telemetry(rows),
    )
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=True,
    )
    return summary


def _run_native_worker_group(
    config: NativeCheckpointGauntletConfig,
    *,
    root: Path,
    library_path: Path,
    checkpoint_paths: Mapping[str, Path],
    candidate_decks: Sequence[Any],
    baseline_decks: Sequence[Any],
    games: Sequence[ScheduledCrossCheckpointGame],
    parts_dir: Path,
    campaign_fingerprint: str,
    progress_path: Path,
    total_games: int,
    resumed_games: int,
    started_clock: float,
) -> None:
    """Run persistent CUDA replicas against one balanced shared work queue."""
    worker_count = min(config.native_worker_replicas, len(games))
    chunks = partition_replica_chunks(
        games,
        chunk_size=config.collection_part_games,
        wave_size=worker_count,
    )
    worker_count = min(worker_count, len(chunks))
    context = mp.get_context("spawn")
    work_queue = context.Queue()
    processes = tuple(
        context.Process(
            target=_native_collection_replica_main,
            args=(
                config,
                worker_index,
                root,
                library_path,
                dict(checkpoint_paths),
                tuple(candidate_decks),
                tuple(baseline_decks),
                work_queue,
                parts_dir,
                campaign_fingerprint,
            ),
            name=f"native-gauntlet-worker-{worker_index}",
        )
        for worker_index in range(worker_count)
    )
    try:
        for process in processes:
            process.start()
        for chunk in chunks:
            work_queue.put(chunk)
        for _worker_index in range(worker_count):
            work_queue.put(None)
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.0)
                if process.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"native gauntlet worker {process.name} failed with "
                        f"exit code {process.exitcode}"
                    )
            write_progress(
                progress_path,
                total_games=total_games,
                completed_games=_committed_game_count(parts_dir),
                resumed_games=resumed_games,
                started_clock=started_clock,
                complete=False,
            )
            time.sleep(5.0)
        failed = tuple(
            process for process in processes if process.exitcode not in (None, 0)
        )
        if failed:
            details = ", ".join(
                f"{process.name}={process.exitcode}" for process in failed
            )
            raise RuntimeError(f"native gauntlet worker group failed: {details}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        join_deadline = time.monotonic() + 30.0
        for process in processes:
            process.join(timeout=max(0.0, join_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        kill_deadline = time.monotonic() + 5.0
        for process in processes:
            if process.is_alive():
                process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
        work_queue.cancel_join_thread()
        work_queue.close()


def _native_collection_replica_main(
    config: NativeCheckpointGauntletConfig,
    worker_index: int,
    root: Path,
    library_path: Path,
    checkpoint_paths: Mapping[str, Path],
    candidate_decks: Sequence[Any],
    baseline_decks: Sequence[Any],
    work_queue: Any,
    parts_dir: Path,
    campaign_fingerprint: str,
) -> None:
    """Own one model pair and consume balanced cohorts until the queue drains."""
    if config.native_inductor_compile_threads is not None:
        torch._inductor.config.compile_threads = (
            config.native_inductor_compile_threads
        )
    torch.set_num_threads(config.native_worker_torch_threads)
    torch.set_num_interop_threads(config.native_worker_torch_threads)
    runtime: _CollectionRuntime | None = None
    try:
        runtime = _CollectionRuntime(
            config,
            root=root,
            library_path=library_path,
            checkpoint_paths=checkpoint_paths,
            candidate_decks=candidate_decks,
            baseline_decks=baseline_decks,
        )
        while True:
            cohort = work_queue.get()
            if cohort is None:
                break
            before_rows = runtime.actor.intervention_rows
            before_resets = runtime.actor.history_reset_rows
            result = runtime.collect(cohort)
            part_rows = _result_rows(
                cohort,
                result.outcomes,
                report=result.report,
                campaign_fingerprint=campaign_fingerprint,
                candidate_label=config.candidate.label,
                baseline_label=config.baseline.label,
            )
            first_game = min(game.game_index for game in cohort)
            for index, row in enumerate(part_rows):
                row["candidate_intervention"] = config.candidate_intervention
                row["intervention_rows"] = (
                    runtime.actor.intervention_rows - before_rows if index == 0 else 0
                )
                row["history_reset_rows"] = (
                    runtime.actor.history_reset_rows - before_resets if index == 0 else 0
                )
            write_part(
                parts_dir
                / f"part-worker-{worker_index:02d}-game-{first_game:08d}.parquet",
                part_rows,
                compression=config.compression,
            )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if runtime is not None:
            runtime.close()


def _partition_replica_games(
    games: Sequence[ScheduledCrossCheckpointGame],
    *,
    workers: int,
) -> tuple[tuple[ScheduledCrossCheckpointGame, ...], ...]:
    """Stripe deterministic schedule rows evenly across persistent replicas."""
    return partition_replica_items(games, workers=workers)


def _committed_game_count(parts_dir: Path) -> int:
    """Read only Parquet footers while workers atomically publish parts."""
    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in parts_dir.glob("part-*.parquet")
    )


class _CollectionRuntime:
    """Own two immutable models and reusable training-path collection services."""

    def __init__(
        self,
        config: NativeCheckpointGauntletConfig,
        *,
        root: Path,
        library_path: Path,
        checkpoint_paths: Mapping[str, Path],
        candidate_decks: Sequence[Any],
        baseline_decks: Sequence[Any],
    ) -> None:
        self.config = config
        candidate = load_stateless_policy_checkpoint(checkpoint_paths["candidate"])
        if config.candidate.expected_checkpoint_sha256 == (
            config.baseline.expected_checkpoint_sha256
        ):
            baseline = candidate
        else:
            baseline = load_stateless_policy_checkpoint(checkpoint_paths["baseline"])
        _validate_loaded_checkpoint(
            candidate, config.candidate.expected_checkpoint_sha256
        )
        _validate_loaded_checkpoint(
            baseline, config.baseline.expected_checkpoint_sha256
        )
        self.identity = _fragment_identity(candidate)
        baseline_identity = _fragment_identity(baseline)
        candidate_catalog, _ = load_public_deck_catalog(
            (root / config.candidate.public_catalog_manifest_path).resolve()
        )
        baseline_catalog, _ = load_public_deck_catalog(
            (root / config.baseline.public_catalog_manifest_path).resolve()
        )
        if candidate_catalog.fingerprint != (
            candidate.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("candidate checkpoint differs from its public catalog")
        if baseline_catalog.fingerprint != (
            baseline.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("baseline checkpoint differs from its public catalog")
        candidate_sequence = candidate.model_config_value.sequence
        baseline_sequence = baseline.model_config_value.sequence
        if candidate_sequence is None or baseline_sequence is None:
            raise ValueError("native collection gauntlet requires sequence checkpoints")
        if candidate_sequence.engine_facts != baseline_sequence.engine_facts:
            raise ValueError("cross-checkpoint engine-fact contracts differ")

        model = materialize_simple_stateless_checkpoint_model(
            candidate.model_config_value,
            candidate.model_state,
        )
        # The checkpoint bytes and FP32 model state were already authenticated.
        # Convert on CPU so evaluation follows the distributed worker's
        # single-resident BF16 path instead of briefly owning FP32 + BF16 on CUDA.
        model.to(dtype=torch.bfloat16)
        self.actor = InterventionSequenceActor(
            model,
            identity=self.identity,
            device=config.device,
            verify_model_state=False,
            retain_raw_blocks=False,
            temporal_cache_slots=config.native_arena_capacity,
            rollout_precision="bf16",
        )
        self.actor.intervention = config.candidate_intervention

        self.candidate_decks = _canonical_deck_map(candidate_decks)
        self.baseline_decks = _canonical_deck_map(baseline_decks)
        self.members = _baseline_members(baseline, self.baseline_decks)
        self.past_self = PastSelfPolicyPool(
            device=config.device,
            fragment_horizon=baseline_identity.horizon,
            archive_sequence_models_on_cpu=True,
        )
        prepared = self.past_self.prepare(
            self.members,
            preloaded={baseline.artifact.policy_path: baseline},
        )
        prepared.commit()
        fact_config = candidate_sequence.engine_facts
        fact_producer = ProspectiveEngineFactProducer(
            sampler=BeliefSampler(config=fact_config.sampler),
            config=fact_config,
        )
        self.collector = NativeStatelessCollector(
            actor=self.actor,
            identity=self.identity,
            catalog=candidate_catalog,
            active_decks=self.candidate_decks,
            opponent_decks=self.baseline_decks,
            members=self.members,
            past_self_pool=self.past_self,
            historical_pool=NativeHistoricalPolicyPool({}),
            scripted_policies={},
            scripted_bindings={},
            maximum_engine_steps=config.maximum_engine_steps,
            seed=config.seed,
            fragments_per_part=64,
            mirror_bilateral_trajectories=False,
            arena_capacity=config.native_arena_capacity,
            engine_shards=config.native_engine_shards,
            policy_cohort_slots=config.native_policy_cohort_slots,
            policy_group_bank_limit=config.native_policy_group_bank_limit,
            policy_cohort_wait_ms=config.native_policy_cohort_wait_ms,
            frozen_batch_min_rows=config.native_frozen_batch_min_rows,
            frozen_batch_max_wait_waves=(config.native_frozen_batch_max_wait_waves),
            sequence_rollout_precision="bf16",
            library_path=library_path,
            engine_fact_producer=fact_producer,
            engine_fact_workers=config.native_engine_fact_workers,
            route_input_contracts={
                baseline.artifact.policy_sha256: (
                    baseline_catalog,
                    baseline.identity.input_contract_fingerprint,
                )
            },
            retain_trajectories=False,
            current_policy_temperature=config.candidate.policy_temperature,
            frozen_policy_temperatures={
                baseline.artifact.policy_sha256: config.baseline.policy_temperature
            },
            evaluation_action_only=config.evaluation_action_only,
        )
        self.member_by_deck = {
            member.exact_deck_digest: member for member in self.members
        }
        self._closed = False

    def collect(self, games: Sequence[ScheduledCrossCheckpointGame]) -> Any:
        """Collect one durable result part without rebuilding either model."""
        assignments = tuple(
            _assignment(
                game,
                member=self.member_by_deck[game.baseline_deck.deck_digest],
            )
            for game in games
        )
        return collect_native_banked_assigned(self.collector, assignments)

    def close(self) -> None:
        """Release CUDA actors and the archived baseline source once."""
        if self._closed:
            return
        self._closed = True
        try:
            self.collector.close()
        finally:
            self.past_self.unload(tuple(member.member_id for member in self.members))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _validate_loaded_checkpoint(
    loaded: LoadedStatelessPolicyCheckpoint,
    expected_sha256: str,
) -> None:
    if loaded.artifact.policy_sha256 != expected_sha256:
        raise ValueError("loaded checkpoint bytes differ from frozen identity")


def _fragment_identity(
    loaded: LoadedStatelessPolicyCheckpoint,
) -> StatelessFragmentIdentity:
    identity: StatelessPolicyIdentity = loaded.identity
    for horizon in range(1, 4_097):
        result = StatelessFragmentIdentity(
            schema_version=(
                2 if identity.sequence_contract_fingerprint is not None else 1
            ),
            horizon=horizon,
            behavior_policy_version=loaded.artifact.version,
            behavior_policy_fingerprint=loaded.artifact.policy_model_fingerprint,
            model_config_fingerprint=identity.model_config_fingerprint,
            action_schema_fingerprint=identity.action_schema_fingerprint,
            public_context_fingerprint=identity.public_context_fingerprint,
            card_catalog_fingerprint=identity.card_catalog_fingerprint,
            public_deck_catalog_fingerprint=(identity.public_deck_catalog_fingerprint),
            exact_registry_fingerprint=identity.exact_registry_fingerprint,
            belief_target_semantics_fingerprint=(
                identity.belief_target_semantics_fingerprint
            ),
            input_contract_fingerprint=identity.input_contract_fingerprint,
            resolved_config_fingerprint=identity.resolved_config_fingerprint,
            sequence_contract_fingerprint=identity.sequence_contract_fingerprint,
        )
        if result.static_contract_fingerprint == (
            identity.fragment_static_contract_fingerprint
        ):
            return result
    raise ValueError("checkpoint fragment horizon could not be recovered")


def _canonical_deck_map(decks: Sequence[Any]) -> dict[str, CanonicalDeck]:
    return {
        deck.deck_digest: canonicalize_deck(records.read_deck(deck.path))
        for deck in decks
    }


def _baseline_members(
    baseline: LoadedStatelessPolicyCheckpoint,
    decks: Mapping[str, CanonicalDeck],
) -> tuple[PfspMember, ...]:
    artifact = baseline.artifact
    members = []
    for deck_digest in sorted(decks):
        member_id = f"gauntlet-{deck_digest}"
        bundle_fingerprint = _sha256(
            f"gauntlet-bundle\0{artifact.policy_sha256}\0{deck_digest}"
        )
        members.append(
            PfspMember(
                member_id=member_id,
                snapshot_id=f"gauntlet-v{artifact.version}",
                source="past_self",
                pilot_artifact_fingerprint=artifact.policy_model_fingerprint,
                bundle_fingerprint=bundle_fingerprint,
                exact_deck_digest=deck_digest,
                policy_path=artifact.policy_path,
                policy_size_bytes=artifact.policy_size_bytes,
                policy_sha256=artifact.policy_sha256,
                input_contract_fingerprint=artifact.input_contract_fingerprint,
                exact_registry_fingerprint=artifact.exact_registry_fingerprint,
                pair=artifact,
            )
        )
    return tuple(members)


def _assignment(
    game: ScheduledCrossCheckpointGame,
    *,
    member: PfspMember,
) -> StatelessAssignedGame:
    balance_id = _sha256(f"gauntlet-balance\0{game.match_id}")
    curriculum_id = _sha256(f"gauntlet-curriculum\0{game.match_id}")
    return StatelessAssignedGame(
        balance=DeckSeatAssignment(
            assignment_id=balance_id,
            assignment_cursor=game.game_index,
            deck_digest=game.candidate_deck.deck_digest,
            seat=game.candidate_seat,
        ),
        curriculum=CurriculumAssignment(
            assignment_id=curriculum_id,
            cursor=game.game_index,
            generation=0,
            lane="pfsp",
            candidate_deck_digest=game.candidate_deck.deck_digest,
            candidate_seat=game.candidate_seat,
            opponent_id=member.member_id,
            opponent_artifact_fingerprint=member.bundle_fingerprint,
            opponent_pilot_fingerprint=member.pilot_artifact_fingerprint,
            opponent_deck_digest=game.baseline_deck.deck_digest,
            member_id=member.member_id,
        ),
    )


def _result_rows(
    games: Sequence[ScheduledCrossCheckpointGame],
    outcomes: Sequence[StatelessGameOutcome],
    *,
    report: StatelessCollectionReport,
    campaign_fingerprint: str,
    candidate_label: str,
    baseline_label: str,
) -> list[dict[str, Any]]:
    if len(games) != len(outcomes):
        raise RuntimeError("native collection outcomes differ from scheduled games")
    now = datetime.now(UTC).isoformat()
    baseline_reports = tuple(
        item
        for item in report.native_artifact_inference
        if item.route_kind == "past_self"
    )
    telemetry = {
        "native_candidate_policy_rows": report.current_policy_rows,
        "native_candidate_policy_batches": report.current_policy_batches,
        "native_candidate_policy_seconds": report.current_policy_seconds,
        "native_baseline_policy_rows": sum(item.rows for item in baseline_reports),
        "native_baseline_policy_batches": sum(
            item.batches for item in baseline_reports
        ),
        "native_baseline_policy_seconds": report.past_self_policy_seconds,
        "native_engine_steps": report.engine_steps,
        "native_elapsed_seconds": report.elapsed_seconds,
        "native_policy_cohort_rows": report.native_policy_cohort_rows,
        "native_policy_cohort_batches": report.native_policy_cohort_batches,
        "native_policy_cohort_max_rows": report.native_policy_cohort_max_rows,
    }
    rows: list[dict[str, Any]] = []
    for index, (game, outcome) in enumerate(zip(games, outcomes, strict=True)):
        terminal = outcome.status == "engine_terminal"
        score = outcome.candidate_score if terminal else None
        candidate_result = (
            "unresolved"
            if score is None
            else "draw"
            if score == 0.5
            else "win"
            if score == 1.0
            else "loss"
        )
        baseline_result = (
            "loss"
            if candidate_result == "win"
            else "win"
            if candidate_result == "loss"
            else candidate_result
        )
        row: dict[str, Any] = {
            "format": RESULTS_FORMAT,
            "campaign_fingerprint": campaign_fingerprint,
            "game_index": game.game_index,
            "match_id": game.match_id,
            "candidate_checkpoint": candidate_label,
            "baseline_checkpoint": baseline_label,
            "candidate_deck_id": game.candidate_deck.deck_digest,
            "candidate_deck_hash": game.candidate_deck.deck_hash,
            "candidate_deck_label": game.candidate_deck.label,
            "candidate_deck_signature": game.candidate_deck.deck_signature,
            "candidate_deck_source": records.display_path(game.candidate_deck.path),
            "baseline_deck_id": game.baseline_deck.deck_digest,
            "baseline_deck_hash": game.baseline_deck.deck_hash,
            "baseline_deck_label": game.baseline_deck.label,
            "baseline_deck_signature": game.baseline_deck.deck_signature,
            "baseline_deck_source": records.display_path(game.baseline_deck.path),
            "candidate_seat": game.candidate_seat,
            "baseline_seat": 1 - game.candidate_seat,
            "candidate_result": candidate_result,
            "baseline_result": baseline_result,
            "candidate_score": float("nan") if score is None else score,
            "baseline_score": float("nan") if score is None else 1.0 - score,
            "outcome": outcome.status,
            "terminal_reason": outcome.status,
            "started_at": now,
            "finished_at": now,
            "steps": 0,
            "duration_seconds": 0.0,
            "candidate_decisions": outcome.candidate_decisions,
            "baseline_decisions": 0,
            "candidate_action_seconds": 0.0,
            "baseline_action_seconds": 0.0,
            "candidate_mean_action_seconds": 0.0,
            "baseline_mean_action_seconds": 0.0,
            "candidate_policy_decisions": 0,
            "candidate_policy_batch_size_sum": 0,
            "candidate_policy_batch_size_max": 0,
            "candidate_policy_batch_queue_seconds": 0.0,
            "candidate_policy_batch_service_seconds": 0.0,
            "baseline_policy_decisions": 0,
            "baseline_policy_batch_size_sum": 0,
            "baseline_policy_batch_size_max": 0,
            "baseline_policy_batch_queue_seconds": 0.0,
            "baseline_policy_batch_service_seconds": 0.0,
        }
        row.update(telemetry if index == 0 else dict.fromkeys(telemetry, 0))
        rows.append(row)
    return rows


def _execution_telemetry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> float:
        return sum(float(row.get(key, 0) or 0) for row in rows)

    return {
        "candidate_policy_rows": int(total("native_candidate_policy_rows")),
        "candidate_policy_batches": int(total("native_candidate_policy_batches")),
        "candidate_policy_seconds": total("native_candidate_policy_seconds"),
        "baseline_policy_rows": int(total("native_baseline_policy_rows")),
        "baseline_policy_batches": int(total("native_baseline_policy_batches")),
        "baseline_policy_seconds": total("native_baseline_policy_seconds"),
        "engine_steps": int(total("native_engine_steps")),
        "collection_elapsed_seconds": total("native_elapsed_seconds"),
        "policy_cohort_rows": int(total("native_policy_cohort_rows")),
        "policy_cohort_batches": int(total("native_policy_cohort_batches")),
        "policy_cohort_max_rows": int(
            max(
                (int(row.get("native_policy_cohort_max_rows", 0) or 0) for row in rows),
                default=0,
            )
        ),
    }


def _execution_fingerprint(config: NativeCheckpointGauntletConfig) -> str:
    fields = [
        "native-collection-wave-balanced-work-queue-v3",
        str(config.native_worker_replicas),
        str(config.native_worker_torch_threads),
        str(config.native_arena_capacity),
        str(config.native_engine_shards),
        str(config.native_engine_fact_workers),
        str(config.native_policy_cohort_slots),
        str(config.native_policy_group_bank_limit),
        str(config.maximum_engine_steps),
        str(config.candidate.policy_temperature),
        str(config.baseline.policy_temperature),
        str(config.evaluation_action_only),
    ]
    if config.native_inductor_compile_threads is not None:
        fields[0] = "native-collection-wave-balanced-work-queue-v4"
        fields.insert(3, str(config.native_inductor_compile_threads))
    if config.candidate_intervention != "baseline":
        fields.extend(("candidate-input-intervention-v1", config.candidate_intervention))
    return _sha256("\0".join(fields))


def _catalog_pair_fingerprint(config: NativeCheckpointGauntletConfig) -> str:
    return _sha256(
        "\0".join(
            (
                "catalog-pair-v1",
                config.candidate.expected_public_catalog_manifest_sha256,
                config.baseline.expected_public_catalog_manifest_sha256,
            )
        )
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["run_native_collection_checkpoint_gauntlet"]
