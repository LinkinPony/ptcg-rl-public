"""Parallel engine actors with centralized stateless H200 inference."""

from __future__ import annotations

import queue
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.prospective_facts import (
    ProspectiveEngineFactConfig,
    ProspectiveEngineFactProducer,
)
from ptcg_rl.rl.policy_inputs import PolicyInputContract
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.stateless_actor import SimpleStatelessActorPolicy
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionReport,
    StatelessCollectionResult,
    StatelessEngineCollector,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_fragment import (
    StatelessFragment,
    StatelessFragmentIdentity,
)
from ptcg_rl.rl.stateless_fragment_io import (
    CompactFragmentPart,
    CompactFragmentShardWriter,
    load_compact_fragment_part,
)
from ptcg_rl.rl.stateless_inference import (
    CentralHistoricalPolicyExecutor,
    RemoteStatelessActorPolicy,
    StatelessInferenceBroker,
)
from ptcg_rl.rl.stateless_opponents import (
    PastSelfPolicyPool,
    RemoteHistoricalPolicyPool,
    RemotePastSelfPolicyPool,
    past_self_policy_route,
)
from ptcg_rl.rl.stateless_replay import reconstruct_stateless_fragment_part
from ptcg_rl.rl.stateless_training_config import (
    StatelessHistoricalAnchorConfig,
    StatelessScriptedOpponentConfig,
)


@dataclass(frozen=True)
class StatelessParallelWorkerSpec:
    """Immutable resources loaded once by one engine actor process."""

    actor_index: int
    candidate_contract: PolicyInputContract
    catalog: PublicDeckCatalog
    active_decks: Mapping[str, CanonicalDeck]
    opponent_decks: Mapping[str, CanonicalDeck]
    historical_resources: Mapping[
        str,
        tuple[StatelessHistoricalAnchorConfig, tuple[int, ...]],
    ]
    scripted: Mapping[str, StatelessScriptedOpponentConfig]
    fragment_horizon: int
    maximum_engine_steps: int
    seed: int
    inference_timeout_seconds: float
    mirror_bilateral_trajectories: bool
    engine_fact_config: ProspectiveEngineFactConfig | None = None


@dataclass(frozen=True)
class StatelessParallelCollectCommand:
    """One behavior-bound assigned cohort sent to an engine actor."""

    shard_index: int
    identity: StatelessFragmentIdentity
    assignments: tuple[StatelessAssignedGame, ...]
    members: tuple[PfspMember, ...]
    shard_dir: Path
    fragments_per_part: int


@dataclass(frozen=True)
class StatelessParallelWorkerResult:
    """Worker response with one explicit compact transport representation."""

    actor_index: int
    shard_index: int
    report: StatelessCollectionReport | None = None
    outcomes: tuple[StatelessGameOutcome, ...] = ()
    part_paths: tuple[Path, ...] = ()
    compact_parts: tuple[CompactFragmentPart, ...] = ()
    error: str | None = None


class StatelessParallelCollector:
    """Own persistent CPU engine actors and one inference queue topology."""

    def __init__(
        self,
        *,
        context: Any,
        worker_specs: Sequence[StatelessParallelWorkerSpec],
        request_queue: Any,
        response_queues: Mapping[int, Any],
        past_self_pool: PastSelfPolicyPool,
        max_batch_rows: int,
        batch_wait_seconds: float,
        result_timeout_seconds: float,
        game_chunk_size: int,
    ) -> None:
        """Spawn all engine actors before the first collection window."""
        if len(worker_specs) <= 1:
            raise ValueError("parallel collector requires at least two workers")
        if result_timeout_seconds <= 0.0:
            raise ValueError("parallel result timeout must be positive")
        self.worker_specs = tuple(worker_specs)
        self.request_queue = request_queue
        self.response_queues = dict(response_queues)
        self.past_self_pool = past_self_pool
        self.historical_executor = CentralHistoricalPolicyExecutor(
            self.worker_specs[0].historical_resources
        )
        self.historical_executor.preload()
        self.max_batch_rows = int(max_batch_rows)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.result_timeout_seconds = float(result_timeout_seconds)
        if game_chunk_size <= 0:
            raise ValueError("parallel game chunk size must be positive")
        self.game_chunk_size = int(game_chunk_size)
        self.command_queues = {
            spec.actor_index: context.Queue(maxsize=1) for spec in self.worker_specs
        }
        self.result_queue = context.Queue()
        self.processes = {
            spec.actor_index: context.Process(
                target=_parallel_worker_main,
                args=(
                    spec,
                    self.command_queues[spec.actor_index],
                    self.result_queue,
                    self.request_queue,
                    self.response_queues[spec.actor_index],
                ),
                name=f"stateless-engine-actor-{spec.actor_index}",
            )
            for spec in self.worker_specs
        }
        self._closed = False
        for process in self.processes.values():
            process.start()

    def collect(
        self,
        *,
        actor: SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
        identity: StatelessFragmentIdentity,
        assignments: Sequence[StatelessAssignedGame],
        members: Sequence[PfspMember],
        temporary_root: Path,
        fragments_per_part: int,
        serve_inference: bool = True,
    ) -> StatelessCollectionResult:
        """Collect all shards while the central broker batches current-policy rows."""
        if not assignments:
            raise ValueError("parallel collection requires assigned games")
        started_at = time.perf_counter()
        shards = _partition_assignments(
            assignments,
            chunk_size=self.game_chunk_size,
            workers=len(self.worker_specs),
        )
        broker: StatelessInferenceBroker | None = None
        if serve_inference:
            broker = StatelessInferenceBroker(
                actor=actor,
                request_queue=self.request_queue,
                response_queues=self.response_queues,
                max_batch_rows=self.max_batch_rows,
                batch_wait_seconds=self.batch_wait_seconds,
                routed_actors=_central_past_self_routes(
                    self.past_self_pool,
                    members,
                ),
                historical_executor=self.historical_executor,
            )
            broker.start()
        elif members:
            raise ValueError(
                "externally served parallel collection cannot use PFSP routes"
            )
        results: dict[int, StatelessParallelWorkerResult] = {}
        next_shard_index = 0
        active_shards: dict[int, int] = {}

        def dispatch(actor_index: int, shard_index: int) -> None:
            shard_dir = temporary_root / (
                f"shard-{shard_index:04d}-actor-{actor_index:02d}"
            )
            self.command_queues[actor_index].put(
                StatelessParallelCollectCommand(
                    shard_index=shard_index,
                    identity=identity,
                    assignments=shards[shard_index],
                    members=tuple(members),
                    shard_dir=shard_dir,
                    fragments_per_part=fragments_per_part,
                )
            )
            active_shards[actor_index] = shard_index

        try:
            for spec in self.worker_specs[: len(shards)]:
                dispatch(spec.actor_index, next_shard_index)
                next_shard_index += 1
            deadline = time.monotonic() + self.result_timeout_seconds
            while len(results) < len(shards):
                self._raise_failed_worker()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("parallel stateless collection timed out")
                try:
                    result = self.result_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                if not isinstance(result, StatelessParallelWorkerResult):
                    raise TypeError("parallel actor returned an invalid result")
                expected_shard = active_shards.pop(result.actor_index, None)
                if expected_shard != result.shard_index:
                    raise RuntimeError("parallel actor crossed its assigned shard")
                if result.shard_index in results:
                    raise RuntimeError("parallel actor returned a duplicate shard")
                if result.error is not None:
                    raise RuntimeError(
                        "parallel actor "
                        f"{result.actor_index} shard {result.shard_index} "
                        f"failed:\n{result.error}"
                    )
                results[result.shard_index] = result
                if next_shard_index < len(shards):
                    dispatch(result.actor_index, next_shard_index)
                    next_shard_index += 1
        finally:
            if broker is not None:
                broker.close()
        if broker is not None and broker.fatal_error is not None:
            raise RuntimeError("central stateless inference broker failed") from (
                broker.fatal_error
            )
        ordered = tuple(results[index] for index in range(len(shards)))
        compact_part_paths = tuple(
            part_path for result in ordered for part_path in result.part_paths
        )
        fragments: tuple[StatelessFragment, ...] = ()
        compact_parts = _load_worker_parts(
            ordered,
            expected_identity=identity,
        )
        elapsed = max(time.perf_counter() - started_at, 1.0e-9)
        return StatelessCollectionResult(
            fragments=fragments,
            assignments=tuple(assignments),
            outcomes=tuple(
                outcome for result in ordered for outcome in result.outcomes
            ),
            report=_merge_reports(
                ordered,
                elapsed_seconds=elapsed,
                current_policy_batches=broker.batches if broker is not None else 0,
                current_policy_rows=broker.rows if broker is not None else 0,
                current_policy_seconds=(
                    broker.inference_seconds if broker is not None else 0.0
                ),
                past_self_policy_batches=(
                    broker.past_self_batches if broker is not None else 0
                ),
                past_self_policy_rows=(
                    broker.past_self_rows if broker is not None else 0
                ),
                past_self_policy_seconds=(
                    broker.past_self_inference_seconds
                    if broker is not None
                    else 0.0
                ),
                historical_policy_batches=(
                    broker.historical_batches if broker is not None else 0
                ),
                historical_policy_rows=(
                    broker.historical_rows if broker is not None else 0
                ),
                historical_policy_seconds=(
                    broker.historical_inference_seconds
                    if broker is not None
                    else 0.0
                ),
            ),
            compact_part_paths=compact_part_paths,
            compact_parts=compact_parts,
        )

    def close(self) -> None:
        """Stop, join, and if needed terminate all owned actor processes."""
        if self._closed:
            return
        self._closed = True
        for command_queue in self.command_queues.values():
            with suppress(queue.Full):
                command_queue.put_nowait(None)
        deadline = time.monotonic() + 30.0
        for process in self.processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
        for process in self.processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        for command_queue in self.command_queues.values():
            command_queue.close()
        self.result_queue.close()
        self.request_queue.close()
        for response_queue in self.response_queues.values():
            response_queue.close()

    def _raise_failed_worker(self) -> None:
        failed = {
            index: process.exitcode
            for index, process in self.processes.items()
            if process.exitcode not in (None, 0)
        }
        if failed:
            raise RuntimeError(f"parallel stateless actor exited: {failed}")


def _parallel_worker_main(
    spec: StatelessParallelWorkerSpec,
    command_queue: Any,
    result_queue: Any,
    request_queue: Any,
    response_queue: Any,
) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    past_self = RemotePastSelfPolicyPool(
        actor_index=spec.actor_index,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_seconds=spec.inference_timeout_seconds,
        engine_fact_config=spec.engine_fact_config,
    )
    historical = RemoteHistoricalPolicyPool(
        spec.historical_resources,
        actor_index=spec.actor_index,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_seconds=spec.inference_timeout_seconds,
    )
    while True:
        command = command_queue.get()
        if command is None:
            return
        if not isinstance(command, StatelessParallelCollectCommand):
            result_queue.put(
                StatelessParallelWorkerResult(
                    actor_index=spec.actor_index,
                    shard_index=-1,
                    error="invalid parallel collection command",
                )
            )
            continue
        try:
            past_self.sync(
                command.members,
                current_identity=command.identity,
            )
            actor = RemoteStatelessActorPolicy(
                actor_index=spec.actor_index,
                identity=command.identity,
                request_queue=request_queue,
                response_queue=response_queue,
                timeout_seconds=spec.inference_timeout_seconds,
                engine_fact_config=spec.engine_fact_config,
            )
            engine_fact_producer = (
                None
                if spec.engine_fact_config is None
                or not spec.engine_fact_config.enabled
                else ProspectiveEngineFactProducer(
                    sampler=BeliefSampler(
                        config=spec.engine_fact_config.sampler
                    ),
                    config=spec.engine_fact_config,
                )
            )
            collector = StatelessEngineCollector(
                actor=actor,
                identity=command.identity,
                candidate_contract=spec.candidate_contract,
                catalog=spec.catalog,
                active_decks=spec.active_decks,
                opponent_decks=spec.opponent_decks,
                curriculum=None,
                deck_balance=None,
                past_self_pool=past_self,
                historical_pool=historical,
                scripted=spec.scripted,
                maximum_engine_steps=spec.maximum_engine_steps,
                seed=spec.seed + command.shard_index,
                mirror_bilateral_trajectories=(spec.mirror_bilateral_trajectories),
                opponent_members=command.members,
                defer_controller_updates=True,
                engine_fact_producer=engine_fact_producer,
            )
            collected = collector.collect_assigned(command.assignments)
            writer = CompactFragmentShardWriter(
                command.shard_dir,
                static_contract_fingerprint=(
                    command.identity.static_contract_fingerprint
                ),
                horizon=command.identity.horizon,
                fragments_per_part=command.fragments_per_part,
                sequence=command.identity.schema_version == 2,
            )
            for fragment in collected.fragments:
                writer.add(fragment)
            manifest = writer.close()
            result_queue.put(
                StatelessParallelWorkerResult(
                    actor_index=spec.actor_index,
                    shard_index=command.shard_index,
                    report=collected.report,
                    outcomes=collected.outcomes,
                    part_paths=tuple(
                        command.shard_dir / "parts" / part.filename
                        for part in manifest.parts
                    ),
                )
            )
        except BaseException:
            result_queue.put(
                StatelessParallelWorkerResult(
                    actor_index=spec.actor_index,
                    shard_index=command.shard_index,
                    error=traceback.format_exc(),
                )
            )


def _central_past_self_routes(
    pool: PastSelfPolicyPool,
    members: Sequence[PfspMember],
) -> dict[
    str,
    SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
]:
    """Resolve each immutable past-self artifact once in the GPU owner."""
    routes: dict[
        str,
        SimpleStatelessActorPolicy | GeneralistSequenceActorPolicy,
    ] = {}
    for member in members:
        if member.source != "past_self":
            continue
        route = past_self_policy_route(member)
        actor = pool.actor(member.member_id)
        existing = routes.get(route)
        if existing is not None and existing is not actor:
            raise ValueError("one past-self artifact resolves to multiple actors")
        routes[route] = actor
    return routes


def _partition_assignments(
    assignments: Sequence[StatelessAssignedGame],
    *,
    chunk_size: int,
    workers: int,
) -> tuple[tuple[StatelessAssignedGame, ...], ...]:
    if not assignments or chunk_size <= 0 or workers <= 0:
        raise ValueError("parallel assignment chunk size must be positive")
    effective_chunk_size = min(
        chunk_size,
        max(1, (len(assignments) + workers - 1) // workers),
    )
    shard_count = (len(assignments) + effective_chunk_size - 1) // (
        effective_chunk_size
    )
    shards: list[list[StatelessAssignedGame]] = [
        [] for _ in range(shard_count)
    ]

    def place_balanced(rows: Sequence[StatelessAssignedGame]) -> None:
        for assignment in rows:
            candidates = tuple(
                index
                for index, shard in enumerate(shards)
                if len(shard) < effective_chunk_size
            )
            if not candidates:
                raise RuntimeError("parallel assignment shards exhausted capacity")
            shard_index = min(
                candidates,
                key=lambda index: (len(shards[index]), index),
            )
            shards[shard_index].append(assignment)

    mirror = tuple(
        assignment
        for assignment in assignments
        if assignment.curriculum.lane == "mirror"
    )
    scripted = tuple(
        assignment
        for assignment in assignments
        if assignment.curriculum.lane == "scripted"
    )
    place_balanced(mirror)
    place_balanced(scripted)

    pfsp_by_artifact: defaultdict[str, list[StatelessAssignedGame]] = defaultdict(
        list
    )
    for assignment in assignments:
        if assignment.curriculum.lane == "pfsp":
            pfsp_by_artifact[
                assignment.curriculum.opponent_pilot_fingerprint
            ].append(assignment)
    for artifact in sorted(
        pfsp_by_artifact,
        key=lambda value: (-len(pfsp_by_artifact[value]), value),
    ):
        remaining = pfsp_by_artifact[artifact]
        while remaining:
            candidates = tuple(
                index
                for index, shard in enumerate(shards)
                if len(shard) < effective_chunk_size
            )
            if not candidates:
                raise RuntimeError("parallel PFSP shards exhausted capacity")
            shard_index = min(
                candidates,
                key=lambda index: (len(shards[index]), index),
            )
            available = effective_chunk_size - len(shards[shard_index])
            shards[shard_index].extend(remaining[:available])
            del remaining[:available]

    covered = tuple(
        id(assignment) for shard in shards for assignment in shard
    )
    if sorted(covered) != sorted(id(assignment) for assignment in assignments):
        raise RuntimeError("parallel assignment partition changed its cohort")
    return tuple(tuple(shard) for shard in shards)


def _load_worker_fragments(
    results: Sequence[StatelessParallelWorkerResult],
    *,
    expected_identity: StatelessFragmentIdentity,
) -> tuple[StatelessFragment, ...]:
    fragments: list[StatelessFragment] = []
    seen_part_paths: set[Path] = set()
    seen_fragment_ids: set[str] = set()
    for result in results:
        report = result.report
        if report is None:
            raise RuntimeError("parallel actor omitted its collection report")
        worker_fragments: list[StatelessFragment] = []
        worker_decisions = 0
        for part_path in result.part_paths:
            try:
                resolved_path = part_path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(
                    f"parallel actor compact fragment part is unavailable: {part_path}"
                ) from error
            if resolved_path in seen_part_paths:
                raise RuntimeError(
                    "parallel actors returned a duplicate compact fragment part"
                )
            seen_part_paths.add(resolved_path)
            try:
                part = load_compact_fragment_part(resolved_path)
                part_fragments = reconstruct_stateless_fragment_part(part)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "parallel actor returned an invalid compact fragment part: "
                    f"{part_path}"
                ) from error
            part_decisions = sum(len(fragment.decisions) for fragment in part_fragments)
            if (
                len(part_fragments) != part.fragment_count
                or part_decisions != part.decision_count
            ):
                raise RuntimeError(
                    "parallel actor compact fragment part count mismatch"
                )
            for fragment in part_fragments:
                if fragment.identity != expected_identity:
                    raise RuntimeError(
                        "parallel actor compact fragment identity mismatch"
                    )
                if fragment.fragment_id in seen_fragment_ids:
                    raise RuntimeError("parallel actors returned a duplicate fragment")
                seen_fragment_ids.add(fragment.fragment_id)
            worker_fragments.extend(part_fragments)
            worker_decisions += part_decisions
        if len(worker_fragments) != report.fragments:
            raise RuntimeError(
                "parallel actor compact fragment count differs from its report"
            )
        if worker_decisions != report.candidate_decisions:
            raise RuntimeError(
                "parallel actor compact decision count differs from its report"
            )
        fragments.extend(worker_fragments)
    return tuple(fragments)


def _load_worker_parts(
    results: Sequence[StatelessParallelWorkerResult],
    *,
    expected_identity: StatelessFragmentIdentity,
) -> tuple[CompactFragmentPart, ...]:
    """Load compact worker parts without reconstructing trajectory objects."""
    parts: list[CompactFragmentPart] = []
    seen_part_paths: set[Path] = set()
    seen_fragment_ids: set[str] = set()
    expected_fields: list[tuple[str, object]] = [
        ("horizons", expected_identity.horizon),
        ("behavior_policy_versions", expected_identity.behavior_policy_version),
        (
            "behavior_policy_fingerprints",
            expected_identity.behavior_policy_fingerprint,
        ),
        ("model_config_fingerprints", expected_identity.model_config_fingerprint),
        ("action_schema_fingerprints", expected_identity.action_schema_fingerprint),
        (
            "public_context_fingerprints",
            expected_identity.public_context_fingerprint,
        ),
        ("card_catalog_fingerprints", expected_identity.card_catalog_fingerprint),
        (
            "public_deck_catalog_fingerprints",
            expected_identity.public_deck_catalog_fingerprint,
        ),
        (
            "exact_registry_fingerprints",
            expected_identity.exact_registry_fingerprint,
        ),
        (
            "belief_target_semantics_fingerprints",
            expected_identity.belief_target_semantics_fingerprint,
        ),
        (
            "input_contract_fingerprints",
            expected_identity.input_contract_fingerprint,
        ),
        (
            "resolved_config_fingerprints",
            expected_identity.resolved_config_fingerprint,
        ),
    ]
    if expected_identity.schema_version == 2:
        expected_fields.extend(
            (
                ("fragment_schema_versions", 2),
                (
                    "sequence_contract_fingerprints",
                    expected_identity.sequence_contract_fingerprint,
                ),
            )
        )
    for result in results:
        report = result.report
        if report is None:
            raise RuntimeError("parallel actor omitted its collection report")
        worker_fragments = 0
        worker_decisions = 0
        for part_path in result.part_paths:
            try:
                resolved_path = part_path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(
                    f"parallel actor compact fragment part is unavailable: {part_path}"
                ) from error
            if resolved_path in seen_part_paths:
                raise RuntimeError(
                    "parallel actors returned a duplicate compact fragment part"
                )
            seen_part_paths.add(resolved_path)
            try:
                part = load_compact_fragment_part(resolved_path)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "parallel actor returned an invalid compact fragment part: "
                    f"{part_path}"
                ) from error
            if any(
                not np.all(np.asarray(part.arrays[field]) == expected)
                for field, expected in expected_fields
            ):
                raise RuntimeError("parallel actor compact fragment identity mismatch")
            fragment_ids = tuple(str(value) for value in part.arrays["fragment_ids"])
            if len(fragment_ids) != len(set(fragment_ids)):
                raise RuntimeError(
                    "parallel actor returned duplicate fragments in one part"
                )
            if seen_fragment_ids.intersection(fragment_ids):
                raise RuntimeError("parallel actors returned a duplicate fragment")
            seen_fragment_ids.update(fragment_ids)
            worker_fragments += part.fragment_count
            worker_decisions += part.decision_count
            parts.append(part)
        if worker_fragments != report.fragments:
            raise RuntimeError(
                "parallel actor compact fragment count differs from its report"
            )
        if worker_decisions != report.candidate_decisions:
            raise RuntimeError(
                "parallel actor compact decision count differs from its report"
            )
    return tuple(parts)


def cleanup_parallel_collection_parts(part_paths: Sequence[Path]) -> None:
    """Discard detached worker parts and their now-empty transport metadata.

    Parent-side loading copies every NPZ array before returning a compact part,
    so callers may release these process-transport files after the optimizer
    has consumed the in-memory window.  Missing files and directories are
    accepted to keep cleanup safely retryable after a partial failure.
    """
    declared_paths = tuple(part_path.resolve(strict=False) for part_path in part_paths)
    if not declared_paths:
        return
    if len(declared_paths) != len(set(declared_paths)):
        raise RuntimeError("parallel fragment transport repeats a part path")
    if any(
        part_path.parent.name != "parts" or part_path.suffix != ".npz"
        for part_path in declared_paths
    ):
        raise RuntimeError("parallel fragment transport part path is malformed")

    shard_roots = {part_path.parent.parent for part_path in declared_paths}
    update_roots = {shard_root.parent for shard_root in shard_roots}
    if len(update_roots) != 1:
        raise RuntimeError("parallel fragment transport spans multiple windows")

    declared_by_directory = {
        shard_root / "parts": {
            part_path
            for part_path in declared_paths
            if part_path.parent.parent == shard_root
        }
        for shard_root in shard_roots
    }
    for parts_dir, declared in declared_by_directory.items():
        if not parts_dir.exists():
            continue
        if not parts_dir.is_dir():
            raise RuntimeError("parallel fragment parts path is not a directory")
        actual = {path.resolve(strict=False) for path in parts_dir.iterdir()}
        if actual - declared:
            raise RuntimeError("parallel fragment transport contains unknown files")
        if any(path.exists() and not path.is_file() for path in declared):
            raise RuntimeError("parallel fragment transport part is not a file")

    for shard_root in shard_roots:
        if not shard_root.exists():
            continue
        if not shard_root.is_dir():
            raise RuntimeError("parallel fragment shard path is not a directory")
        allowed = {shard_root / "manifest.json", shard_root / "parts"}
        actual = {path.resolve(strict=False) for path in shard_root.iterdir()}
        if actual - allowed:
            raise RuntimeError("parallel fragment shard contains unknown files")

    update_root = next(iter(update_roots))
    if update_root.exists():
        if not update_root.is_dir():
            raise RuntimeError("parallel fragment window path is not a directory")
        actual = {path.resolve(strict=False) for path in update_root.iterdir()}
        if actual - shard_roots:
            raise RuntimeError("parallel fragment window contains unknown shards")

    for part_path in declared_paths:
        part_path.unlink(missing_ok=True)
    for shard_root in shard_roots:
        parts_dir = shard_root / "parts"
        (shard_root / "manifest.json").unlink(missing_ok=True)
        if parts_dir.exists():
            parts_dir.rmdir()
        if shard_root.exists():
            shard_root.rmdir()
    if update_root.exists():
        update_root.rmdir()


def _merge_reports(
    results: Sequence[StatelessParallelWorkerResult],
    *,
    elapsed_seconds: float,
    current_policy_batches: int,
    current_policy_rows: int,
    current_policy_seconds: float,
    past_self_policy_batches: int,
    past_self_policy_rows: int,
    past_self_policy_seconds: float,
    historical_policy_batches: int,
    historical_policy_rows: int,
    historical_policy_seconds: float,
) -> StatelessCollectionReport:
    reports = tuple(result.report for result in results if result.report is not None)
    if len(reports) != len(results):
        raise RuntimeError("parallel actor omitted its collection report")
    lane_games: Counter[str] = Counter()
    seat_games: Counter[str] = Counter()
    member_games: Counter[str] = Counter()
    lane_score_total: defaultdict[str, float] = defaultdict(float)
    lane_score_games: Counter[str] = Counter()
    for report in reports:
        lane_games.update(report.lane_games)
        seat_games.update(report.seat_games)
        member_games.update(report.pfsp_member_games)
        for lane, score in report.candidate_score_by_lane.items():
            games = report.lane_games.get(lane, 0)
            lane_score_total[lane] += score * games
            lane_score_games[lane] += games
    decisions = sum(report.candidate_decisions for report in reports)
    return StatelessCollectionReport(
        games_started=sum(report.games_started for report in reports),
        games_finished=sum(report.games_finished for report in reports),
        games_cancelled=sum(report.games_cancelled for report in reports),
        engine_steps=sum(report.engine_steps for report in reports),
        candidate_decisions=decisions,
        mirror_opponent_decisions=sum(
            report.mirror_opponent_decisions for report in reports
        ),
        fragments=sum(report.fragments for report in reports),
        elapsed_seconds=elapsed_seconds,
        decisions_per_second=decisions / elapsed_seconds,
        input_seconds=sum(report.input_seconds for report in reports),
        current_policy_seconds=current_policy_seconds,
        past_self_policy_seconds=past_self_policy_seconds,
        historical_policy_seconds=historical_policy_seconds,
        scripted_policy_seconds=sum(
            report.scripted_policy_seconds for report in reports
        ),
        engine_fact_seconds=sum(report.engine_fact_seconds for report in reports),
        engine_fact_wait_seconds=sum(
            report.engine_fact_wait_seconds for report in reports
        ),
        engine_fact_overlap_seconds=sum(
            report.engine_fact_overlap_seconds for report in reports
        ),
        engine_fact_roots=sum(report.engine_fact_roots for report in reports),
        engine_fact_eligible_options=sum(
            report.engine_fact_eligible_options for report in reports
        ),
        engine_fact_native_batch_calls=sum(
            report.engine_fact_native_batch_calls for report in reports
        ),
        engine_fact_native_transitions=sum(
            report.engine_fact_native_transitions for report in reports
        ),
        engine_fact_unresolved_worlds=sum(
            report.engine_fact_unresolved_worlds for report in reports
        ),
        engine_fact_resolved_options=sum(
            report.engine_fact_resolved_options for report in reports
        ),
        engine_control_seconds=sum(report.engine_control_seconds for report in reports),
        current_policy_batches=current_policy_batches,
        current_policy_rows=current_policy_rows,
        past_self_policy_batches=past_self_policy_batches,
        past_self_policy_rows=past_self_policy_rows,
        historical_policy_batches=historical_policy_batches,
        historical_policy_rows=historical_policy_rows,
        lane_games=dict(sorted(lane_games.items())),
        seat_games=dict(sorted(seat_games.items())),
        pfsp_member_games=dict(sorted(member_games.items())),
        candidate_score_by_lane={
            lane: lane_score_total[lane] / games
            for lane, games in sorted(lane_score_games.items())
            if games
        },
    )


__all__ = [
    "StatelessParallelCollector",
    "StatelessParallelWorkerResult",
    "StatelessParallelWorkerSpec",
    "cleanup_parallel_collection_parts",
]
