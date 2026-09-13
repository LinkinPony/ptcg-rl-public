"""Privacy-safe prospective engine facts for native rollout arenas."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import queue
import random
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ptcg_rl.belief.observation import ObservationEvidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.belief.state import OpponentBeliefState
from ptcg_rl.engine.constants import AreaType, OptionType
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.native_probe import NativeProbeBackend
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
)
from ptcg_rl.engine.prospective_facts import (
    ProspectiveEngineFactConfig,
    ProspectiveEngineFactProducer,
)
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.rl.native_collection_games import NativeLiveGame

Int64Array = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_ELIGIBLE_OPTION_TYPES = frozenset((int(OptionType.ATTACK), int(OptionType.ABILITY)))
_VIRTUAL_AREA = 0
_PROCESS_SAMPLER: BeliefSampler | None = None
_PROCESS_BACKEND: NativeProbeBackend | None = None
_PROCESS_CONFIG: ProspectiveEngineFactConfig | None = None
_MAX_CONCURRENT_FACT_WAVES = 8
_FACT_WAVE_COALESCE_SECONDS = 0.005
_MAX_COALESCED_ROOTS = 4096


@dataclass(frozen=True, slots=True)
class NativeEngineFactSource:
    """One selected arena segment aligned with a combined policy batch."""

    lane: NativeTrainingLane
    view: NativeTrainingBatchView
    rows: Int64Array
    known: NativeKnownOpponentBatch
    live: Mapping[int, NativeLiveGame]


@dataclass(frozen=True, slots=True)
class NativeEngineFactBatch:
    """Model-ready exact fact tensors and auditable probe counters."""

    features: Float32Array
    masks: BoolArray
    roots: int
    eligible_options: int
    native_batch_calls: int
    native_transitions: int
    unresolved_worlds: int
    elapsed_seconds: float
    # Stage decomposition charged to the wave owning each chunk's first root.
    sample_seconds: float = 0.0
    native_seconds: float = 0.0
    queue_seconds: float = 0.0
    chunk_wall_seconds: float = 0.0

    def inject(self, options: OptionBatch) -> None:
        """Copy exact facts into the native model-ready host batch."""
        expected_features = tuple(options.dynamic_effect_features.shape)
        expected_masks = tuple(options.dynamic_effect_masks.shape)
        if self.features.shape != expected_features:
            raise ValueError("native fact feature tensor is misaligned")
        if self.masks.shape != expected_masks:
            raise ValueError("native fact mask tensor is misaligned")
        feature_source = torch.from_numpy(self.features)
        mask_source = torch.from_numpy(self.masks)
        options.dynamic_effect_features.copy_(
            feature_source,
            non_blocking=(
                options.dynamic_effect_features.device.type == "cuda"
                and feature_source.is_pinned()
            ),
        )
        options.dynamic_effect_masks.copy_(
            mask_source,
            non_blocking=(
                options.dynamic_effect_masks.device.type == "cuda"
                and mask_source.is_pinned()
            ),
        )


@dataclass(frozen=True, slots=True)
class _RootProbe:
    """One privacy-erased root plus its public determinization evidence."""

    state_token: bytes
    evidence: ObservationEvidence
    opponent_state: OpponentBeliefState
    own_deck: tuple[int, ...]
    option_indices: tuple[int, ...]
    output_row: int
    seed: int


@dataclass(frozen=True, slots=True)
class _RootChunkResult:
    """Sparse candidate facts from one multi-root native process call."""

    root_output_rows: Int64Array
    output_rows: Int64Array
    option_indices: Int64Array
    features: Float32Array
    masks: BoolArray
    root_count: int
    eligible_options: int
    transition_count: int
    unresolved_worlds: int
    root_unresolved_worlds: Int64Array
    started_at: float = 0.0
    finished_at: float = 0.0
    sample_seconds: float = 0.0
    native_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _QueuedFactWave:
    """One prepared route wave waiting at the shared process-call boundary."""

    roots: tuple[_RootProbe, ...]
    output_shape: tuple[int, int]
    started_at: float
    result: Future[NativeEngineFactBatch]


class NativeProspectiveEngineFactProducer:
    """Probe native arena roots without reconstructing observations."""

    def __init__(
        self,
        producer: ProspectiveEngineFactProducer,
        *,
        maximum_workers: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        """Bind the existing sampler/config/fingerprint to native roots."""
        if not isinstance(producer.backend, NativeProbeBackend):
            raise RuntimeError("native rollout engine facts require libcg_probe.so")
        configured_workers = (
            max(1, _available_cpu_count() - 8)
            if maximum_workers is None
            else int(maximum_workers)
        )
        if configured_workers <= 0:
            raise ValueError("native fact worker count must be positive")
        self.producer = producer
        self.backend = producer.backend
        self.maximum_workers = configured_workers
        self.fingerprint = producer.fingerprint
        # A multiprocess rollout feeder does not own CUDA. Even probing CUDA or
        # asking PyTorch for pinned storage initializes a CUDA/MPS client in
        # that subprocess. At server-scale feeder counts this can exhaust the
        # MPS client limit and fail unrelated GPU inference workers. Callers
        # that cross a CUDA transfer frontier may retain the automatic default;
        # host-only feeder processes opt out without touching the CUDA runtime.
        self.pin_memory = (
            torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        )
        # Current and immutable-past routes are prepared independently. A
        # single coordinator thread serialized those routes before work even
        # reached the GIL-independent process pool, leaving most fact workers
        # idle between small ragged batches. Multiple lightweight coordinators
        # may safely enqueue against ProcessPoolExecutor concurrently; each
        # result remains isolated in its route-owned Future.
        self._wave_executor = ThreadPoolExecutor(
            max_workers=min(self.maximum_workers, _MAX_CONCURRENT_FACT_WAVES),
            thread_name_prefix="native-engine-fact-wave",
        )
        self._root_executor = ProcessPoolExecutor(
            max_workers=self.maximum_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_process_worker,
            initargs=(
                producer.sampler,
                producer.config,
                producer.backend.engine_abi_fingerprint,
            ),
        )
        self._dispatch_queue: queue.Queue[_QueuedFactWave | None] = queue.Queue()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_fact_waves,
            name="native-engine-fact-dispatch",
            daemon=True,
        )
        self._closed = False
        self._dispatch_thread.start()

    def submit(
        self,
        sources: Sequence[NativeEngineFactSource],
        *,
        output_shape: tuple[int, int],
    ) -> Future[NativeEngineFactBatch]:
        """Start one fact wave so native policy encoding can overlap it."""
        if self._closed:
            raise RuntimeError("native fact producer is closed")
        return self._wave_executor.submit(
            self.run,
            tuple(sources),
            output_shape=output_shape,
        )

    def close(self) -> None:
        """Drain in-flight waves and release their coordinator threads."""
        if self._closed:
            return
        self._closed = True
        self._wave_executor.shutdown(wait=True, cancel_futures=False)
        self._dispatch_queue.put(None)
        self._dispatch_thread.join()
        self._root_executor.shutdown(wait=True, cancel_futures=False)

    def run(
        self,
        sources: Sequence[NativeEngineFactSource],
        *,
        output_shape: tuple[int, int],
    ) -> NativeEngineFactBatch:
        """Probe every eligible root once and return dense model tensors."""
        started_at = time.perf_counter()
        if len(output_shape) != 2 or output_shape[0] <= 0 or output_shape[1] <= 0:
            raise ValueError("native fact output shape must be positive rank two")
        source_rows = sum(int(source.rows.size) for source in sources)
        if source_rows != output_shape[0]:
            raise ValueError("native fact sources differ from policy batch rows")
        roots = self._roots(sources)
        if not roots:
            features, masks = _fact_output_arrays(
                output_shape,
                pin_memory=self.pin_memory,
            )
            return NativeEngineFactBatch(
                features=features,
                masks=masks,
                roots=0,
                eligible_options=0,
                native_batch_calls=0,
                native_transitions=0,
                unresolved_worlds=0,
                elapsed_seconds=time.perf_counter() - started_at,
            )
        result: Future[NativeEngineFactBatch] = Future()
        self._dispatch_queue.put(
            _QueuedFactWave(
                roots=roots,
                output_shape=output_shape,
                started_at=started_at,
                result=result,
            )
        )
        return result.result()

    def _dispatch_fact_waves(self) -> None:
        """Coalesce small route waves before crossing the native process ABI."""
        while True:
            first = self._dispatch_queue.get()
            if first is None:
                return
            pending = [first]
            roots = len(first.roots)
            deadline = time.perf_counter() + _FACT_WAVE_COALESCE_SECONDS
            stop_after_batch = False
            while roots < _MAX_COALESCED_ROOTS:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = self._dispatch_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                pending.append(item)
                roots += len(item.roots)
            try:
                self._resolve_coalesced_waves(pending)
            except BaseException as error:
                for item in pending:
                    item.result.set_exception(error)
            if stop_after_batch:
                return

    def _resolve_coalesced_waves(
        self,
        pending: Sequence[_QueuedFactWave],
    ) -> None:
        """Execute one cross-route root batch and demultiplex exact results."""
        row_offsets = [0]
        combined_roots: list[_RootProbe] = []
        root_owners: list[int] = []
        for owner, item in enumerate(pending):
            row_offset = row_offsets[-1]
            combined_roots.extend(
                replace(root, output_row=root.output_row + row_offset)
                for root in item.roots
            )
            root_owners.extend((owner,) * len(item.roots))
            row_offsets.append(row_offset + item.output_shape[0])
        maximum_options = max(item.output_shape[1] for item in pending)
        features, masks = _fact_output_arrays(
            (row_offsets[-1], maximum_options),
            pin_memory=self.pin_memory,
        )
        chunks = _root_chunks(
            tuple(combined_roots),
            maximum_workers=self.maximum_workers,
        )
        dispatched_at = time.perf_counter()
        results = tuple(self._root_executor.map(_probe_root_chunk_in_process, chunks))
        calls = [0] * len(pending)
        unresolved = [0] * len(pending)
        sample_seconds = [0.0] * len(pending)
        native_seconds = [0.0] * len(pending)
        queue_seconds = [0.0] * len(pending)
        chunk_wall_seconds = [0.0] * len(pending)
        transitions = [
            sum(len(root.option_indices) for root in item.roots)
            * self.producer.config.worlds
            for item in pending
        ]
        for result in results:
            usable = np.flatnonzero(result.masks)
            if usable.size:
                rows = result.output_rows[usable]
                indices = result.option_indices[usable]
                features[rows, indices] = result.features[usable]
                masks[rows, indices] = True
            owners = np.searchsorted(
                np.asarray(row_offsets[1:], dtype=np.int64),
                result.root_output_rows,
                side="right",
            )
            # One coalesced native call is charged once, to the wave owning its
            # first root. Summing per-wave reports then remains the exact
            # process-call count instead of multiplying calls by every wave
            # represented in the chunk.
            calls[int(owners[0])] += 1
            sample_seconds[int(owners[0])] += result.sample_seconds
            native_seconds[int(owners[0])] += result.native_seconds
            queue_seconds[int(owners[0])] += max(
                result.started_at - dispatched_at,
                0.0,
            )
            chunk_wall_seconds[int(owners[0])] += max(
                result.finished_at - result.started_at,
                0.0,
            )
            for owner in np.unique(owners):
                selected = owners == owner
                unresolved[int(owner)] += int(
                    result.root_unresolved_worlds[selected].sum()
                )
        for owner, item in enumerate(pending):
            start, stop = row_offsets[owner : owner + 2]
            option_count = item.output_shape[1]
            item.result.set_result(
                NativeEngineFactBatch(
                    features=features[start:stop, :option_count],
                    masks=masks[start:stop, :option_count],
                    roots=len(item.roots),
                    eligible_options=sum(
                        len(root.option_indices) for root in item.roots
                    ),
                    native_batch_calls=calls[owner],
                    native_transitions=transitions[owner],
                    unresolved_worlds=unresolved[owner],
                    elapsed_seconds=time.perf_counter() - item.started_at,
                    sample_seconds=sample_seconds[owner],
                    native_seconds=native_seconds[owner],
                    queue_seconds=queue_seconds[owner],
                    chunk_wall_seconds=chunk_wall_seconds[owner],
                )
            )

    def _roots(
        self,
        sources: Sequence[NativeEngineFactSource],
    ) -> tuple[_RootProbe, ...]:
        roots: list[_RootProbe] = []
        output_row = 0
        for source in sources:
            source_rows = _fact_source_rows(
                source.rows,
                size=source.view.batch_size,
            )
            if source.known.batch_size != source_rows.size:
                raise ValueError("native fact known evidence is misaligned")
            options_by_row = tuple(
                _eligible_options(source.view, int(source_row))
                for source_row in source_rows
            )
            eligible_rows = tuple(
                row
                for row, option_indices in enumerate(options_by_row)
                if option_indices
            )
            tokens_by_row: dict[int, bytes] = {}
            if eligible_rows:
                eligible_source_rows = source_rows[
                    np.asarray(eligible_rows, dtype=np.int64)
                ]
                tokens = source.lane.export_public_state_tokens(
                    source.view.slots[eligible_source_rows]
                )
                tokens_by_row = dict(zip(eligible_rows, tokens, strict=True))
            for row, option_indices in enumerate(options_by_row):
                if option_indices:
                    source_row = int(source_rows[row])
                    slot = int(source.view.slots[source_row])
                    perspective = int(source.view.select_player[source_row])
                    game = source.live[slot]
                    own_deck = (
                        game.candidate
                        if perspective == game.candidate_seat
                        else game.opponent
                    )
                    evidence = _observation_evidence(source.view, source_row)
                    known = _known_counts(source.known, row)
                    current = evidence.opponent_current_visible_counts
                    for card_id, count in current.items():
                        known[card_id] = max(known[card_id], count)
                    decision_index = game.sequence_decisions_by_seat[perspective]
                    roots.append(
                        _RootProbe(
                            state_token=tokens_by_row[row],
                            evidence=evidence,
                            opponent_state=OpponentBeliefState(
                                revealed_no_serial_counts=known
                            ),
                            own_deck=tuple(int(card) for card in own_deck.card_ids),
                            option_indices=option_indices,
                            output_row=output_row + row,
                            seed=_root_seed(
                                self.producer.config.seed,
                                game.game_id,
                                perspective,
                                decision_index,
                            ),
                        )
                    )
            output_row += int(source_rows.size)
        return tuple(roots)


def _initialize_process_worker(
    sampler: BeliefSampler,
    config: ProspectiveEngineFactConfig,
    expected_engine_abi_fingerprint: str,
) -> None:
    """Initialize one GIL-independent native fact worker."""
    global _PROCESS_BACKEND, _PROCESS_CONFIG, _PROCESS_SAMPLER
    backend = NativeProbeBackend(manual_coin=config.manual_coin)
    if backend.engine_abi_fingerprint != expected_engine_abi_fingerprint:
        raise RuntimeError("native fact worker loaded a different engine artifact")
    _PROCESS_SAMPLER = sampler
    _PROCESS_BACKEND = backend
    _PROCESS_CONFIG = config


def _probe_root_chunk_in_process(
    roots: tuple[_RootProbe, ...],
) -> _RootChunkResult:
    """Probe one root chunk using the process-local immutable runtime."""
    if _PROCESS_SAMPLER is None or _PROCESS_BACKEND is None or _PROCESS_CONFIG is None:
        raise RuntimeError("native fact process worker is not initialized")
    profiler = _process_chunk_profiler()
    if profiler is not None:
        profiler.enable()
    try:
        return _probe_root_chunk(
            roots,
            sampler=_PROCESS_SAMPLER,
            backend=_PROCESS_BACKEND,
            config=_PROCESS_CONFIG,
        )
    finally:
        if profiler is not None:
            profiler.disable()
            _maybe_dump_chunk_profile(profiler)


_PROCESS_PROFILER: Any | None = None
_PROCESS_PROFILED_CHUNKS = 0


def _process_chunk_profiler() -> Any | None:
    """Return the env-gated per-process cProfile for fact chunk work."""
    global _PROCESS_PROFILER
    if _PROCESS_PROFILER is None and os.environ.get("PTCG_RL_FACT_PROFILE_DIR"):
        import cProfile

        _PROCESS_PROFILER = cProfile.Profile()
    return _PROCESS_PROFILER


def _maybe_dump_chunk_profile(profiler: Any) -> None:
    """Snapshot profile stats every few chunks to survive hard kills."""
    global _PROCESS_PROFILED_CHUNKS
    _PROCESS_PROFILED_CHUNKS += 1
    if _PROCESS_PROFILED_CHUNKS % 25 != 0:
        return
    directory = Path(os.environ["PTCG_RL_FACT_PROFILE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"fact_{os.getpid()}.prof"
    temporary = target.with_suffix(".prof.tmp")
    profiler.dump_stats(temporary)
    temporary.replace(target)


def _probe_root_chunk(
    roots: tuple[_RootProbe, ...],
    *,
    sampler: BeliefSampler,
    backend: NativeProbeBackend,
    config: ProspectiveEngineFactConfig,
) -> _RootChunkResult:
    """Sample a root chunk and execute one exact ragged native fact call."""
    if not roots:
        raise ValueError("native fact root chunk must not be empty")
    chunk_started_at = time.perf_counter()
    hidden_counts: list[int] = []
    hidden_values: list[int] = []
    hidden_value_offsets = [0]
    candidate_offsets = [0]
    candidate_values: list[int] = []
    output_rows: list[int] = []
    option_indices: list[int] = []
    for root in roots:
        rng = random.Random(root.seed)
        for _world in range(config.worlds):
            hidden = sampler.sample_from_evidence(
                root.evidence,
                your_deck=root.own_deck,
                opponent_state=root.opponent_state,
                rng=rng,
            ).hidden
            for cards in (
                hidden.your_deck,
                hidden.your_prize,
                hidden.opponent_deck,
                hidden.opponent_prize,
                hidden.opponent_hand,
                hidden.opponent_active,
            ):
                hidden_counts.append(len(cards))
                hidden_values.extend(cards)
        hidden_value_offsets.append(len(hidden_values))
        candidate_values.extend(root.option_indices)
        candidate_offsets.append(len(candidate_values))
        output_rows.extend((root.output_row,) * len(root.option_indices))
        option_indices.extend(root.option_indices)
    native_started_at = time.perf_counter()
    result = backend.run_fact_batch(
        tuple(root.state_token for root in roots),
        hidden_counts=hidden_counts,
        hidden_values=hidden_values,
        hidden_value_offsets=hidden_value_offsets,
        candidate_offsets=candidate_offsets,
        candidate_values=candidate_values,
        worlds=config.worlds,
        require_byte_identical_worlds=config.require_byte_identical_worlds,
    )
    native_finished_at = time.perf_counter()
    root_errors = np.asarray(result.root_errors)
    root_unresolved_worlds = np.asarray(result.root_unresolved_worlds)
    if root_errors.shape != (len(roots),):
        raise RuntimeError("native engine fact root error shape is invalid")
    if root_unresolved_worlds.shape != (len(roots),):
        raise RuntimeError("native engine fact unresolved shape is invalid")
    failed_roots = np.flatnonzero(root_errors)
    if failed_roots.size:
        failed = int(failed_roots[0])
        raise RuntimeError(
            "native engine fact transition failed with "
            f"error={int(root_errors[failed])}"
        )
    return _RootChunkResult(
        root_output_rows=np.asarray(
            [root.output_row for root in roots],
            dtype=np.int64,
        ),
        output_rows=np.asarray(output_rows, dtype=np.int64),
        option_indices=np.asarray(option_indices, dtype=np.int64),
        features=result.features,
        masks=result.masks,
        root_count=len(roots),
        eligible_options=len(candidate_values),
        transition_count=len(candidate_values) * config.worlds,
        unresolved_worlds=int(root_unresolved_worlds.sum()),
        root_unresolved_worlds=np.asarray(
            root_unresolved_worlds,
            dtype=np.int64,
        ),
        started_at=chunk_started_at,
        finished_at=native_finished_at,
        sample_seconds=native_started_at - chunk_started_at,
        native_seconds=native_finished_at - native_started_at,
    )


def _root_chunks(
    roots: tuple[_RootProbe, ...],
    *,
    maximum_workers: int,
) -> tuple[tuple[_RootProbe, ...], ...]:
    """Create enough coarse tasks to balance workers without per-root IPC."""
    # Each task is already one ragged native call. Four tasks per process kept
    # CPUs busy but fragmented small route waves back into per-root ABI
    # crossings. At most one coarse call per two fact processes keeps useful
    # parallelism while giving the native backend enough roots to amortize
    # setup.
    target_chunks = max(1, min(len(roots), max(1, maximum_workers // 2)))
    chunk_size = (len(roots) + target_chunks - 1) // target_chunks
    return tuple(
        roots[start : start + chunk_size] for start in range(0, len(roots), chunk_size)
    )


def _eligible_options(
    view: NativeTrainingBatchView,
    row: int,
) -> tuple[int, ...]:
    start = int(view.option_offsets[row])
    stop = int(view.option_offsets[row + 1])
    return tuple(
        index - start
        for index in range(start, stop)
        if int(view.option_type[index]) in _ELIGIBLE_OPTION_TYPES
    )


def _fact_output_arrays(
    output_shape: tuple[int, int],
    *,
    pin_memory: bool,
) -> tuple[Float32Array, BoolArray]:
    """Allocate fact output in pinned host memory when CUDA will consume it."""
    if not pin_memory:
        return (
            np.zeros(
                (*output_shape, DYNAMIC_EFFECT_FEATURE_SIZE),
                dtype=np.float32,
            ),
            np.zeros(output_shape, dtype=np.bool_),
        )
    features = torch.zeros(
        (*output_shape, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=torch.float32,
        device="cpu",
        pin_memory=True,
    )
    masks = torch.zeros(
        output_shape,
        dtype=torch.bool,
        device="cpu",
        pin_memory=True,
    )
    return features.numpy(), masks.numpy()


def _fact_source_rows(
    values: Int64Array,
    *,
    size: int,
) -> Int64Array:
    """Validate a source selection without copying its nested engine columns."""
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("native fact source rows are invalid")
    selected = rows.astype(np.int64, copy=False)
    if (
        np.any(selected < 0)
        or np.any(selected >= size)
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("native fact source rows are invalid")
    return selected


def _observation_evidence(
    view: NativeTrainingBatchView,
    row: int,
) -> ObservationEvidence:
    perspective = int(view.select_player[row])
    if perspective not in (0, 1):
        raise ValueError("native fact root has no acting perspective")
    opponent = 1 - perspective
    own_non_deck: Counter[int] = Counter()
    own_deck: Counter[int] = Counter()
    opponent_visible: Counter[int] = Counter()
    visible_start = int(view.visible_card_offsets[row])
    visible_stop = int(view.visible_card_offsets[row + 1])
    for index in range(visible_start, visible_stop):
        owner = int(view.visible_card_owner[index])
        area = int(view.visible_card_area[index])
        card_id = int(view.visible_card_id[index])
        if card_id <= 0 or area == _VIRTUAL_AREA:
            continue
        if owner == perspective:
            if area == int(AreaType.DECK):
                own_deck[card_id] += 1
            else:
                own_non_deck[card_id] += 1
        elif owner == opponent and area != int(AreaType.DECK):
            opponent_visible[card_id] += 1
    attachment_start = int(view.attachment_offsets[row])
    attachment_stop = int(view.attachment_offsets[row + 1])
    for index in range(attachment_start, attachment_stop):
        card_id = int(view.attachment_card_id[index])
        parent = int(view.attachment_parent[index])
        if card_id <= 0 or not visible_start <= parent < visible_stop:
            continue
        owner = int(view.visible_card_owner[parent])
        if owner == perspective:
            own_non_deck[card_id] += 1
        elif owner == opponent:
            opponent_visible[card_id] += 1
    opponent_active_facedown = any(
        int(view.visible_card_owner[index]) == opponent
        and int(view.visible_card_area[index]) == int(AreaType.ACTIVE)
        and int(view.visible_card_area_index[index]) == 0
        and int(view.visible_card_id[index]) <= 0
        for index in range(visible_start, visible_stop)
    )
    return ObservationEvidence(
        your_index=perspective,
        opponent_index=opponent,
        your_deck_count=int(view.player_deck_counts[perspective][row]),
        your_prize_count=int(view.player_prize_counts[perspective][row]),
        opponent_deck_count=int(view.player_deck_counts[opponent][row]),
        opponent_prize_count=int(view.player_prize_counts[opponent][row]),
        opponent_hand_count=int(view.player_hand_counts[opponent][row]),
        opponent_active_facedown=opponent_active_facedown,
        setup_requires_basic_in_opponent_deck=(
            int(view.turn[row]) == 0 or opponent_active_facedown
        ),
        search_ignores_your_deck=bool(view.select_deck_visible[row]),
        your_non_deck_visible_counts=own_non_deck,
        your_visible_deck_counts=own_deck,
        opponent_current_visible_counts=opponent_visible,
    )


def _known_counts(
    known: NativeKnownOpponentBatch,
    row: int,
) -> Counter[int]:
    start = int(known.offsets[row])
    stop = int(known.offsets[row + 1])
    return Counter(
        {
            int(known.card_ids[index]): int(known.counts[index])
            for index in range(start, stop)
            if int(known.card_ids[index]) > 0 and int(known.counts[index]) > 0
        }
    )


def _root_seed(
    base_seed: int,
    game_id: str,
    perspective: int,
    decision_index: int,
) -> int:
    payload = (
        f"{base_seed}\x00{game_id}\x00{perspective}\x00{decision_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


__all__ = [
    "NativeEngineFactBatch",
    "NativeEngineFactSource",
    "NativeProspectiveEngineFactProducer",
]
