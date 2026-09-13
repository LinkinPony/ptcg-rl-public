"""Actor-side helpers for polling learner-published policy weights."""

from __future__ import annotations

import math
import pickle
import queue
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.profiling import StageTimer
from ptcg_rl.rl.engine_teacher import EngineTeacherProducer
from ptcg_rl.rl.experience import GameTrajectory, TensorTrajectoryRecorder
from ptcg_rl.rl.factual import FactualLaneConfig
from ptcg_rl.rl.inference_publication import read_inference_served_policy
from ptcg_rl.rl.learner import PublishedWeights, read_latest_published_weights
from ptcg_rl.rl.macro_credit import MacroCreditConfig
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.rollout import (
    RolloutActors,
    RolloutBeliefConfig,
    RolloutPlannerBehaviorService,
    RolloutPolicy,
    RolloutProbeConfig,
    RolloutRecorder,
    RolloutStepper,
    VectorPoolLike,
)
from ptcg_rl.rl.shared_weights import SharedMemoryWeightLoader
from ptcg_rl.rl.trajectory import CompletedTrajectory

if TYPE_CHECKING:
    from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import ReanalysisRoot


class TrajectoryQueue(Protocol):
    """Queue-like protocol used by actor trajectory producers."""

    def put(
        self,
        item: GameTrajectory,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put one completed trajectory, raising ``queue.Full`` on timeout."""


class CollectionGate(Protocol):
    """Process-shared event controlling whether an actor may start another step."""

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for one collection credit and report whether it became available."""


class ArchiveTrajectoryRecorder(RolloutRecorder, Protocol):
    """Recorder that emits BC-compatible completed trajectories for archiving."""

    def pop_completed(self) -> tuple[CompletedTrajectory, ...]:
        """Return completed archive trajectories waiting to be written."""


class ActorWeightLoaderConfig(BaseModel):
    """Config for actor-side policy weight loading."""

    model_config = ConfigDict(extra="forbid")

    map_location: str = "cpu"
    strict: bool = True

    @field_validator("map_location")
    @classmethod
    def valid_map_location(cls, value: str) -> str:
        """Reject empty map locations."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("map_location must be non-empty")
        return cleaned


class TrajectoryQueueProducerConfig(BaseModel):
    """Config for bounded actor-to-learner trajectory queue puts."""

    model_config = ConfigDict(extra="forbid")

    put_timeout_seconds: float = 0.1
    max_full_retries: int | None = None
    profile_pickle: bool = False

    @field_validator("put_timeout_seconds")
    @classmethod
    def valid_put_timeout(cls, value: float) -> float:
        """Reject invalid queue put timeouts."""
        if value <= 0.0:
            raise ValueError("put_timeout_seconds must be positive")
        return value

    @field_validator("max_full_retries")
    @classmethod
    def valid_optional_retry_count(cls, value: int | None) -> int | None:
        """Reject invalid optional retry counts."""
        if value is not None and value < 0:
            raise ValueError("max_full_retries must be non-negative")
        return value


class ActorLoopConfig(BaseModel):
    """Config for the single-process body of an actor rollout loop."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = 100_000
    sampling_temperature: float = 1.0
    queue: TrajectoryQueueProducerConfig = TrajectoryQueueProducerConfig()
    rollout_probe: RolloutProbeConfig = RolloutProbeConfig()
    rollout_belief: RolloutBeliefConfig = RolloutBeliefConfig()
    factual: FactualLaneConfig = FactualLaneConfig()
    macro_credit: MacroCreditConfig = MacroCreditConfig()
    stats_callback_interval_seconds: float = 0.0
    seed: int = 0
    reanalysis_root_probability: float = 0.0
    record_public_event_deltas: bool = False

    @field_validator("max_iterations")
    @classmethod
    def valid_max_iterations(cls, value: int) -> int:
        """Reject invalid loop limits."""
        if value <= 0:
            raise ValueError("max_iterations must be positive")
        return value

    @field_validator("sampling_temperature")
    @classmethod
    def valid_sampling_temperature(cls, value: float) -> float:
        """Reject invalid sampling temperatures."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("sampling_temperature must be finite and positive")
        return value

    @field_validator("stats_callback_interval_seconds")
    @classmethod
    def valid_stats_callback_interval(cls, value: float) -> float:
        """Reject negative actor progress callback intervals."""
        if value < 0.0:
            raise ValueError("stats_callback_interval_seconds must be non-negative")
        return value

    @field_validator("reanalysis_root_probability")
    @classmethod
    def valid_reanalysis_probability(cls, value: float) -> float:
        """Require a fixed non-adaptive auxiliary root share."""
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("reanalysis_root_probability must be in [0, 1]")
        return value


@dataclass(frozen=True)
class LoadedPolicyWeights:
    """One actor-side successful weight load."""

    version: int
    path: Path
    model_fingerprint: str | None


@dataclass(frozen=True)
class TrajectoryQueuePutStats:
    """Counters from putting completed trajectories into a bounded queue."""

    trajectories: int
    queued_trajectory_decisions: int
    full_retries: int
    put_seconds: float
    pickle_probe_seconds: float
    pickle_probe_bytes: int


@dataclass(frozen=True)
class ActorLoopStats:
    """Counters from one actor rollout loop."""

    elapsed_seconds: float
    iterations: int
    policy_actions: int
    forced_actions: int
    scripted_actions: int
    recorded_decisions: int
    finished_games: int
    queued_trajectories: int
    queued_trajectory_decisions: int
    queue_full_retries: int
    queue_put_seconds: float
    queue_pickle_probe_seconds: float
    queue_pickle_probe_bytes: int
    loaded_weight_versions: tuple[int, ...]
    stage_timings: Mapping[str, Mapping[str, float | int]]
    inference_client: Mapping[str, Any] = field(default_factory=dict)
    rollout_features: Mapping[str, Any] = field(default_factory=dict)


class WeightPollingLoader:
    """Poll ``latest.json`` and load newer policy weights into an actor model."""

    def __init__(
        self,
        directory: Path,
        config: ActorWeightLoaderConfig | None = None,
    ) -> None:
        """Initialize loader state."""
        self.directory = Path(directory)
        self.config = config or ActorWeightLoaderConfig()
        self._loaded_version: int | None = None

    @property
    def loaded_version(self) -> int | None:
        """Return the newest version loaded by this actor."""
        return self._loaded_version

    def has_newer_publication(self) -> bool:
        """Check the publication pointer without mutating the actor model."""
        latest = read_latest_published_weights(self.directory)
        return bool(
            latest is not None
            and (self._loaded_version is None or latest.version > self._loaded_version)
        )

    def poll(self, model: torch.nn.Module) -> LoadedPolicyWeights | None:
        """Load the latest published weights if they are newer than the model."""
        latest = read_latest_published_weights(self.directory)
        if latest is None or (
            self._loaded_version is not None and latest.version <= self._loaded_version
        ):
            return None
        model_fingerprint = self._load(model, latest)
        self._loaded_version = latest.version
        return LoadedPolicyWeights(
            version=latest.version,
            path=latest.path,
            model_fingerprint=model_fingerprint,
        )

    def _load(
        self,
        model: torch.nn.Module,
        latest: PublishedWeights,
    ) -> str | None:
        checkpoint = torch.load(latest.path, map_location=self.config.map_location)
        state_dict = _state_dict_from_checkpoint(checkpoint)
        expected_fingerprint = latest.model_fingerprint
        recurrent_config = getattr(getattr(model, "config", None), "recurrent", None)
        if expected_fingerprint is None and recurrent_config is not None:
            raise RuntimeError(
                "recurrent actor publication omitted its model fingerprint"
            )
        if expected_fingerprint is not None:
            actual_fingerprint = canonical_model_state_fingerprint(
                cast(Mapping[str, torch.Tensor], state_dict)
            )
            if actual_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    "published actor checkpoint differs from latest fingerprint"
                )
        model.load_state_dict(
            state_dict,
            strict=self.config.strict,
        )
        return expected_fingerprint


@dataclass(frozen=True)
class RecurrentPublicationVersions:
    """One due learner/inference version observation."""

    learner_latest_version: int | None
    inference_served_version: int | None


class RecurrentStaleGameWatcher:
    """Poll lightweight publication pointers at a bounded actor interval."""

    def __init__(
        self,
        weights_dir: Path,
        *,
        poll_interval_seconds: float,
    ) -> None:
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0.0:
            raise ValueError("recurrent stale-game poll interval must be positive")
        self.weights_dir = Path(weights_dir)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._shared_loader = SharedMemoryWeightLoader(self.weights_dir)
        self._last_poll_at: float | None = None

    def poll(self) -> RecurrentPublicationVersions | None:
        """Read pointers only when the configured monotonic interval is due."""
        now = time.monotonic()
        if (
            self._last_poll_at is not None
            and now - self._last_poll_at < self.poll_interval_seconds
        ):
            return None
        shared_version = self._shared_loader.latest_version()
        disk = read_latest_published_weights(self.weights_dir)
        candidate_versions = [
            version
            for version in (
                shared_version,
                None if disk is None else disk.version,
            )
            if version is not None
        ]
        served = read_inference_served_policy(self.weights_dir)
        self._last_poll_at = now
        return RecurrentPublicationVersions(
            learner_latest_version=(
                max(candidate_versions) if candidate_versions else None
            ),
            inference_served_version=None if served is None else served.version,
        )


class TrajectoryQueueProducer:
    """Put completed trajectories into the learner queue with bounded backpressure."""

    def __init__(
        self,
        trajectory_queue: TrajectoryQueue,
        config: TrajectoryQueueProducerConfig | None = None,
    ) -> None:
        """Initialize a queue producer."""
        self._queue = trajectory_queue
        self._config = config or TrajectoryQueueProducerConfig()

    def put_many(
        self,
        trajectories: Iterable[GameTrajectory],
    ) -> TrajectoryQueuePutStats:
        """Put every trajectory, retrying when the queue is temporarily full."""
        full_retries = 0
        pickle_probe_bytes = 0
        pickle_probe_seconds = 0.0
        put_count = 0
        queued_trajectory_decisions = 0
        put_seconds = 0.0
        for trajectory in trajectories:
            trajectory_decisions = trajectory.decision_count
            if self._config.profile_pickle:
                pickle_start = time.perf_counter()
                payload = pickle.dumps(trajectory, protocol=pickle.HIGHEST_PROTOCOL)
                pickle_probe_seconds += time.perf_counter() - pickle_start
                pickle_probe_bytes += len(payload)
            while True:
                try:
                    put_start = time.perf_counter()
                    self._queue.put(
                        trajectory,
                        timeout=self._config.put_timeout_seconds,
                    )
                    put_seconds += time.perf_counter() - put_start
                    put_count += 1
                    queued_trajectory_decisions += trajectory_decisions
                    break
                except queue.Full as exc:
                    put_seconds += time.perf_counter() - put_start
                    full_retries += 1
                    if (
                        self._config.max_full_retries is not None
                        and full_retries > self._config.max_full_retries
                    ):
                        raise TimeoutError("trajectory queue stayed full") from exc
        return TrajectoryQueuePutStats(
            trajectories=put_count,
            queued_trajectory_decisions=queued_trajectory_decisions,
            full_retries=full_retries,
            put_seconds=put_seconds,
            pickle_probe_seconds=pickle_probe_seconds,
            pickle_probe_bytes=pickle_probe_bytes,
        )


def run_actor_loop(
    *,
    pool: VectorPoolLike,
    actors: RolloutActors,
    trajectory_queue: TrajectoryQueue,
    config: ActorLoopConfig | None = None,
    weight_loader: WeightPollingLoader | None = None,
    weight_model: torch.nn.Module | None = None,
    recurrent_stale_game_watcher: RecurrentStaleGameWatcher | None = None,
    recurrent_max_staleness: int | None = None,
    device: torch.device | str | None = None,
    archive_recorder: ArchiveTrajectoryRecorder | None = None,
    decision_callback: Callable[[Any], None] | None = None,
    finished_callback: Callable[[Any], None] | None = None,
    discard_callback: Callable[[Sequence[str], str], None] | None = None,
    post_step_callback: Callable[[], None] | None = None,
    stats_callback: Callable[[ActorLoopStats], None] | None = None,
    trajectory_transform: Callable[[GameTrajectory], GameTrajectory] | None = None,
    engine_teacher_producer: EngineTeacherProducer | None = None,
    planner_behavior_service: RolloutPlannerBehaviorService | None = None,
    collection_gate: CollectionGate | None = None,
    reanalysis_root_callback: (Callable[[ReanalysisRoot], bool | None] | None) = None,
) -> ActorLoopStats:
    """Run the actor loop body and enqueue completed tensor trajectories."""
    cfg = config or ActorLoopConfig()
    if (recurrent_stale_game_watcher is None) != (recurrent_max_staleness is None):
        raise ValueError(
            "recurrent stale-game watcher and max_staleness must be configured together"
        )
    if recurrent_max_staleness is not None and recurrent_max_staleness < 0:
        raise ValueError("recurrent max_staleness must be non-negative")
    if cfg.record_public_event_deltas and archive_recorder is not None:
        raise ValueError(
            "recurrent public events cannot use the legacy archive recorder"
        )
    recorder = TensorTrajectoryRecorder(
        reanalysis_root_callback=reanalysis_root_callback,
    )
    recorders: list[RolloutRecorder] = [recorder]
    if archive_recorder is not None:
        recorders.append(archive_recorder)
    if decision_callback is not None or finished_callback is not None:
        recorders.append(
            _FinishedCallbackRecorder(
                finished_callback=finished_callback,
                decision_callback=decision_callback,
            )
        )
    rollout_recorder: RolloutRecorder = (
        recorders[0]
        if len(recorders) == 1
        else _MultiplexRolloutRecorder(tuple(recorders))
    )
    producer = TrajectoryQueueProducer(trajectory_queue, cfg.queue)
    timer = StageTimer()
    stepper = RolloutStepper(
        pool=pool,
        actors=actors,
        recorder=rollout_recorder,
        temperature=cfg.sampling_temperature,
        device=device,
        timer=timer,
        probe_config=cfg.rollout_probe,
        belief_config=cfg.rollout_belief,
        factual_config=cfg.factual,
        macro_credit_config=cfg.macro_credit,
        engine_teacher_producer=engine_teacher_producer,
        planner_behavior_service=planner_behavior_service,
        seed=cfg.seed,
        record_public_event_deltas=cfg.record_public_event_deltas,
        reanalysis_root_probability=cfg.reanalysis_root_probability,
    )

    started = time.perf_counter()
    policy_actions = 0
    forced_actions = 0
    scripted_actions = 0
    recorded_decisions = 0
    finished_games = 0
    queued_trajectories = 0
    queued_trajectory_decisions = 0
    queue_full_retries = 0
    queue_pickle_probe_bytes = 0
    queue_pickle_probe_seconds = 0.0
    queue_put_seconds = 0.0
    loaded_versions: list[int] = []
    iterations = 0
    last_stats_callback_at = 0.0

    def current_stats() -> ActorLoopStats:
        return ActorLoopStats(
            elapsed_seconds=time.perf_counter() - started,
            iterations=iterations,
            policy_actions=policy_actions,
            forced_actions=forced_actions,
            scripted_actions=scripted_actions,
            recorded_decisions=recorded_decisions,
            finished_games=finished_games,
            queued_trajectories=queued_trajectories,
            queued_trajectory_decisions=queued_trajectory_decisions,
            queue_full_retries=queue_full_retries,
            queue_put_seconds=queue_put_seconds,
            queue_pickle_probe_seconds=queue_pickle_probe_seconds,
            queue_pickle_probe_bytes=queue_pickle_probe_bytes,
            loaded_weight_versions=tuple(loaded_versions),
            stage_timings=_merged_stage_timings(
                timer.summary(),
                _rollout_inference_client_stage_timings(actors),
            ),
            inference_client=_rollout_inference_client_summary(actors),
            rollout_features=stepper.feature_summary,
        )

    def maybe_report_stats(*, force: bool = False) -> None:
        """Build the potentially expensive snapshot only when it will be used."""
        nonlocal last_stats_callback_at
        if stats_callback is None:
            return
        now = time.perf_counter()
        if (
            not force
            and last_stats_callback_at > 0.0
            and now - last_stats_callback_at < cfg.stats_callback_interval_seconds
        ):
            return
        last_stats_callback_at = now
        stats_callback(current_stats())

    actor_error: BaseException | None = None
    try:
        while iterations < cfg.max_iterations:
            if collection_gate is not None:
                while not collection_gate.wait(timeout=1.0):
                    maybe_report_stats()
            loaded = _poll_actor_weights_at_game_barrier(
                weight_loader=weight_loader,
                weight_model=weight_model,
                stepper=stepper,
            )
            if loaded is not None:
                loaded_versions.append(loaded.version)
                _publish_loaded_policy_identity(
                    actors.candidate_policy,
                    loaded,
                )
            step_stats = stepper.step()
            if recurrent_stale_game_watcher is not None:
                versions = recurrent_stale_game_watcher.poll()
                if versions is not None:
                    stepper.recycle_stale_recurrent_games(
                        learner_latest_version=versions.learner_latest_version,
                        inference_served_version=versions.inference_served_version,
                        max_staleness=cast(int, recurrent_max_staleness),
                        lifecycle_discard_callback=discard_callback,
                    )
            if post_step_callback is not None:
                post_step_callback()
            iterations += 1
            policy_actions += step_stats.policy_actions
            forced_actions += step_stats.forced_actions
            scripted_actions += step_stats.scripted_actions
            recorded_decisions += step_stats.recorded_decisions
            finished_games += step_stats.finished_games
            put_stats = producer.put_many(
                _queue_trajectories(
                    recorder,
                    archive_recorder,
                    trajectory_transform,
                )
            )
            queued_trajectories += put_stats.trajectories
            queued_trajectory_decisions += put_stats.queued_trajectory_decisions
            queue_full_retries += put_stats.full_retries
            queue_put_seconds += put_stats.put_seconds
            queue_pickle_probe_seconds += put_stats.pickle_probe_seconds
            queue_pickle_probe_bytes += put_stats.pickle_probe_bytes
            maybe_report_stats()
            if step_stats.pending_games == 0 and step_stats.finished_games == 0:
                break

        stepper.flush_engine_teacher()
        put_stats = producer.put_many(
            _queue_trajectories(
                recorder,
                archive_recorder,
                trajectory_transform,
            )
        )
        queued_trajectories += put_stats.trajectories
        queued_trajectory_decisions += put_stats.queued_trajectory_decisions
        queue_full_retries += put_stats.full_retries
        queue_put_seconds += put_stats.put_seconds
        queue_pickle_probe_seconds += put_stats.pickle_probe_seconds
        queue_pickle_probe_bytes += put_stats.pickle_probe_bytes
        maybe_report_stats(force=True)
        return current_stats()
    except BaseException as exc:
        actor_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            stepper.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        close_teacher = getattr(engine_teacher_producer, "close", None)
        if callable(close_teacher):
            try:
                close_teacher()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if actor_error is not None:
                for cleanup_error in cleanup_errors:
                    actor_error.add_note(
                        "actor cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            else:
                primary_cleanup_error = cleanup_errors[0]
                for cleanup_error in cleanup_errors[1:]:
                    primary_cleanup_error.add_note(
                        "additional actor cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise primary_cleanup_error


class _MultiplexRolloutRecorder:
    """Forward rollout recorder events to multiple recorders."""

    def __init__(self, recorders: tuple[RolloutRecorder, ...]) -> None:
        self._recorders = recorders

    def record(self, decision: Any) -> None:
        for recorder in self._recorders:
            recorder.record(decision)

    def finalize(self, finished: Any) -> None:
        for recorder in self._recorders:
            recorder.finalize(finished)

    def discard(self, game_id: str, *, reason: str) -> None:
        """Discard one game from each sink in recorder order."""
        self.discard_many((game_id,), reason=reason)

    def discard_many(self, game_ids: Sequence[str], *, reason: str) -> None:
        """Discard a batch from data sinks before lifecycle callbacks."""
        concrete = tuple(str(game_id) for game_id in game_ids)
        for recorder in self._recorders:
            discard_many = getattr(recorder, "discard_many", None)
            if callable(discard_many):
                discard_many(concrete, reason=reason)
                continue
            for game_id in concrete:
                recorder.discard(game_id, reason=reason)


def _rollout_inference_client_summary(
    actors: RolloutActors,
) -> dict[str, Mapping[str, Any]]:
    summaries: dict[str, Mapping[str, Any]] = {}
    for index, policy in enumerate(_iter_unique_rollout_policies(actors)):
        summary_fn = getattr(policy, "inference_client_summary", None)
        if not callable(summary_fn):
            continue
        summary = cast(Mapping[str, Any], summary_fn())
        policy_id = str(summary.get("policy_id", f"policy-{index}"))
        key = policy_id if policy_id not in summaries else f"{policy_id}-{index}"
        summaries[key] = summary
    return summaries


def _rollout_inference_client_stage_timings(
    actors: RolloutActors,
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = {}
    for policy in _iter_unique_rollout_policies(actors):
        summary_fn = getattr(policy, "inference_client_stage_timings", None)
        if not callable(summary_fn):
            continue
        _merge_stage_timing_into(
            merged,
            cast(Mapping[str, Mapping[str, float | int]], summary_fn()),
        )
    return merged


def _iter_unique_rollout_policies(actors: RolloutActors) -> Iterable[RolloutPolicy]:
    seen: set[int] = set()
    policies: list[RolloutPolicy] = [actors.candidate_policy]
    if actors.frozen_policy is not None:
        policies.append(actors.frozen_policy)
    policies.extend(actors.frozen_policies.values())
    for policy in policies:
        policy_id = id(policy)
        if policy_id in seen:
            continue
        seen.add(policy_id)
        yield policy


def _merged_stage_timings(
    first: Mapping[str, Mapping[str, float | int]],
    second: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = {}
    _merge_stage_timing_into(merged, first)
    _merge_stage_timing_into(merged, second)
    return merged


def _merge_stage_timing_into(
    target: dict[str, dict[str, float | int]],
    source: Mapping[str, Mapping[str, float | int]],
) -> None:
    for stage, timing in source.items():
        seconds = float(timing.get("seconds", 0.0))
        count = int(timing.get("count", 0))
        if count <= 0 and seconds <= 0.0:
            continue
        current = target.setdefault(stage, {"seconds": 0.0, "count": 0, "mean_ms": 0.0})
        current["seconds"] = float(current["seconds"]) + seconds
        current["count"] = int(current["count"]) + count
        current_count = int(current["count"])
        current["mean_ms"] = (
            1000.0 * float(current["seconds"]) / float(current_count)
            if current_count > 0
            else 0.0
        )


class _FinishedCallbackRecorder:
    """Recorder shim that forwards decision and finished-game callbacks."""

    def __init__(
        self,
        *,
        finished_callback: Callable[[Any], None] | None,
        decision_callback: Callable[[Any], None] | None = None,
    ) -> None:
        self._finished_callback = finished_callback
        self._decision_callback = decision_callback

    def record(self, decision: Any) -> None:
        if self._decision_callback is not None:
            self._decision_callback(decision)

    def finalize(self, finished: Any) -> None:
        if self._finished_callback is not None:
            self._finished_callback(finished)

    def discard(self, game_id: str, *, reason: str) -> None:
        """Ignore non-terminal data-sink disposal in the lifecycle observer."""
        del game_id, reason


def _pop_completed_trajectories(
    recorder: TensorTrajectoryRecorder,
    archive_recorder: ArchiveTrajectoryRecorder | None,
) -> tuple[GameTrajectory, ...]:
    completed = recorder.pop_completed()
    if archive_recorder is None:
        return completed

    archive_by_game_id = {
        trajectory.game_id: trajectory
        for trajectory in archive_recorder.pop_completed()
    }
    archived: list[GameTrajectory] = []
    for trajectory in completed:
        archive = archive_by_game_id.pop(trajectory.game_id, None)
        if archive is None:
            raise RuntimeError(
                f"archive recorder missed completed game {trajectory.game_id}"
            )
        archived.append(replace(trajectory, archive=archive))
    if archive_by_game_id:
        extra = ", ".join(sorted(archive_by_game_id))
        raise RuntimeError(f"archive recorder emitted unmatched games: {extra}")
    return tuple(archived)


def _queue_trajectories(
    recorder: TensorTrajectoryRecorder,
    archive_recorder: ArchiveTrajectoryRecorder | None,
    trajectory_transform: Callable[[GameTrajectory], GameTrajectory] | None,
) -> tuple[GameTrajectory, ...]:
    trajectories = _pop_completed_trajectories(recorder, archive_recorder)
    if trajectory_transform is None:
        return trajectories
    return tuple(trajectory_transform(trajectory) for trajectory in trajectories)


def _state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError("published weights must be a state_dict or checkpoint mapping")


def _set_policy_version(policy: object, version: int) -> None:
    current = getattr(policy, "policy_version", None)
    if current is not None and not callable(current):
        cast(Any, policy).policy_version = int(version)


def _publish_loaded_policy_identity(
    policy: object,
    loaded: LoadedPolicyWeights,
) -> None:
    """Refresh version and full artifact identity after a verified load."""
    reset_contexts = getattr(policy, "reset_planner_contexts", None)
    if callable(reset_contexts):
        reset_contexts(
            policy_version=loaded.version,
            model_fingerprint=loaded.model_fingerprint,
        )
        return
    recurrent_enabled = getattr(policy, "recurrent_enabled", False)
    if callable(recurrent_enabled):
        recurrent_enabled = recurrent_enabled()
    if recurrent_enabled:
        raise RuntimeError(
            "recurrent actor policy cannot refresh its artifact identity"
        )
    _set_policy_version(policy, loaded.version)


def _poll_actor_weights_at_game_barrier(
    *,
    weight_loader: WeightPollingLoader | None,
    weight_model: torch.nn.Module | None,
    stepper: RolloutStepper,
) -> LoadedPolicyWeights | None:
    """Load only outside a live recurrent cohort, draining when necessary."""
    if weight_loader is None or weight_model is None:
        return None
    if stepper.recurrent_drain_active:
        if stepper.has_active_recurrent_sequences:
            return None
        loaded = weight_loader.poll(weight_model)
        stepper.finish_recurrent_drain()
        return loaded
    if stepper.has_active_recurrent_sequences and weight_loader.has_newer_publication():
        stepper.begin_recurrent_drain()
        return None
    return weight_loader.poll(weight_model)
