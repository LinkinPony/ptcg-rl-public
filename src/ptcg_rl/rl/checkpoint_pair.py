"""Asynchronous publication of exact policy and learner-state pairs."""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, cast

import torch

from ptcg_rl.rl.checkpoint_pair_io import (
    atomic_write_bytes,
    fsync_directory,
    json_payload,
    pair_manifest_version,
    publish_bytes_file,
    publish_torch_file,
)
from ptcg_rl.rl.durable_writer import AsyncWriteBusyError, freeze_durable_value
from ptcg_rl.rl.learner import (
    PublishedWeights,
    WeightPublisher,
    prepared_checkpoint_payload,
)
from ptcg_rl.rl.model_publication import PreparedModelState, prepare_model_state
from ptcg_rl.rl.training_resume import (
    PreparedTrainingState,
    commit_prepared_training_state,
    prepare_training_state,
    prepared_training_state_payload,
    prune_training_states,
    training_state_path,
)

_PAIR_SCHEMA_VERSION = 1
_PAIR_FORMAT = "exact_policy_learner_pair_v1"


@dataclass(frozen=True, slots=True)
class CheckpointPairTiming:
    """Foreground freeze and background publication wall times."""

    freeze_seconds: float
    policy_write_seconds: float
    training_state_write_seconds: float
    commit_seconds: float
    background_seconds: float


@dataclass(frozen=True, slots=True)
class PublishedCheckpointPair:
    """One durable policy and exact learner-state pair."""

    version: int
    policy: PublishedWeights
    training_state_path: Path
    pair_manifest_path: Path
    latest_pair_path: Path
    policy_size_bytes: int
    policy_sha256: str
    training_state_size_bytes: int
    training_state_sha256: str
    published_at: str
    timing: CheckpointPairTiming
    post_commit_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _FrozenCheckpointPair:
    """All mutable state detached before the background handoff."""

    policy: PreparedModelState
    training: PreparedTrainingState
    metadata: Mapping[str, Any]
    checkpoint_fields: Mapping[str, Any] | None
    freeze_seconds: float


class AsyncCheckpointPairPublisher:
    """Publish at most one exact checkpoint pair in a background thread.

    ``submit`` performs the expensive correctness boundary synchronously: all
    live tensors are copied to independent CPU storage. The worker owns only
    that immutable snapshot and performs serialization, hashing, fsync, and
    atomic pointer publication. A second submission either waits for the first
    or raises :class:`AsyncWriteBusyError`.
    """

    def __init__(
        self,
        weight_publisher: WeightPublisher,
        resume_directory: Path,
        *,
        keep_last: int,
        retain_every_versions: int | None = None,
        pair_directory: Path | None = None,
    ) -> None:
        """Initialize a single-inflight exact checkpoint lane."""
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        if retain_every_versions is not None and retain_every_versions <= 0:
            raise ValueError("retain_every_versions must be positive when set")
        if (
            weight_publisher.config.keep_last != keep_last
            or weight_publisher.config.retain_every_versions != retain_every_versions
        ):
            raise ValueError(
                "policy, learner-state, and pair retention policies must match"
            )
        self.weight_publisher = weight_publisher
        self.resume_directory = Path(resume_directory).resolve()
        weights_parent = Path(weight_publisher.directory).resolve().parent
        resume_parent = self.resume_directory.parent
        if pair_directory is None:
            if weights_parent != resume_parent:
                raise ValueError(
                    "weights and resume directories require an explicit pair directory"
                )
            pair_directory = resume_parent / "checkpoint_pairs"
        self.pair_directory = Path(pair_directory).resolve()
        self.keep_last = int(keep_last)
        self.retain_every_versions = retain_every_versions
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="checkpoint-pair",
        )
        self._lock = threading.Lock()
        self._pending: Future[PublishedCheckpointPair] | None = None
        self._last_result: PublishedCheckpointPair | None = None
        self._closed = False

    @property
    def latest_pair_path(self) -> Path:
        """Return the authoritative atomic pointer to a complete pair."""
        return self.pair_directory / "latest.json"

    @property
    def busy(self) -> bool:
        """Return whether one pair is still in flight."""
        with self._lock:
            return self._pending is not None and not self._pending.done()

    @property
    def last_result(self) -> PublishedCheckpointPair | None:
        """Return the last pair observed at a barrier."""
        with self._lock:
            return self._last_result

    def submit(
        self,
        state_dict: Mapping[str, Any] | PreparedModelState,
        *,
        version: int,
        metadata: Mapping[str, Any] | None,
        checkpoint_fields: Mapping[str, Any] | None,
        completed_iterations: int,
        total_optimizer_updates: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        planned_scheduler_steps: int,
        resume_config_sha256: str,
        optimizer_parameter_names: Sequence[Sequence[str]],
        auxiliary_state: Mapping[str, Any] | None = None,
        take_prepared_ownership: bool = False,
        take_auxiliary_ownership: bool = False,
        block: bool = True,
    ) -> Future[PublishedCheckpointPair]:
        """Freeze one exact pair and hand only immutable CPU state to the worker.

        ``take_prepared_ownership`` avoids a second model-sized copy only when
        the caller transfers a newly created :class:`PreparedModelState` and
        will neither mutate nor reuse its staging tensors before the barrier.
        ``take_auxiliary_ownership`` provides the same explicit transfer for a
        freshly returned CPU-only target-network state mapping.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint pair publisher is closed")
            previous = self._pending
            if previous is not None and not previous.done() and not block:
                raise AsyncWriteBusyError("one checkpoint pair is already in flight")
            if previous is not None:
                self._last_result = previous.result()
                self._pending = None
            self._validate_new_version(version)
            frozen = _freeze_checkpoint_pair(
                state_dict,
                version=version,
                metadata=metadata,
                checkpoint_fields=checkpoint_fields,
                completed_iterations=completed_iterations,
                total_optimizer_updates=total_optimizer_updates,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                planned_scheduler_steps=planned_scheduler_steps,
                resume_config_sha256=resume_config_sha256,
                optimizer_parameter_names=optimizer_parameter_names,
                auxiliary_state=auxiliary_state,
                take_prepared_ownership=take_prepared_ownership,
                take_auxiliary_ownership=take_auxiliary_ownership,
            )
            future = self._executor.submit(self._publish_frozen_pair, frozen)
            self._pending = future
            return future

    def publish(
        self,
        state_dict: Mapping[str, Any] | PreparedModelState,
        **kwargs: Any,
    ) -> PublishedCheckpointPair:
        """Synchronous wrapper retaining a simple migration path."""
        self.submit(state_dict, **kwargs)
        result = self.barrier()
        if result is None:
            raise RuntimeError("synchronous checkpoint publication returned no pair")
        return result

    def barrier(self) -> PublishedCheckpointPair | None:
        """Wait for the current pair and surface its worker exception."""
        with self._lock:
            pending = self._pending
            if pending is None:
                return self._last_result
            result = pending.result()
            self._last_result = result
            self._pending = None
            return result

    def close(self) -> None:
        """Wait for durability, propagate failures, and stop the worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.barrier()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()

    def _validate_new_version(self, version: int) -> None:
        if version < 0:
            raise ValueError("version must be non-negative")
        paths = (
            self.weight_publisher.path_for_version(version),
            training_state_path(
                self.resume_directory,
                policy_version=version,
            ),
            self._pair_manifest_path(version),
        )
        existing = tuple(path for path in paths if path.exists())
        if existing:
            raise FileExistsError(
                "immutable checkpoint pair version already exists: "
                + ", ".join(str(path) for path in existing)
            )

    def _pair_manifest_path(self, version: int) -> Path:
        return self.pair_directory / f"checkpoint_pair_v{version}.json"

    def _publish_frozen_pair(
        self,
        frozen: _FrozenCheckpointPair,
    ) -> PublishedCheckpointPair:
        started_at = time.perf_counter()
        version = frozen.training.policy_version
        policy_path = self.weight_publisher.path_for_version(version)
        state_path = training_state_path(
            self.resume_directory,
            policy_version=version,
        )
        pair_manifest_path = self._pair_manifest_path(version)
        published_paths: list[Path] = []
        committed = False
        try:
            policy_started_at = time.perf_counter()
            policy_size, policy_sha256 = publish_torch_file(
                policy_path,
                prepared_checkpoint_payload(
                    frozen.policy,
                    checkpoint_fields=frozen.checkpoint_fields,
                ),
            )
            published_paths.append(policy_path)
            policy_seconds = time.perf_counter() - policy_started_at
            state_started_at = time.perf_counter()
            state_size, state_sha256 = publish_torch_file(
                state_path,
                prepared_training_state_payload(
                    frozen.training,
                    policy_size_bytes=policy_size,
                    policy_sha256=policy_sha256,
                ),
            )
            published_paths.append(state_path)
            state_seconds = time.perf_counter() - state_started_at
            published_at = datetime.now(UTC).isoformat()
            pair_record = {
                "format": _PAIR_FORMAT,
                "schema_version": _PAIR_SCHEMA_VERSION,
                "version": version,
                "published_at": published_at,
                "policy": {
                    "path": str(policy_path.resolve()),
                    "size_bytes": policy_size,
                    "sha256": policy_sha256,
                    "model_fingerprint": frozen.policy.model_fingerprint,
                },
                "training_state": {
                    "path": str(state_path.resolve()),
                    "size_bytes": state_size,
                    "sha256": state_sha256,
                    "policy_sha256": policy_sha256,
                },
                "metadata": dict(frozen.metadata),
            }
            pair_payload = json_payload(pair_record)
            commit_started_at = time.perf_counter()
            publish_bytes_file(pair_manifest_path, pair_payload)
            published_paths.append(pair_manifest_path)
            # This is the transaction commit point. Readers either retain the
            # previous complete pair or discover both immutable numbered files.
            # Legacy pointers are compatibility indexes repaired after this
            # authoritative pair pointer advances.
            try:
                atomic_write_bytes(
                    self.latest_pair_path,
                    pair_payload,
                    overwrite=True,
                )
            except BaseException:
                # ``atomic_write_bytes`` can fail while fsyncing the directory
                # after ``os.replace`` has already exposed this exact pointer.
                # Preserve the complete pair in that uncertain-but-visible
                # state; deleting it would turn the new pointer into a dangling
                # reference.
                committed = _pair_pointer_maybe_published(
                    self.latest_pair_path,
                    pair_payload,
                )
                raise
            committed = True

            # The authoritative pair is now complete. Legacy pointers and
            # retention are repairable compatibility maintenance: failure in
            # any one must not turn this successfully committed pair into a
            # failed Future or prevent the other maintenance steps.
            post_commit_errors: list[str] = []
            policy = PublishedWeights(
                version=version,
                path=policy_path,
                latest_path=self.weight_publisher.latest_path,
                published_at=published_at,
                metadata=frozen.metadata,
                model_fingerprint=frozen.policy.model_fingerprint,
            )
            try:
                policy = self.weight_publisher.commit_prepared(
                    policy_path,
                    version=version,
                    metadata=frozen.metadata,
                    model_fingerprint=frozen.policy.model_fingerprint,
                    prune=False,
                )
            except Exception as exc:
                post_commit_errors.append(_post_commit_error("policy_pointer", exc))
            try:
                commit_prepared_training_state(
                    self.resume_directory,
                    path=state_path,
                    prepared=frozen.training,
                    policy_size_bytes=policy_size,
                    policy_sha256=policy_sha256,
                    keep_last=self.keep_last,
                    retain_every_versions=self.retain_every_versions,
                    prune=False,
                )
            except Exception as exc:
                post_commit_errors.append(_post_commit_error("resume_pointer", exc))
            for stage, prune in (
                ("policy_prune", self.weight_publisher.prune),
                (
                    "resume_prune",
                    lambda: prune_training_states(
                        self.resume_directory,
                        keep_last=self.keep_last,
                        retain_every_versions=self.retain_every_versions,
                    ),
                ),
                ("pair_manifest_prune", self._prune_pair_manifests),
            ):
                try:
                    prune()
                except Exception as exc:
                    post_commit_errors.append(_post_commit_error(stage, exc))
            if post_commit_errors:
                # Warning filters may promote RuntimeWarning to an exception;
                # diagnostics must not reverse an already completed commit.
                with suppress(RuntimeWarning):
                    warnings.warn(
                        "checkpoint pair committed with repairable post-commit "
                        f"errors: {'; '.join(post_commit_errors)}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            commit_seconds = time.perf_counter() - commit_started_at
            return PublishedCheckpointPair(
                version=version,
                policy=policy,
                training_state_path=state_path,
                pair_manifest_path=pair_manifest_path,
                latest_pair_path=self.latest_pair_path,
                policy_size_bytes=policy_size,
                policy_sha256=policy_sha256,
                training_state_size_bytes=state_size,
                training_state_sha256=state_sha256,
                published_at=published_at,
                timing=CheckpointPairTiming(
                    freeze_seconds=frozen.freeze_seconds,
                    policy_write_seconds=policy_seconds,
                    training_state_write_seconds=state_seconds,
                    commit_seconds=commit_seconds,
                    background_seconds=time.perf_counter() - started_at,
                ),
                post_commit_errors=tuple(post_commit_errors),
            )
        except BaseException:
            if not committed:
                _remove_uncommitted_pair(*published_paths)
            raise

    def _prune_pair_manifests(self) -> None:
        versioned = sorted(
            (
                (version, path)
                for path in self.pair_directory.glob("checkpoint_pair_v*.json")
                if (version := pair_manifest_version(path)) is not None
            ),
            reverse=True,
        )
        protected = {version for version, _path in versioned[: self.keep_last]}
        interval = self.retain_every_versions
        if interval is not None:
            protected.update(
                version for version, _path in versioned if version % interval == 0
            )
        for version, path in versioned:
            if version not in protected:
                path.unlink(missing_ok=True)


def _remove_uncommitted_pair(*paths: Path) -> None:
    """Remove runtime-failure leftovers before the pair commit point."""
    parents: set[Path] = set()
    for path in paths:
        if path.exists():
            path.unlink()
            parents.add(path.parent)
    for parent in parents:
        fsync_directory(parent)


def _pair_pointer_maybe_published(path: Path, payload: bytes) -> bool:
    """Return whether a failed atomic write may have exposed this exact pair."""
    try:
        return path.read_bytes() == payload
    except FileNotFoundError:
        return False
    except OSError:
        # A read failure leaves the replace outcome unknowable. Preserve the
        # complete pair rather than risk deleting files referenced by ``path``.
        return True


def _post_commit_error(stage: str, exc: Exception) -> str:
    """Render one repairable compatibility-maintenance failure."""
    return f"{stage}: {type(exc).__name__}: {exc}"


def _freeze_checkpoint_pair(
    state_dict: Mapping[str, Any] | PreparedModelState,
    *,
    version: int,
    metadata: Mapping[str, Any] | None,
    checkpoint_fields: Mapping[str, Any] | None,
    completed_iterations: int,
    total_optimizer_updates: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_scheduler_steps: int,
    resume_config_sha256: str,
    optimizer_parameter_names: Sequence[Sequence[str]],
    auxiliary_state: Mapping[str, Any] | None,
    take_prepared_ownership: bool,
    take_auxiliary_ownership: bool,
) -> _FrozenCheckpointPair:
    started_at = time.perf_counter()
    policy = (
        state_dict
        if isinstance(state_dict, PreparedModelState) and take_prepared_ownership
        else prepare_model_state(
            state_dict.state_dict
            if isinstance(state_dict, PreparedModelState)
            else state_dict
        )
    )
    frozen_metadata = freeze_durable_value(dict(metadata or {}))
    frozen_fields = (
        None
        if checkpoint_fields is None
        else freeze_durable_value(dict(checkpoint_fields))
    )
    if not isinstance(frozen_metadata, Mapping):
        raise TypeError("frozen checkpoint metadata must be a mapping")
    if frozen_fields is not None and not isinstance(frozen_fields, Mapping):
        raise TypeError("frozen checkpoint fields must be a mapping")
    training = prepare_training_state(
        policy_version=version,
        completed_iterations=completed_iterations,
        total_optimizer_updates=total_optimizer_updates,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        planned_scheduler_steps=planned_scheduler_steps,
        resume_config_sha256=resume_config_sha256,
        optimizer_parameter_names=optimizer_parameter_names,
        auxiliary_state=auxiliary_state,
        take_auxiliary_ownership=take_auxiliary_ownership,
    )
    return _FrozenCheckpointPair(
        policy=policy,
        training=training,
        metadata=cast(Mapping[str, Any], frozen_metadata),
        checkpoint_fields=(
            None if frozen_fields is None else cast(Mapping[str, Any], frozen_fields)
        ),
        freeze_seconds=time.perf_counter() - started_at,
    )


__all__ = [
    "AsyncCheckpointPairPublisher",
    "CheckpointPairTiming",
    "PublishedCheckpointPair",
]
