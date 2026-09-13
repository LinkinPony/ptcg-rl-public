"""Controller loading and bounded checkpoint reuse for native league matches."""

from __future__ import annotations

import gc
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ptcg_rl.agent.runtime import PolicyRuntimeAgent
from ptcg_rl.agent.simple_stateless_batching import RoutedPolicyInferenceBatcher
from ptcg_rl.evaluation.continuous_league.models import (
    BundleIdentity,
    ControllerKind,
    NativeMatchConfig,
)
from ptcg_rl.opponents.spec import BattleAgent
from ptcg_rl.rl.scripted_manifest import builtin_scripted_implementations

if TYPE_CHECKING:
    from ptcg_rl.agent.simple_stateless_runtime import RoutedDeckStatelessPolicy


class MatchAgent(Protocol):
    """Game-local controller surface consumed by the native executor."""

    def act(self, observation: Any) -> Sequence[int]:
        """Select one complete engine action."""

    def telemetry(self) -> Mapping[str, Any]:
        """Return compact telemetry for the most recent callback."""

    def close(self) -> None:
        """Release game-local state without discarding cached weights."""


@dataclass
class _CachedCheckpoint:
    path: Path
    size_bytes: int
    mtime_ns: int
    prototype: RoutedDeckStatelessPolicy


class CheckpointPolicyCache:
    """Small LRU whose keys preserve independent per-bundle policy state."""

    def __init__(
        self,
        *,
        maximum_entries: int,
        device: str,
        public_catalog_manifest_path: Path | None,
        resident_precision: Literal["source", "bfloat16"] = "source",
        rollout_inductor: bool = False,
        inference_batcher: RoutedPolicyInferenceBatcher | None = None,
    ) -> None:
        if maximum_entries <= 0:
            raise ValueError("checkpoint cache must retain at least one entry")
        self.maximum_entries = maximum_entries
        self.device = device
        self.public_catalog_manifest_path = public_catalog_manifest_path
        self.resident_precision = resident_precision
        self.rollout_inductor = rollout_inductor
        self.inference_batcher = inference_batcher
        self._entries: OrderedDict[str, _CachedCheckpoint] = OrderedDict()
        self._lock = threading.RLock()

    def acquire(
        self,
        *,
        bundle: BundleIdentity,
        checkpoint_path: Path,
        deck: Sequence[int],
    ) -> RoutedDeckStatelessPolicy:
        """Fork independently mutable state over one cached routed model."""
        with self._lock:
            return self._acquire_locked(
                bundle=bundle,
                checkpoint_path=checkpoint_path,
                deck=deck,
            )

    def _acquire_locked(
        self,
        *,
        bundle: BundleIdentity,
        checkpoint_path: Path,
        deck: Sequence[int],
    ) -> RoutedDeckStatelessPolicy:
        """Resolve one prototype while serializing cache mutation and first load."""
        expected_sha = _checkpoint_sha(bundle.controller_id)
        key = expected_sha
        cached = self._entries.pop(key, None)
        if cached is not None:
            if cached.path != checkpoint_path:
                raise ValueError("checkpoint identity resolved to a different path")
            stat = checkpoint_path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                cached.size_bytes,
                cached.mtime_ns,
            ):
                raise ValueError("cached checkpoint asset changed on disk")
            self._entries[key] = cached
            policy = cached.prototype.fork()
            policy.bind_own_deck(deck)
            return policy
        actual_sha = _file_sha256(checkpoint_path)
        if actual_sha != expected_sha:
            raise ValueError("checkpoint bytes differ from controller identity")
        catalog_path = self.public_catalog_manifest_path
        if catalog_path is None:
            raise ValueError("routed checkpoint requires a frozen public catalog")
        from ptcg_rl.agent.simple_stateless_runtime import RoutedDeckStatelessPolicy

        prototype = RoutedDeckStatelessPolicy(
            checkpoint_path,
            public_catalog_manifest_path=catalog_path,
            device=self.device,
            own_deck=deck,
            resident_precision=self.resident_precision,
            rollout_inductor=self.rollout_inductor,
        )
        prototype.bind_inference_batcher(self.inference_batcher)
        stat = checkpoint_path.stat()
        self._entries[key] = _CachedCheckpoint(
            path=checkpoint_path,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            prototype=prototype,
        )
        self._evict_if_needed()
        return prototype.fork()

    def close(self) -> None:
        """Drop every retained model and return cached CUDA pages when possible."""
        with self._lock:
            while self._entries:
                _, entry = self._entries.popitem(last=False)
                entry.prototype.close()
        _release_model_memory()

    def _evict_if_needed(self) -> None:
        evicted = False
        while len(self._entries) > self.maximum_entries:
            _, entry = self._entries.popitem(last=False)
            entry.prototype.close()
            del entry
            evicted = True
        if evicted:
            _release_model_memory()


class ControllerFactory:
    """Build one game-local ActTime or allowlisted scripted controller."""

    def __init__(self, config: NativeMatchConfig, *, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root.resolve()
        catalog_path = (
            None
            if config.public_catalog_manifest_path is None
            else _resolve_path(
                config.public_catalog_manifest_path,
                self.repo_root,
            )
        )
        expected_catalog_sha = config.expected_public_catalog_manifest_sha256
        if (
            catalog_path is not None
            and _file_sha256(catalog_path) != expected_catalog_sha
        ):
            raise ValueError("public catalog manifest differs from frozen SHA-256")
        self.checkpoints = CheckpointPolicyCache(
            maximum_entries=config.checkpoint_cache_entries,
            device=config.checkpoint_device,
            public_catalog_manifest_path=catalog_path,
            resident_precision=config.checkpoint_resident_precision,
            rollout_inductor=config.checkpoint_rollout_inductor,
            inference_batcher=(
                RoutedPolicyInferenceBatcher(
                    maximum_rows=config.policy_batch_max_rows,
                    maximum_wait_seconds=config.policy_batch_wait_ms / 1_000.0,
                    coalesce_temperatures=(config.policy_batch_coalesce_temperatures),
                )
                if config.policy_batch_max_rows > 1
                else None
            ),
        )

    def build(
        self,
        *,
        bundle: BundleIdentity,
        kind: ControllerKind,
        controller_path: Path | None,
        deck_path: Path,
        deck: Sequence[int],
        seat: int,
        seed: int,
        policy_temperature: float | None = None,
    ) -> MatchAgent:
        """Construct and initialize one controller for a single game."""
        if kind == "script":
            name = _script_name(bundle.controller_id)
            implementation = builtin_scripted_implementations((name,))[0]
            agent = implementation.factory(seed, {})
            agent.reset()
            return _ScriptMatchAgent(agent)
        if controller_path is None:
            raise ValueError("checkpoint controller has no policy asset path")
        resolved_checkpoint = _resolve_path(controller_path, self.repo_root)
        resolved_deck = _resolve_path(deck_path, self.repo_root)
        policy = self.checkpoints.acquire(
            bundle=bundle,
            checkpoint_path=resolved_checkpoint,
            deck=deck,
        )
        runtime_updates: dict[str, object] = {
            "checkpoint_path": resolved_checkpoint,
            "deck_path": resolved_deck,
            "seed": seed,
        }
        if policy_temperature is not None:
            runtime_updates["policy_temperature"] = policy_temperature
        runtime_config = self.config.act_time.model_copy(update=runtime_updates)
        runtime = PolicyRuntimeAgent(
            config=runtime_config,
            policy=policy,
            strict_runtime_errors=True,
        )
        runtime.begin_game(player_index=seat, own_deck=deck)
        return _PolicyMatchAgent(runtime)

    def close(self) -> None:
        """Close all cache-owned checkpoint policies."""
        batcher = self.checkpoints.inference_batcher
        if batcher is not None:
            batcher.close()
        self.checkpoints.close()


class _PolicyMatchAgent:
    def __init__(self, runtime: PolicyRuntimeAgent) -> None:
        self.runtime = runtime

    def act(self, observation: Any) -> Sequence[int]:
        return self.runtime.act(observation)

    def telemetry(self) -> Mapping[str, Any]:
        return self.runtime.last_act_telemetry()

    def close(self) -> None:
        self.runtime.close()


class _ScriptMatchAgent:
    def __init__(self, agent: BattleAgent) -> None:
        self.agent = agent

    def act(self, observation: Any) -> Sequence[int]:
        return self.agent.act(observation)

    def telemetry(self) -> Mapping[str, Any]:
        return {}

    def close(self) -> None:
        close = getattr(self.agent, "close", None)
        if callable(close):
            close()


def _checkpoint_sha(controller_id: str) -> str:
    prefix = "checkpoint:"
    if not controller_id.startswith(prefix):
        raise ValueError("checkpoint controller identity has an invalid prefix")
    return controller_id.removeprefix(prefix)


def _script_name(controller_id: str) -> str:
    prefix = "script:"
    if not controller_id.startswith(prefix):
        raise ValueError("script controller identity has an invalid prefix")
    name = controller_id.removeprefix(prefix)
    if not name:
        raise ValueError("script controller identity has no runtime name")
    return name


def _resolve_path(path: Path, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        cast(Any, torch.cuda).empty_cache()


__all__ = ["ControllerFactory", "MatchAgent"]
