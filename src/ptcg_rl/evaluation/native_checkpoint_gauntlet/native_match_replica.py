"""One persistent native-match replica and its checkpoint routing contract."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any, cast

from ptcg_rl.evaluation.continuous_league.models import (
    BundleIdentity,
    MatchLease,
    NativeMatchConfig,
)
from ptcg_rl.evaluation.continuous_league.native_match import NativeMatchExecutor
from ptcg_rl.evaluation.continuous_league.native_runtime import ControllerFactory
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    CheckpointRosterParticipantConfig,
    NativeCheckpointGauntletConfig,
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.results import result_row
from ptcg_rl.evaluation.native_deck_elo.storage import resolve_file


class ParticipantControllerRouter:
    """Route both participants through checkpoint-local persistent factories."""

    def __init__(
        self,
        *,
        candidate_label: str,
        candidate_sha256: str,
        candidate_config: NativeMatchConfig,
        candidate_temperature: float,
        baseline_label: str,
        baseline_sha256: str,
        baseline_config: NativeMatchConfig,
        baseline_temperature: float,
        root: Path,
    ) -> None:
        self._candidate_id = f"checkpoint:{candidate_sha256}"
        self._baseline_id = f"checkpoint:{baseline_sha256}"
        self._candidate_bundle_prefix = f"{candidate_label}:"
        self._baseline_bundle_prefix = f"{baseline_label}:"
        self._candidate_temperature = candidate_temperature
        self._baseline_temperature = baseline_temperature
        self._shared_factory = candidate_sha256 == baseline_sha256 and (
            _without_policy_temperature(candidate_config)
            == _without_policy_temperature(baseline_config)
        )
        with ExitStack() as resources:
            self._candidate = ControllerFactory(candidate_config, repo_root=root)
            resources.callback(self._candidate.close)
            self._baseline = (
                self._candidate
                if self._shared_factory
                else ControllerFactory(baseline_config, repo_root=root)
            )
            if not self._shared_factory:
                resources.callback(self._baseline.close)
            self._resources = resources.pop_all()

    def build(self, **kwargs: Any) -> Any:
        """Delegate one game-local controller to its checkpoint owner."""
        bundle = cast(BundleIdentity, kwargs["bundle"])
        if bundle.controller_id == self._candidate_id and bundle.bundle_id.startswith(
            self._candidate_bundle_prefix
        ):
            return self._candidate.build(
                **kwargs,
                policy_temperature=self._candidate_temperature,
            )
        if bundle.controller_id == self._baseline_id and bundle.bundle_id.startswith(
            self._baseline_bundle_prefix
        ):
            return self._baseline.build(
                **kwargs,
                policy_temperature=self._baseline_temperature,
            )
        raise ValueError("cross-checkpoint lease references an unknown controller")

    def close(self) -> None:
        """Release both independently batched resident checkpoint models."""
        self._resources.close()


def build_native_match_config(
    config: NativeCheckpointGauntletConfig,
    participant: CheckpointRosterParticipantConfig,
    *,
    root: Path,
    library_path: Path,
) -> NativeMatchConfig:
    """Resolve one participant's immutable native-match runtime contract."""
    return NativeMatchConfig(
        library_path=library_path,
        expected_library_sha256=config.expected_native_library_sha256,
        public_catalog_manifest_path=resolve_file(
            participant.public_catalog_manifest_path, root=root
        ),
        expected_public_catalog_manifest_sha256=(
            participant.expected_public_catalog_manifest_sha256
        ),
        checkpoint_device=config.device,
        checkpoint_resident_precision=config.checkpoint_resident_precision,
        checkpoint_rollout_inductor=config.checkpoint_rollout_inductor,
        lane_worker_count=config.native_lane_worker_count,
        option_capacity=config.native_option_capacity,
        maximum_engine_steps=config.maximum_engine_steps,
        checkpoint_cache_entries=2,
        policy_batch_max_rows=config.policy_batch_max_rows,
        policy_batch_wait_ms=config.policy_batch_wait_ms,
        policy_batch_coalesce_temperatures=(config.policy_batch_coalesce_temperatures),
        act_time=config.act_time.model_copy(
            update={"policy_temperature": participant.policy_temperature}
        ),
    )


class NativeMatchReplica:
    """Own one resident checkpoint pair and a reusable native lane pool."""

    def __init__(
        self,
        config: NativeCheckpointGauntletConfig,
        *,
        root: Path,
        candidate_native: NativeMatchConfig,
        baseline_native: NativeMatchConfig,
        checkpoint_paths: Mapping[str, Path],
        campaign_fingerprint: str,
        runtime_fingerprint: str,
        belief_fingerprint: str,
        game_capacity: int,
    ) -> None:
        self.config = config
        self.checkpoint_paths = dict(checkpoint_paths)
        self.campaign_fingerprint = campaign_fingerprint
        self.runtime_fingerprint = runtime_fingerprint
        self.belief_fingerprint = belief_fingerprint
        with ExitStack() as resources:
            self.router = ParticipantControllerRouter(
                candidate_label=config.candidate.label,
                candidate_sha256=config.candidate.expected_checkpoint_sha256,
                candidate_config=candidate_native,
                candidate_temperature=config.candidate.policy_temperature,
                baseline_label=config.baseline.label,
                baseline_sha256=config.baseline.expected_checkpoint_sha256,
                baseline_config=baseline_native,
                baseline_temperature=config.baseline.policy_temperature,
                root=root,
            )
            resources.callback(self.router.close)
            executors: list[NativeMatchExecutor] = []
            for _index in range(min(config.concurrency, game_capacity)):
                executor = NativeMatchExecutor(
                    candidate_native,
                    repo_root=root,
                    controllers=cast(ControllerFactory, self.router),
                )
                resources.callback(executor.close)
                executors.append(executor)
            self.executors = tuple(executors)
            self._resources = resources.pop_all()
        self.available: Queue[NativeMatchExecutor] = Queue()
        for executor in self.executors:
            self.available.put(executor)

    def execute(self, game: ScheduledCrossCheckpointGame) -> dict[str, Any]:
        """Execute one game with exclusive ownership of one native lane."""
        executor = self.available.get()
        try:
            response = executor.execute(
                _lease(
                    game,
                    config=self.config,
                    checkpoint_paths=self.checkpoint_paths,
                    runtime_fingerprint=self.runtime_fingerprint,
                    belief_fingerprint=self.belief_fingerprint,
                )
            )
            return result_row(
                game,
                response,
                campaign_fingerprint=self.campaign_fingerprint,
                candidate_label=self.config.candidate.label,
                baseline_label=self.config.baseline.label,
            )
        finally:
            self.available.put(executor)

    def execute_many(
        self,
        games: Sequence[ScheduledCrossCheckpointGame],
    ) -> Iterator[dict[str, Any]]:
        """Yield completion-order rows with a bounded number of live futures."""
        iterator = iter(games)
        with ThreadPoolExecutor(
            max_workers=len(self.executors),
            thread_name_prefix="native-match-replica",
        ) as pool:
            pending = {
                pool.submit(self.execute, game)
                for _, game in zip(range(len(self.executors)), iterator, strict=False)
            }
            while pending:
                future = next(as_completed(pending))
                pending.remove(future)
                yield future.result()
                game = next(iterator, None)
                if game is not None:
                    pending.add(pool.submit(self.execute, game))

    def close(self) -> None:
        """Release native lanes before dropping the shared resident model."""
        self._resources.close()


def _without_policy_temperature(config: NativeMatchConfig) -> NativeMatchConfig:
    """Normalize the one participant-local field before factory sharing."""
    return config.model_copy(
        update={
            "act_time": config.act_time.model_copy(update={"policy_temperature": 0.0})
        }
    )


def _lease(
    game: ScheduledCrossCheckpointGame,
    *,
    config: NativeCheckpointGauntletConfig,
    checkpoint_paths: Mapping[str, Path],
    runtime_fingerprint: str,
    belief_fingerprint: str,
) -> MatchLease:
    candidate_first = game.candidate_seat == 0
    side_a = game.candidate_deck if candidate_first else game.baseline_deck
    side_b = game.baseline_deck if candidate_first else game.candidate_deck
    return MatchLease(
        match_id=game.match_id,
        side_a=side_a.bundle,
        side_b=side_b.bundle,
        side_a_controller_kind="checkpoint",
        side_b_controller_kind="checkpoint",
        side_a_controller_path=(
            checkpoint_paths["candidate"]
            if candidate_first
            else checkpoint_paths["baseline"]
        ),
        side_b_controller_path=(
            checkpoint_paths["baseline"]
            if candidate_first
            else checkpoint_paths["candidate"]
        ),
        side_a_deck_path=side_a.path,
        side_b_deck_path=side_b.path,
        requires_cuda=True,
        runtime_fingerprint=runtime_fingerprint,
        belief_fingerprint=belief_fingerprint,
        lease_expires_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "NativeMatchReplica",
    "ParticipantControllerRouter",
    "build_native_match_config",
]
