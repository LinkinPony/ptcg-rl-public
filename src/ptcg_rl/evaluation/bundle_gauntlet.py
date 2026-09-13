"""Exact bundle-vs-bundle gauntlet evaluation.

A bundle is the deployable combination of a deck, a pilot, and its runtime
configuration.  Unlike deck ladders, this runner never substitutes one shared
controller on both sides: every configured candidate is evaluated against the
same explicit opponent bundles and seat strata.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.agent.runtime import ActTimeConfig, PolicyRuntimeAgent, SelectPolicy
from ptcg_rl.agent.search.budget import ActTimeLedgerConfig
from ptcg_rl.context import ContextBeliefTracker, OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.evaluation.bundle_models import (
    BundleAgentConfig,
    BundleAgentKind,
    BundleGamePlan,
    BundleGauntletConfig,
    BundleScheduleConfig,
    EvaluationBundle,
    EvaluationBundleConfig,
)
from ptcg_rl.evaluation.h2h_summary import summarize_bundle_h2h
from ptcg_rl.evaluation.release_agent import (
    IsolatedReleaseAgent,
    ReleaseCapsule,
    materialize_release_capsule,
)
from ptcg_rl.evaluation.search_identity import (
    environment_identity,
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.evaluation.streaming_games import StreamingGameStore
from ptcg_rl.opponents.spec import build_opponent, opponent_registry
from ptcg_rl.rl.scripted_manifest import (
    builtin_scripted_implementations,
    fingerprint_scripted_sources,
)
from ptcg_rl.training.arena import (
    BattleSessionFactory,
    GamePlan,
    run_arena_game,
)
from ptcg_rl.training.arena_agents import ArenaAgent
from ptcg_rl.training.arena_decks import DeckPoolConfig, load_deck_pool
from ptcg_rl.training.arena_telemetry import RuntimeTelemetryAccumulator
from ptcg_rl.training.arena_utils import field_value
from ptcg_rl.training.run_config import (
    resolve_training_output_dir,
    resolved_training_config_dump,
)

_REGISTERED_DECK_PATH_ENV = "POKEMON_TCG_DECK_PATH"
_DIAGNOSTICS_TIMEOUT_SECONDS = 1.0

__all__ = [
    "BundleAgentConfig",
    "BundleAgentKind",
    "BundleGamePlan",
    "BundleGauntletConfig",
    "BundleScheduleConfig",
    "EvaluationBundle",
    "EvaluationBundleConfig",
    "expand_bundle_gauntlet_plans",
    "run_bundle_gauntlet",
]


@dataclass(frozen=True)
class _BundleTask:
    """Pickleable worker payload for one planned bundle game."""

    plan: BundleGamePlan
    max_steps: int
    act_time_ledger: ActTimeLedgerConfig
    run_timeout_seconds: float | None


class GameRunTimeoutError(TimeoutError):
    """Raised when one local game exceeds its pre-registered wall timeout."""


class _RuntimeBundleAgent:
    """Arena adapter for the complete Kaggle-like policy runtime."""

    def __init__(
        self,
        *,
        name: str,
        config: ActTimeConfig,
        policy: SelectPolicy,
    ) -> None:
        self.name = name
        self._config = config
        self._policy = policy
        self._agent: PolicyRuntimeAgent | None = None
        self.reset()

    def reset(self) -> None:
        """Recreate per-game runtime state without reloading model weights."""
        self._agent = PolicyRuntimeAgent(
            config=self._config,
            policy=self._policy,
            strict_runtime_errors=True,
        )

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Seed runtime context with the local seat and exact deck."""
        if self._agent is None:
            self.reset()
        cast(PolicyRuntimeAgent, self._agent).begin_game(
            player_index=player_index,
            own_deck=own_deck,
        )

    def act(self, observation: Any) -> Sequence[int]:
        """Return one runtime action."""
        if self._agent is None:
            self.reset()
        return tuple(
            int(index)
            for index in cast(PolicyRuntimeAgent, self._agent).act(observation)
        )

    def last_act_telemetry(self) -> Mapping[str, Any]:
        """Expose compact runtime timing/search telemetry to the arena."""
        if self._agent is None:
            return {}
        return self._agent.last_act_telemetry()


class _GreedyBundleAgent:
    """Context-aware greedy adapter backed by a cached checkpoint policy."""

    def __init__(
        self,
        *,
        name: str,
        policy: SelectPolicy,
        belief: OpponentBeliefFeatureConfig | None,
    ) -> None:
        self.name = name
        self._policy = policy
        self._tracker = ContextBeliefTracker(belief=belief)

    def reset(self) -> None:
        """Reset per-game evidence."""
        self._tracker.begin_game()

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Reset context with the local seat and exact deck."""
        self._tracker.begin_game(player_index=player_index, own_deck=own_deck)
        bind_deck = getattr(self._policy, "bind_own_deck", None)
        if own_deck is not None and callable(bind_deck):
            bind_deck(own_deck)

    def act(self, observation: Any) -> Sequence[int]:
        """Greedily decode one legal policy action."""
        action = forced_action(field_value(observation, "select"))
        context_observation = self._tracker.observation_with_context(observation)
        if action is not None:
            return action
        return self._policy.select_action(context_observation)


class _SimpleStatelessBundleAgent:
    """Arena adapter for one fixed-route simple-stateless deployment policy."""

    def __init__(self, *, name: str, policy: SelectPolicy) -> None:
        self.name = name
        self._policy = policy

    def reset(self) -> None:
        """Reset the policy's public-only per-game tracker."""
        reset = getattr(self._policy, "reset_runtime_episode", None)
        if callable(reset):
            reset()

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Bind the exact route and begin a fresh public history."""
        del player_index
        if own_deck is None:
            raise ValueError("simple-stateless bundle requires its exact deck")
        bind_deck = getattr(self._policy, "bind_own_deck", None)
        if not callable(bind_deck):
            raise TypeError("simple-stateless policy cannot bind its exact deck")
        bind_deck(own_deck)

    def act(self, observation: Any) -> Sequence[int]:
        """Greedily decode one complete legal engine action."""
        return self._policy.select_action(observation)


_POLICY_CACHE: dict[tuple[str, ...], SelectPolicy] = {}
_RELEASE_CAPSULE_CACHE: dict[str, ReleaseCapsule] = {}
_RELEASE_CACHE_CLEANUPS: set[Path] = set()


def expand_bundle_gauntlet_plans(
    config: BundleGauntletConfig,
) -> tuple[BundleGamePlan, ...]:
    """Resolve exact bundles and create common opponent/seat strata.

    A stratum seed intentionally excludes candidate identity.  Thus every
    candidate receives the same opponent, repeat, logical seat, and agent RNG
    seed.  The bundled engine does not expose a battle seed, so this does not
    imply paired deck shuffles.
    """
    candidates = tuple(_resolve_bundle(bundle) for bundle in config.candidates)
    opponents = tuple(_resolve_bundle(bundle) for bundle in config.opponents)
    plans: list[BundleGamePlan] = []
    for opponent in opponents:
        games = config.schedule.games_per_opponent.get(
            opponent.config.bundle_id,
            config.schedule.games_per_matchup,
        )
        for repeat_index in range(games):
            candidate_seat = repeat_index % 2 if config.schedule.mirror_sides else 0
            matched_block_index = (
                repeat_index // 2 if config.schedule.mirror_sides else repeat_index
            )
            matched_block_seed = _shared_stratum_seed(
                config.seed,
                opponent.config.bundle_id,
                matched_block_index,
                candidate_seat,
            )
            matched_block_id = (
                f"{opponent.config.bundle_id}:"
                f"block_{matched_block_index}:seat_{candidate_seat}"
            )
            for candidate in candidates:
                arena_plan = GamePlan(
                    game_index=len(plans),
                    seed=matched_block_seed,
                    candidate_deck=candidate.deck,
                    opponent_deck=opponent.deck,
                    candidate_seat=candidate_seat,
                )
                plans.append(
                    BundleGamePlan(
                        arena_plan=arena_plan,
                        candidate=candidate,
                        opponent=opponent,
                        repeat_index=repeat_index,
                        matched_block_index=matched_block_index,
                        matched_block_id=matched_block_id,
                        matched_block_seed=matched_block_seed,
                    )
                )
    return tuple(plans)


def run_bundle_gauntlet(
    config: BundleGauntletConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or resume an exact-bundle gauntlet with durable streamed results."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="bundle_gauntlet",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = expand_bundle_gauntlet_plans(config)
    evaluation_fingerprint = _evaluation_fingerprint(config, plans)
    identity = _bundle_evaluation_identity(
        config,
        plans,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    with StreamingGameStore(
        output_dir,
        evaluation_fingerprint=evaluation_fingerprint,
        total_games=len(plans),
        compression=config.compression,
        result_shard_size=config.result_shard_size,
    ) as store:
        write_identity_atomic(output_dir / "fingerprints.json", identity)
        write_identity_atomic(output_dir / "environment.json", identity["environment"])
        completed_indices = store.completed_indices
        tasks = tuple(
            _BundleTask(
                plan=plan,
                max_steps=config.max_steps_per_game,
                act_time_ledger=config.act_time_ledger,
                run_timeout_seconds=config.run_timeout_seconds,
            )
            for plan in plans
            if plan.arena_plan.game_index not in completed_indices
        )
        games_finished = store.resumed_games
        _report_progress(
            store,
            games_finished=games_finished,
            progress_callback=progress_callback,
        )
        buffered_rows: list[dict[str, Any]] = []
        try:
            for row in _iter_task_rows(
                config,
                tasks,
                battle_session_factory=battle_session_factory,
            ):
                _add_evaluation_identity(row, identity)
                buffered_rows.append(row)
                games_finished += 1
                if len(buffered_rows) >= config.result_shard_size:
                    store.append(buffered_rows)
                    buffered_rows.clear()
                _report_progress(
                    store,
                    games_finished=games_finished,
                    progress_callback=progress_callback,
                )
        except BaseException:
            if buffered_rows:
                store.append(buffered_rows)
            raise
        if buffered_rows:
            store.append(buffered_rows)
        games_path = store.compact()
        h2h_summary = summarize_bundle_h2h(
            games_path,
            output_dir=output_dir,
            compression=config.compression,
        )
        _report_progress(
            store,
            games_finished=games_finished,
            progress_callback=progress_callback,
        )
        error_actor_counts = store.error_actor_counts
        summary: dict[str, Any] = {
            "runner": "bundle_gauntlet",
            "games": len(store.completed_indices),
            "planned_games": len(plans),
            "resumed_games": store.resumed_games,
            "result_parts": store.result_parts,
            "candidate_bundles": len(config.candidates),
            "opponent_bundles": len(config.opponents),
            "terminal_reason_counts": store.terminal_reason_counts,
            "error_actor_counts": error_actor_counts,
            "evaluation_fingerprint": evaluation_fingerprint,
            "campaign_fp": identity["campaign_fp"],
            "stage_fp": identity["stage_fp"],
            "identity": identity,
            "engine_rng_controlled": False,
            "matched_randomness_scope": "python agent and fallback RNG only",
            "games_path": records.display_path(games_path),
            "games_parts_dir": records.display_path(store.parts_dir),
            "h2h": h2h_summary,
            "progress_path": records.display_path(store.progress_path),
            "output_dir": records.display_path(output_dir),
            "run": config.run.model_dump(mode="json"),
            "config": resolved_training_config_dump(
                config,
                task_name="bundle_gauntlet",
                run=config.run,
                output_dir=config.output_dir,
            ),
        }
        summary_path = output_dir / "summary.json"
        summary["summary_path"] = records.display_path(summary_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if config.fail_on_error and error_actor_counts:
            raise RuntimeError(f"bundle gauntlet had errors: {error_actor_counts}")
        return summary


def _report_progress(
    store: StreamingGameStore,
    *,
    games_finished: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    if progress_callback is not None:
        progress_callback(store.progress_payload(games_finished=games_finished))


def _iter_task_rows(
    config: BundleGauntletConfig,
    tasks: Sequence[_BundleTask],
    *,
    battle_session_factory: BattleSessionFactory | None,
) -> Iterator[dict[str, Any]]:
    if config.num_workers <= 1 or battle_session_factory is not None:
        for task in tasks:
            yield _run_task(task, battle_session_factory=battle_session_factory)
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        mp_context=context,
        initializer=_initialize_worker,
    ) as executor:
        task_iterator = iter(tasks)
        futures: set[Future[dict[str, Any]]] = set()
        for _ in range(config.num_workers * 2):
            initial_task = next(task_iterator, None)
            if initial_task is None:
                break
            futures.add(executor.submit(_run_task, initial_task))
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                replacement_task = next(task_iterator, None)
                if replacement_task is not None:
                    futures.add(executor.submit(_run_task, replacement_task))


def _initialize_worker() -> None:
    """Keep spawned engine/model workers from oversubscribing CPU threads."""
    import torch

    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _run_task(
    task: _BundleTask,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    plan = task.plan
    try:
        candidate_agent = _build_agent(
            plan.candidate,
            seed=plan.matched_block_seed + 11,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a candidate failure row.
        return _task_error_row(task, exc, error_stage="candidate_build")

    with _registered_deck_env(plan.opponent):
        try:
            opponent_agent = _build_agent(
                plan.opponent,
                seed=plan.matched_block_seed + 29,
            )
        except Exception as exc:  # noqa: BLE001 - preserve an opponent failure row.
            output = _task_error_row(task, exc, error_stage="opponent_build")
            _add_agent_diagnostics(
                output,
                candidate_agent=candidate_agent,
                opponent_agent=None,
            )
            _close_agent(candidate_agent)
            return output
        try:
            try:
                with _game_timeout(task.run_timeout_seconds):
                    row = run_arena_game(
                        plan.arena_plan,
                        candidate_agent=candidate_agent,
                        opponent_agent=opponent_agent,
                        max_steps=task.max_steps,
                        battle_session_factory=(
                            battle_session_factory or _default_battle_session
                        ),
                        on_agent_error=_stop_on_agent_error,
                        reset_agents=True,
                        act_time_ledger_config=task.act_time_ledger,
                    )
            except Exception as exc:  # noqa: BLE001 - isolate battle errors.
                output = _task_error_row(task, exc, error_stage="battle")
            else:
                output = dict(row)
                output["error_stage"] = (
                    "act"
                    if output.get("error_actor") in {"candidate", "opponent"}
                    else ""
                )
                _add_bundle_metadata(output, plan)
            _add_agent_diagnostics(
                output,
                candidate_agent=candidate_agent,
                opponent_agent=opponent_agent,
            )
            return output
        finally:
            _close_agent(candidate_agent)
            _close_agent(opponent_agent)


def _build_agent(bundle: EvaluationBundle, *, seed: int) -> ArenaAgent:
    config = bundle.config.agent
    if config.kind == "registered":
        name = cast(str, config.registered_name)
        registered = opponent_registry().get(name)
        if registered is not None:
            return cast(ArenaAgent, build_opponent(registered, seed=seed))
        implementation = builtin_scripted_implementations((name,))[0]
        return cast(ArenaAgent, implementation.factory(seed, {}))
    if config.kind == "release":
        manifest_path = cast(Path, config.release_manifest_path)
        resolved_manifest = records.repo_path(manifest_path).resolve()
        cache_key = f"{resolved_manifest}\0{bundle.config.pilot_id}"
        capsule = _RELEASE_CAPSULE_CACHE.get(cache_key)
        if capsule is None:
            capsule = materialize_release_capsule(
                label=bundle.config.pilot_id,
                manifest_path=resolved_manifest,
                cache_root=_release_process_root() / "cache",
            )
            _RELEASE_CAPSULE_CACHE[cache_key] = capsule
        safe_bundle_id = hashlib.sha256(
            bundle.config.bundle_id.encode("utf-8")
        ).hexdigest()[:16]
        episode_root = (
            _release_process_root() / "episodes" / (f"{safe_bundle_id}-{seed}")
        )
        return cast(
            ArenaAgent,
            IsolatedReleaseAgent(
                capsule,
                log_path=episode_root / "agent.log",
                work_dir=episode_root / "work",
                startup_timeout_seconds=300.0,
                action_timeout_seconds=30.0,
            ),
        )

    checkpoint_path = cast(Path, config.checkpoint_path)
    deck_cards = records.read_deck(records.repo_path(bundle.config.deck_path))
    policy = _deck_checkpoint_policy(
        checkpoint_path,
        device=config.device,
        own_deck=deck_cards,
        public_catalog_manifest_path=(
            config.public_catalog_manifest_path
            if config.kind == "simple_stateless_greedy"
            else None
        ),
    )
    if config.kind == "simple_stateless_greedy":
        configure_sampling = getattr(policy, "configure_deployment_sampling", None)
        if callable(configure_sampling):
            configure_sampling(
                temperature=config.act_time.policy_temperature,
                seed=seed,
            )
        return _SimpleStatelessBundleAgent(
            name=bundle.config.pilot_id,
            policy=policy,
        )
    belief = _belief_config(config.belief_summary_path)
    if config.kind == "policy_greedy":
        return _GreedyBundleAgent(
            name=bundle.config.pilot_id,
            policy=policy,
            belief=belief,
        )

    act_time = config.act_time.model_copy(deep=True)
    act_time = act_time.model_copy(
        update={
            "checkpoint_path": records.repo_path(checkpoint_path),
            "deck_path": records.repo_path(bundle.config.deck_path),
            "seed": seed,
        }
    )
    if config.belief_summary_path is not None:
        belief_path = records.repo_path(config.belief_summary_path)
        act_time.belief.deck_signature_summary_path = belief_path
        act_time.search.sampler.prior_deck_signature_summary_path = belief_path
    return _RuntimeBundleAgent(
        name=bundle.config.pilot_id,
        config=act_time,
        policy=policy,
    )


def _deck_checkpoint_policy(
    checkpoint_path: Path,
    *,
    device: str,
    own_deck: Sequence[int],
    public_catalog_manifest_path: Path | None = None,
) -> SelectPolicy:
    resolved = records.repo_path(checkpoint_path)
    signature = records.deck_signature(list(own_deck))
    resolved_catalog = (
        None
        if public_catalog_manifest_path is None
        else records.repo_path(public_catalog_manifest_path)
    )
    key = (
        str(resolved),
        device,
        signature,
        "" if resolved_catalog is None else str(resolved_catalog),
    )
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    policy = (
        _checkpoint_policy(resolved, device=device)
        if resolved_catalog is None
        else _fixed_stateless_policy(
            resolved,
            public_catalog_manifest_path=resolved_catalog,
            device=device,
        )
    )
    bind_own_deck = getattr(policy, "bind_own_deck", None)
    if callable(bind_own_deck):
        bind_own_deck(own_deck)
    _POLICY_CACHE[key] = policy
    return policy


def _fixed_stateless_policy(
    checkpoint_path: Path,
    *,
    public_catalog_manifest_path: Path,
    device: str,
) -> SelectPolicy:
    """Load one routed learner policy for diagnostic evaluation."""
    from ptcg_rl.agent.simple_stateless_runtime import routed_stateless_policy

    return cast(
        SelectPolicy,
        routed_stateless_policy(
            checkpoint_path,
            public_catalog_manifest_path=public_catalog_manifest_path,
            device=device,
        ),
    )


def _checkpoint_policy(checkpoint_path: Path, *, device: str) -> SelectPolicy:
    """Load one policy; exact-deck caching is handled by its caller."""
    from ptcg_rl.agent.runtime import CheckpointPolicy

    return cast(
        SelectPolicy,
        CheckpointPolicy(records.repo_path(checkpoint_path), device=device),
    )


def _belief_config(path: Path | None) -> OpponentBeliefFeatureConfig | None:
    if path is None:
        return None
    return OpponentBeliefFeatureConfig(
        deck_signature_summary_path=records.repo_path(path),
    )


def _resolve_bundle(config: EvaluationBundleConfig) -> EvaluationBundle:
    decks = load_deck_pool(DeckPoolConfig(deck_paths=(config.deck_path,)))
    if len(decks) != 1:
        raise ValueError(
            f"bundle {config.bundle_id} must resolve exactly one deck, got {len(decks)}"
        )
    if config.agent.kind == "registered":
        registered_name = cast(str, config.agent.registered_name)
        fixed_signature = _registered_fixed_deck_signature(registered_name)
        if fixed_signature is not None and fixed_signature != decks[0].signature:
            raise ValueError(
                f"registered opponent {registered_name} requires its fixed "
                f"deck signature; bundle {config.bundle_id} configured a "
                "different deck"
            )
    if config.agent.kind == "release":
        from ptcg_rl.submission.release_assets import (
            load_release_bundle_for_native_execution,
        )

        manifest = load_release_bundle_for_native_execution(
            records.repo_path(cast(Path, config.agent.release_manifest_path))
        )
        release_decks = load_deck_pool(DeckPoolConfig(deck_paths=(manifest.deck_path,)))
        if len(release_decks) != 1 or release_decks[0].signature != decks[0].signature:
            raise ValueError(
                f"release bundle {config.bundle_id} deck does not match its "
                "immutable manifest"
            )
    return EvaluationBundle(config=config, deck=decks[0])


def _registered_fixed_deck_signature(registered_name: str) -> str | None:
    """Resolve a fixed registered deck without coupling its code to the registry."""
    registered = opponent_registry().get(registered_name)
    if registered is not None:
        if registered.deck_path is None:
            return None
        fixed_decks = load_deck_pool(
            DeckPoolConfig(deck_paths=(registered.deck_path,))
        )
        if len(fixed_decks) != 1:
            raise ValueError(
                f"registered opponent {registered_name} must resolve one fixed deck"
            )
        return fixed_decks[0].signature

    # Names outside the legacy registry, including versioned local scripts,
    # remain explicit members of the immutable scripted allowlist.
    builtin_scripted_implementations((registered_name,))

    if registered_name == "slowking_copy_v1":
        from ptcg_rl.opponents import slowking_copy

        return records.deck_signature(list(slowking_copy.DECK))
    if registered_name == "slowking_copy_v2":
        from ptcg_rl.opponents import slowking_copy_v2

        return records.deck_signature(list(slowking_copy_v2.DECK))
    if registered_name == "lopunny_dudunsparce_v1":
        from ptcg_rl.opponents import lopunny_dudunsparce

        return records.deck_signature(list(lopunny_dudunsparce.DECK))
    return None


def _evaluation_fingerprint(
    config: BundleGauntletConfig,
    plans: Sequence[BundleGamePlan],
) -> str:
    """Hash game semantics and referenced model inputs for safe resumption."""
    semantic_config = config.model_dump(
        mode="json",
        exclude={
            "output_dir",
            "num_workers",
            "compression",
            "result_shard_size",
            "fail_on_error",
        },
    )
    resolved_decks = sorted(
        {
            (
                "candidate",
                plan.candidate.config.bundle_id,
                plan.candidate.deck.signature,
            )
            for plan in plans
        }
        | {
            (
                "opponent",
                plan.opponent.config.bundle_id,
                plan.opponent.deck.signature,
            )
            for plan in plans
        }
    )
    referenced_files: dict[str, str] = {}
    for bundle in (*config.candidates, *config.opponents):
        for input_kind, configured_path in (
            ("checkpoint", bundle.agent.checkpoint_path),
            ("belief", bundle.agent.belief_summary_path),
            ("public_catalog", bundle.agent.public_catalog_manifest_path),
            ("release_manifest", bundle.agent.release_manifest_path),
        ):
            if configured_path is None:
                continue
            key = f"{input_kind}:{configured_path}"
            if key not in referenced_files:
                referenced_files[key] = _sha256_file(records.repo_path(configured_path))
    payload = {
        "store_schema_version": 3,
        "semantic_config": semantic_config,
        "resolved_decks": resolved_decks,
        "referenced_files": referenced_files,
        "registered_implementations": {
            f"{role}:{bundle.bundle_id}": _registered_implementation_identity(bundle)
            for role, bundles in (
                ("candidate", config.candidates),
                ("opponent", config.opponents),
            )
            for bundle in bundles
            if bundle.agent.kind == "registered"
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle_evaluation_identity(
    config: BundleGauntletConfig,
    plans: Sequence[BundleGamePlan],
    *,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    """Build row-level exact bundle, host, campaign, and stage provenance."""
    bundles_by_id: dict[str, EvaluationBundle] = {}
    for plan in plans:
        for bundle in (plan.candidate, plan.opponent):
            bundles_by_id.setdefault(bundle.config.bundle_id, bundle)
    asset_fingerprints: dict[Path, str] = {}
    resolved_bundles = {
        bundle_id: _exact_bundle_fingerprint(
            bundle,
            asset_fingerprints=asset_fingerprints,
        )
        for bundle_id, bundle in bundles_by_id.items()
    }
    environment = environment_identity()
    env_fp = fingerprint_payload(environment)
    experiment_id = config.experiment_id or config.run.version
    campaign_payload = {
        "protocol": config.protocol,
        "experiment_id": experiment_id,
        "bundle_fingerprints": resolved_bundles,
        "registered_implementations": {
            bundle_id: _registered_implementation_identity(bundle.config)
            for bundle_id, bundle in bundles_by_id.items()
            if bundle.config.agent.kind == "registered"
        },
        "env_fp": env_fp,
    }
    campaign_fp = fingerprint_payload(campaign_payload)
    stage_payload = {
        "campaign_fp": campaign_fp,
        "stage": config.stage or "unspecified",
        "evaluation_fingerprint": evaluation_fingerprint,
    }
    return {
        **campaign_payload,
        "stage": config.stage or "unspecified",
        "evaluation_fingerprint": evaluation_fingerprint,
        "campaign_fp": campaign_fp,
        "stage_fp": fingerprint_payload(stage_payload),
        "environment": environment,
    }


def _exact_bundle_fingerprint(
    bundle: EvaluationBundle,
    *,
    asset_fingerprints: dict[Path, str] | None = None,
) -> str:
    config = bundle.config
    fingerprint_cache = asset_fingerprints if asset_fingerprints is not None else {}

    def asset_fingerprint(path: Path) -> str:
        resolved = records.repo_path(path).resolve()
        fingerprint = fingerprint_cache.get(resolved)
        if fingerprint is None:
            fingerprint = file_sha256(resolved)
            fingerprint_cache[resolved] = fingerprint
        return fingerprint

    assets: dict[str, str] = {"deck": asset_fingerprint(config.deck_path)}
    for label, path in (
        ("checkpoint", config.agent.checkpoint_path),
        ("belief", config.agent.belief_summary_path),
        ("public_catalog", config.agent.public_catalog_manifest_path),
        ("release_manifest", config.agent.release_manifest_path),
    ):
        if path is not None:
            assets[label] = asset_fingerprint(path)
    payload = {
        "config": config.model_dump(mode="json"),
        "deck_signature": bundle.deck.signature,
        "assets": assets,
    }
    if config.agent.kind == "registered":
        payload["registered_implementation"] = _registered_implementation_identity(
            config
        )
    return fingerprint_payload(payload)


def _registered_implementation_identity(
    config: EvaluationBundleConfig,
) -> dict[str, Any]:
    """Fingerprint the complete allowlisted source inventory of one script."""
    if config.agent.kind != "registered":
        raise ValueError("registered implementation identity requires registered agent")
    registered_name = cast(str, config.agent.registered_name)
    try:
        implementation = builtin_scripted_implementations((registered_name,))[0]
    except KeyError as exc:
        raise ValueError(
            f"registered bundle has no immutable source inventory: {registered_name}"
        ) from exc
    return {
        "script_name": implementation.script_name,
        "implementation_files": implementation.implementation_files,
        "implementation_code_fingerprint": fingerprint_scripted_sources(
            records.REPO_ROOT,
            implementation.implementation_files,
        ),
        "vector_safe": implementation.vector_safe,
        "requires_search": implementation.requires_search,
    }


def _add_evaluation_identity(row: dict[str, Any], identity: Mapping[str, Any]) -> None:
    bundle_fingerprints = cast(Mapping[str, str], identity["bundle_fingerprints"])
    environment = cast(Mapping[str, Any], identity["environment"])
    row.update(
        {
            "experiment_protocol": identity["protocol"],
            "experiment_id": identity["experiment_id"],
            "experiment_stage": identity["stage"],
            "campaign_fp": identity["campaign_fp"],
            "stage_fp": identity["stage_fp"],
            "evaluation_fingerprint": identity["evaluation_fingerprint"],
            "candidate_bundle_fp": bundle_fingerprints[str(row["candidate_bundle_id"])],
            "opponent_bundle_fp": bundle_fingerprints[str(row["opponent_bundle_id"])],
            "env_fp": fingerprint_payload(environment),
            "host_platform": environment.get("platform"),
            "host_processor": environment.get("processor"),
            "git_revision": environment.get("git_revision"),
            "git_dirty": environment.get("git_dirty"),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_stratum_seed(
    base_seed: int,
    opponent_id: str,
    block_index: int,
    candidate_seat: int,
) -> int:
    payload = f"{base_seed}\0{opponent_id}\0{block_index}\0{candidate_seat}".encode()
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _add_bundle_metadata(row: dict[str, Any], plan: BundleGamePlan) -> None:
    candidate = plan.candidate.config
    opponent = plan.opponent.config
    row.update(
        {
            "candidate_bundle_id": candidate.bundle_id,
            "opponent_bundle_id": opponent.bundle_id,
            "candidate_pilot_id": candidate.pilot_id,
            "opponent_pilot_id": opponent.pilot_id,
            "candidate_archetype": candidate.archetype,
            "opponent_archetype": opponent.archetype,
            "candidate_variant_weight": candidate.variant_weight,
            "opponent_variant_weight": opponent.variant_weight,
            "candidate_agent_kind": candidate.agent.kind,
            "opponent_agent_kind": opponent.agent.kind,
            "bundle_repeat_index": plan.repeat_index,
            "matched_block_index": plan.matched_block_index,
            "matched_block_id": plan.matched_block_id,
            "matched_block_seed": plan.matched_block_seed,
            "stratum_id": plan.matched_block_id,
            "stratum_seed": plan.matched_block_seed,
        }
    )


def _task_error_row(
    task: _BundleTask,
    exc: Exception,
    *,
    error_stage: Literal["candidate_build", "opponent_build", "battle"],
) -> dict[str, Any]:
    plan = task.plan
    arena_plan = plan.arena_plan
    startup_charge = (
        task.act_time_ledger.startup_charge_seconds
        if task.act_time_ledger.enabled
        else 0.0
    )
    remaining_overage = max(
        0.0,
        task.act_time_ledger.total_seconds - startup_charge,
    )
    row: dict[str, Any] = {
        "game_index": arena_plan.game_index,
        "seed": arena_plan.seed,
        "candidate_agent": plan.candidate.config.pilot_id,
        "opponent_agent": plan.opponent.config.pilot_id,
        "candidate_seat": arena_plan.candidate_seat,
        "winner_index": -1,
        "candidate_result": "truncated",
        "terminal_reason": "infrastructure_error",
        "steps": 0,
        "candidate_deck_id": arena_plan.candidate_deck.deck_id,
        "candidate_deck_hash": arena_plan.candidate_deck.deck_hash,
        "candidate_deck_label": arena_plan.candidate_deck.label,
        "candidate_deck_signature": arena_plan.candidate_deck.signature,
        "opponent_deck_id": arena_plan.opponent_deck.deck_id,
        "opponent_deck_hash": arena_plan.opponent_deck.deck_hash,
        "opponent_deck_label": arena_plan.opponent_deck.label,
        "opponent_deck_signature": arena_plan.opponent_deck.signature,
        "candidate_decisions": 0,
        "opponent_decisions": 0,
        "candidate_action_seconds": 0.0,
        "opponent_action_seconds": 0.0,
        "candidate_mean_action_seconds": 0.0,
        "opponent_mean_action_seconds": 0.0,
        "candidate_illegal_actions": 0,
        "opponent_illegal_actions": 0,
        "candidate_act_time_used_seconds": startup_charge,
        "opponent_act_time_used_seconds": startup_charge,
        "candidate_remaining_overage_time": remaining_overage,
        "opponent_remaining_overage_time": remaining_overage,
        "act_time_startup_charge_seconds": startup_charge,
        "act_time_timeout_seat": -1,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_player_index": -1,
        "error_actor": "infrastructure",
        "error_stage": error_stage,
        "candidate_agent_diagnostics": "{}",
        "opponent_agent_diagnostics": "{}",
    }
    row.update(
        RuntimeTelemetryAccumulator(
            expected=plan.candidate.config.agent.kind == "runtime"
        ).row_fields("candidate")
    )
    row.update(
        RuntimeTelemetryAccumulator(
            expected=plan.opponent.config.agent.kind == "runtime"
        ).row_fields("opponent")
    )
    _add_bundle_metadata(row, plan)
    return row


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: ArenaAgent,
    observation: Mapping[str, Any],
) -> Sequence[int] | None:
    del exc, player_index, agent, observation
    return None


def _close_agent(agent: ArenaAgent) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()
    if isinstance(agent, IsolatedReleaseAgent):
        shutil.rmtree(agent.log_path.parent, ignore_errors=True)


def _add_agent_diagnostics(
    row: dict[str, Any],
    *,
    candidate_agent: ArenaAgent | None,
    opponent_agent: ArenaAgent | None,
) -> None:
    """Add failure-isolated, canonical per-game agent diagnostics."""
    row["candidate_agent_diagnostics"] = _agent_diagnostics_json(candidate_agent)
    row["opponent_agent_diagnostics"] = _agent_diagnostics_json(opponent_agent)


def _agent_diagnostics_json(agent: ArenaAgent | None) -> str:
    """Read optional diagnostics without allowing them to affect game results."""
    if agent is None:
        return "{}"
    try:
        with _game_timeout(_DIAGNOSTICS_TIMEOUT_SECONDS):
            diagnostics = getattr(agent, "diagnostics", None)
            if not callable(diagnostics):
                return "{}"
            payload = diagnostics()
        if not payload:
            return "{}"
        if not isinstance(payload, Mapping):
            raise TypeError("agent diagnostics must be a mapping")
        return json.dumps(
            dict(payload),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not alter outcomes.
        return json.dumps(
            {"error_type": type(exc).__name__},
            separators=(",", ":"),
            sort_keys=True,
        )


def _release_process_root() -> Path:
    root = records.repo_path(
        Path("tmp/bundle_gauntlet_release/processes") / str(os.getpid())
    ).resolve()
    if root not in _RELEASE_CACHE_CLEANUPS:
        _RELEASE_CACHE_CLEANUPS.add(root)
        atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


@contextmanager
def _game_timeout(timeout_seconds: float | None) -> Iterator[None]:
    """Enforce a Unix wall-clock timeout around one complete local game."""
    if timeout_seconds is None:
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise GameRunTimeoutError(
            f"local battle exceeded run timeout of {timeout_seconds:g} seconds"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def _registered_deck_env(bundle: EvaluationBundle) -> Iterator[None]:
    if bundle.config.agent.kind != "registered":
        yield
        return
    previous = os.environ.get(_REGISTERED_DECK_PATH_ENV)
    os.environ[_REGISTERED_DECK_PATH_ENV] = str(
        records.repo_path(bundle.config.deck_path)
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_REGISTERED_DECK_PATH_ENV, None)
        else:
            os.environ[_REGISTERED_DECK_PATH_ENV] = previous


def _default_battle_session(
    deck0: Sequence[int],
    deck1: Sequence[int],
) -> Any:
    return BattleSession(deck0, deck1)
