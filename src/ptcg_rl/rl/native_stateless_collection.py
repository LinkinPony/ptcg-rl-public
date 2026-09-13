"""Route-aware zero-observation native stateless rollout collection."""

from __future__ import annotations

import copy
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.native_prospective_facts import (
    NativeProspectiveEngineFactProducer,
)
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.rl.native_collection_games import NativeLiveGame
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_policy_bank import (
    NativePolicyCudaStreamPool,
    NativePolicyInferenceBank,
)
from ptcg_rl.rl.native_policy_batch import NativeDeckBatchCache
from ptcg_rl.rl.native_policy_inference import NativePolicyInferenceExecutor
from ptcg_rl.rl.native_route_arena import NativeArenaResourcePool
from ptcg_rl.rl.native_route_scheduler import NativeArenaKey
from ptcg_rl.rl.native_scripted_policy import NativeScriptedPolicy
from ptcg_rl.rl.sequence_actor import (
    GeneralistSequenceActorPolicy,
    SequenceRolloutPrecision,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessCollectionResult,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart
from ptcg_rl.rl.stateless_opponents import (
    PastSelfArchivedSequenceSource,
    PastSelfNativeSource,
    PastSelfPolicyPool,
)

_DEFAULT_OPTIONS_PER_LANE = 256


class NativeStatelessCollector:
    """Run one fixed assignment window through bounded mixed engine shards."""

    def __init__(
        self,
        *,
        actor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        identity: StatelessFragmentIdentity,
        catalog: PublicDeckCatalog,
        active_decks: Mapping[str, CanonicalDeck],
        opponent_decks: Mapping[str, CanonicalDeck],
        members: Sequence[PfspMember],
        past_self_pool: PastSelfPolicyPool,
        historical_pool: NativeHistoricalPolicyPool,
        scripted_policies: Mapping[str, NativeScriptedPolicy],
        scripted_bindings: Mapping[str, tuple[str, str]],
        maximum_engine_steps: int,
        seed: int,
        fragments_per_part: int,
        mirror_bilateral_trajectories: bool,
        arena_capacity: int | None = None,
        engine_shards: int = 1,
        policy_cohort_slots: int | None = None,
        policy_group_bank_limit: int = 2,
        policy_cohort_wait_ms: float = 0.0,
        trainable_decision_budget: int | None = None,
        frozen_batch_min_rows: int = 1,
        frozen_batch_max_wait_waves: int = 1,
        sequence_rollout_precision: SequenceRolloutPrecision = "fp32",
        options_per_lane: int = _DEFAULT_OPTIONS_PER_LANE,
        library_path: Path | None = None,
        engine_fact_producer: (
            ProspectiveEngineFactProducer | NativeProspectiveEngineFactProducer | None
        ) = None,
        engine_fact_workers: int | None = None,
        owns_engine_fact_producer: bool = True,
        policy_stream_pool: NativePolicyCudaStreamPool | None = None,
        arena_resource_pool: NativeArenaResourcePool | None = None,
        route_input_contracts: Mapping[str, tuple[PublicDeckCatalog, str]]
        | None = None,
        retain_trajectories: bool = True,
        compact_part_sink: Callable[[CompactFragmentPart], None] | None = None,
        external_drain_signal: Callable[[], bool] | None = None,
        immediate_whole_game_cutoff_on_drain: bool = False,
        current_policy_temperature: float = 1.0,
        frozen_policy_temperatures: Mapping[str, float] | None = None,
        evaluation_action_only: bool = False,
    ) -> None:
        """Bind one behavior publication and every exact opponent runtime."""
        if actor.identity != identity:
            raise ValueError("native collector actor identity differs")
        if identity.public_deck_catalog_fingerprint != catalog.fingerprint:
            raise ValueError("native collector catalog identity differs")
        if not 1 <= engine_shards <= 8:
            raise ValueError("native engine shards must be within [1, 8]")
        if (
            maximum_engine_steps <= 0
            or options_per_lane <= 0
            or (arena_capacity is not None and arena_capacity <= 0)
            or (policy_cohort_slots is not None and policy_cohort_slots <= 0)
            or not 1 <= policy_group_bank_limit <= 4
            or not 0.0 <= policy_cohort_wait_ms <= 100.0
            or (engine_fact_workers is not None and engine_fact_workers <= 0)
            or (
                trainable_decision_budget is not None and trainable_decision_budget <= 0
            )
        ):
            raise ValueError("native collection capacities must be positive")
        frozen_temperatures = dict(frozen_policy_temperatures or {})
        temperatures = (float(current_policy_temperature), *frozen_temperatures.values())
        if any(not math.isfinite(value) or value < 0.0 for value in temperatures):
            raise ValueError("native policy temperatures must be finite and non-negative")
        if evaluation_action_only:
            if retain_trajectories:
                raise ValueError(
                    "evaluation action-only collection cannot retain trajectories"
                )
            if current_policy_temperature != 0.0 or any(
                value != 0.0 for value in frozen_temperatures.values()
            ):
                raise ValueError("evaluation action-only collection requires T=0")
            frozen_artifacts = {
                member.policy_sha256
                for member in members
                if member.source in {"past_self", "fixed_stateless_anchor"}
            }
            missing_temperatures = frozen_artifacts - set(frozen_temperatures)
            if missing_temperatures:
                raise ValueError(
                    "evaluation action-only collection requires every frozen "
                    "policy temperature"
                )
            if any(
                member.source not in {"past_self", "fixed_stateless_anchor"}
                for member in members
            ) or scripted_policies:
                raise ValueError(
                    "evaluation action-only collection supports exact model policies"
                )
        elif current_policy_temperature <= 0.0 or any(
            value <= 0.0 for value in frozen_temperatures.values()
        ):
            raise ValueError("T=0 collection is evaluation action-only")
        if frozen_batch_min_rows < 1 or frozen_batch_max_wait_waves < 1:
            raise ValueError("native frozen batch controls must be positive")
        if sequence_rollout_precision not in {"fp32", "bf16"}:
            raise ValueError("sequence rollout precision must be fp32 or bf16")
        if (
            isinstance(actor, GeneralistSequenceActorPolicy)
            and actor.rollout_precision != sequence_rollout_precision
        ):
            raise ValueError("native collector sequence rollout precision differs")
        if (
            not isinstance(actor, GeneralistSequenceActorPolicy)
            and sequence_rollout_precision != "fp32"
        ):
            raise ValueError("BF16 sequence rollout requires a sequence actor")
        self.actor = actor
        self.identity = identity
        self.catalog = catalog
        self.active_decks = dict(active_decks)
        self.opponent_decks = dict(opponent_decks)
        self.members = {member.member_id: member for member in members}
        self.past_self_pool = past_self_pool
        self.historical_pool = historical_pool
        self.scripted_policies = dict(scripted_policies)
        self.scripted_bindings = dict(scripted_bindings)
        self.maximum_engine_steps = int(maximum_engine_steps)
        self.seed = int(seed)
        self.fragments_per_part = int(fragments_per_part)
        self.mirror_bilateral_trajectories = bool(mirror_bilateral_trajectories)
        self.arena_capacity = None if arena_capacity is None else int(arena_capacity)
        self.engine_shards = int(engine_shards)
        self.policy_cohort_slots = (
            None if policy_cohort_slots is None else int(policy_cohort_slots)
        )
        self.policy_group_bank_limit = int(policy_group_bank_limit)
        self.policy_cohort_wait_ms = float(policy_cohort_wait_ms)
        self.trainable_decision_budget = (
            None
            if trainable_decision_budget is None
            else int(trainable_decision_budget)
        )
        self.frozen_batch_min_rows = int(frozen_batch_min_rows)
        self.frozen_batch_max_wait_waves = int(frozen_batch_max_wait_waves)
        self.sequence_rollout_precision = sequence_rollout_precision
        self.options_per_lane = int(options_per_lane)
        self.library_path = library_path
        self.arena_resource_pool = arena_resource_pool
        self.route_input_contracts = dict(route_input_contracts or {})
        self.retain_trajectories = bool(retain_trajectories)
        self.compact_part_sink = compact_part_sink
        # Advisory external latch: once it reports True the collection window
        # drains at the next safe wave boundary even below its local budget.
        self.external_drain_signal = external_drain_signal
        # A distributed learner-clocked window may discard every nonterminal
        # game at that boundary. No bootstrap row is needed because the whole
        # game's already-streamed and local fragments are revoked together.
        self.immediate_whole_game_cutoff_on_drain = bool(
            immediate_whole_game_cutoff_on_drain
        )
        self.current_policy_temperature = float(current_policy_temperature)
        self.frozen_policy_temperatures = frozen_temperatures
        self.evaluation_action_only = bool(evaluation_action_only)
        self.engine_fact_producer: NativeProspectiveEngineFactProducer | None
        if isinstance(engine_fact_producer, NativeProspectiveEngineFactProducer):
            self.engine_fact_producer = engine_fact_producer
        elif engine_fact_producer is None:
            self.engine_fact_producer = None
        else:
            self.engine_fact_producer = NativeProspectiveEngineFactProducer(
                engine_fact_producer,
                maximum_workers=engine_fact_workers,
            )
        self._owns_engine_fact_producer = bool(owns_engine_fact_producer)
        self.engine_fact_producer_fingerprint = (
            None
            if self.engine_fact_producer is None
            else self.engine_fact_producer.fingerprint
        )
        self.deck_batch_cache = NativeDeckBatchCache()
        self.policy_bank = (
            NativePolicyInferenceBank()
            if policy_stream_pool is None
            else NativePolicyInferenceBank(stream_pool=policy_stream_pool)
        )
        self._past_executors: dict[
            str,
            NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy,
        ] = {}
        self._past_temporal_cache_slots: dict[str, int] = {}
        self._closed = False

    def frozen_policy_temperature(self, artifact_sha256: str) -> float:
        """Return one exact frozen artifact's immutable decode temperature."""
        return self.frozen_policy_temperatures.get(artifact_sha256, 1.0)

    def _cohort_group_key(self, game: NativeLiveGame) -> NativeArenaKey:
        """Return the validated opponent execution route for arena packing."""
        cached = getattr(game, "cached_route_key", None)
        if isinstance(cached, NativeArenaKey):
            return cached
        key = self._build_cohort_group_key(game)
        game.cached_route_key = key
        return key

    def _build_cohort_group_key(self, game: NativeLiveGame) -> NativeArenaKey:
        curriculum = game.assignment.curriculum
        if curriculum.lane == "mirror":
            return NativeArenaKey(
                kind="current",
                artifact_sha256=self.identity.behavior_policy_fingerprint,
                policy_fingerprint=self.identity.behavior_policy_fingerprint,
                input_contract_fingerprint=self.identity.input_contract_fingerprint,
                exact_registry_fingerprint=self.identity.exact_registry_fingerprint,
            )
        if curriculum.lane == "pfsp":
            member = game.required_member()
            if member.source in {"past_self", "fixed_stateless_anchor"}:
                return NativeArenaKey(
                    kind="past_self",
                    artifact_sha256=member.policy_sha256,
                    policy_fingerprint=member.pilot_artifact_fingerprint,
                    input_contract_fingerprint=(member.input_contract_fingerprint),
                    exact_registry_fingerprint=(member.exact_registry_fingerprint),
                )
            if member.source == "historical_anchor":
                return NativeArenaKey(
                    kind="historical",
                    artifact_sha256=member.policy_sha256,
                    policy_fingerprint=member.pilot_artifact_fingerprint,
                    input_contract_fingerprint=(member.input_contract_fingerprint),
                    exact_registry_fingerprint=(member.exact_registry_fingerprint),
                )
            raise ValueError(f"unsupported native PFSP source: {member.source}")
        if curriculum.lane == "scripted":
            return NativeArenaKey(
                kind="scripted",
                artifact_sha256=curriculum.opponent_artifact_fingerprint,
                policy_fingerprint=curriculum.opponent_artifact_fingerprint,
                runtime_id=curriculum.opponent_id,
            )
        raise ValueError(f"unsupported native curriculum lane: {curriculum.lane}")

    def collect_assigned(
        self,
        assignments: Sequence[StatelessAssignedGame],
        *,
        cancellation_event: threading.Event | None = None,
    ) -> StatelessCollectionResult:
        """Collect one PPO window without changing its leased assignments."""
        from ptcg_rl.rl.native_route_collection import (
            collect_native_route_assigned,
        )

        if self._closed:
            raise RuntimeError("native collector is closed")
        try:
            return collect_native_route_assigned(
                self,
                assignments,
                cancellation_event=cancellation_event,
            )
        finally:
            if (
                getattr(self, "_owns_engine_fact_producer", True)
                and self.engine_fact_producer is not None
            ):
                self.engine_fact_producer.close()

    def close(self) -> None:
        """Release all window-local inference state and recurrent KV pools."""
        if self._closed:
            return
        # Outstanding policy tickets still own deferred actor transactions and
        # CUDA transfers.  Treat the policy bank as a teardown gate: if it
        # cannot close, leave the collector and every dependent actor intact so
        # the ticket can be resolved and close() can be retried safely.
        policy_bank = self.policy_bank
        if policy_bank is not None:
            policy_bank.close()
        self._closed = True
        first_error: BaseException | None = None

        def clean(operation: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                operation()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(
                        "additional native collector teardown failure: "
                        f"{type(error).__name__}: {error}"
                    )

        producer = self.engine_fact_producer
        self.engine_fact_producer = None
        if getattr(self, "_owns_engine_fact_producer", True) and producer is not None:
            clean(producer.close)
        actors = (self.actor, *self._past_executors.values())
        self._past_executors.clear()
        self._past_temporal_cache_slots.clear()
        closed_actor_ids: set[int] = set()
        for actor in actors:
            if (
                isinstance(actor, GeneralistSequenceActorPolicy)
                and id(actor) not in closed_actor_ids
            ):
                closed_actor_ids.add(id(actor))
                clean(actor.close)
        if first_error is not None:
            raise first_error

    def _dispatch_rows(
        self,
        view: NativeTrainingBatchView,
        live: Mapping[int, NativeLiveGame],
        forced_rows: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        dict[str, npt.NDArray[np.int64]],
        dict[str, npt.NDArray[np.int64]],
        dict[str, npt.NDArray[np.int64]],
    ]:
        forced = {int(row) for row in forced_rows}
        current: list[int] = []
        past: defaultdict[str, list[int]] = defaultdict(list)
        historical: defaultdict[str, list[int]] = defaultdict(list)
        scripted: defaultdict[str, list[int]] = defaultdict(list)
        for row in range(view.batch_size):
            if row in forced:
                continue
            game = live[int(view.slots[row])]
            if int(view.select_player[row]) == game.candidate_seat:
                current.append(row)
                continue
            lane = game.assignment.curriculum.lane
            if lane == "mirror":
                current.append(row)
            elif lane == "pfsp":
                member = game.required_member()
                if member.source in {"past_self", "fixed_stateless_anchor"}:
                    past[member.policy_sha256].append(row)
                elif member.source == "historical_anchor":
                    historical[member.policy_sha256].append(row)
                else:
                    raise ValueError(f"unsupported native PFSP source: {member.source}")
            elif lane == "scripted":
                scripted[game.assignment.curriculum.opponent_id].append(row)
            else:
                raise ValueError(f"unsupported native curriculum lane: {lane}")
        return (
            np.asarray(current, dtype=np.int64),
            {key: np.asarray(rows, dtype=np.int64) for key, rows in past.items()},
            {key: np.asarray(rows, dtype=np.int64) for key, rows in historical.items()},
            {key: np.asarray(rows, dtype=np.int64) for key, rows in scripted.items()},
        )

    def _trajectory_local_rows(
        self,
        view: NativeTrainingBatchView,
        rows: npt.NDArray[np.int64],
        live: Mapping[int, NativeLiveGame],
    ) -> npt.NDArray[np.int64]:
        """Select current-policy rows retained as learner trajectories."""
        if not self.retain_trajectories:
            return np.asarray((), dtype=np.int64)
        return cast(
            npt.NDArray[np.int64],
            np.asarray(
                [
                    local
                    for local, row in enumerate(rows)
                    if (
                        not live[int(view.slots[row])].trajectory_sealed(
                            int(view.select_player[row])
                        )
                        and (
                            int(view.select_player[row])
                            == live[int(view.slots[row])].candidate_seat
                            or (
                                self.mirror_bilateral_trajectories
                                and live[
                                    int(view.slots[row])
                                ].assignment.curriculum.lane
                                == "mirror"
                            )
                        )
                    )
                ],
                dtype=np.int64,
            ),
        )

    def _past_executor(
        self,
        artifact_sha256: str,
    ) -> NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy:
        existing = self._past_executors.get(artifact_sha256)
        if existing is not None:
            return existing
        members = tuple(
            member
            for member in self.members.values()
            if (
                member.source in {"past_self", "fixed_stateless_anchor"}
                and member.policy_sha256 == artifact_sha256
            )
        )
        if not members:
            raise KeyError("native past-self actor binding is absent")
        source_lookup = getattr(self.past_self_pool, "native_source", None)

        def resolve(member_id: str) -> PastSelfNativeSource:
            if callable(source_lookup):
                return cast(PastSelfNativeSource, source_lookup(member_id))
            return self.past_self_pool.actor(member_id)

        source = resolve(members[0].member_id)
        if any(resolve(member.member_id) is not source for member in members[1:]):
            raise ValueError(
                "one native past-self artifact has conflicting actor bindings"
            )
        behavior_fingerprints = {
            member.pilot_artifact_fingerprint for member in members
        }
        if behavior_fingerprints != {source.identity.behavior_policy_fingerprint}:
            raise ValueError(
                "native past-self actor differs from its checkpoint members"
            )
        executor: NativePolicyInferenceExecutor | GeneralistSequenceActorPolicy
        if isinstance(source, PastSelfArchivedSequenceSource):
            model = copy.deepcopy(source.model)
            if self.sequence_rollout_precision == "bf16":
                # The shared FP32 CPU archive already binds the exact checkpoint
                # identity.  Round the window-local copy before CUDA residency so
                # workers never reserve both FP32 and BF16 model storage.
                model.to(dtype=torch.bfloat16)
            executor = GeneralistSequenceActorPolicy(
                model,
                identity=source.identity,
                device=source.device,
                verify_model_state=False,
                retain_raw_blocks=False,
                temporal_cache_slots=(
                    getattr(self, "_past_temporal_cache_slots", {}).get(
                        artifact_sha256,
                        self.actor.temporal_cache_slot_capacity,
                    )
                    if isinstance(self.actor, GeneralistSequenceActorPolicy)
                    else None
                ),
                rollout_precision=self.sequence_rollout_precision,
            )
            if self.sequence_rollout_precision == "bf16":
                model = executor.rollout_model
                executor.model = model
        elif isinstance(source, GeneralistSequenceActorPolicy):
            executor = GeneralistSequenceActorPolicy(
                source.model,
                identity=source.identity,
                device=source.device,
                verify_model_state=False,
                retain_raw_blocks=False,
                temporal_cache_slots=(
                    getattr(self, "_past_temporal_cache_slots", {}).get(
                        artifact_sha256,
                        self.actor.temporal_cache_slot_capacity,
                    )
                    if isinstance(self.actor, GeneralistSequenceActorPolicy)
                    else None
                ),
                rollout_precision=self.sequence_rollout_precision,
            )
        else:
            executor = NativePolicyInferenceExecutor(
                source.model,
                identity=source.identity,
                device=source.device,
                verify_model_state=False,
            )
        self._past_executors[artifact_sha256] = executor
        return executor

    def _configure_past_temporal_cache_slots(
        self,
        assignments: Sequence[StatelessAssignedGame],
        *,
        arena_capacity: int,
    ) -> None:
        """Right-size frozen recurrent pools to their maximum live game quota."""
        if not isinstance(self.actor, GeneralistSequenceActorPolicy):
            return
        counts: Counter[str] = Counter()
        for assignment in assignments:
            curriculum = assignment.curriculum
            if curriculum.lane != "pfsp":
                continue
            member = self.members.get(curriculum.member_id)
            if member is None:
                raise KeyError("native PFSP assignment member is absent")
            if member.source in {"past_self", "fixed_stateless_anchor"}:
                counts[member.policy_sha256] += 1
        temporal_cache_slots = {
            artifact_sha256: min(int(count), arena_capacity)
            for artifact_sha256, count in counts.items()
            if count > 0
        }
        past_executors = getattr(self, "_past_executors", {})
        for artifact_sha256, executor in tuple(past_executors.items()):
            if not isinstance(executor, GeneralistSequenceActorPolicy):
                continue
            if (
                temporal_cache_slots.get(artifact_sha256)
                != executor.temporal_cache_slot_capacity
            ):
                executor.close()
                del past_executors[artifact_sha256]
        self._past_temporal_cache_slots = temporal_cache_slots

    def _release_sequence_game(self, game: NativeLiveGame) -> None:
        """Release current and frozen recurrent state before slot reuse."""
        actors = (
            self.actor,
            *self._past_executors.values(),
        )
        for actor in actors:
            if not isinstance(actor, GeneralistSequenceActorPolicy):
                continue
            actor.release_game(game_id=game.game_id, seat=0)
            actor.release_game(game_id=game.game_id, seat=1)


__all__ = ["NativeStatelessCollector"]
