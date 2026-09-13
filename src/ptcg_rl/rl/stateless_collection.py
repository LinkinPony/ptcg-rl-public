"""Batched engine collection for fixed-horizon clean stateless fragments."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.actions.selection import forced_action, is_legal_action
from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.context import PublicCatalogContext, PublicEventDecisionToken
from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.engine.vector_battle import (
    VectorGame,
    finish_pointer_battle,
    load_cg_sim_lib,
    result_index,
    select_pointer_battle,
    start_pointer_battle,
)
from ptcg_rl.opponents import BattleAgent, build_opponent, opponent_registry
from ptcg_rl.rl.policy_inputs import (
    PolicyInputContract,
    SimpleStatelessActorRow,
    SimpleStatelessPublicInputAdapter,
    SimpleStatelessTensorizeRequest,
    tensorize_simple_stateless_request,
)
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_actor import (
    StatelessActorDecisionTrace,
    StatelessActorPolicy,
)
from ptcg_rl.rl.stateless_curriculum import (
    CurriculumAssignment,
    PfspMember,
    StatelessCurriculumController,
    TerminalStatus,
)
from ptcg_rl.rl.stateless_deck_balance import (
    DeckSeatAssignment,
    StatelessDeckBalanceSampler,
)
from ptcg_rl.rl.stateless_fragment import (
    FixedHorizonFragmentBuilder,
    StatelessFragment,
    StatelessFragmentContext,
    StatelessFragmentIdentity,
    next_fragment_context,
)
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart
from ptcg_rl.rl.stateless_opponents import (
    HistoricalPolicyRequest,
    HistoricalPolicyRuntime,
    PastSelfPolicyRuntime,
    past_self_input_adapter,
)
from ptcg_rl.rl.stateless_training_config import StatelessScriptedOpponentConfig

NativeInferenceRouteKind = Literal["current", "past_self", "historical"]


def _uses_generalist_sequence(actor: object | None) -> bool:
    """Return whether an actor owns transactional temporal state."""
    return bool(getattr(actor, "uses_generalist_sequence", False))


class NativeArtifactInferenceReport(BaseModel):
    """Bounded per-artifact batch geometry for one native collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_kind: NativeInferenceRouteKind
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batches: int = Field(gt=0)
    rows: int = Field(gt=0)


class StatelessCollectionReport(BaseModel):
    """Measured game/lane/seat throughput for one optimizer collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games_started: int = Field(gt=0)
    assignment_reservations: int = Field(default=0, ge=0)
    unstarted_reservations_released: int = Field(default=0, ge=0)
    games_finished: int = Field(ge=0)
    games_cancelled: int = Field(ge=0)
    games_window_cutoff: int = Field(default=0, ge=0)
    games_immediate_window_cutoff: int = Field(default=0, ge=0)
    native_engine_arenas: int = Field(default=0, ge=0)
    engine_steps: int = Field(ge=0)
    candidate_decisions: int = Field(ge=0)
    mirror_opponent_decisions: int = Field(default=0, ge=0)
    native_trainable_decisions: int = Field(default=0, ge=0)
    native_trainable_decision_budget: int | None = Field(default=None, gt=0)
    native_trainable_decision_budget_reached: bool = False
    native_trainable_decision_budget_overshoot: int = Field(default=0, ge=0)
    fragments: int = Field(ge=0)
    elapsed_seconds: float = Field(gt=0.0)
    decisions_per_second: float = Field(ge=0.0)
    native_phase_seconds: float = Field(default=0.0, ge=0.0)
    scripted_phase_seconds: float = Field(default=0.0, ge=0.0)
    input_seconds: float = Field(ge=0.0)
    current_policy_seconds: float = Field(ge=0.0)
    past_self_policy_seconds: float = Field(ge=0.0)
    historical_policy_seconds: float = Field(ge=0.0)
    scripted_policy_seconds: float = Field(ge=0.0)
    native_policy_route_overlap_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_route_overlap_waves: int = Field(default=0, ge=0)
    native_policy_cohort_batches: int = Field(default=0, ge=0)
    native_policy_cohort_rows: int = Field(default=0, ge=0)
    native_policy_cohort_max_rows: int = Field(default=0, ge=0)
    native_policy_cohort_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_cohort_wait_events: int = Field(default=0, ge=0)
    native_policy_cohort_wait_harvests: int = Field(default=0, ge=0)
    native_gpu_feed_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_gpu_feed_wait_events: int = Field(default=0, ge=0)
    native_policy_host_prepare_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_host_prepare_wall_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_host_prepare_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_host_prepare_overlap_seconds: float = Field(default=0.0, ge=0.0)
    native_policy_host_prepare_past_bypasses: int = Field(default=0, ge=0)
    native_policy_completion_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_scripted_prefetch_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_scripted_prefetch_queue_seconds: float = Field(default=0.0, ge=0.0)
    native_scripted_prefetch_overlap_seconds: float = Field(default=0.0, ge=0.0)
    native_scripted_prefetch_batches: int = Field(default=0, ge=0)
    native_scripted_prefetch_rows: int = Field(default=0, ge=0)
    native_bank_engine_wait_seconds: float = Field(default=0.0, ge=0.0)
    native_bank_policy_overlap_seconds: float = Field(default=0.0, ge=0.0)
    native_bank_policy_prefetches: int = Field(default=0, ge=0)
    native_bank_engine_barriers: int = Field(default=0, ge=0)
    native_bank_policy_groups: int = Field(default=0, ge=0)
    native_bank_policy_group_members: int = Field(default=0, ge=0)
    native_bank_policy_group_max_size: int = Field(default=0, ge=0)
    native_bank_policy_coalescing_misses: int = Field(default=0, ge=0)
    native_bank_gpu_feed_gap_seconds: float = Field(default=0.0, ge=0.0)
    native_bank_gpu_feed_gap_events: int = Field(default=0, ge=0)
    native_startup_seconds: float = Field(default=0.0, ge=0.0)
    native_budget_fence_wait_seconds: float = Field(default=0.0, ge=0.0)
    engine_fact_seconds: float = Field(default=0.0, ge=0.0)
    engine_fact_wait_seconds: float = Field(default=0.0, ge=0.0)
    engine_fact_overlap_seconds: float = Field(default=0.0, ge=0.0)
    engine_fact_roots: int = Field(default=0, ge=0)
    engine_fact_eligible_options: int = Field(default=0, ge=0)
    engine_fact_native_batch_calls: int = Field(default=0, ge=0)
    engine_fact_native_transitions: int = Field(default=0, ge=0)
    engine_fact_unresolved_worlds: int = Field(default=0, ge=0)
    engine_fact_resolved_options: int = Field(default=0, ge=0)
    engine_control_seconds: float = Field(ge=0.0)
    current_policy_batches: int = Field(ge=0)
    current_policy_rows: int = Field(ge=0)
    past_self_policy_batches: int = Field(ge=0)
    past_self_policy_rows: int = Field(ge=0)
    historical_policy_batches: int = Field(ge=0)
    historical_policy_rows: int = Field(ge=0)
    frozen_pending_row_waves: int = Field(default=0, ge=0)
    frozen_threshold_releases: int = Field(default=0, ge=0)
    frozen_deadline_releases: int = Field(default=0, ge=0)
    frozen_forced_releases: int = Field(default=0, ge=0)
    native_process_workers: int = Field(default=0, ge=0)
    native_inference_requests: int = Field(default=0, ge=0)
    native_inference_threshold_batches: int = Field(default=0, ge=0)
    native_inference_deadline_batches: int = Field(default=0, ge=0)
    native_inference_unblock_batches: int = Field(default=0, ge=0)
    native_shared_batch_rows: int = Field(default=0, ge=0)
    integrated_scripted_inference_requests: int = Field(default=0, ge=0)
    integrated_scripted_inference_batches: int = Field(default=0, ge=0)
    integrated_scripted_inference_rows: int = Field(default=0, ge=0)
    integrated_scripted_mixed_batches: int = Field(default=0, ge=0)
    integrated_scripted_mixed_rows: int = Field(default=0, ge=0)
    native_artifact_inference: tuple[NativeArtifactInferenceReport, ...] = ()
    lane_games: dict[str, int]
    seat_games: dict[str, int]
    pfsp_member_games: dict[str, int]
    candidate_score_by_lane: dict[str, float]


@dataclass
class StatelessCollectionResult:
    """Fragments and monitoring evidence from one complete game cohort."""

    fragments: tuple[StatelessFragment, ...]
    report: StatelessCollectionReport
    assignments: tuple[StatelessAssignedGame, ...]
    outcomes: tuple[StatelessGameOutcome, ...]
    compact_part_paths: tuple[Path, ...] = ()
    compact_parts: tuple[CompactFragmentPart, ...] = ()


@dataclass(frozen=True)
class StatelessAssignedGame:
    """One centrally leased deck/seat/curriculum assignment for a worker."""

    balance: DeckSeatAssignment
    curriculum: CurriculumAssignment


class StatelessGameOutcome(BaseModel):
    """Outcome and retained candidate-row evidence for the single writer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    balance_assignment_id: str
    curriculum_assignment_id: str
    status: Literal[
        "engine_terminal",
        "infrastructure_error",
        "step_limit",
        "window_cutoff",
    ]
    candidate_score: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_decisions: int = Field(ge=0)


@dataclass
class _LiveGame:
    game: VectorGame
    balance_assignment: DeckSeatAssignment
    curriculum_assignment: CurriculumAssignment
    candidate_deck: CanonicalDeck
    opponent_deck: CanonicalDeck
    candidate_adapter: SimpleStatelessPublicInputAdapter
    builder: FixedHorizonFragmentBuilder
    candidate_decisions: int = 0
    mirror_builder: FixedHorizonFragmentBuilder | None = None
    mirror_decisions: int = 0
    past_self_decisions: int = 0
    opponent_adapter: SimpleStatelessPublicInputAdapter | None = None
    opponent_member: PfspMember | None = None
    scripted_agent: BattleAgent | None = None


@dataclass(frozen=True)
class _PolicyTurn:
    live: _LiveGame
    role: Literal["candidate", "mirror", "past_self"]
    row: SimpleStatelessActorRow
    adapter: SimpleStatelessPublicInputAdapter
    public_event_token: PublicEventDecisionToken
    known_opponent_counts: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _PendingPolicyTurn:
    live: _LiveGame
    role: Literal["candidate", "mirror", "past_self"]
    request: SimpleStatelessTensorizeRequest
    adapter: SimpleStatelessPublicInputAdapter
    context: PublicCatalogContext
    public_event_token: PublicEventDecisionToken
    known_opponent_counts: tuple[tuple[int, int], ...] = ()


class StatelessEngineCollector:
    """Run a fixed assigned cohort to terminal with batched clean inference."""

    def __init__(
        self,
        *,
        actor: StatelessActorPolicy,
        identity: StatelessFragmentIdentity,
        candidate_contract: PolicyInputContract,
        catalog: PublicDeckCatalog,
        active_decks: Mapping[str, CanonicalDeck],
        opponent_decks: Mapping[str, CanonicalDeck],
        curriculum: StatelessCurriculumController | None,
        deck_balance: StatelessDeckBalanceSampler | None,
        past_self_pool: PastSelfPolicyRuntime,
        historical_pool: HistoricalPolicyRuntime,
        scripted: Mapping[str, StatelessScriptedOpponentConfig],
        maximum_engine_steps: int,
        seed: int,
        mirror_bilateral_trajectories: bool = False,
        opponent_members: Sequence[PfspMember] = (),
        defer_controller_updates: bool = False,
        engine_fact_producer: ProspectiveEngineFactProducer | None = None,
    ) -> None:
        """Bind immutable behavior and all three isolated opponent runtimes."""
        if maximum_engine_steps <= 0:
            raise ValueError("maximum engine steps must be positive")
        self.actor = actor
        self.identity = identity
        self.candidate_contract = candidate_contract
        self.catalog = catalog
        self.active_decks = dict(active_decks)
        self.opponent_decks = dict(opponent_decks)
        self.curriculum = curriculum
        self.deck_balance = deck_balance
        self.defer_controller_updates = defer_controller_updates
        if not defer_controller_updates and (
            curriculum is None or deck_balance is None
        ):
            raise ValueError("local collection requires writable controllers")
        resolved_members = (
            tuple(opponent_members)
            if curriculum is None
            else tuple(curriculum.state.members)
        )
        self.opponent_members = {
            member.member_id: member for member in resolved_members
        }
        self.past_self_pool = past_self_pool
        self.historical_pool = historical_pool
        self.scripted = dict(scripted)
        self.maximum_engine_steps = maximum_engine_steps
        self.seed = seed
        self.mirror_bilateral_trajectories = bool(mirror_bilateral_trajectories)
        self.engine_fact_producer = engine_fact_producer
        if identity.input_contract_fingerprint != candidate_contract.fingerprint:
            raise ValueError("collector candidate input contract mismatch")
        if identity.public_deck_catalog_fingerprint != catalog.fingerprint:
            raise ValueError("collector public catalog identity mismatch")

    def collect(self, *, games: int) -> StatelessCollectionResult:
        """Collect one complete cohort without crossing a behavior publication."""
        if games <= 0:
            raise ValueError("collection game count must be positive")
        if self.curriculum is None or self.deck_balance is None:
            raise RuntimeError("detached collector requires explicit assignments")
        assignments = assign_stateless_games(
            games=games,
            curriculum=self.curriculum,
            deck_balance=self.deck_balance,
            active_decks=self.active_decks,
            opponent_decks=self.opponent_decks,
            mirror_policy_fingerprint=(self.identity.behavior_policy_fingerprint),
        )
        return self.collect_assigned(assignments)

    def collect_assigned(
        self,
        assignments: Sequence[StatelessAssignedGame],
    ) -> StatelessCollectionResult:
        """Collect centrally assigned games without mutating worker-side control."""
        if not assignments:
            raise ValueError("assigned collection requires at least one game")
        games = len(assignments)
        started_at = time.perf_counter()
        lib = load_cg_sim_lib()
        live_games: dict[str, _LiveGame] = {}
        fragments: list[StatelessFragment] = []
        outcomes: list[StatelessGameOutcome] = []
        lane_games: Counter[str] = Counter()
        seat_games: Counter[str] = Counter()
        member_games: Counter[str] = Counter()
        lane_scores: defaultdict[str, list[float]] = defaultdict(list)
        finished_games = 0
        cancelled_games = 0
        mirror_trajectory_decisions = 0
        engine_steps = 0
        input_seconds = 0.0
        current_policy_seconds = 0.0
        past_self_policy_seconds = 0.0
        historical_policy_seconds = 0.0
        scripted_policy_seconds = 0.0
        current_policy_batches = 0
        current_policy_rows = 0
        past_self_policy_batches = 0
        past_self_policy_rows = 0
        historical_policy_batches = 0
        historical_policy_rows = 0
        try:
            for game_index, assignment in enumerate(assignments):
                live = self._start_game(
                    lib,
                    game_index=game_index,
                    assignment=assignment,
                )
                live_games[live.game.game_id] = live
                lane = live.curriculum_assignment.lane
                lane_games[lane] += 1
                seat_games[str(live.curriculum_assignment.candidate_seat)] += 1
                if live.curriculum_assignment.member_id:
                    member_games[live.curriculum_assignment.member_id] += 1
            while live_games:
                current_turns: list[_PolicyTurn] = []
                past_turns: defaultdict[int, list[_PolicyTurn]] = defaultdict(list)
                past_actors: dict[
                    int,
                    StatelessActorPolicy | GeneralistSequenceActorPolicy,
                ] = {}
                pending_turns: list[_PendingPolicyTurn] = []
                historical_turns: list[tuple[_LiveGame, HistoricalPolicyRequest]] = []
                for live in tuple(live_games.values()):
                    observation = live.game.observation
                    seat = _player_index(observation)
                    if seat not in (0, 1):
                        raise ValueError(
                            "engine observation has an invalid acting seat"
                        )
                    select = _field(observation, "select")
                    forced = forced_action(select)
                    if seat == live.curriculum_assignment.candidate_seat:
                        input_started_at = time.perf_counter()
                        context = live.candidate_adapter.observe(observation)
                        input_seconds += time.perf_counter() - input_started_at
                        if forced is not None:
                            self._submit(lib, live, forced)
                            engine_steps += 1
                        else:
                            pending_turns.append(
                                _PendingPolicyTurn(
                                    live=live,
                                    role="candidate",
                                    request=(
                                        live.candidate_adapter.tensorize_request(
                                            observation,
                                            context=context,
                                        )
                                    ),
                                    adapter=live.candidate_adapter,
                                    context=context,
                                    public_event_token=(
                                        live.candidate_adapter.prepare_decision()
                                    ),
                                    known_opponent_counts=(
                                        live.candidate_adapter.known_opponent_counts
                                    ),
                                )
                            )
                        continue
                    lane = live.curriculum_assignment.lane
                    if lane == "mirror":
                        if live.opponent_adapter is None:
                            raise RuntimeError("mirror game has no public adapter")
                        input_started_at = time.perf_counter()
                        context = live.opponent_adapter.observe(observation)
                        input_seconds += time.perf_counter() - input_started_at
                        if forced is not None:
                            self._submit(lib, live, forced)
                            engine_steps += 1
                        else:
                            pending_turns.append(
                                _PendingPolicyTurn(
                                    live=live,
                                    role="mirror",
                                    request=(
                                        live.opponent_adapter.tensorize_request(
                                            observation,
                                            context=context,
                                        )
                                    ),
                                    adapter=live.opponent_adapter,
                                    context=context,
                                    public_event_token=(
                                        live.opponent_adapter.prepare_decision()
                                    ),
                                    known_opponent_counts=(
                                        live.opponent_adapter.known_opponent_counts
                                    ),
                                )
                            )
                    elif lane == "pfsp":
                        member = live.opponent_member
                        if member is None:
                            raise RuntimeError("PFSP game has no member")
                        if member.source == "past_self":
                            if live.opponent_adapter is None:
                                raise RuntimeError("past-self game has no adapter")
                            input_started_at = time.perf_counter()
                            context = live.opponent_adapter.observe(observation)
                            input_seconds += time.perf_counter() - input_started_at
                            if forced is not None:
                                self._submit(lib, live, forced)
                                engine_steps += 1
                            else:
                                pending_turns.append(
                                    _PendingPolicyTurn(
                                        live=live,
                                        role="past_self",
                                        request=(
                                            live.opponent_adapter.tensorize_request(
                                                observation,
                                                context=context,
                                            )
                                        ),
                                        adapter=live.opponent_adapter,
                                        context=context,
                                        public_event_token=(
                                            live.opponent_adapter.prepare_decision()
                                        ),
                                    )
                                )
                        else:
                            historical_turns.append(
                                (
                                    live,
                                    HistoricalPolicyRequest(
                                        member_id=member.member_id,
                                        game_id=live.game.game_id,
                                        observation=observation,
                                        forced_action=forced,
                                    ),
                                )
                            )
                    else:
                        scripted_started_at = time.perf_counter()
                        action = self._scripted_action(
                            live,
                            observation=observation,
                            forced=forced,
                        )
                        scripted_policy_seconds += (
                            time.perf_counter() - scripted_started_at
                        )
                        self._submit(lib, live, action)
                        engine_steps += 1

                if pending_turns:
                    input_started_at = time.perf_counter()
                    rows = tuple(
                        tensorize_simple_stateless_request(turn.request)
                        for turn in pending_turns
                    )
                    input_seconds += time.perf_counter() - input_started_at
                    for pending, tensorized in zip(
                        pending_turns,
                        rows,
                        strict=True,
                    ):
                        row = pending.adapter.complete_tensorized(
                            tensorized,
                            context=pending.context,
                        )
                        member = pending.live.opponent_member
                        past_actor = (
                            None
                            if pending.role != "past_self" or member is None
                            else self.past_self_pool.actor(member.member_id)
                        )
                        sequence_actor = (
                            self.actor if pending.role != "past_self" else past_actor
                        )
                        if _uses_generalist_sequence(sequence_actor):
                            if pending.role == "candidate":
                                decision_index = pending.live.candidate_decisions
                            elif pending.role == "mirror":
                                decision_index = pending.live.mirror_decisions
                            else:
                                decision_index = pending.live.past_self_decisions
                            seat = (
                                pending.live.curriculum_assignment.candidate_seat
                                if pending.role == "candidate"
                                else 1
                                - pending.live.curriculum_assignment.candidate_seat
                            )
                            game_id = pending.live.game.game_id
                            row = replace(
                                row,
                                sequence_identity=SequenceDecisionIdentity(
                                    game_id=game_id,
                                    seat=cast(Literal[0, 1], seat),
                                    decision_index=decision_index,
                                    request_id=(f"{game_id}:{seat}:{decision_index}"),
                                ),
                            )
                        turn = _PolicyTurn(
                            live=pending.live,
                            role=pending.role,
                            row=row,
                            adapter=pending.adapter,
                            public_event_token=pending.public_event_token,
                            known_opponent_counts=(pending.known_opponent_counts),
                        )
                        if pending.role != "past_self":
                            current_turns.append(turn)
                            continue
                        member = pending.live.opponent_member
                        if member is None:
                            raise RuntimeError("past-self turn has no PFSP member")
                        opponent_actor = (
                            past_actor
                            if past_actor is not None
                            else self.past_self_pool.actor(member.member_id)
                        )
                        actor_key = id(opponent_actor)
                        past_actors[actor_key] = opponent_actor
                        past_turns[actor_key].append(turn)

                if current_turns:
                    policy_started_at = time.perf_counter()
                    trace = self.actor.sample(
                        tuple(turn.row for turn in current_turns),
                        temperature=1.0,
                    )
                    current_policy_seconds += time.perf_counter() - policy_started_at
                    current_policy_batches += 1
                    current_policy_rows += len(current_turns)
                    current_pairs = tuple(
                        zip(current_turns, trace.decisions, strict=True)
                    )
                    for pair_index, (turn, decision) in enumerate(current_pairs):
                        sequence_committed = False
                        events_committed = False
                        try:
                            self._submit(lib, turn.live, decision.action)
                            if _uses_generalist_sequence(self.actor):
                                cast(Any, self.actor).commit_decision(
                                    turn.row,
                                    decision,
                                )
                                sequence_committed = True
                            turn.adapter.commit_decision(turn.public_event_token)
                            events_committed = True
                        except BaseException:
                            abort_start = pair_index + int(sequence_committed)
                            for pending_turn, pending_decision in current_pairs[
                                abort_start:
                            ]:
                                if _uses_generalist_sequence(self.actor):
                                    cast(Any, self.actor).abort_decision(
                                        pending_turn.row,
                                        pending_decision,
                                    )
                                pending_turn.adapter.abort_decision(
                                    pending_turn.public_event_token
                                )
                            if sequence_committed and not events_committed:
                                turn.adapter.abort_decision(turn.public_event_token)
                            raise
                        engine_steps += 1
                        if turn.role == "candidate":
                            self._record_candidate(
                                turn.live,
                                turn.row,
                                decision,
                                fragments,
                                known_opponent_counts=(turn.known_opponent_counts),
                            )
                        elif (
                            turn.role == "mirror" and self.mirror_bilateral_trajectories
                        ):
                            self._record_mirror(
                                turn.live,
                                turn.row,
                                decision,
                                fragments,
                                known_opponent_counts=(turn.known_opponent_counts),
                            )
                for actor_key, turns in past_turns.items():
                    opponent_actor = past_actors[actor_key]
                    policy_started_at = time.perf_counter()
                    trace = opponent_actor.sample(
                        tuple(turn.row for turn in turns),
                        temperature=1.0,
                    )
                    past_self_policy_seconds += time.perf_counter() - policy_started_at
                    past_self_policy_batches += 1
                    past_self_policy_rows += len(turns)
                    pairs = tuple(zip(turns, trace.decisions, strict=True))
                    for pair_index, (turn, decision) in enumerate(pairs):
                        sequence_committed = False
                        events_committed = False
                        try:
                            self._submit(lib, turn.live, decision.action)
                            if _uses_generalist_sequence(opponent_actor):
                                cast(Any, opponent_actor).commit_decision(
                                    turn.row,
                                    decision,
                                )
                                sequence_committed = True
                                turn.live.past_self_decisions += 1
                            turn.adapter.commit_decision(turn.public_event_token)
                            events_committed = True
                        except BaseException:
                            abort_start = pair_index + int(sequence_committed)
                            for pending_turn, pending_decision in pairs[abort_start:]:
                                if _uses_generalist_sequence(opponent_actor):
                                    cast(Any, opponent_actor).abort_decision(
                                        pending_turn.row,
                                        pending_decision,
                                    )
                                pending_turn.adapter.abort_decision(
                                    pending_turn.public_event_token
                                )
                            if sequence_committed and not events_committed:
                                turn.adapter.abort_decision(turn.public_event_token)
                            raise
                        engine_steps += 1
                if historical_turns:
                    historical_requests = tuple(
                        request for _live, request in historical_turns
                    )
                    policy_started_at = time.perf_counter()
                    actions = self.historical_pool.act_many(historical_requests)
                    historical_policy_seconds += time.perf_counter() - policy_started_at
                    non_forced = tuple(
                        request
                        for request in historical_requests
                        if request.forced_action is None
                    )
                    historical_policy_batches += len(
                        {request.member_id for request in non_forced}
                    )
                    historical_policy_rows += len(non_forced)
                    for (live, _request), action in zip(
                        historical_turns,
                        actions,
                        strict=True,
                    ):
                        self._submit(lib, live, action)
                        engine_steps += 1

                for game_id, live in tuple(live_games.items()):
                    winner = result_index(live.game.observation)
                    if winner >= 0:
                        score = _candidate_score(
                            winner,
                            live.curriculum_assignment.candidate_seat,
                        )
                        self._finish_terminal(
                            lib,
                            live,
                            score=score,
                            fragments=fragments,
                            outcomes=outcomes,
                        )
                        mirror_trajectory_decisions += live.mirror_decisions
                        lane_scores[live.curriculum_assignment.lane].append(score)
                        finished_games += 1
                        del live_games[game_id]
                    elif live.game.steps >= self.maximum_engine_steps:
                        mirror_trajectory_decisions += self._cancel(
                            lib,
                            live,
                            status="step_limit",
                            fragments=fragments,
                            outcomes=outcomes,
                        )
                        cancelled_games += 1
                        del live_games[game_id]
        except BaseException:
            for live in tuple(live_games.values()):
                self._cancel(
                    lib,
                    live,
                    status="infrastructure_error",
                    fragments=fragments,
                    outcomes=outcomes,
                )
            raise
        elapsed = max(time.perf_counter() - started_at, 1.0e-9)
        decisions = sum(len(fragment.decisions) for fragment in fragments)
        measured_non_engine_seconds = (
            input_seconds
            + current_policy_seconds
            + past_self_policy_seconds
            + historical_policy_seconds
            + scripted_policy_seconds
        )
        return StatelessCollectionResult(
            fragments=tuple(fragments),
            assignments=tuple(assignments),
            outcomes=tuple(outcomes),
            report=StatelessCollectionReport(
                games_started=games,
                games_finished=finished_games,
                games_cancelled=cancelled_games,
                engine_steps=engine_steps,
                candidate_decisions=decisions,
                mirror_opponent_decisions=mirror_trajectory_decisions,
                fragments=len(fragments),
                elapsed_seconds=elapsed,
                decisions_per_second=decisions / elapsed,
                input_seconds=input_seconds,
                current_policy_seconds=current_policy_seconds,
                past_self_policy_seconds=past_self_policy_seconds,
                historical_policy_seconds=historical_policy_seconds,
                scripted_policy_seconds=scripted_policy_seconds,
                engine_control_seconds=max(
                    elapsed - measured_non_engine_seconds,
                    0.0,
                ),
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
                    lane: sum(scores) / len(scores)
                    for lane, scores in sorted(lane_scores.items())
                    if scores
                },
            ),
        )

    def _start_game(
        self,
        lib: Any,
        *,
        game_index: int,
        assignment: StatelessAssignedGame,
    ) -> _LiveGame:
        balance = assignment.balance
        candidate = self.active_decks[balance.deck_digest]
        curriculum_assignment = assignment.curriculum
        opponent = self.opponent_decks.get(curriculum_assignment.opponent_deck_digest)
        if opponent is None:
            raise KeyError("curriculum opponent deck is not registered")
        pair = (
            (candidate.card_ids, opponent.card_ids)
            if balance.seat == 0
            else (opponent.card_ids, candidate.card_ids)
        )
        game_id = (
            f"stateless-{self.identity.behavior_policy_version}-"
            f"{balance.assignment_cursor}-{game_index}"
        )
        try:
            game = start_pointer_battle(
                lib,
                pair,
                game_id=game_id,
                include_search_input=self.engine_fact_producer is not None,
            )
        except BaseException:
            self._commit_outcome(
                StatelessGameOutcome(
                    balance_assignment_id=balance.assignment_id,
                    curriculum_assignment_id=(curriculum_assignment.assignment_id),
                    status="infrastructure_error",
                    candidate_decisions=0,
                )
            )
            raise
        candidate_adapter = SimpleStatelessPublicInputAdapter(
            self.catalog,
            contract=self.candidate_contract,
            player_index=balance.seat,
            own_deck=candidate.card_ids,
            engine_fact_producer=self.engine_fact_producer,
        )
        context = StatelessFragmentContext(
            game_id=game_id,
            seat=balance.seat,
            start_decision_index=0,
            own_deck=candidate.card_ids,
            own_deck_digest=candidate.deck_digest,
            opponent_deck=opponent.card_ids,
            opponent_deck_digest=opponent.deck_digest,
            curriculum_generation=curriculum_assignment.generation,
            assignment_id=curriculum_assignment.assignment_id,
            opponent_artifact_fingerprint=(
                curriculum_assignment.opponent_artifact_fingerprint
            ),
        )
        mirror_builder = None
        member = _member_for_assignment(
            self.opponent_members,
            curriculum_assignment,
        )
        opponent_adapter = None
        scripted_agent = None
        opponent_seat: Literal[0, 1] = 1 if balance.seat == 0 else 0
        if curriculum_assignment.lane == "mirror":
            opponent_adapter = SimpleStatelessPublicInputAdapter(
                self.catalog,
                contract=self.candidate_contract,
                player_index=opponent_seat,
                own_deck=opponent.card_ids,
                engine_fact_producer=self.engine_fact_producer,
            )
            if self.mirror_bilateral_trajectories:
                mirror_context = StatelessFragmentContext(
                    game_id=game_id,
                    seat=opponent_seat,
                    start_decision_index=0,
                    own_deck=opponent.card_ids,
                    own_deck_digest=opponent.deck_digest,
                    opponent_deck=candidate.card_ids,
                    opponent_deck_digest=candidate.deck_digest,
                    curriculum_generation=curriculum_assignment.generation,
                    assignment_id=curriculum_assignment.assignment_id,
                    opponent_artifact_fingerprint=(
                        curriculum_assignment.opponent_artifact_fingerprint
                    ),
                )
                mirror_builder = FixedHorizonFragmentBuilder(
                    self.identity,
                    mirror_context,
                )
        elif member is not None and member.source == "past_self":
            opponent_actor = self.past_self_pool.actor(member.member_id)
            opponent_adapter = past_self_input_adapter(
                opponent_actor,
                catalog=self.catalog,
                player_index=opponent_seat,
                own_deck=opponent.card_ids,
            )
        elif member is not None:
            self.historical_pool.start_game(
                member.member_id,
                game_id=game_id,
                seat=opponent_seat,
            )
        else:
            scripted_agent = self._new_scripted_agent(
                curriculum_assignment.opponent_id,
                seat=opponent_seat,
                own_deck=opponent.card_ids,
            )
        return _LiveGame(
            game=game,
            balance_assignment=balance,
            curriculum_assignment=curriculum_assignment,
            candidate_deck=candidate,
            opponent_deck=opponent,
            candidate_adapter=candidate_adapter,
            builder=FixedHorizonFragmentBuilder(self.identity, context),
            mirror_builder=mirror_builder,
            opponent_adapter=opponent_adapter,
            opponent_member=member,
            scripted_agent=scripted_agent,
        )

    def _record_candidate(
        self,
        live: _LiveGame,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
        fragments: list[StatelessFragment],
        *,
        known_opponent_counts: tuple[tuple[int, int], ...],
    ) -> None:
        if live.builder.full:
            completed = live.builder.truncate(
                bootstrap_value=trace.root_value,
                bootstrap_policy_version=self.identity.behavior_policy_version,
                bootstrap_policy_fingerprint=(
                    self.identity.behavior_policy_fingerprint
                ),
            )
            fragments.append(completed)
            live.builder = FixedHorizonFragmentBuilder(
                self.identity,
                next_fragment_context(completed),
            )
        decision = trace.fragment_decision(
            decision_index=live.candidate_decisions,
            actor_row=row,
            known_opponent_counts=dict(known_opponent_counts),
        )
        live.builder.append(decision)
        live.candidate_decisions += 1

    def _record_mirror(
        self,
        live: _LiveGame,
        row: SimpleStatelessActorRow,
        trace: StatelessActorDecisionTrace,
        fragments: list[StatelessFragment],
        *,
        known_opponent_counts: tuple[tuple[int, int], ...],
    ) -> None:
        builder = live.mirror_builder
        if builder is None:
            raise RuntimeError("bilateral mirror game has no opponent builder")
        if builder.full:
            completed = builder.truncate(
                bootstrap_value=trace.root_value,
                bootstrap_policy_version=self.identity.behavior_policy_version,
                bootstrap_policy_fingerprint=(
                    self.identity.behavior_policy_fingerprint
                ),
            )
            fragments.append(completed)
            builder = FixedHorizonFragmentBuilder(
                self.identity,
                next_fragment_context(completed),
            )
            live.mirror_builder = builder
        decision = trace.fragment_decision(
            decision_index=live.mirror_decisions,
            actor_row=row,
            known_opponent_counts=dict(known_opponent_counts),
        )
        builder.append(decision)
        live.mirror_decisions += 1

    def _finish_terminal(
        self,
        lib: Any,
        live: _LiveGame,
        *,
        score: float,
        fragments: list[StatelessFragment],
        outcomes: list[StatelessGameOutcome],
    ) -> None:
        if live.builder.decision_count > 0:
            fragments.append(
                live.builder.finish_terminal(engine_reward=2.0 * score - 1.0)
            )
        if live.mirror_builder is not None and live.mirror_builder.decision_count > 0:
            fragments.append(
                live.mirror_builder.finish_terminal(engine_reward=1.0 - 2.0 * score)
            )
        outcome = StatelessGameOutcome(
            balance_assignment_id=live.balance_assignment.assignment_id,
            curriculum_assignment_id=(live.curriculum_assignment.assignment_id),
            status="engine_terminal",
            candidate_score=score,
            candidate_decisions=live.candidate_decisions,
        )
        outcomes.append(outcome)
        self._commit_outcome(outcome)
        self._release_sequence_cache(live)
        self._release_opponent(live)
        finish_pointer_battle(lib, live.game.battle_ptr)

    def _cancel(
        self,
        lib: Any,
        live: _LiveGame,
        *,
        status: Literal["infrastructure_error", "step_limit"],
        fragments: list[StatelessFragment],
        outcomes: list[StatelessGameOutcome],
    ) -> int:
        """Retire one game and return its retained mirror decision count."""
        retained_candidate_decisions = 0
        retained_mirror_decisions = 0
        if status == "step_limit":
            candidate_seat = live.curriculum_assignment.candidate_seat
            for fragment in fragments:
                if fragment.context.game_id != live.game.game_id:
                    continue
                decisions = len(fragment.decisions)
                if fragment.context.seat == candidate_seat:
                    retained_candidate_decisions += decisions
                else:
                    retained_mirror_decisions += decisions
            if (
                retained_candidate_decisions > live.candidate_decisions
                or retained_mirror_decisions > live.mirror_decisions
            ):
                raise RuntimeError(
                    "retained fragment rows exceed collected game decisions"
                )
        else:
            fragments[:] = [
                fragment
                for fragment in fragments
                if fragment.context.game_id != live.game.game_id
            ]
        outcome = StatelessGameOutcome(
            balance_assignment_id=live.balance_assignment.assignment_id,
            curriculum_assignment_id=(live.curriculum_assignment.assignment_id),
            status=status,
            candidate_decisions=retained_candidate_decisions,
        )
        outcomes.append(outcome)
        self._commit_outcome(outcome)
        self._release_sequence_cache(live)
        self._release_opponent(live)
        finish_pointer_battle(lib, live.game.battle_ptr)
        return retained_mirror_decisions

    def _commit_outcome(self, outcome: StatelessGameOutcome) -> None:
        if self.defer_controller_updates:
            return
        if self.curriculum is None or self.deck_balance is None:
            raise RuntimeError("controller update has no writable owner")
        if outcome.status != "infrastructure_error" and outcome.candidate_decisions > 0:
            self.deck_balance.finish(
                outcome.balance_assignment_id,
                finished_decisions=outcome.candidate_decisions,
            )
        else:
            self.deck_balance.cancel(outcome.balance_assignment_id)
        self.curriculum.observe_terminal(
            outcome.curriculum_assignment_id,
            status=outcome.status,
            candidate_score=outcome.candidate_score,
        )

    def _release_opponent(self, live: _LiveGame) -> None:
        member = live.opponent_member
        if member is not None and member.source == "historical_anchor":
            self.historical_pool.release_game(
                member.member_id,
                live.game.game_id,
            )
        elif member is not None and member.source == "past_self":
            actor = self.past_self_pool.actor(member.member_id)
            if _uses_generalist_sequence(actor):
                cast(Any, actor).release_game(
                    game_id=live.game.game_id,
                    seat=1 - live.curriculum_assignment.candidate_seat,
                )

    def _release_sequence_cache(self, live: _LiveGame) -> None:
        """Release current-policy KV only after terminal or whole-game cancel."""
        actor = getattr(self, "actor", None)
        if not _uses_generalist_sequence(actor):
            return
        candidate_seat = live.curriculum_assignment.candidate_seat
        cast(Any, actor).release_game(
            game_id=live.game.game_id,
            seat=candidate_seat,
        )
        if live.curriculum_assignment.lane == "mirror":
            cast(Any, actor).release_game(
                game_id=live.game.game_id,
                seat=1 - candidate_seat,
            )

    def _submit(self, lib: Any, live: _LiveGame, action: Sequence[int]) -> None:
        select = _field(live.game.observation, "select")
        if not is_legal_action(select, action):
            raise ValueError("opponent or policy emitted an engine-illegal action")
        live.game.observation = select_pointer_battle(
            lib,
            live.game.battle_ptr,
            action,
            include_search_input=self.engine_fact_producer is not None,
        )
        live.game.steps += 1

    def _scripted_action(
        self,
        live: _LiveGame,
        *,
        observation: Mapping[str, Any],
        forced: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if forced is not None:
            return forced
        if live.scripted_agent is None:
            raise RuntimeError("scripted game has no agent")
        return tuple(int(index) for index in live.scripted_agent.act(observation))

    def _new_scripted_agent(
        self,
        opponent_id: str,
        *,
        seat: int,
        own_deck: Sequence[int],
    ) -> BattleAgent:
        try:
            configured = self.scripted[opponent_id]
            spec = opponent_registry()[configured.opponent_name]
        except KeyError as error:
            raise KeyError(
                f"scripted runtime is not registered: {opponent_id}"
            ) from error
        agent = build_opponent(
            spec,
            seed=self.seed,
        )
        begin = getattr(agent, "begin_game", None)
        if callable(begin):
            begin(player_index=seat, own_deck=own_deck)
        else:
            agent.reset()
        return agent


def assign_stateless_games(
    *,
    games: int,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
    active_decks: Mapping[str, CanonicalDeck],
    opponent_decks: Mapping[str, CanonicalDeck],
    mirror_policy_fingerprint: str,
) -> tuple[StatelessAssignedGame, ...]:
    """Lease a complete cohort from the controller's single writer."""
    if games <= 0:
        raise ValueError("assignment game count must be positive")
    mirror_decks = tuple(active_decks.values())
    if not mirror_decks:
        raise ValueError("assignment requires at least one active deck")
    balances: tuple[DeckSeatAssignment, ...] = ()
    try:
        balances = deck_balance.assign_many(games)
        curriculum_assignments = curriculum.assign_many(
            tuple(
                (
                    active_decks[balance.deck_digest].deck_digest,
                    balance.seat,
                    mirror_policy_fingerprint,
                    mirror_decks[
                        balance.assignment_cursor % len(mirror_decks)
                    ].deck_digest,
                )
                for balance in balances
            )
        )
        assignments = tuple(
            StatelessAssignedGame(balance=balance, curriculum=assigned)
            for balance, assigned in zip(
                balances,
                curriculum_assignments,
                strict=True,
            )
        )
        for assignment in assignments:
            assigned = assignment.curriculum
            if assigned.opponent_deck_digest not in opponent_decks:
                curriculum.observe_terminals(
                    tuple(
                        (
                            item.curriculum.assignment_id,
                            "cancelled",
                            None,
                        )
                        for item in assignments
                    )
                )
                for item in assignments:
                    deck_balance.cancel(item.balance.assignment_id)
                raise KeyError("curriculum opponent deck is not registered")
    except BaseException:
        for balance in balances:
            if balance in deck_balance.state.inflight_assignments:
                deck_balance.cancel(balance.assignment_id)
        raise
    return assignments


def commit_stateless_outcomes(
    assignments: Sequence[StatelessAssignedGame],
    outcomes: Sequence[StatelessGameOutcome],
    *,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
) -> None:
    """Atomically apply exactly one worker result per central lease."""
    expected = {
        assignment.curriculum.assignment_id: assignment for assignment in assignments
    }
    actual = {outcome.curriculum_assignment_id: outcome for outcome in outcomes}
    if len(expected) != len(assignments) or len(actual) != len(outcomes):
        raise ValueError("parallel collection produced duplicate assignment IDs")
    if set(actual) != set(expected):
        raise ValueError("parallel collection outcomes do not cover leased games")
    if not assignments:
        return
    active_balances = {
        assignment.assignment_id: assignment
        for assignment in deck_balance.state.inflight_assignments
    }
    active_curriculum = curriculum.state.inflight
    deck_outcomes: list[tuple[str, int | None, float | None]] = []
    terminal_observations: list[tuple[str, TerminalStatus, float | None]] = []
    for assignment in assignments:
        outcome = actual[assignment.curriculum.assignment_id]
        if active_balances.get(assignment.balance.assignment_id) != assignment.balance:
            raise ValueError("deck-balance assignment is not the active lease")
        if (
            active_curriculum.get(assignment.curriculum.assignment_id)
            != assignment.curriculum
        ):
            raise ValueError("curriculum assignment is not the active lease")
        if outcome.balance_assignment_id != assignment.balance.assignment_id:
            raise ValueError("parallel outcome crossed deck-balance identity")
        _validate_stateless_outcome(outcome)
        if outcome.status != "infrastructure_error" and outcome.candidate_decisions > 0:
            deck_outcomes.append(
                (
                    outcome.balance_assignment_id,
                    outcome.candidate_decisions,
                    outcome.candidate_score,
                )
            )
        else:
            deck_outcomes.append(
                (
                    outcome.balance_assignment_id,
                    None,
                    outcome.candidate_score,
                )
            )
        terminal_observations.append(
            (
                outcome.curriculum_assignment_id,
                outcome.status,
                outcome.candidate_score,
            )
        )
    prepared_deck = deck_balance.prepare_outcomes(deck_outcomes)
    prepared_curriculum = curriculum.prepare_terminals(terminal_observations)
    deck_balance.validate_prepared_outcomes(prepared_deck)
    curriculum.validate_prepared_terminals(prepared_curriculum)

    # This is the only fallible publication step.  The two immutable in-memory
    # snapshots remain untouched if the atomic curriculum write fails.
    curriculum.persist_prepared_terminals(prepared_curriculum)
    deck_balance.publish_prepared_outcomes(prepared_deck)
    curriculum.publish_prepared_terminals(prepared_curriculum)


def _validate_stateless_outcome(outcome: StatelessGameOutcome) -> None:
    decisions = outcome.candidate_decisions
    if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 0:
        raise ValueError("candidate decision count must be a non-negative integer")
    score = outcome.candidate_score
    if outcome.status == "engine_terminal":
        if score is None or not math.isfinite(score) or score not in (0.0, 0.5, 1.0):
            raise ValueError("engine terminal score must be 0, 0.5, or 1")
    elif outcome.status == "step_limit" and score is not None:
        if not math.isfinite(score) or score not in (0.0, 1.0):
            raise ValueError("step-limit score must identify a winner")
    elif score is not None:
        raise ValueError("non-terminal outcome cannot carry a candidate score")


def cancel_stateless_assignments(
    assignments: Sequence[StatelessAssignedGame],
    *,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
) -> None:
    """Release central leases after a failed parallel collection."""
    for assignment in assignments:
        deck_balance.cancel(assignment.balance.assignment_id)
        curriculum.observe_terminal(
            assignment.curriculum.assignment_id,
            status="infrastructure_error",
        )


def _member_for_assignment(
    members: Mapping[str, PfspMember],
    assignment: CurriculumAssignment,
) -> PfspMember | None:
    if not assignment.member_id:
        return None
    try:
        return members[assignment.member_id]
    except KeyError as error:
        raise RuntimeError(
            "curriculum assignment member disappeared while leased"
        ) from error


def _candidate_score(winner: int, candidate_seat: int) -> float:
    if winner == 2:
        return 0.5
    if winner not in (0, 1):
        raise ValueError("engine terminal result is invalid")
    return 1.0 if winner == candidate_seat else 0.0


def _player_index(observation: Mapping[str, Any]) -> int:
    return int(_field(_field(observation, "current"), "yourIndex", -1))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "NativeArtifactInferenceReport",
    "NativeInferenceRouteKind",
    "StatelessAssignedGame",
    "StatelessCollectionReport",
    "StatelessCollectionResult",
    "StatelessEngineCollector",
    "StatelessGameOutcome",
    "assign_stateless_games",
    "cancel_stateless_assignments",
    "commit_stateless_outcomes",
]
