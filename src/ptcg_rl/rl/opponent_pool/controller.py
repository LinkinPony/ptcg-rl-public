"""Small in-process coordinator over the pure plan, commit, and transition API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ptcg_rl.rl.opponent_pool.commit import (
    OpponentOutcome,
    commit_quota_window,
    commit_window,
)
from ptcg_rl.rl.opponent_pool.models import CandidateSeat, QuotaStratum
from ptcg_rl.rl.opponent_pool.planner import (
    ArtifactCapacity,
    PlannerPolicy,
    QuotaWindowPlan,
    WindowPlan,
    plan_adaptive_quota_window_for_candidate_counts,
    plan_quota_window_for_candidate_counts,
    plan_window,
    plan_window_for_candidate_cells,
)
from ptcg_rl.rl.opponent_pool.state import (
    LeagueState,
    RevisionTransitionReceipt,
    RevisionTransitionRequest,
    transition_revision,
)


class OpponentPoolControllerError(RuntimeError):
    """Raised when caller ordering violates the one-pending-window protocol."""


class OpponentPoolController:
    """Own one immutable state and at most one pending executable plan."""

    def __init__(self, state: LeagueState) -> None:
        self._state = state
        self._pending_plan: WindowPlan | QuotaWindowPlan | None = None

    @property
    def state(self) -> LeagueState:
        """Return the current committed state."""
        return self._state

    @property
    def pending_plan(self) -> WindowPlan | QuotaWindowPlan | None:
        """Return the uncommitted plan, if one exists."""
        return self._pending_plan

    def begin_window(
        self,
        *,
        capacity: ArtifactCapacity,
        policy: PlannerPolicy,
        total_games: int,
        artifact_priorities: Mapping[str, float] | None = None,
        matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
    ) -> WindowPlan:
        """Plan one window and reserve it until commit or abort."""
        if self._pending_plan is not None:
            raise OpponentPoolControllerError("a window plan is already pending")
        plan = plan_window(
            self._state,
            capacity=capacity,
            policy=policy,
            total_games=total_games,
            artifact_priorities=artifact_priorities,
            matchup_priorities=matchup_priorities,
        )
        self._pending_plan = plan
        return plan

    def commit(
        self,
        plan: WindowPlan,
        outcomes: Sequence[OpponentOutcome],
    ) -> LeagueState:
        """Commit exactly the reserved plan and clear the reservation."""
        if self._pending_plan is None:
            raise OpponentPoolControllerError("no window plan is pending")
        if self._pending_plan.plan_id != plan.plan_id:
            raise OpponentPoolControllerError("plan is not the pending window")
        successor = commit_window(self._state, plan, outcomes)
        self._state = successor
        self._pending_plan = None
        return successor

    def begin_window_for_candidate_cells(
        self,
        *,
        capacity: ArtifactCapacity,
        policy: PlannerPolicy,
        candidate_cells: Sequence[tuple[str, CandidateSeat]],
        stratum_decision_coverage: Mapping[QuotaStratum, int],
        mandatory_artifact_ids: frozenset[str] = frozenset(),
        artifact_priorities: Mapping[str, float] | None = None,
        matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
    ) -> WindowPlan:
        """Reserve one plan whose candidate cells came from deck balancing."""
        if self._pending_plan is not None:
            raise OpponentPoolControllerError("a window plan is already pending")
        plan = plan_window_for_candidate_cells(
            self._state,
            capacity=capacity,
            policy=policy,
            candidate_cells=candidate_cells,
            stratum_decision_coverage=stratum_decision_coverage,
            mandatory_artifact_ids=mandatory_artifact_ids,
            artifact_priorities=artifact_priorities,
            matchup_priorities=matchup_priorities,
        )
        self._pending_plan = plan
        return plan

    def begin_quota_window_for_candidate_counts(
        self,
        *,
        capacity: ArtifactCapacity,
        policy: PlannerPolicy,
        candidate_cell_counts: Mapping[tuple[str, CandidateSeat], int],
        stratum_decision_coverage: Mapping[QuotaStratum, int],
        mandatory_artifact_ids: frozenset[str] = frozenset(),
        artifact_priorities: Mapping[str, float] | None = None,
        matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
    ) -> QuotaWindowPlan:
        """Reserve one aggregate plan without per-game controller objects."""
        if self._pending_plan is not None:
            raise OpponentPoolControllerError("a window plan is already pending")
        plan = plan_quota_window_for_candidate_counts(
            self._state,
            capacity=capacity,
            policy=policy,
            candidate_cell_counts=candidate_cell_counts,
            stratum_decision_coverage=stratum_decision_coverage,
            mandatory_artifact_ids=mandatory_artifact_ids,
            artifact_priorities=artifact_priorities,
            matchup_priorities=matchup_priorities,
        )
        self._pending_plan = plan
        return plan

    def begin_adaptive_quota_window_for_candidate_counts(
        self,
        *,
        capacity: ArtifactCapacity,
        candidate_cell_counts: Mapping[tuple[str, CandidateSeat], int],
        matchup_targets: Mapping[tuple[str, str, int], float],
        matchup_decision_mass: Mapping[tuple[str, str, int], float],
        expected_decisions_per_game: Mapping[tuple[str, str, int], float],
        minimum_artifact_games: int,
        matchup_game_batch_size: int = 1,
        matchup_last_learning_windows: (
            Mapping[tuple[str, str, int], int | None] | None
        ) = None,
        matchup_coverage_windows: int | None = None,
    ) -> QuotaWindowPlan:
        """Reserve one joint adaptive decision-mass plan."""
        if self._pending_plan is not None:
            raise OpponentPoolControllerError("a window plan is already pending")
        plan = plan_adaptive_quota_window_for_candidate_counts(
            self._state,
            capacity=capacity,
            candidate_cell_counts=candidate_cell_counts,
            matchup_targets=matchup_targets,
            matchup_decision_mass=matchup_decision_mass,
            expected_decisions_per_game=expected_decisions_per_game,
            minimum_artifact_games=minimum_artifact_games,
            matchup_game_batch_size=matchup_game_batch_size,
            matchup_last_learning_windows=matchup_last_learning_windows,
            matchup_coverage_windows=matchup_coverage_windows,
        )
        self._pending_plan = plan
        return plan

    def commit_quota(
        self,
        plan: QuotaWindowPlan,
        outcomes: Sequence[OpponentOutcome],
    ) -> LeagueState:
        """Commit observations issued from exactly one aggregate plan."""
        if self._pending_plan is None:
            raise OpponentPoolControllerError("no window plan is pending")
        if self._pending_plan.plan_id != plan.plan_id:
            raise OpponentPoolControllerError("plan is not the pending window")
        successor = commit_quota_window(self._state, plan, outcomes)
        self._state = successor
        self._pending_plan = None
        return successor

    def abort(self) -> None:
        """Discard a pending plan without changing committed state."""
        self._pending_plan = None

    def transition(
        self,
        request: RevisionTransitionRequest,
    ) -> RevisionTransitionReceipt:
        """Change revision only when no execution window is pending."""
        if self._pending_plan is not None:
            raise OpponentPoolControllerError(
                "cannot transition while a window plan is pending"
            )
        result = transition_revision(self._state, request)
        self._state = result.state
        return result.receipt
