"""Normalized committed state and explicit opponent-pool transitions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool._identity import (
    Sha256,
    cached_fingerprint,
    canonical_fingerprint,
)
from ptcg_rl.rl.opponent_pool.models import ActiveMatchup, PoolRevision


class MatchupStat(BaseModel):
    """The sole committed fact row for one candidate/route/seat cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matchup: ActiveMatchup
    executed_games: int = Field(default=0, ge=0)
    learning_exposures: int = Field(default=0, ge=0)
    trainable_decisions: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    last_executed_window: int | None = Field(default=None, ge=0)
    last_learning_exposure_window: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def counters_are_consistent(self) -> Self:
        """Keep execution, learning, and terminal statistics coherent."""
        terminal_games = self.wins + self.draws + self.losses
        if terminal_games > self.executed_games:
            raise ValueError("terminal games cannot exceed executed games")
        if self.learning_exposures > self.executed_games:
            raise ValueError("learning exposures cannot exceed executed games")
        if self.trainable_decisions < self.learning_exposures:
            raise ValueError("each learning exposure needs a trainable decision")
        if (self.learning_exposures == 0) != (self.trainable_decisions == 0):
            raise ValueError("learning exposure and decision totals must agree")
        if (self.executed_games == 0) != (self.last_executed_window is None):
            raise ValueError("last executed window must match execution count")
        if (self.learning_exposures == 0) != (
            self.last_learning_exposure_window is None
        ):
            raise ValueError("last learning window must match exposure count")
        return self


class LeagueState(BaseModel):
    """Compact learner-side state paired with the active checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    revision: PoolRevision
    generation: int = Field(ge=0)
    next_window_sequence: int = Field(ge=0)
    matchup_stats: tuple[MatchupStat, ...] = ()
    last_committed_plan_id: Sha256 | None = None
    last_transition_id: Sha256 | None = None

    @classmethod
    def initial(cls, revision: PoolRevision) -> Self:
        """Create an empty committed state for a first active revision."""
        return cls(
            revision=revision,
            generation=0,
            next_window_sequence=0,
        )

    @model_validator(mode="after")
    def normalized_stats_are_valid(self) -> Self:
        """Require canonical rows drawn only from the active revision."""
        stat_keys = tuple(stat.matchup.key for stat in self.matchup_stats)
        if stat_keys != tuple(sorted(stat_keys)):
            raise ValueError("matchup stats must be sorted by cell key")
        if len(set(stat_keys)) != len(stat_keys):
            raise ValueError("matchup stats must have unique cell keys")
        active_keys = {matchup.key for matchup in self.revision.active_matchups}
        if not set(stat_keys).issubset(active_keys):
            raise ValueError("matchup stats reference inactive cells")
        for stat in self.matchup_stats:
            windows = (
                stat.last_executed_window,
                stat.last_learning_exposure_window,
            )
            if any(
                window is not None and window >= self.next_window_sequence
                for window in windows
            ):
                raise ValueError("committed stat refers to an uncommitted window")
        if (self.next_window_sequence == 0) != (self.last_committed_plan_id is None):
            raise ValueError("last plan identity must match committed windows")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the immutable state identity used by stale-plan gates."""
        return cached_fingerprint(
            "league-state",
            self,
            lambda: league_state_fingerprint(self),
        )

    def matchup_stat(self, matchup: ActiveMatchup) -> MatchupStat | None:
        """Look up one normalized fact row, returning implicit zero as None."""
        return next(
            (stat for stat in self.matchup_stats if stat.matchup.key == matchup.key),
            None,
        )

    def artifact_learning_exposures(self, artifact_id: str) -> int:
        """Aggregate learning exposure on demand from normalized cells."""
        route_ids = {
            route.route_id
            for route in self.revision.routes
            if route.artifact_id == artifact_id
        }
        return sum(
            stat.learning_exposures
            for stat in self.matchup_stats
            if stat.matchup.route_id in route_ids
        )

    def artifact_executed_games(self, artifact_id: str) -> int:
        """Aggregate executed games on demand from normalized cells."""
        route_ids = {
            route.route_id
            for route in self.revision.routes
            if route.artifact_id == artifact_id
        }
        return sum(
            stat.executed_games
            for stat in self.matchup_stats
            if stat.matchup.route_id in route_ids
        )

    def artifact_trainable_decisions(self, artifact_id: str) -> int:
        """Aggregate accepted trainable decisions for one active artifact."""
        route_ids = {
            route.route_id
            for route in self.revision.routes
            if route.artifact_id == artifact_id
        }
        return sum(
            stat.trainable_decisions
            for stat in self.matchup_stats
            if stat.matchup.route_id in route_ids
        )

    def artifact_last_learning_window(self, artifact_id: str) -> int | None:
        """Return the most recent learning window across an artifact's cells."""
        route_ids = {
            route.route_id
            for route in self.revision.routes
            if route.artifact_id == artifact_id
        }
        windows = [
            stat.last_learning_exposure_window
            for stat in self.matchup_stats
            if stat.matchup.route_id in route_ids
            and stat.last_learning_exposure_window is not None
        ]
        return max(windows, default=None)


def league_state_fingerprint(state: LeagueState) -> str:
    """Fingerprint one complete committed state."""
    return canonical_fingerprint("league-state", state.model_dump(mode="json"))


class RevisionTransitionRequest(BaseModel):
    """Explicit compare-and-swap request for changing the active revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predecessor_state_fingerprint: Sha256
    predecessor_revision_fingerprint: Sha256
    successor_revision: PoolRevision
    admitted_artifact_ids: tuple[Sha256, ...]
    retired_artifact_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def declared_sets_are_canonical(self) -> Self:
        """Keep declared topology changes canonical and disjoint."""
        for name, values in (
            ("admitted_artifact_ids", self.admitted_artifact_ids),
            ("retired_artifact_ids", self.retired_artifact_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.admitted_artifact_ids) & set(self.retired_artifact_ids):
            raise ValueError("admitted and retired artifact sets must be disjoint")
        return self


class RevisionTransitionReceipt(BaseModel):
    """Bounded audit receipt suitable for immutable external archiving."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: Sha256
    predecessor_state_fingerprint: Sha256
    predecessor_revision_fingerprint: Sha256
    successor_revision_fingerprint: Sha256
    admitted_artifact_ids: tuple[Sha256, ...]
    retired_artifact_ids: tuple[Sha256, ...]
    preserved_artifact_ids: tuple[Sha256, ...]
    predecessor_generation: int = Field(ge=0)
    successor_generation: int = Field(ge=1)

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        """Reject receipt aliases or inconsistent generations."""
        if self.successor_generation != self.predecessor_generation + 1:
            raise ValueError("a transition must advance one generation")
        if self.transition_id != revision_transition_fingerprint(self):
            raise ValueError("transition_id does not match receipt content")
        return self


def revision_transition_fingerprint(receipt: RevisionTransitionReceipt) -> str:
    """Return the immutable transition receipt identity."""
    return canonical_fingerprint(
        "revision-transition",
        receipt.model_dump(mode="json", exclude={"transition_id"}),
    )


class RevisionTransitionResult(BaseModel):
    """The successor state and its externally archivable receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: LeagueState
    receipt: RevisionTransitionReceipt


def transition_revision(
    state: LeagueState,
    request: RevisionTransitionRequest,
) -> RevisionTransitionResult:
    """Atomically replace the active revision after exact stale/diff checks."""
    if request.predecessor_state_fingerprint != state.fingerprint:
        raise ValueError("transition predecessor state is stale")
    if request.predecessor_revision_fingerprint != state.revision.fingerprint:
        raise ValueError("transition predecessor revision is stale")
    successor = request.successor_revision
    if successor.revision_sequence != state.revision.revision_sequence + 1:
        raise ValueError("successor revision sequence must advance exactly once")

    predecessor_ids = {artifact.artifact_id for artifact in state.revision.artifacts}
    successor_ids = {artifact.artifact_id for artifact in successor.artifacts}
    admitted = tuple(sorted(successor_ids - predecessor_ids))
    retired = tuple(sorted(predecessor_ids - successor_ids))
    preserved = tuple(sorted(predecessor_ids & successor_ids))
    if admitted != request.admitted_artifact_ids:
        raise ValueError("declared admitted artifacts do not match revision diff")
    if retired != request.retired_artifact_ids:
        raise ValueError("declared retired artifacts do not match revision diff")
    predecessor_entries = {entry.artifact_id: entry for entry in state.revision.entries}
    successor_entries = {entry.artifact_id: entry for entry in successor.entries}
    if any(
        successor_entries[artifact_id].admission_generation
        != predecessor_entries[artifact_id].admission_generation
        for artifact_id in preserved
    ):
        raise ValueError("preserved artifact changed its pool admission generation")
    if any(
        successor_entries[artifact_id].admission_generation != state.generation + 1
        for artifact_id in admitted
    ):
        raise ValueError("admitted artifact uses a non-pool generation")

    receipt_content = {
        "predecessor_state_fingerprint": state.fingerprint,
        "predecessor_revision_fingerprint": state.revision.fingerprint,
        "successor_revision_fingerprint": successor.fingerprint,
        "admitted_artifact_ids": admitted,
        "retired_artifact_ids": retired,
        "preserved_artifact_ids": preserved,
        "predecessor_generation": state.generation,
        "successor_generation": state.generation + 1,
    }
    receipt = RevisionTransitionReceipt(
        transition_id=canonical_fingerprint(
            "revision-transition",
            receipt_content,
        ),
        predecessor_state_fingerprint=state.fingerprint,
        predecessor_revision_fingerprint=state.revision.fingerprint,
        successor_revision_fingerprint=successor.fingerprint,
        admitted_artifact_ids=admitted,
        retired_artifact_ids=retired,
        preserved_artifact_ids=preserved,
        predecessor_generation=state.generation,
        successor_generation=state.generation + 1,
    )
    successor_keys = {matchup.key for matchup in successor.active_matchups}
    preserved_route_ids = {
        route.route_id
        for route in successor.routes
        if route.artifact_id in set(preserved)
    }
    retained_stats = tuple(
        stat
        for stat in state.matchup_stats
        if stat.matchup.key in successor_keys
        and stat.matchup.route_id in preserved_route_ids
    )
    new_state = LeagueState(
        revision=successor,
        generation=state.generation + 1,
        next_window_sequence=state.next_window_sequence,
        matchup_stats=retained_stats,
        last_committed_plan_id=state.last_committed_plan_id,
        last_transition_id=receipt.transition_id,
    )
    return RevisionTransitionResult(state=new_state, receipt=receipt)
