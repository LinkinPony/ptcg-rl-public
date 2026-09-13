"""Immutable assignment and outcome bindings for native collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.rl.native_route_scheduler import NativeArenaKey
from ptcg_rl.rl.native_trajectory_page import NativeTrajectoryGame
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_curriculum import PfspMember
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity


@dataclass(slots=True)
class NativeLiveGame:
    """One validated fixed-slot game and its mutable rollout counters."""

    slot: int
    assignment: StatelessAssignedGame
    candidate: CanonicalDeck
    opponent: CanonicalDeck
    member: PfspMember | None
    game_id: str
    engine_steps: int = 0
    candidate_decisions: int = 0
    mirror_opponent_decisions: int = 0
    sequence_decisions_by_seat: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    window_draining: bool = False
    sealed_trajectory_seats: set[int] = field(default_factory=set)
    # Immutable route identity memoized by the collector on first resolution.
    cached_route_key: NativeArenaKey | None = None

    @property
    def candidate_seat(self) -> int:
        return int(self.assignment.balance.seat)

    def required_member(self) -> PfspMember:
        if self.member is None:
            raise RuntimeError("native PFSP game has no member")
        return self.member

    def trajectory_games(
        self,
        *,
        mirror_bilateral: bool,
        include_sealed: bool = False,
    ) -> tuple[NativeTrajectoryGame, ...]:
        """Return every current-policy perspective retained for PPO."""
        assignment = self.assignment.curriculum
        candidate = self._trajectory_game(
            seat=self.candidate_seat,
            own_deck=self.candidate,
            opponent_deck=self.opponent,
        )
        trajectories: tuple[NativeTrajectoryGame, ...] = (candidate,)
        if not mirror_bilateral or assignment.lane != "mirror":
            return self._filter_sealed_trajectories(
                trajectories,
                include_sealed=include_sealed,
            )
        trajectories = (
            candidate,
            self._trajectory_game(
                seat=1 - self.candidate_seat,
                own_deck=self.opponent,
                opponent_deck=self.candidate,
            ),
        )
        return self._filter_sealed_trajectories(
            trajectories,
            include_sealed=include_sealed,
        )

    def begin_window_drain(self) -> bool:
        """Latch fragment-boundary draining for this game."""
        if self.window_draining:
            return False
        if self.sealed_trajectory_seats:
            raise RuntimeError("undrained game already has sealed trajectories")
        self.window_draining = True
        return True

    def seal_trajectory_seat(self, seat: int) -> bool:
        """Mark one current-policy perspective as safely bootstrap-closed."""
        normalized_seat = int(seat)
        if normalized_seat not in (0, 1):
            raise ValueError("trajectory seat must be 0 or 1")
        if not self.window_draining:
            raise RuntimeError("trajectory cannot seal before window drain")
        if (
            normalized_seat != self.candidate_seat
            and self.assignment.curriculum.lane != "mirror"
        ):
            raise ValueError("non-mirror game has no opposite-seat trajectory")
        if normalized_seat in self.sealed_trajectory_seats:
            return False
        self.sealed_trajectory_seats.add(normalized_seat)
        return True

    def trajectory_sealed(self, seat: int) -> bool:
        """Return whether one perspective has closed its final fragment."""
        return int(seat) in self.sealed_trajectory_seats

    def is_window_drain_complete(self, *, mirror_bilateral: bool) -> bool:
        """Return whether every trainable perspective is bootstrap-closed."""
        if not self.window_draining:
            return False
        required = {self.candidate_seat}
        if mirror_bilateral and self.assignment.curriculum.lane == "mirror":
            required.add(1 - self.candidate_seat)
        return required.issubset(self.sealed_trajectory_seats)

    def _filter_sealed_trajectories(
        self,
        trajectories: tuple[NativeTrajectoryGame, ...],
        *,
        include_sealed: bool,
    ) -> tuple[NativeTrajectoryGame, ...]:
        if include_sealed or not self.sealed_trajectory_seats:
            return trajectories
        return tuple(
            trajectory
            for trajectory in trajectories
            if trajectory.candidate_seat not in self.sealed_trajectory_seats
        )

    def _trajectory_game(
        self,
        *,
        seat: int,
        own_deck: CanonicalDeck,
        opponent_deck: CanonicalDeck,
    ) -> NativeTrajectoryGame:
        assignment = self.assignment.curriculum
        return NativeTrajectoryGame(
            slot=self.slot,
            game_id=self.game_id,
            candidate_seat=seat,
            own_deck=own_deck,
            opponent_deck=opponent_deck,
            curriculum_generation=assignment.generation,
            assignment_id=assignment.assignment_id,
            opponent_artifact_fingerprint=(assignment.opponent_artifact_fingerprint),
        )

    def outcome(self, *, score: float) -> StatelessGameOutcome:
        return StatelessGameOutcome(
            balance_assignment_id=self.assignment.balance.assignment_id,
            curriculum_assignment_id=self.assignment.curriculum.assignment_id,
            status="engine_terminal",
            candidate_score=score,
            candidate_decisions=self.candidate_decisions,
        )

    def step_limit_outcome(self, *, losing_seat: int) -> StatelessGameOutcome:
        """Adjudicate a selection-limit violation against its acting seat."""
        normalized_seat = int(losing_seat)
        if normalized_seat not in (0, 1):
            raise ValueError("step-limit losing seat must be 0 or 1")
        return StatelessGameOutcome(
            balance_assignment_id=self.assignment.balance.assignment_id,
            curriculum_assignment_id=self.assignment.curriculum.assignment_id,
            status="step_limit",
            candidate_score=float(normalized_seat != self.candidate_seat),
            candidate_decisions=self.candidate_decisions,
        )

    def cancelled_outcome(
        self,
        *,
        status: Literal["infrastructure_error", "step_limit", "window_cutoff"],
        retained_candidate_decisions: int,
    ) -> StatelessGameOutcome:
        """Return unresolved controller evidence for retained candidate rows."""
        if not 0 <= retained_candidate_decisions <= self.candidate_decisions:
            raise ValueError("retained candidate decisions exceed collected rows")
        return StatelessGameOutcome(
            balance_assignment_id=self.assignment.balance.assignment_id,
            curriculum_assignment_id=self.assignment.curriculum.assignment_id,
            status=status,
            candidate_decisions=retained_candidate_decisions,
        )


def build_native_live_games(
    assignments: Sequence[StatelessAssignedGame],
    *,
    identity: StatelessFragmentIdentity,
    active_decks: Mapping[str, CanonicalDeck],
    opponent_decks: Mapping[str, CanonicalDeck],
    members: Mapping[str, PfspMember],
    scripted_bindings: Mapping[str, tuple[str, str]],
    scripted_policy_ids: frozenset[str],
) -> dict[int, NativeLiveGame]:
    """Resolve all exact artifacts before any native slot is mutated."""
    live: dict[int, NativeLiveGame] = {}
    for slot, assignment in enumerate(assignments):
        balance = assignment.balance
        curriculum = assignment.curriculum
        if (
            int(balance.seat) != int(curriculum.candidate_seat)
            or balance.deck_digest != curriculum.candidate_deck_digest
        ):
            raise ValueError("native assignment candidate bindings differ")
        try:
            candidate = active_decks[balance.deck_digest]
            opponent = opponent_decks[curriculum.opponent_deck_digest]
        except KeyError as error:
            raise KeyError("native assignment deck is not registered") from error
        member = None if not curriculum.member_id else members.get(curriculum.member_id)
        _validate_opponent_binding(
            assignment,
            identity=identity,
            active_decks=active_decks,
            member=member,
            scripted_bindings=scripted_bindings,
            scripted_policy_ids=scripted_policy_ids,
        )
        live[slot] = NativeLiveGame(
            slot=slot,
            assignment=assignment,
            candidate=candidate,
            opponent=opponent,
            member=member,
            game_id=(
                f"stateless-{identity.behavior_policy_version}-"
                f"{balance.assignment_cursor}-{slot}"
            ),
        )
    return live


def native_deck_rows(live: Mapping[int, NativeLiveGame]) -> np.ndarray:
    """Return absolute player-order decks for one fixed native slot cohort."""
    ordered = tuple(
        game for _slot, game in sorted(live.items(), key=lambda item: item[0])
    )
    if tuple(live) != tuple(range(len(live))) or any(
        game.slot != slot for slot, game in enumerate(ordered)
    ):
        raise ValueError("native initial deck rows require contiguous slots")
    return native_deck_rows_for_games(ordered)


def native_deck_rows_for_games(
    games: Sequence[NativeLiveGame],
) -> np.ndarray:
    """Return player-order decks aligned with an arbitrary game sequence."""
    rows = np.empty((len(games), 2, 60), dtype=np.int32)
    for row, game in enumerate(games):
        pair = (
            (game.candidate, game.opponent)
            if game.candidate_seat == 0
            else (game.opponent, game.candidate)
        )
        rows[row, 0] = pair[0].card_ids
        rows[row, 1] = pair[1].card_ids
    return rows


def native_candidate_score(result: int, candidate_seat: int) -> float:
    """Map source-engine winner index to the candidate's [0, 1] score."""
    if result == 2:
        return 0.5
    if result in (0, 1):
        return float(result == candidate_seat)
    raise ValueError(f"native terminal has an invalid engine result: {result}")


def _validate_opponent_binding(
    assignment: StatelessAssignedGame,
    *,
    identity: StatelessFragmentIdentity,
    active_decks: Mapping[str, CanonicalDeck],
    member: PfspMember | None,
    scripted_bindings: Mapping[str, tuple[str, str]],
    scripted_policy_ids: frozenset[str],
) -> None:
    curriculum = assignment.curriculum
    if curriculum.lane == "mirror":
        if (
            curriculum.opponent_id != "current_policy_mirror"
            or curriculum.member_id
            or curriculum.opponent_artifact_fingerprint
            != identity.behavior_policy_fingerprint
            or curriculum.opponent_pilot_fingerprint
            != identity.behavior_policy_fingerprint
            or curriculum.opponent_deck_digest not in active_decks
        ):
            raise ValueError("native mirror assignment binding differs")
        return
    if curriculum.lane == "pfsp":
        if member is None:
            raise KeyError("native PFSP assignment member is not registered")
        if (
            curriculum.opponent_id != member.member_id
            or curriculum.member_id != member.member_id
            or curriculum.opponent_deck_digest != member.exact_deck_digest
            or curriculum.opponent_artifact_fingerprint != member.bundle_fingerprint
            or curriculum.opponent_pilot_fingerprint
            != member.pilot_artifact_fingerprint
        ):
            raise ValueError("native PFSP assignment binding differs")
        return
    if curriculum.lane == "scripted":
        try:
            artifact, exact_deck = scripted_bindings[curriculum.opponent_id]
        except KeyError as error:
            raise KeyError("native scripted assignment binding is absent") from error
        if (
            curriculum.member_id
            or curriculum.opponent_artifact_fingerprint != artifact
            or curriculum.opponent_pilot_fingerprint != artifact
            or curriculum.opponent_deck_digest != exact_deck
            or curriculum.opponent_id not in scripted_policy_ids
        ):
            raise ValueError("native scripted assignment binding differs")
        return
    raise ValueError(f"unsupported native curriculum lane: {curriculum.lane}")


__all__ = [
    "NativeLiveGame",
    "build_native_live_games",
    "native_candidate_score",
    "native_deck_rows",
    "native_deck_rows_for_games",
]
