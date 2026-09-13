"""Aggregate assignment planning with lease-local object materialization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.rl.stateless_collection import StatelessAssignedGame
from ptcg_rl.rl.stateless_curriculum import (
    CurriculumLane,
    StatelessCurriculumController,
    stateless_curriculum_state_fingerprint,
)
from ptcg_rl.rl.stateless_deck_balance import StatelessDeckBalanceSampler
from ptcg_rl.rl.stateless_opponent_pool_v2 import (
    BoundOpponentQuotaWindow,
    StatelessOpponentPoolV2,
)

_PLAN_DOMAIN = b"ptcg-rl/stateless-assignment-quota-plan/v1\x00"


class StatelessAssignmentQuotaRow(BaseModel):
    """One aggregate execution row independent of total window games."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: int = Field(ge=0)
    candidate_deck_digest: str
    candidate_seat: Literal[0, 1]
    lane: CurriculumLane
    member_id: str = ""
    frozen_artifact_id: str = ""
    scheduler_cohort_id: str = ""
    opponent_cell_index: int | None = Field(default=None, ge=0)
    game_count: int = Field(gt=0)

    @model_validator(mode="after")
    def coherent_lane(self) -> Self:
        """Bind PFSP-only fields and reject partial frozen routes."""
        values = (self.member_id, self.frozen_artifact_id)
        if self.lane == "pfsp":
            if any(not value for value in values) or self.opponent_cell_index is None:
                raise ValueError("PFSP quota row requires member, artifact, and cell")
            if self.scheduler_cohort_id != self.frozen_artifact_id:
                raise ValueError("PFSP quota row must use its artifact cohort")
        elif any(values) or self.opponent_cell_index is not None:
            raise ValueError("non-PFSP quota row cannot bind a frozen opponent")
        for value in (
            self.candidate_deck_digest,
            self.frozen_artifact_id,
            self.scheduler_cohort_id,
        ):
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("quota row fingerprint is invalid")
        return self


@dataclass(slots=True)
class StatelessQuotaAssignmentPlan:
    """Materialize only rows selected by the native lease scheduler."""

    rows: tuple[StatelessAssignmentQuotaRow, ...]
    revision: str
    opponent_window: BoundOpponentQuotaWindow
    curriculum: StatelessCurriculumController
    deck_balance: StatelessDeckBalanceSampler
    active_decks: Mapping[str, CanonicalDeck]
    opponent_decks: Mapping[str, CanonicalDeck]
    mirror_policy_fingerprint: str
    initial_balance_cursor: int
    initial_curriculum_cursor: int
    curriculum_generation: int
    exposure_cohort_games: int = 0
    candidate_target_shares: Mapping[str, float] = field(default_factory=dict)
    _issued: list[StatelessAssignedGame] = field(default_factory=list)
    _adopted: bool = False

    @property
    def total_games(self) -> int:
        """Return the complete unmaterialized window capacity."""
        return sum(item.game_count for item in self.rows)

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        """Return the canonical frozen artifact inventory."""
        return tuple(
            sorted({item.frozen_artifact_id for item in self.rows if item.member_id})
        )

    @property
    def issued_assignments(self) -> tuple[StatelessAssignedGame, ...]:
        """Return only leases materialized so far."""
        return tuple(self._issued)

    def artifact_for_row(self, row_index: int) -> str | None:
        """Return the row's frozen artifact, if any."""
        value = self.rows[row_index].frozen_artifact_id
        return value or None

    def cohort_for_row(self, row_index: int) -> str | None:
        """Return the row's scheduling cohort without changing its policy."""
        value = self.rows[row_index].scheduler_cohort_id
        return value or None

    def materialize(
        self, row_indices: Sequence[int]
    ) -> tuple[StatelessAssignedGame, ...]:
        """Expand one native recipe without mutating controller state."""
        if not row_indices:
            raise ValueError("quota recipe cannot be empty")
        if self._adopted:
            raise RuntimeError("quota plan was already adopted")
        indexes = tuple(int(index) for index in row_indices)
        if any(index < 0 or index >= len(self.rows) for index in indexes):
            raise ValueError("quota recipe references an absent row")
        selected = tuple(self.rows[index] for index in indexes)
        balance_cursor = self.initial_balance_cursor + len(self._issued)
        curriculum_cursor = self.initial_curriculum_cursor + len(self._issued)
        balances = self.deck_balance.materialize_cells(
            tuple(
                (item.candidate_deck_digest, item.candidate_seat) for item in selected
            ),
            start_cursor=balance_cursor,
        )
        mirror_decks = tuple(self.active_decks.values())
        if not mirror_decks:
            raise ValueError("quota plan active deck roster is empty")
        curriculum_assignments = self.curriculum.materialize_planned_assignments(
            tuple(
                (
                    balance.deck_digest,
                    balance.seat,
                    self.mirror_policy_fingerprint,
                    mirror_decks[
                        balance.assignment_cursor % len(mirror_decks)
                    ].deck_digest,
                )
                for balance in balances
            ),
            pfsp_member_ids=tuple(item.member_id or None for item in selected),
            planned_lanes=tuple(item.lane for item in selected),
            start_cursor=curriculum_cursor,
            generation=self.curriculum_generation,
        )
        assignments = tuple(
            StatelessAssignedGame(balance=balance, curriculum=curriculum_assignment)
            for balance, curriculum_assignment in zip(
                balances,
                curriculum_assignments,
                strict=True,
            )
        )
        if any(
            item.curriculum.opponent_deck_digest not in self.opponent_decks
            for item in assignments
        ):
            raise KeyError("quota opponent deck is not registered")
        for row, assignment in zip(selected, assignments, strict=True):
            if row.opponent_cell_index is not None:
                self.opponent_window.bind(
                    assignment.curriculum.assignment_id,
                    row.opponent_cell_index,
                )
        self._issued.extend(assignments)
        return assignments

    def adopt_issued(
        self,
        assignment_ids: Collection[str] | None = None,
    ) -> tuple[StatelessAssignedGame, ...]:
        """Attach only accepted started reservations to the controllers."""
        if self._adopted:
            raise RuntimeError("quota plan was already adopted")
        issued_by_id = {item.curriculum.assignment_id: item for item in self._issued}
        if len(issued_by_id) != len(self._issued):
            raise RuntimeError("quota plan issued duplicate assignment identities")
        selected_ids = (
            set(issued_by_id)
            if assignment_ids is None
            else {str(value) for value in assignment_ids}
        )
        if not selected_ids <= set(issued_by_id):
            raise ValueError("quota adoption references an unissued reservation")
        assignments = tuple(
            item
            for item in self._issued
            if item.curriculum.assignment_id in selected_ids
        )
        self.opponent_window.curriculum_assignment_cells = {
            assignment_id: cell_index
            for assignment_id, cell_index in (
                self.opponent_window.curriculum_assignment_cells.items()
            )
            if assignment_id in selected_ids
        }
        balances = tuple(item.balance for item in assignments)
        curriculum_assignments = tuple(item.curriculum for item in assignments)
        reservation_count = len(self._issued)
        self.deck_balance.adopt_assignments(
            balances,
            reservation_count=reservation_count,
        )
        try:
            self.curriculum.adopt_assignments(
                curriculum_assignments,
                reservation_count=reservation_count,
            )
        except BaseException:
            for balance in balances:
                self.deck_balance.cancel(balance.assignment_id)
            raise
        self._adopted = True
        return assignments


def plan_stateless_assignment_quotas(
    *,
    games: int,
    opponent_pool: StatelessOpponentPoolV2,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
    active_decks: Mapping[str, CanonicalDeck],
    opponent_decks: Mapping[str, CanonicalDeck],
    mirror_policy_fingerprint: str,
    minimum_cohort_games: int = 0,
) -> StatelessQuotaAssignmentPlan:
    """Create a small quota table without issuing any controller lease."""
    if games <= 0:
        raise ValueError("quota assignment count must be positive")
    if minimum_cohort_games < 0:
        raise ValueError("minimum cohort games cannot be negative")
    opponent_pool.set_deck_target_shares(deck_balance.target_share_mapping)
    candidate_target_shares = opponent_pool.candidate_target_shares()
    cells = deck_balance.preview_cells(
        games,
        target_shares=candidate_target_shares,
    )
    lanes = curriculum.preview_lanes(games)
    portfolio = Counter(zip(lanes, cells, strict=True))
    pfsp_counts: Counter[tuple[str, Literal[0, 1]]] = Counter()
    for (lane, cell), count in portfolio.items():
        if lane == "pfsp":
            pfsp_counts[cell] += count
    opponent_window = opponent_pool.begin_quota_window(
        candidate_cell_counts=pfsp_counts,
    )
    members_by_id = {member.member_id: member for member in curriculum.state.members}
    rows: list[StatelessAssignmentQuotaRow] = []
    global_cohort_weights: Counter[str] = Counter()
    for opponent_cell in opponent_window.plan.cells:
        member_id = opponent_pool.member_id_for_route(opponent_cell.route_id)
        try:
            frozen_artifact_id = members_by_id[member_id].policy_sha256
        except KeyError as exc:
            opponent_pool.abort(opponent_window)
            raise ValueError("quota route member is absent from curriculum") from exc
        rows.append(
            StatelessAssignmentQuotaRow(
                row_index=0,
                candidate_deck_digest=opponent_cell.candidate_deck_digest,
                candidate_seat=opponent_cell.candidate_seat,
                lane="pfsp",
                member_id=member_id,
                frozen_artifact_id=frozen_artifact_id,
                scheduler_cohort_id=frozen_artifact_id,
                opponent_cell_index=opponent_cell.cell_index,
                game_count=opponent_cell.game_count,
            )
        )
        global_cohort_weights[frozen_artifact_id] += opponent_cell.game_count
    non_pfsp_rows = tuple(
        sorted(
            (
                (lane, cell, count)
                for (lane, cell), count in portfolio.items()
                if lane != "pfsp"
            ),
            key=lambda item: (item[0], item[1][0], item[1][1]),
        )
    )
    non_pfsp_total = sum(count for _lane, _cell, count in non_pfsp_rows)
    if global_cohort_weights:
        if opponent_window.role_budget_snapshot is not None:
            try:
                non_pfsp_targets = _balanced_role_cohort_fill(
                    global_cohort_weights,
                    non_pfsp_total=non_pfsp_total,
                    minimum_cohort_games=minimum_cohort_games,
                )
            except ValueError:
                opponent_pool.abort(opponent_window)
                raise
        else:
            minimum_fill = {
                artifact_id: max(
                    minimum_cohort_games - planned_games,
                    0,
                )
                for artifact_id, planned_games in global_cohort_weights.items()
            }
            required_fill = sum(minimum_fill.values())
            if required_fill > non_pfsp_total:
                opponent_pool.abort(opponent_window)
                raise ValueError(
                    "non-PFSP quota cannot fill every artifact exposure cohort"
                )
            extra = dict(
                _apportion_count(
                    non_pfsp_total - required_fill,
                    global_cohort_weights,
                )
            )
            non_pfsp_targets = {
                artifact_id: minimum_fill[artifact_id] + extra.get(artifact_id, 0)
                for artifact_id in sorted(global_cohort_weights)
            }
        row_splits = _apportion_rows_to_cohorts(
            tuple(count for _lane, _cell, count in non_pfsp_rows),
            non_pfsp_targets,
        )
    else:
        row_splits = tuple((("", count),) for _lane, _cell, count in non_pfsp_rows)
    for (lane, cell, _count), splits in zip(
        non_pfsp_rows,
        row_splits,
        strict=True,
    ):
        rows.extend(
            StatelessAssignmentQuotaRow(
                row_index=0,
                candidate_deck_digest=cell[0],
                candidate_seat=cell[1],
                lane=lane,
                scheduler_cohort_id=cohort_id,
                game_count=cohort_count,
            )
            for cohort_id, cohort_count in splits
        )
    # The native deficit scheduler uses row order only to break mathematically
    # exact ties. Hashing the immutable plan identity with row content makes
    # singleton matchup selection deterministic for recovery, but rotates it
    # between windows instead of permanently favoring lexical route IDs.
    ordered_values = sorted(
        rows,
        key=lambda item: (
            hashlib.sha256(
                _PLAN_DOMAIN
                + opponent_window.plan.plan_id.encode()
                + b"\x00"
                + json.dumps(
                    item.model_dump(mode="json", exclude={"row_index"}),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).digest(),
            item.lane,
            item.scheduler_cohort_id,
            item.frozen_artifact_id,
            item.member_id,
            item.candidate_deck_digest,
            item.candidate_seat,
        ),
    )
    ordered = tuple(
        item.model_copy(update={"row_index": index})
        for index, item in enumerate(ordered_values)
    )
    if sum(item.game_count for item in ordered) != games:
        opponent_pool.abort(opponent_window)
        raise RuntimeError("quota assignment plan failed to conserve games")
    content = {
        "opponent_plan_id": opponent_window.plan.plan_id,
        "curriculum_state_fingerprint": stateless_curriculum_state_fingerprint(
            curriculum.state
        ),
        "deck_balance_state_fingerprint": hashlib.sha256(
            json.dumps(
                deck_balance.state.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "mirror_policy_fingerprint": mirror_policy_fingerprint,
        "exposure_cohort_games": minimum_cohort_games,
        "rows": [item.model_dump(mode="json") for item in ordered],
    }
    revision = hashlib.sha256(
        _PLAN_DOMAIN
        + json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return StatelessQuotaAssignmentPlan(
        rows=ordered,
        revision=revision,
        opponent_window=opponent_window,
        curriculum=curriculum,
        deck_balance=deck_balance,
        active_decks=active_decks,
        opponent_decks=opponent_decks,
        mirror_policy_fingerprint=mirror_policy_fingerprint,
        initial_balance_cursor=deck_balance.state.assignment_cursor,
        initial_curriculum_cursor=curriculum.state.assignment_cursor,
        curriculum_generation=curriculum.state.generation,
        exposure_cohort_games=minimum_cohort_games,
        candidate_target_shares=candidate_target_shares,
    )


def _apportion_count(
    total: int,
    weights: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    """Split an integer row exactly by deterministic largest remainder."""
    positive = tuple(
        sorted((key, value) for key, value in weights.items() if value > 0)
    )
    weight_total = sum(value for _key, value in positive)
    if total <= 0 or weight_total <= 0:
        return ()
    counts = {key: total * weight // weight_total for key, weight in positive}
    remainder = total - sum(counts.values())
    order = sorted(
        positive,
        key=lambda item: (-(total * item[1] % weight_total), item[0]),
    )
    for key, _weight in order[:remainder]:
        counts[key] += 1
    return tuple((key, counts[key]) for key, _weight in positive if counts[key] > 0)


def _balanced_role_cohort_fill(
    pfsp_counts: Mapping[str, int],
    *,
    non_pfsp_total: int,
    minimum_cohort_games: int,
) -> dict[str, int]:
    """Fill V4 artifact cohorts to equal total execution capacity.

    Native V3 leases deliberately contain one frozen artifact, and the first
    worker wave gives every active artifact one equal-sized shard.  If current-
    policy filler follows PFSP mass, every cohort has the same PFSP density and
    that wave samples roles in proportion to artifact count instead of the V4
    role budget.  Equal cohort totals retain homogeneous full-arena leases while
    making each cohort's PFSP density carry its planned role share.

    A large PFSP cohort can make exact equality infeasible.  Deterministic
    water-filling then returns the closest capacity-balanced totals without
    moving or changing any PFSP game.
    """
    if non_pfsp_total < 0 or minimum_cohort_games < 0:
        raise ValueError("role cohort fill inputs cannot be negative")
    if not pfsp_counts or any(value <= 0 for value in pfsp_counts.values()):
        raise ValueError("role cohort PFSP counts must be positive")
    totals = {
        artifact_id: max(planned_games, minimum_cohort_games)
        for artifact_id, planned_games in sorted(pfsp_counts.items())
    }
    required_fill = sum(
        totals[artifact_id] - planned_games
        for artifact_id, planned_games in pfsp_counts.items()
    )
    if required_fill > non_pfsp_total:
        raise ValueError("non-PFSP quota cannot fill every artifact exposure cohort")
    for _index in range(non_pfsp_total - required_fill):
        selected = min(
            totals,
            key=lambda artifact_id: (totals[artifact_id], artifact_id),
        )
        totals[selected] += 1
    return {
        artifact_id: totals[artifact_id] - pfsp_counts[artifact_id]
        for artifact_id in sorted(pfsp_counts)
    }


def _apportion_rows_to_cohorts(
    row_counts: Sequence[int],
    targets: Mapping[str, int],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Make every row prefix track exact global cohort totals."""
    cohort_ids = tuple(sorted(key for key, value in targets.items() if value > 0))
    total = sum(row_counts)
    if total != sum(targets.values()):
        raise ValueError("cohort targets do not conserve non-PFSP rows")
    assigned: Counter[str] = Counter()
    issued = 0
    result: list[tuple[tuple[str, int], ...]] = []
    for row_count in row_counts:
        local: Counter[str] = Counter()
        for _index in range(row_count):
            available = tuple(
                cohort_id
                for cohort_id in cohort_ids
                if assigned[cohort_id] < targets[cohort_id]
            )
            chosen = min(
                available,
                key=lambda cohort_id: (
                    -(targets[cohort_id] * (issued + 1) - assigned[cohort_id] * total),
                    cohort_id,
                ),
            )
            assigned[chosen] += 1
            local[chosen] += 1
            issued += 1
        result.append(
            tuple(
                (cohort_id, local[cohort_id])
                for cohort_id in cohort_ids
                if local[cohort_id] > 0
            )
        )
    if dict(assigned) != {cohort_id: targets[cohort_id] for cohort_id in cohort_ids}:
        raise RuntimeError("cohort row apportionment did not reach its targets")
    return tuple(result)


__all__ = [
    "StatelessAssignmentQuotaRow",
    "StatelessQuotaAssignmentPlan",
    "plan_stateless_assignment_quotas",
]
