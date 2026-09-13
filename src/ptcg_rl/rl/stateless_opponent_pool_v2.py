"""Formal bridge from stateless resources to Historical Opponent Pool V2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ptcg_rl.decks.identity import CanonicalDeck
from ptcg_rl.rl.opponent_pool import (
    ActiveMatchup,
    ArtifactCapacity,
    LeagueState,
    OpponentArtifact,
    OpponentOutcome,
    OpponentPoolController,
    OpponentRoute,
    PlannerPolicy,
    PoolEntry,
    PoolRevision,
    QuotaStratum,
    QuotaWindowPlan,
    RevisionTransitionRequest,
    StratumPolicy,
    WindowPlan,
)
from ptcg_rl.rl.opponent_pool._identity import canonical_fingerprint
from ptcg_rl.rl.opponent_pool.adaptive import (
    AdaptiveAllocationSnapshot,
    AdaptiveCellScore,
    AdaptiveEvidenceAllocationConfig,
    AdaptiveEvidenceIndex,
    AdaptiveMatchupEvidence,
    AdaptiveMatchupIdentity,
    PortfolioName,
    bounded_candidate_shares,
    cvar,
    normalized_matchup_targets,
    observe_evidence,
    portfolio_mass,
)
from ptcg_rl.rl.opponent_pool.adaptive_report import (
    AdaptiveOpponentAllocationReport,
    build_adaptive_allocation_report,
)
from ptcg_rl.rl.opponent_pool.role_budget import (
    ROLE_ORDER,
    ROLE_TARGET_SHARES,
    RoleBudgetAllocationSnapshot,
    RoleBudgetName,
    hierarchical_role_targets,
    role_for_stratum,
    role_priority,
)
from ptcg_rl.rl.opponent_pool.role_budget_report import (
    RoleBudgetOpponentAllocationReport,
    build_role_budget_allocation_report,
)
from ptcg_rl.rl.stateless_collection import (
    StatelessAssignedGame,
    StatelessGameOutcome,
)
from ptcg_rl.rl.stateless_curriculum import (
    PfspMember,
    StatelessCurriculumController,
)
from ptcg_rl.rl.stateless_deck_balance import StatelessDeckBalanceSampler
from ptcg_rl.rl.stateless_opponent_pool_state import (
    OpponentPoolCheckpointState,
    StatelessOpponentPoolAdaptiveState,
    StatelessOpponentPoolLineageState,
    StratumDecisionTotal,
)
from ptcg_rl.rl.stateless_training_config import StatelessOpponentPoolV2Config

STATELESS_OPPONENT_POOL_V2_COMPATIBILITY_FINGERPRINT = canonical_fingerprint(
    "stateless-opponent-pool-v2-compatibility",
    {
        "native_shard_protocol": 2,
        "past_self_runtime": "wire_bf16",
        "historical_runtime": "legacy_resident",
        "route_adapter": "stateless-exact-deck-v1",
    },
)


@dataclass(frozen=True)
class BoundOpponentPoolWindow:
    """One immutable V2 plan bound to issued curriculum assignment IDs."""

    plan: WindowPlan
    curriculum_assignment_ids: tuple[str, ...]
    lineage_state_fingerprint: str | None = None


@dataclass(slots=True)
class BoundOpponentQuotaWindow:
    """One aggregate plan plus only the PFSP leases actually materialized."""

    plan: QuotaWindowPlan
    lineage_state_fingerprint: str | None = None
    curriculum_assignment_cells: dict[str, int] = field(default_factory=dict)
    allocation_snapshot: AdaptiveAllocationSnapshot | None = None
    role_budget_snapshot: RoleBudgetAllocationSnapshot | None = None
    portfolio_by_matchup: dict[tuple[str, str, int], PortfolioName] = field(
        default_factory=dict
    )

    def bind(self, assignment_id: str, cell_index: int) -> None:
        """Bind one issued curriculum lease to its immutable quota cell."""
        previous = self.curriculum_assignment_cells.setdefault(
            assignment_id,
            cell_index,
        )
        if previous != cell_index:
            raise ValueError("curriculum assignment crossed quota cells")


@dataclass(frozen=True)
class PlannedStatelessAssignments:
    """Full three-lane assignment pool and its PFSP V2 binding."""

    assignments: tuple[StatelessAssignedGame, ...]
    opponent_pool_window: BoundOpponentPoolWindow


@dataclass(frozen=True)
class _ArtifactGroup:
    policy_sha256: str
    members: tuple[PfspMember, ...]
    source_policy_version: int

    @property
    def source(
        self,
    ) -> Literal["historical_anchor", "fixed_stateless_anchor", "past_self"]:
        return self.members[0].source


@dataclass(frozen=True)
class _PreparedAdaptiveAllocation:
    """Ephemeral scoring products for one immutable pending window."""

    revision_fingerprint: str
    state_fingerprint: str
    snapshot: AdaptiveAllocationSnapshot
    scores: tuple[AdaptiveCellScore, ...]
    matchup_targets: dict[tuple[str, str, int], float]
    matchup_decision_mass: dict[tuple[str, str, int], float]
    expected_decisions_per_game: dict[tuple[str, str, int], float]
    portfolio_by_matchup: dict[tuple[str, str, int], PortfolioName]


@dataclass(frozen=True)
class _PreparedRoleBudgetAllocation:
    """Ephemeral posterior and hierarchical targets for one role-budget window."""

    revision_fingerprint: str
    state_fingerprint: str
    snapshot: RoleBudgetAllocationSnapshot
    scores: tuple[AdaptiveCellScore, ...]
    matchup_targets: dict[tuple[str, str, int], float]
    matchup_decision_mass: dict[tuple[str, str, int], float]
    expected_decisions_per_game: dict[tuple[str, str, int], float]
    matchup_last_learning_windows: dict[tuple[str, str, int], int | None]


def _source_policy_version(member: PfspMember) -> int:
    if member.pair is not None:
        return member.pair.version
    suffix = member.snapshot_id.rsplit("v", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _group_members(members: Sequence[PfspMember]) -> tuple[_ArtifactGroup, ...]:
    grouped: dict[str, list[PfspMember]] = defaultdict(list)
    for member in members:
        if member.status == "active":
            grouped[member.policy_sha256].append(member)
    return tuple(
        _ArtifactGroup(
            policy_sha256=policy_sha256,
            members=tuple(sorted(values, key=lambda item: item.member_id)),
            source_policy_version=max(_source_policy_version(item) for item in values),
        )
        for policy_sha256, values in sorted(grouped.items())
    )


def _evenly_spaced(
    values: Sequence[_ArtifactGroup],
    count: int,
) -> tuple[_ArtifactGroup, ...]:
    if count <= 0 or count > len(values):
        raise ValueError("age-diverse opponent-pool selection is infeasible")
    if count == 1:
        return (values[-1],)
    indexes = tuple(
        round(index * (len(values) - 1) / (count - 1)) for index in range(count)
    )
    if len(set(indexes)) != count:
        raise RuntimeError("age-diverse opponent-pool indexes are not unique")
    return tuple(values[index] for index in indexes)


def _artifact(group: _ArtifactGroup) -> OpponentArtifact:
    return OpponentArtifact.from_content(
        runtime_kind=(
            "legacy_resident" if group.source == "historical_anchor" else "wire_bf16"
        ),
        source_fingerprint=group.policy_sha256,
        execution_fingerprint=canonical_fingerprint(
            "stateless-opponent-execution",
            {
                "policy_sha256": group.policy_sha256,
                "members": [
                    {
                        "member_id": member.member_id,
                        "pilot": member.pilot_artifact_fingerprint,
                        "registry": member.exact_registry_fingerprint,
                        "input": member.input_contract_fingerprint,
                    }
                    for member in group.members
                ],
            },
        ),
        compatibility_fingerprint=(
            STATELESS_OPPONENT_POOL_V2_COMPATIBILITY_FINGERPRINT
        ),
        source_policy_version=group.source_policy_version,
    )


def _route(artifact: OpponentArtifact, member: PfspMember) -> OpponentRoute:
    return OpponentRoute.from_content(
        artifact_id=artifact.artifact_id,
        exact_deck_digest=member.exact_deck_digest,
        route_contract_fingerprint=canonical_fingerprint(
            "stateless-opponent-route-contract",
            {
                "member_id": member.member_id,
                "bundle": member.bundle_fingerprint,
                "pilot": member.pilot_artifact_fingerprint,
                "input": member.input_contract_fingerprint,
                "registry": member.exact_registry_fingerprint,
            },
        ),
    )


def _lineage_initial_state(
    state: OpponentPoolCheckpointState,
) -> StatelessOpponentPoolLineageState:
    """Narrow a validated behavior-two checkpoint for static type checking."""
    if not isinstance(state, StatelessOpponentPoolLineageState):
        raise TypeError("lineage opponent pool requires a version-two state")
    return state


def _migrate_adaptive_state(
    state: StatelessOpponentPoolLineageState,
) -> StatelessOpponentPoolAdaptiveState:
    """Convert settled V2 counters into conservative V3 sufficient statistics."""
    revision = state.league_state.revision
    routes = {item.route_id: item for item in revision.routes}
    route_artifacts = {item.route_id: item.artifact_id for item in revision.routes}
    evidence: list[AdaptiveMatchupEvidence] = []
    for stat in state.league_state.matchup_stats:
        route = routes[stat.matchup.route_id]
        terminal_games = stat.wins + stat.draws + stat.losses
        score_sum = float(stat.wins) + 0.5 * float(stat.draws)
        evidence.append(
            AdaptiveMatchupEvidence(
                identity=AdaptiveMatchupIdentity(
                    candidate_deck_digest=stat.matchup.candidate_deck_digest,
                    artifact_id=route_artifacts[stat.matchup.route_id],
                    route_id=stat.matchup.route_id,
                    opponent_deck_digest=route.exact_deck_digest,
                    candidate_seat=stat.matchup.candidate_seat,
                ),
                fast_score_sum=score_sum,
                fast_score_weight=float(terminal_games),
                slow_score_sum=score_sum,
                slow_score_weight=float(terminal_games),
                decision_sum=float(stat.trainable_decisions),
                exposure_weight=float(stat.learning_exposures),
                total_terminal_games=terminal_games,
                total_trainable_decisions=stat.trainable_decisions,
                last_update_window=state.league_state.next_window_sequence,
                last_learning_window=stat.last_learning_exposure_window,
            )
        )
    return StatelessOpponentPoolAdaptiveState(
        league_state=state.league_state,
        founder_policy_sha256=state.founder_policy_sha256,
        evidence=tuple(sorted(evidence, key=lambda item: item.identity.evidence_key)),
        allocation_decision_clock=sum(
            item.trainable_decisions for item in state.league_state.matchup_stats
        ),
    )


def _migrate_role_budget_state(
    state: StatelessOpponentPoolLineageState | StatelessOpponentPoolAdaptiveState,
    *,
    target_schema_version: Literal[4, 5] = 4,
) -> StatelessOpponentPoolAdaptiveState:
    """Preserve exact evidence across an explicit role-budget semantic epoch."""
    adaptive = (
        _migrate_adaptive_state(state)
        if isinstance(state, StatelessOpponentPoolLineageState)
        else state
    )
    if adaptive.schema_version == target_schema_version:
        return adaptive
    if adaptive.schema_version > target_schema_version:
        raise ValueError("role-budget opponent-pool state cannot be downgraded")
    payload = adaptive.model_dump(mode="json")
    payload.update(
        {
            "schema_version": target_schema_version,
            "last_target_weights": [],
            "candidate_target_shares": {},
            "portfolio_target_mass": {},
            "portfolio_decisions": {},
            "role_target_mass": {},
            # Prior allocation semantics are not attribution-compatible with
            # the target epoch. Preserve posterior evidence, but restart role
            # decision diagnostics at the declared behavior boundary.
            "role_decisions": dict.fromkeys(ROLE_ORDER, 0),
            "last_target_fingerprint": None,
        }
    )
    return StatelessOpponentPoolAdaptiveState.model_validate(payload)


class StatelessOpponentPoolV2:
    """Own the settled V2 revision, plans, and normalized exposure state."""

    def __init__(
        self,
        config: StatelessOpponentPoolV2Config,
        *,
        active_deck_digests: Sequence[str],
        curriculum: StatelessCurriculumController,
        initial_state: OpponentPoolCheckpointState | None = None,
        founder_policy_sha256: str | None = None,
        deck_target_shares: Mapping[str, float] | None = None,
        synchronize_revision: bool = True,
    ) -> None:
        if not config.enabled:
            raise ValueError("opponent-pool V2 runtime requires an enabled config")
        self.config = config
        self.active_deck_digests = tuple(sorted(active_deck_digests))
        self.curriculum = curriculum
        self.deck_target_shares = (
            {
                digest: 1.0 / float(len(self.active_deck_digests))
                for digest in self.active_deck_digests
            }
            if deck_target_shares is None
            else dict(deck_target_shares)
        )
        if set(self.deck_target_shares) != set(self.active_deck_digests):
            raise ValueError("opponent-pool deck targets differ from active decks")
        self._route_member_ids: dict[str, str] = {}
        self._lineage_state: StatelessOpponentPoolLineageState | None = None
        self._adaptive_state: StatelessOpponentPoolAdaptiveState | None = None
        self._prepared_adaptive: _PreparedAdaptiveAllocation | None = None
        self._prepared_role_budget: _PreparedRoleBudgetAllocation | None = None
        self._last_adaptive_report: (
            AdaptiveOpponentAllocationReport | RoleBudgetOpponentAllocationReport | None
        ) = None
        self._selection_cache: (
            tuple[tuple[object, ...], tuple[tuple[_ArtifactGroup, str], ...]] | None
        ) = None
        self._revision_cache: (
            tuple[tuple[object, ...], PoolRevision, dict[str, str]] | None
        ) = None
        self._founder_policy_sha256: str | None
        if config.behavior_version in {2, 3, 4, 5}:
            if isinstance(initial_state, LeagueState):
                raise ValueError("lineage opponent pool cannot resume a v1 state")
            resolved_founder: str | None = (
                initial_state.founder_policy_sha256
                if isinstance(
                    initial_state,
                    (
                        StatelessOpponentPoolLineageState,
                        StatelessOpponentPoolAdaptiveState,
                    ),
                )
                else founder_policy_sha256
            )
            if resolved_founder is None:
                raise ValueError("lineage opponent pool requires a founder policy")
            if (
                founder_policy_sha256 is not None
                and founder_policy_sha256 != resolved_founder
            ):
                raise ValueError("lineage opponent-pool founder changed")
            self._founder_policy_sha256 = resolved_founder
            core_initial = None if initial_state is None else initial_state.league_state
            if config.behavior_version == 2 and isinstance(
                initial_state, StatelessOpponentPoolAdaptiveState
            ):
                raise ValueError("behavior two cannot resume adaptive pool state")
            if config.behavior_version == 3:
                if isinstance(initial_state, StatelessOpponentPoolLineageState):
                    self._adaptive_state = _migrate_adaptive_state(initial_state)
                elif isinstance(initial_state, StatelessOpponentPoolAdaptiveState):
                    if initial_state.schema_version != 3:
                        raise ValueError("behavior three cannot resume role state")
                    self._adaptive_state = initial_state
            elif config.behavior_version in {4, 5} and isinstance(
                initial_state,
                (StatelessOpponentPoolLineageState, StatelessOpponentPoolAdaptiveState),
            ):
                self._adaptive_state = _migrate_role_budget_state(
                    initial_state,
                    target_schema_version=(4 if config.behavior_version == 4 else 5),
                )
        else:
            if isinstance(
                initial_state,
                (
                    StatelessOpponentPoolLineageState,
                    StatelessOpponentPoolAdaptiveState,
                ),
            ):
                raise ValueError("legacy opponent pool cannot resume a lineage state")
            if founder_policy_sha256 is not None:
                raise ValueError("legacy opponent pool cannot bind a founder policy")
            self._founder_policy_sha256 = None
            core_initial = initial_state
        if core_initial is None and not synchronize_revision:
            raise ValueError("a fresh opponent pool must synchronize its revision")
        self.controller = OpponentPoolController(
            LeagueState.initial(self._build_revision(sequence=0))
            if core_initial is None
            else core_initial
        )
        if config.behavior_version == 2:
            self._lineage_state = (
                StatelessOpponentPoolLineageState(
                    league_state=self.controller.state,
                    founder_policy_sha256=str(self._founder_policy_sha256),
                )
                if initial_state is None
                else _lineage_initial_state(initial_state)
            )
        elif config.behavior_version == 3 and self._adaptive_state is None:
            self._adaptive_state = StatelessOpponentPoolAdaptiveState(
                league_state=self.controller.state,
                founder_policy_sha256=str(self._founder_policy_sha256),
            )
        elif config.behavior_version in {4, 5} and self._adaptive_state is None:
            self._adaptive_state = StatelessOpponentPoolAdaptiveState(
                schema_version=(4 if config.behavior_version == 4 else 5),
                league_state=self.controller.state,
                founder_policy_sha256=str(self._founder_policy_sha256),
                role_decisions=dict.fromkeys(ROLE_ORDER, 0),
            )
        if synchronize_revision:
            self._sync_revision()
        else:
            self._restore_route_member_ids()

    @property
    def state(self) -> OpponentPoolCheckpointState:
        """Return the current settled state."""
        if self._adaptive_state is not None:
            return self._adaptive_state
        if self._lineage_state is not None:
            return self._lineage_state
        return self.controller.state

    @property
    def league_state(self) -> LeagueState:
        """Return the normalized active-revision state used by the core planner."""
        return self.controller.state

    @property
    def revision_fingerprint(self) -> str:
        """Return the active immutable pool revision identity."""
        return self.league_state.revision.fingerprint

    def set_deck_target_shares(
        self,
        target_shares: Mapping[str, float],
    ) -> None:
        """Set candidate-deck priorities for the next settled window plan."""
        normalized = dict(target_shares)
        if set(normalized) != set(self.active_deck_digests):
            raise ValueError("opponent-pool deck targets differ from active decks")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in normalized.values()
        ):
            raise ValueError("opponent-pool deck targets must be finite and positive")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
            raise ValueError("opponent-pool deck targets must sum to one")
        if normalized == self.deck_target_shares:
            return
        self.deck_target_shares = normalized
        self._prepared_adaptive = None
        self._prepared_role_budget = None
        self._selection_cache = None

    def _selected_groups(self) -> tuple[tuple[_ArtifactGroup, str], ...]:
        if self.config.behavior_version in {4, 5}:
            return self._selected_role_budget_groups()
        if self.config.behavior_version == 3:
            return self._selected_adaptive_groups()
        if self.config.behavior_version == 2:
            return self._selected_lineage_groups()
        groups = _group_members(self.curriculum.state.members)
        protected = tuple(
            group
            for group in groups
            if group.source in {"historical_anchor", "fixed_stateless_anchor"}
        )
        past_self = tuple(
            sorted(
                (group for group in groups if group.source == "past_self"),
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )
        )
        if len(past_self) < self.config.past_self_artifacts:
            raise ValueError("opponent-pool V2 has too few past-self artifacts")
        retained = past_self[-self.config.past_self_artifacts :]
        recent = retained[-self.config.recent_artifacts :]
        age_candidates = retained[: -self.config.recent_artifacts]
        age_diverse = _evenly_spaced(age_candidates, len(age_candidates))
        selected = (
            *((group, "protected") for group in protected),
            *((group, "recent") for group in recent),
            *((group, "age_diverse") for group in age_diverse),
        )
        if not protected or len(selected) > self.config.maximum_active_artifacts:
            raise ValueError("opponent-pool V2 active artifact partition is invalid")
        return selected

    def _selected_lineage_groups(self) -> tuple[tuple[_ArtifactGroup, str], ...]:
        """Build the bounded active graph from the complete retained lineage."""
        groups = _group_members(self.curriculum.state.members)
        protected = tuple(
            group
            for group in groups
            if group.source in {"historical_anchor", "fixed_stateless_anchor"}
            or group.policy_sha256 == self._founder_policy_sha256
        )
        if not any(
            group.policy_sha256 == self._founder_policy_sha256 for group in protected
        ):
            raise ValueError("lineage opponent-pool founder is not resident")
        past_self = tuple(
            sorted(
                (
                    group
                    for group in groups
                    if group.source == "past_self"
                    and group.policy_sha256 != self._founder_policy_sha256
                ),
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )
        )
        recent = past_self[-2:]
        recent_ids = {group.policy_sha256 for group in recent}
        older = tuple(
            group for group in past_self if group.policy_sha256 not in recent_ids
        )
        counter: tuple[_ArtifactGroup, ...] = ()
        eligible = tuple(
            (self._artifact_priority(group), group)
            for group in older
            if self._artifact_evidence(group)
            >= self.curriculum.config.pfsp.minimum_evidence
        )
        if eligible:
            counter = (
                max(
                    eligible,
                    key=lambda item: (
                        item[0],
                        self._artifact_evidence(item[1]),
                        -item[1].source_policy_version,
                        item[1].policy_sha256,
                    ),
                )[1],
            )
        counter_ids = {group.policy_sha256 for group in counter}
        age_candidates = tuple(
            group for group in older if group.policy_sha256 not in counter_ids
        )
        age_diverse = self._rotating_age_landmarks(age_candidates, count=2)
        selected = (
            *((group, "protected") for group in protected),
            *((group, "counter_frontier") for group in counter),
            *((group, "recent") for group in recent),
            *((group, "age_diverse") for group in age_diverse),
        )
        if not protected or len(selected) > self.config.maximum_active_artifacts:
            raise ValueError("lineage opponent-pool active partition is invalid")
        return selected

    def _selected_adaptive_groups(self) -> tuple[tuple[_ArtifactGroup, str], ...]:
        """Select a capacity-bounded set that covers non-transitive demand."""
        allocation = self.config.adaptive_allocation
        if allocation is None:
            raise RuntimeError("adaptive opponent-pool settings are missing")
        state = self._adaptive_state
        window_sequence = (
            state.league_state.next_window_sequence if state is not None else 0
        )
        cache_key: tuple[object, ...] = (
            self.curriculum.state.generation,
            window_sequence,
            tuple(sorted(self.deck_target_shares.items())),
            0 if state is None else len(state.evidence),
            0 if state is None else state.allocation_decision_clock,
            None if state is None else state.last_target_fingerprint,
        )
        if self._selection_cache is not None and self._selection_cache[0] == cache_key:
            return self._selection_cache[1]
        groups = _group_members(self.curriculum.state.members)
        protected = tuple(
            group
            for group in groups
            if group.source in {"historical_anchor", "fixed_stateless_anchor"}
            or group.policy_sha256 == self._founder_policy_sha256
        )
        if not any(
            group.policy_sha256 == self._founder_policy_sha256 for group in protected
        ):
            raise ValueError("adaptive opponent pool lost its founder")
        if len(protected) > self.config.maximum_active_artifacts:
            raise ValueError("protected artifacts exceed adaptive capacity")
        past_self = tuple(
            sorted(
                (
                    group
                    for group in groups
                    if group.source == "past_self"
                    and group.policy_sha256 != self._founder_policy_sha256
                ),
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )
        )
        remaining_capacity = self.config.maximum_active_artifacts - len(protected)
        recent_count = min(
            allocation.recent_artifacts, remaining_capacity, len(past_self)
        )
        recent = past_self[-recent_count:] if recent_count else ()
        recent_ids = {group.policy_sha256 for group in recent}
        older = tuple(
            group for group in past_self if group.policy_sha256 not in recent_ids
        )
        evidence_index = AdaptiveEvidenceIndex(
            () if state is None else state.evidence,
            window_sequence=window_sequence,
            config=allocation,
        )
        mandatory = protected + recent
        mandatory_demands = {
            group.policy_sha256: self._group_adaptive_demand(group, evidence_index)
            for group in mandatory
        }
        candidate_coverage = dict.fromkeys(self.active_deck_digests, 0.0)
        for demand, _dominant in mandatory_demands.values():
            for candidate, value in demand.items():
                candidate_coverage[candidate] = max(
                    candidate_coverage[candidate], value
                )
        older_demands = {
            group.policy_sha256: self._group_adaptive_demand(group, evidence_index)
            for group in older
        }
        selected_adaptive: list[tuple[_ArtifactGroup, str]] = []
        selectable = list(older)
        adaptive_slots = remaining_capacity - len(recent)
        for _index in range(min(adaptive_slots, len(selectable))):

            def selection_key(group: _ArtifactGroup) -> tuple[float, float, int, str]:
                demand, _dominant = older_demands[group.policy_sha256]
                gain = sum(
                    self.deck_target_shares[candidate]
                    * max(0.0, value - candidate_coverage[candidate])
                    for candidate, value in demand.items()
                )
                baseline = sum(
                    self.deck_target_shares[candidate] * value
                    for candidate, value in demand.items()
                )
                return (
                    gain + 0.25 * baseline,
                    baseline,
                    -group.source_policy_version,
                    group.policy_sha256,
                )

            chosen = max(selectable, key=selection_key)
            demand, dominant = older_demands[chosen.policy_sha256]
            stratum = (
                "probe_reentry"
                if dominant == "probe"
                else "age_diverse"
                if dominant in {"rehearsal", "staleness"}
                else "counter_frontier"
            )
            selected_adaptive.append((chosen, stratum))
            selectable.remove(chosen)
            for candidate, value in demand.items():
                candidate_coverage[candidate] = max(
                    candidate_coverage[candidate], value
                )
        selected = (
            *((group, "protected") for group in protected),
            *((group, "recent") for group in recent),
            *selected_adaptive,
        )
        if not selected or len(selected) > self.config.maximum_active_artifacts:
            raise ValueError("adaptive opponent-pool selection is invalid")
        self._selection_cache = (cache_key, selected)
        return selected

    def _selected_role_budget_groups(self) -> tuple[tuple[_ArtifactGroup, str], ...]:
        """Select one artifact per adaptive role plus protected and recent floors."""
        allocation = self.config.role_budget_allocation
        if allocation is None:
            raise RuntimeError("role-budget opponent-pool settings are missing")
        state = self._adaptive_state
        window_sequence = (
            state.league_state.next_window_sequence if state is not None else 0
        )
        cache_key: tuple[object, ...] = (
            "role-budget-v1",
            self.curriculum.state.generation,
            window_sequence,
            tuple(sorted(self.deck_target_shares.items())),
            0 if state is None else len(state.evidence),
            0 if state is None else state.allocation_decision_clock,
        )
        if self._selection_cache is not None and self._selection_cache[0] == cache_key:
            return self._selection_cache[1]

        groups = _group_members(self.curriculum.state.members)
        protected = tuple(
            group
            for group in groups
            if group.source in {"historical_anchor", "fixed_stateless_anchor"}
            or group.policy_sha256 == self._founder_policy_sha256
        )
        if not any(
            group.policy_sha256 == self._founder_policy_sha256 for group in protected
        ):
            raise ValueError("role-budget opponent pool lost its founder")
        past_self = tuple(
            sorted(
                (
                    group
                    for group in groups
                    if group.source == "past_self"
                    and group.policy_sha256 != self._founder_policy_sha256
                ),
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )
        )
        recent_count = min(allocation.recent_artifacts, len(past_self))
        recent = past_self[-recent_count:] if recent_count else ()
        recent_ids = {group.policy_sha256 for group in recent}
        older = tuple(
            group for group in past_self if group.policy_sha256 not in recent_ids
        )
        # A public-catalog transition deliberately retires every incompatible
        # past-self artifact and admits only the relabeled current policy as
        # the new founder. Keep all five role budgets executable while the new
        # lineage grows by temporarily assigning non-founder protected anchors
        # to otherwise-empty roles. Each later distinct past-self artifact
        # replaces one stand-in, returning that anchor to protected.
        founder_groups = tuple(
            group
            for group in protected
            if group.policy_sha256 == self._founder_policy_sha256
        )
        stand_in_count = (0 if recent else 1) + max(0, 3 - len(older))
        stand_ins = tuple(
            sorted(
                (
                    group
                    for group in protected
                    if group.policy_sha256 != self._founder_policy_sha256
                ),
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )[:stand_in_count]
        )
        if len(founder_groups) != 1 or len(stand_ins) != stand_in_count:
            raise ValueError(
                "role-budget pool needs five distinct bootstrap artifacts"
            )
        stand_in_ids = {group.policy_sha256 for group in stand_ins}
        selected_protected = tuple(
            group for group in protected if group.policy_sha256 not in stand_in_ids
        )
        stand_in_offset = 0
        if not recent:
            recent = (stand_ins[stand_in_offset],)
            stand_in_offset += 1
        adaptive_stand_ins = iter(stand_ins[stand_in_offset:])
        required_count = len(selected_protected) + len(recent) + 3
        if required_count > self.config.maximum_active_artifacts:
            raise ValueError("role-budget pool exceeds its active artifact budget")

        evidence_index = AdaptiveEvidenceIndex(
            () if state is None else state.evidence,
            window_sequence=window_sequence,
            config=allocation,
        )
        scored_demands = {
            group.policy_sha256: self._group_role_demands(group, evidence_index)
            for group in older
        }
        demands = {
            policy_sha256: scored[0] for policy_sha256, scored in scored_demands.items()
        }
        artifact_evidence = {
            policy_sha256: scored[1] for policy_sha256, scored in scored_demands.items()
        }
        selectable = list(older)
        selected_roles: list[tuple[_ArtifactGroup, str]] = []
        role_strata: tuple[tuple[RoleBudgetName, str], ...] = (
            ("counter", "counter_frontier"),
            ("frontier", "age_diverse"),
            ("recovery", "probe_reentry"),
        )
        for role, stratum in role_strata:

            def selection_key(
                group: _ArtifactGroup,
                selected_role: RoleBudgetName = role,
            ) -> tuple[float, float, int, int, str]:
                candidate_demand = demands[group.policy_sha256][selected_role]
                weighted = sum(
                    self.deck_target_shares[candidate] * value
                    for candidate, value in candidate_demand.items()
                )
                worst = min(candidate_demand.values())
                return (
                    weighted,
                    worst,
                    artifact_evidence[group.policy_sha256],
                    -group.source_policy_version,
                    group.policy_sha256,
                )

            chosen = (
                max(selectable, key=selection_key)
                if selectable
                else next(adaptive_stand_ins)
            )
            selected_roles.append((chosen, stratum))
            if chosen in selectable:
                selectable.remove(chosen)

        selected = (
            *((group, "protected") for group in selected_protected),
            *((group, "recent") for group in recent),
            *selected_roles,
        )
        self._selection_cache = (cache_key, selected)
        return selected

    def _group_role_demands(
        self,
        group: _ArtifactGroup,
        evidence_index: AdaptiveEvidenceIndex,
    ) -> tuple[dict[RoleBudgetName, dict[str, float]], int]:
        """Score each archive artifact once under each explicit adaptive role."""
        allocation = self.config.role_budget_allocation
        if allocation is None:
            raise RuntimeError("role-budget opponent-pool settings are missing")
        artifact = _artifact(group)
        routes = tuple(_route(artifact, member) for member in group.members)
        requests = tuple(
            (candidate, member.member_id)
            for candidate in self.active_deck_digests
            for member in group.members
        )
        observations = iter(self.curriculum.pfsp_member_observations(requests))
        total_evidence = 0
        values: dict[
            RoleBudgetName,
            defaultdict[str, list[float]],
        ] = {role: defaultdict(list) for role in ("counter", "frontier", "recovery")}
        for candidate in self.active_deck_digests:
            for member, route in zip(group.members, routes, strict=True):
                fallback_score, exact_evidence = next(observations)
                total_evidence += exact_evidence
                for seat in (0, 1):
                    score = evidence_index.score(
                        AdaptiveMatchupIdentity(
                            candidate_deck_digest=candidate,
                            artifact_id=artifact.artifact_id,
                            route_id=route.route_id,
                            opponent_deck_digest=route.exact_deck_digest,
                            candidate_seat=seat,
                        ),
                        fallback_score=fallback_score,
                        fallback_evidence=exact_evidence,
                        base_weight=member.base_weight,
                    )
                    for role in ("counter", "frontier", "recovery"):
                        values[role][candidate].append(role_priority(score, role))
        demands = {
            role: {
                candidate: cvar(
                    candidate_values,
                    fraction=allocation.artifact_cvar_fraction,
                )
                for candidate, candidate_values in by_candidate.items()
            }
            for role, by_candidate in values.items()
        }
        return demands, total_evidence

    def _group_adaptive_demand(
        self,
        group: _ArtifactGroup,
        evidence_index: AdaptiveEvidenceIndex,
    ) -> tuple[dict[str, float], PortfolioName]:
        """Summarize one archive artifact without averaging away its counters."""
        allocation = self.config.adaptive_allocation
        if allocation is None:
            raise RuntimeError("adaptive opponent-pool settings are missing")
        artifact = _artifact(group)
        routes = tuple(_route(artifact, member) for member in group.members)
        requests = tuple(
            (candidate, member.member_id)
            for candidate in self.active_deck_digests
            for member in group.members
        )
        observations = iter(self.curriculum.pfsp_member_observations(requests))
        demand_values: defaultdict[str, list[float]] = defaultdict(list)
        components: dict[PortfolioName, float] = {
            "counter": 0.0,
            "frontier": 0.0,
            "probe": 0.0,
            "rehearsal": 0.0,
            "staleness": 0.0,
        }
        for candidate in self.active_deck_digests:
            candidate_share = self.deck_target_shares[candidate]
            for member, route in zip(group.members, routes, strict=True):
                fallback_score, exact_evidence = next(observations)
                for seat in (0, 1):
                    score = evidence_index.score(
                        AdaptiveMatchupIdentity(
                            candidate_deck_digest=candidate,
                            artifact_id=artifact.artifact_id,
                            route_id=route.route_id,
                            opponent_deck_digest=route.exact_deck_digest,
                            candidate_seat=seat,
                        ),
                        fallback_score=fallback_score,
                        fallback_evidence=exact_evidence,
                        base_weight=member.base_weight,
                    )
                    demand_values[candidate].append(score.utility)
                    for name, value in score.components.items():
                        components[name] += candidate_share * value
        demands = {
            candidate: cvar(values, fraction=allocation.artifact_cvar_fraction)
            for candidate, values in demand_values.items()
        }
        weights = allocation.portfolios.as_mapping()
        dominant = max(
            components,
            key=lambda name: (components[name] * weights[name], name),
        )
        return demands, dominant

    def _rotating_age_landmarks(
        self,
        values: Sequence[_ArtifactGroup],
        *,
        count: int,
    ) -> tuple[_ArtifactGroup, ...]:
        """Rotate one representative from each age band across full history."""
        ordered = tuple(
            sorted(
                values,
                key=lambda item: (item.source_policy_version, item.policy_sha256),
            )
        )
        if not ordered:
            return ()
        bands = min(count, len(ordered))
        window = (
            self.controller.state.next_window_sequence
            if hasattr(self, "controller")
            else 0
        )
        selected: list[_ArtifactGroup] = []
        for band in range(bands):
            start = band * len(ordered) // bands
            stop = (band + 1) * len(ordered) // bands
            candidates = ordered[start:stop]
            selected.append(candidates[window % len(candidates)])
        return tuple(selected)

    def _artifact_priority(self, group: _ArtifactGroup) -> float:
        """Aggregate unchanged matchup PFSP weights with deck target shares."""
        requests = tuple(
            (candidate, member.member_id)
            for candidate in self.deck_target_shares
            for member in group.members
        )
        priorities = iter(self.curriculum.pfsp_member_priorities(requests))
        base_total = sum(member.base_weight for member in group.members)
        return sum(
            candidate_share
            * sum(next(priorities) for _member in group.members)
            / base_total
            for candidate, candidate_share in self.deck_target_shares.items()
        )

    def _artifact_evidence(self, group: _ArtifactGroup) -> int:
        """Return exact candidate/route terminal evidence for counter admission."""
        return sum(
            self.curriculum.pfsp_member_evidence_many(
                tuple(
                    (candidate, member.member_id)
                    for candidate in self.active_deck_digests
                    for member in group.members
                )
            )
        )

    def _matchup_priorities(
        self, revision: PoolRevision
    ) -> dict[tuple[str, str, int], float]:
        """Resolve all active matchup weights from one curriculum snapshot."""
        requests = tuple(
            (
                matchup.candidate_deck_digest,
                self._route_member_ids[matchup.route_id],
            )
            for matchup in revision.active_matchups
        )
        values = self.curriculum.pfsp_member_priorities(requests)
        return {
            matchup.key: value
            for matchup, value in zip(
                revision.active_matchups,
                values,
                strict=True,
            )
        }

    def candidate_target_shares(self) -> dict[str, float]:
        """Return the joint candidate targets for the next full lane portfolio."""
        if self.config.behavior_version not in {3, 4, 5}:
            return dict(self.deck_target_shares)
        self._sync_revision()
        if self.config.behavior_version in {4, 5}:
            return dict(
                self._prepare_role_budget_allocation().snapshot.candidate_target_shares
            )
        return dict(
            self._prepare_adaptive_allocation().snapshot.candidate_target_shares
        )

    @property
    def adaptive_allocation_report(
        self,
    ) -> AdaptiveOpponentAllocationReport | RoleBudgetOpponentAllocationReport | None:
        """Return the last settled detail artifact without rescoring live state."""
        return self._last_adaptive_report

    def _score_active_matchups(
        self,
        allocation: AdaptiveEvidenceAllocationConfig,
    ) -> tuple[tuple[AdaptiveCellScore, ...], AdaptiveEvidenceIndex]:
        """Build shared posterior evidence once for the active exact graph."""
        if self._adaptive_state is None:
            raise RuntimeError("adaptive opponent-pool state is unavailable")
        revision = self.league_state.revision
        routes = {item.route_id: item for item in revision.routes}
        artifacts = {item.artifact_id: item for item in revision.artifacts}
        route_artifacts = {
            route.route_id: artifacts[route.artifact_id] for route in revision.routes
        }
        members = {
            member.member_id: member
            for member in self.curriculum.state.members
            if member.status == "active"
        }
        requests = tuple(
            (
                matchup.candidate_deck_digest,
                self._route_member_ids[matchup.route_id],
            )
            for matchup in revision.active_matchups
        )
        observations = self.curriculum.pfsp_member_observations(requests)
        evidence_index = AdaptiveEvidenceIndex(
            self._adaptive_state.evidence,
            window_sequence=self.league_state.next_window_sequence,
            config=allocation,
        )
        scores: list[AdaptiveCellScore] = []
        for matchup, observation in zip(
            revision.active_matchups,
            observations,
            strict=True,
        ):
            fallback_score, exact_evidence = observation
            route = routes[matchup.route_id]
            artifact = route_artifacts[matchup.route_id]
            member = members[self._route_member_ids[matchup.route_id]]
            scores.append(
                evidence_index.score(
                    AdaptiveMatchupIdentity(
                        candidate_deck_digest=matchup.candidate_deck_digest,
                        artifact_id=artifact.artifact_id,
                        route_id=route.route_id,
                        opponent_deck_digest=route.exact_deck_digest,
                        candidate_seat=matchup.candidate_seat,
                    ),
                    fallback_score=fallback_score,
                    fallback_evidence=exact_evidence,
                    base_weight=member.base_weight,
                )
            )
        return tuple(scores), evidence_index

    def _prepare_adaptive_allocation(self) -> _PreparedAdaptiveAllocation:
        """Score one active revision once for candidate and opponent planning."""
        if self.config.behavior_version != 3 or self._adaptive_state is None:
            raise RuntimeError("adaptive opponent-pool state is unavailable")
        allocation = self.config.adaptive_allocation
        if allocation is None:
            raise RuntimeError("adaptive opponent-pool settings are missing")
        revision = self.league_state.revision
        if self._prepared_adaptive is not None and (
            self._prepared_adaptive.revision_fingerprint == revision.fingerprint
            and self._prepared_adaptive.state_fingerprint
            == self._adaptive_state.fingerprint
        ):
            return self._prepared_adaptive
        scores, evidence_index = self._score_active_matchups(allocation)
        by_candidate: defaultdict[str, list[float]] = defaultdict(list)
        for score in scores:
            by_candidate[score.identity.candidate_deck_digest].append(score.utility)
        demands = {
            candidate: cvar(
                by_candidate[candidate],
                fraction=allocation.artifact_cvar_fraction,
            )
            for candidate in self.active_deck_digests
        }
        candidate_targets = bounded_candidate_shares(
            self.deck_target_shares,
            demands,
            config=allocation,
        )
        previous = {
            item.matchup_key: item.share
            for item in self._adaptive_state.last_target_weights
        }
        target_weights = normalized_matchup_targets(
            scores,
            previous=previous,
            config=allocation,
        )
        target_mapping = {item.matchup_key: item.share for item in target_weights}
        score_mapping = {item.identity.matchup_key: item for item in scores}
        decision_mass: dict[tuple[str, str, int], float] = {}
        expected_decisions: dict[tuple[str, str, int], float] = {}
        portfolio_by_matchup: dict[tuple[str, str, int], PortfolioName] = {}
        for key, score in score_mapping.items():
            row = evidence_index.rows.get(score.identity.evidence_key)
            decision_mass[key] = 0.0 if row is None else row.decision_sum
            expected_decisions[key] = score.expected_decisions
            portfolio_by_matchup[key] = score.dominant_portfolio
        snapshot = AdaptiveAllocationSnapshot(
            window_sequence=self.league_state.next_window_sequence,
            candidate_target_shares=candidate_targets,
            target_weights=target_weights,
            portfolio_target_mass=portfolio_mass(
                scores,
                target_weights,
                candidate_targets,
            ),
            evidence_cells=sum(item.effective_evidence > 0.0 for item in scores),
            low_evidence_cells=sum(
                item.effective_evidence < allocation.fallback_evidence_cap
                for item in scores
            ),
        )
        self._prepared_adaptive = _PreparedAdaptiveAllocation(
            revision_fingerprint=revision.fingerprint,
            state_fingerprint=self._adaptive_state.fingerprint,
            snapshot=snapshot,
            scores=tuple(scores),
            matchup_targets=target_mapping,
            matchup_decision_mass=decision_mass,
            expected_decisions_per_game=expected_decisions,
            portfolio_by_matchup=portfolio_by_matchup,
        )
        return self._prepared_adaptive

    def _prepare_role_budget_allocation(self) -> _PreparedRoleBudgetAllocation:
        """Build fixed role, artifact-first, and exact-route targets."""
        if (
            self.config.behavior_version not in {4, 5}
            or self._adaptive_state is None
            or self._adaptive_state.schema_version != self.config.behavior_version
        ):
            raise RuntimeError("role-budget opponent-pool state is unavailable")
        allocation = self.config.role_budget_allocation
        if allocation is None:
            raise RuntimeError("role-budget opponent-pool settings are missing")
        revision = self.league_state.revision
        if self._prepared_role_budget is not None and (
            self._prepared_role_budget.revision_fingerprint == revision.fingerprint
            and self._prepared_role_budget.state_fingerprint
            == self._adaptive_state.fingerprint
        ):
            return self._prepared_role_budget

        scores, evidence_index = self._score_active_matchups(allocation)
        by_candidate: defaultdict[str, list[float]] = defaultdict(list)
        for score in scores:
            by_candidate[score.identity.candidate_deck_digest].append(
                max(
                    role_priority(score, "counter"),
                    role_priority(score, "frontier"),
                    role_priority(score, "recovery"),
                )
            )
        demands = {
            candidate: cvar(
                by_candidate[candidate],
                fraction=allocation.artifact_cvar_fraction,
            )
            for candidate in self.active_deck_digests
        }
        candidate_targets = bounded_candidate_shares(
            self.deck_target_shares,
            demands,
            config=allocation,
        )
        entries = {item.artifact_id: item for item in revision.entries}
        artifact_roles = {
            artifact_id: role_for_stratum(entry.stratum)
            for artifact_id, entry in entries.items()
        }
        target_weights = hierarchical_role_targets(
            scores,
            artifact_roles=artifact_roles,
        )
        target_mapping = {item.matchup_key: item.share for item in target_weights}
        # Role budgets are a per-window contract. Historical decisions remain in
        # evidence for posterior estimation, but they were collected under
        # earlier role semantics and must not become allocation debt under the
        # new fixed budget. Starting each window at zero also prevents a role
        # change for an archive artifact from inheriting another role's debt.
        decision_mass: dict[tuple[str, str, int], float] = {}
        expected_decisions: dict[tuple[str, str, int], float] = {}
        last_learning_windows: dict[tuple[str, str, int], int | None] = {}
        for score in scores:
            key = score.identity.matchup_key
            row = evidence_index.rows.get(score.identity.evidence_key)
            decision_mass[key] = 0.0
            expected_decisions[key] = score.expected_decisions
            last_learning_windows[key] = (
                None if row is None else row.last_learning_window
            )
        snapshot = RoleBudgetAllocationSnapshot(
            window_sequence=self.league_state.next_window_sequence,
            candidate_target_shares=candidate_targets,
            target_weights=target_weights,
            role_target_mass=dict(ROLE_TARGET_SHARES),
            evidence_cells=sum(item.effective_evidence > 0.0 for item in scores),
            low_evidence_cells=sum(
                item.effective_evidence < allocation.fallback_evidence_cap
                for item in scores
            ),
        )
        self._prepared_role_budget = _PreparedRoleBudgetAllocation(
            revision_fingerprint=revision.fingerprint,
            state_fingerprint=self._adaptive_state.fingerprint,
            snapshot=snapshot,
            scores=scores,
            matchup_targets=target_mapping,
            matchup_decision_mass=decision_mass,
            expected_decisions_per_game=expected_decisions,
            matchup_last_learning_windows=last_learning_windows,
        )
        return self._prepared_role_budget

    def _build_revision(self, *, sequence: int) -> PoolRevision:
        selected = self._selected_groups()
        cache_key: tuple[object, ...] = (
            sequence,
            tuple(
                (
                    group.policy_sha256,
                    stratum,
                    tuple(member.member_id for member in group.members),
                )
                for group, stratum in selected
            ),
            self.active_deck_digests,
        )
        if self._revision_cache is not None and self._revision_cache[0] == cache_key:
            self._route_member_ids = dict(self._revision_cache[2])
            return self._revision_cache[1]
        artifacts_by_policy = {
            group.policy_sha256: _artifact(group) for group, _stratum in selected
        }
        route_members: dict[str, str] = {}
        routes: list[OpponentRoute] = []
        entries: list[PoolEntry] = []
        for group, stratum in selected:
            artifact = artifacts_by_policy[group.policy_sha256]
            roles: tuple[str, ...]
            if stratum == "protected":
                roles = (
                    ("champion",)
                    if group.policy_sha256 == self._founder_policy_sha256
                    else ("sentinel",)
                )
            elif stratum == "counter_frontier":
                roles = ("counter_train",)
            elif stratum == "recent":
                roles = ("recent",)
            elif stratum == "probe_reentry":
                roles = ("probe",)
            else:
                roles = (
                    ("frontier",)
                    if self.config.behavior_version in {4, 5}
                    else ("age_landmark",)
                )
            entries.append(
                PoolEntry(
                    artifact_id=artifact.artifact_id,
                    semantic_roles=roles,  # type: ignore[arg-type]
                    stratum=stratum,  # type: ignore[arg-type]
                    # Revision synchronization assigns the pool-owned clock.
                    # Curriculum admission generations are a different domain.
                    admission_generation=0,
                )
            )
            for member in group.members:
                route = _route(artifact, member)
                if route.route_id in route_members:
                    raise ValueError("opponent-pool V2 route identity is duplicated")
                route_members[route.route_id] = member.member_id
                routes.append(route)
        ordered_routes = tuple(sorted(routes, key=lambda item: item.route_id))
        matchups = tuple(
            ActiveMatchup(
                candidate_deck_digest=candidate,
                route_id=route.route_id,
                candidate_seat=seat,
            )
            for candidate in self.active_deck_digests
            for route in ordered_routes
            for seat in (0, 1)
        )
        revision = PoolRevision(
            revision_sequence=sequence,
            maximum_active_artifacts=self.config.maximum_active_artifacts,
            compatibility_target_fingerprint=(
                STATELESS_OPPONENT_POOL_V2_COMPATIBILITY_FINGERPRINT
            ),
            artifacts=tuple(
                sorted(artifacts_by_policy.values(), key=lambda item: item.artifact_id)
            ),
            routes=ordered_routes,
            entries=tuple(sorted(entries, key=lambda item: item.artifact_id)),
            active_matchups=tuple(sorted(matchups, key=lambda item: item.key)),
        )
        self._route_member_ids = route_members
        self._revision_cache = (cache_key, revision, dict(route_members))
        return revision

    def _sync_revision(self) -> None:
        current = self.controller.state
        desired = self._build_revision(sequence=current.revision.revision_sequence)
        current_admissions = {
            entry.artifact_id: entry.admission_generation
            for entry in current.revision.entries
        }
        desired = desired.model_copy(
            update={
                "entries": tuple(
                    entry.model_copy(
                        update={
                            "admission_generation": current_admissions.get(
                                entry.artifact_id,
                                current.generation + 1,
                            )
                        }
                    )
                    for entry in desired.entries
                )
            }
        )
        # ``desired`` uses the current sequence and preserves current admission
        # generations, so frozen-model equality is the same comparison without
        # materializing both large active graphs as dictionaries.
        if desired == current.revision:
            return
        successor = desired.model_copy(
            update={
                "revision_sequence": current.revision.revision_sequence + 1,
            }
        )
        previous_ids = {item.artifact_id for item in current.revision.artifacts}
        successor_ids = {item.artifact_id for item in successor.artifacts}
        self.controller.transition(
            RevisionTransitionRequest(
                predecessor_state_fingerprint=current.fingerprint,
                predecessor_revision_fingerprint=current.revision.fingerprint,
                successor_revision=successor,
                admitted_artifact_ids=tuple(sorted(successor_ids - previous_ids)),
                retired_artifact_ids=tuple(sorted(previous_ids - successor_ids)),
            )
        )

        self._replace_lineage_league_state(self.controller.state)
        self._prepared_adaptive = None
        self._prepared_role_budget = None

    def _restore_route_member_ids(self) -> None:
        """Resolve a checkpoint revision without selecting its successor yet."""
        required_route_ids = {
            route.route_id for route in self.controller.state.revision.routes
        }
        route_members: dict[str, str] = {}
        for group in _group_members(self.curriculum.state.members):
            artifact = _artifact(group)
            for member in group.members:
                route = _route(artifact, member)
                if route.route_id in required_route_ids:
                    route_members[route.route_id] = member.member_id
        if set(route_members) != required_route_ids:
            raise ValueError("opponent-pool checkpoint routes lost curriculum members")
        self._route_member_ids = route_members

    def _replace_lineage_league_state(self, state: LeagueState) -> None:
        """Keep the exact-resume envelope synchronized with core transitions."""
        if self._lineage_state is not None:
            self._lineage_state = self._lineage_state.model_copy(
                update={"league_state": state}
            )
        if self._adaptive_state is not None:
            self._adaptive_state = self._adaptive_state.model_copy(
                update={"league_state": state}
            )

    def begin_window(
        self,
        *,
        pfsp_games: int | None = None,
        candidate_cells: Sequence[tuple[str, Literal[0, 1]]] | None = None,
    ) -> WindowPlan:
        """Synchronize resource admission, then reserve one adaptive V2 plan."""
        if self.config.behavior_version in {3, 4, 5}:
            raise ValueError("joint adaptive allocation requires aggregate quotas")
        self._sync_revision()
        revision = self.league_state.revision
        routes = {route.route_id: route for route in revision.routes}
        matchup_priorities = self._matchup_priorities(revision)
        artifact_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for matchup in revision.active_matchups:
            artifact_values[routes[matchup.route_id].artifact_id].append(
                (
                    matchup_priorities[matchup.key],
                    self.deck_target_shares[matchup.candidate_deck_digest],
                )
            )
        artifact_priorities = {
            artifact_id: sum(value * weight for value, weight in values)
            / sum(weight for _value, weight in values)
            for artifact_id, values in artifact_values.items()
        }
        if self.config.behavior_version == 1:
            if pfsp_games is None or candidate_cells is not None:
                raise ValueError("legacy opponent pool requires only a game count")
            selected_slots = sum(
                item.artifact_slots for item in self.config.planner_policy.strata
            )
            return self.controller.begin_window(
                capacity=ArtifactCapacity(maximum_artifacts=selected_slots),
                policy=self.config.planner_policy,
                total_games=pfsp_games,
                artifact_priorities=artifact_priorities,
                matchup_priorities=matchup_priorities,
            )
        if candidate_cells is None or pfsp_games is not None:
            raise ValueError("lineage opponent pool requires fixed candidate cells")
        effective_policy = self._effective_lineage_policy()
        effective_strata = tuple(item.stratum for item in effective_policy.strata)
        if self._lineage_state is None:
            raise RuntimeError("lineage opponent-pool state is missing")
        if self._lineage_state.effective_strata != effective_strata:
            self._lineage_state = self._lineage_state.model_copy(
                update={
                    "effective_strata": effective_strata,
                    "stratum_decisions": tuple(
                        StratumDecisionTotal(stratum=stratum)
                        for stratum in effective_strata
                    ),
                }
            )
        coverage = {
            item.stratum: item.decisions
            for item in self._lineage_state.stratum_decisions
        }
        founder_ids = {
            artifact.artifact_id
            for artifact in revision.artifacts
            if artifact.source_fingerprint == self._founder_policy_sha256
        }
        if len(founder_ids) != 1:
            raise RuntimeError("lineage active revision lost its unique founder")
        selected_slots = sum(item.artifact_slots for item in effective_policy.strata)
        return self.controller.begin_window_for_candidate_cells(
            capacity=ArtifactCapacity(maximum_artifacts=selected_slots),
            policy=effective_policy,
            candidate_cells=tuple(candidate_cells),
            stratum_decision_coverage=coverage,
            mandatory_artifact_ids=frozenset(founder_ids),
            artifact_priorities=artifact_priorities,
            matchup_priorities=matchup_priorities,
        )

    def begin_quota_window(
        self,
        *,
        candidate_cell_counts: Mapping[tuple[str, Literal[0, 1]], int],
    ) -> BoundOpponentQuotaWindow:
        """Reserve an aggregate lineage plan from compact candidate counts."""
        if self.config.behavior_version not in {2, 3, 4, 5}:
            raise ValueError("aggregate quota planning requires lineage behavior")
        self._sync_revision()
        revision = self.league_state.revision
        if self.config.behavior_version in {4, 5}:
            if self._adaptive_state is None:
                raise RuntimeError("role-budget opponent-pool state is missing")
            role_allocation = self.config.role_budget_allocation
            if role_allocation is None:
                raise RuntimeError("role-budget opponent-pool settings are missing")
            prepared_role = self._prepare_role_budget_allocation()
            plan = self.controller.begin_adaptive_quota_window_for_candidate_counts(
                capacity=ArtifactCapacity(
                    maximum_artifacts=len(revision.artifacts),
                ),
                candidate_cell_counts=candidate_cell_counts,
                matchup_targets=prepared_role.matchup_targets,
                matchup_decision_mass=prepared_role.matchup_decision_mass,
                expected_decisions_per_game=(prepared_role.expected_decisions_per_game),
                minimum_artifact_games=(
                    role_allocation.matchup_game_batch_size
                    if self.config.behavior_version == 5
                    else 1
                ),
                matchup_game_batch_size=role_allocation.matchup_game_batch_size,
                matchup_last_learning_windows=(
                    prepared_role.matchup_last_learning_windows
                    if self.config.behavior_version == 5
                    else None
                ),
                matchup_coverage_windows=(
                    role_allocation.matchup_coverage_windows
                    if self.config.behavior_version == 5
                    else None
                ),
            )
            return BoundOpponentQuotaWindow(
                plan=plan,
                lineage_state_fingerprint=self._adaptive_state.fingerprint,
                role_budget_snapshot=prepared_role.snapshot,
            )
        if self.config.behavior_version == 3:
            if self._adaptive_state is None:
                raise RuntimeError("adaptive opponent-pool state is missing")
            allocation = self.config.adaptive_allocation
            if allocation is None:
                raise RuntimeError("adaptive opponent-pool settings are missing")
            prepared = self._prepare_adaptive_allocation()
            plan = self.controller.begin_adaptive_quota_window_for_candidate_counts(
                capacity=ArtifactCapacity(
                    maximum_artifacts=len(revision.artifacts),
                ),
                candidate_cell_counts=candidate_cell_counts,
                matchup_targets=prepared.matchup_targets,
                matchup_decision_mass=prepared.matchup_decision_mass,
                expected_decisions_per_game=prepared.expected_decisions_per_game,
                minimum_artifact_games=allocation.minimum_artifact_games,
            )
            return BoundOpponentQuotaWindow(
                plan=plan,
                lineage_state_fingerprint=self._adaptive_state.fingerprint,
                allocation_snapshot=prepared.snapshot,
                portfolio_by_matchup=prepared.portfolio_by_matchup,
            )
        routes = {route.route_id: route for route in revision.routes}
        matchup_priorities = self._matchup_priorities(revision)
        artifact_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for matchup in revision.active_matchups:
            artifact_values[routes[matchup.route_id].artifact_id].append(
                (
                    matchup_priorities[matchup.key],
                    self.deck_target_shares[matchup.candidate_deck_digest],
                )
            )
        artifact_priorities = {
            artifact_id: sum(value * weight for value, weight in values)
            / sum(weight for _value, weight in values)
            for artifact_id, values in artifact_values.items()
        }
        effective_policy = self._effective_lineage_policy()
        effective_strata = tuple(item.stratum for item in effective_policy.strata)
        if self._lineage_state is None:
            raise RuntimeError("lineage opponent-pool state is missing")
        if self._lineage_state.effective_strata != effective_strata:
            self._lineage_state = self._lineage_state.model_copy(
                update={
                    "effective_strata": effective_strata,
                    "stratum_decisions": tuple(
                        StratumDecisionTotal(stratum=stratum)
                        for stratum in effective_strata
                    ),
                }
            )
        coverage = {
            item.stratum: item.decisions
            for item in self._lineage_state.stratum_decisions
        }
        founder_ids = {
            artifact.artifact_id
            for artifact in revision.artifacts
            if artifact.source_fingerprint == self._founder_policy_sha256
        }
        if len(founder_ids) != 1:
            raise RuntimeError("lineage active revision lost its unique founder")
        selected_slots = sum(item.artifact_slots for item in effective_policy.strata)
        plan = self.controller.begin_quota_window_for_candidate_counts(
            capacity=ArtifactCapacity(maximum_artifacts=selected_slots),
            policy=effective_policy,
            candidate_cell_counts=candidate_cell_counts,
            stratum_decision_coverage=coverage,
            mandatory_artifact_ids=frozenset(founder_ids),
            artifact_priorities=artifact_priorities,
            matchup_priorities=matchup_priorities,
        )
        return BoundOpponentQuotaWindow(
            plan=plan,
            lineage_state_fingerprint=self._lineage_state.fingerprint,
        )

    def _effective_lineage_policy(self) -> PlannerPolicy:
        """Drop unavailable bootstrap strata and renormalize their target mass."""
        available = {entry.stratum for entry in self.league_state.revision.entries}
        configured = tuple(
            item
            for item in self.config.planner_policy.strata
            if item.stratum in available
        )
        total = sum(item.target_fraction for item in configured)
        entries_by_stratum = {
            stratum: sum(
                entry.stratum == stratum for entry in self.league_state.revision.entries
            )
            for stratum in available
        }
        return self.config.planner_policy.model_copy(
            update={
                "strata": tuple(
                    StratumPolicy(
                        stratum=item.stratum,
                        artifact_slots=min(
                            item.artifact_slots,
                            entries_by_stratum[item.stratum],
                        ),
                        target_fraction=item.target_fraction / total,
                    )
                    for item in configured
                )
            }
        )

    def _role_budget_policy(self) -> PlannerPolicy:
        """Expose the fixed role contract using the existing strata ABI."""
        counts: Counter[QuotaStratum] = Counter(
            item.stratum for item in self.league_state.revision.entries
        )
        targets: dict[QuotaStratum, float] = {
            "protected": ROLE_TARGET_SHARES["protected"],
            "counter_frontier": ROLE_TARGET_SHARES["counter"],
            "recent": ROLE_TARGET_SHARES["recent"],
            "age_diverse": ROLE_TARGET_SHARES["frontier"],
            "probe_reentry": ROLE_TARGET_SHARES["recovery"],
        }
        return PlannerPolicy(
            strata=tuple(
                StratumPolicy(
                    stratum=stratum,
                    artifact_slots=counts[stratum],
                    target_fraction=targets[stratum],
                )
                for stratum in (
                    "protected",
                    "counter_frontier",
                    "recent",
                    "age_diverse",
                    "probe_reentry",
                )
            )
        )

    def effective_planner_policy(self) -> PlannerPolicy:
        """Expose the currently executable quota policy for status reporting."""
        return (
            self._effective_lineage_policy()
            if self.config.behavior_version == 2
            else self._role_budget_policy()
            if self.config.behavior_version in {4, 5}
            else self.config.planner_policy
        )

    def member_id_for_route(self, route_id: str) -> str:
        """Resolve a planned immutable route to its execution member."""
        try:
            return self._route_member_ids[route_id]
        except KeyError as exc:
            raise ValueError("opponent-pool V2 planned an unknown route") from exc

    def commit(
        self,
        window: BoundOpponentPoolWindow,
        outcomes: Sequence[StatelessGameOutcome],
    ) -> OpponentPoolCheckpointState:
        """Commit accepted terminal evidence; omitted plan rows stay unresolved."""
        if self._adaptive_state is not None:
            raise ValueError("adaptive opponent pool requires aggregate quota commit")
        if len(window.curriculum_assignment_ids) != len(window.plan.assignments):
            raise ValueError("opponent-pool V2 assignment binding is incomplete")
        indexes = {
            assignment_id: index
            for index, assignment_id in enumerate(window.curriculum_assignment_ids)
        }
        opponent_outcomes = tuple(
            OpponentOutcome(
                assignment_index=indexes[outcome.curriculum_assignment_id],
                status=outcome.status,
                candidate_decisions=outcome.candidate_decisions,
                candidate_score=outcome.candidate_score,
            )
            for outcome in outcomes
            if outcome.curriculum_assignment_id in indexes
        )
        if self._lineage_state is not None and (
            window.lineage_state_fingerprint != self._lineage_state.fingerprint
        ):
            raise ValueError("lineage opponent-pool window state is stale")
        successor = self.controller.commit(window.plan, opponent_outcomes)
        if self._lineage_state is None:
            return successor
        assignments = {
            assignment.assignment_index: assignment
            for assignment in window.plan.assignments
        }
        increments: dict[QuotaStratum, int] = defaultdict(int)
        for outcome in opponent_outcomes:
            if outcome.status in {"engine_terminal", "window_cutoff", "step_limit"}:
                increments[assignments[outcome.assignment_index].stratum] += (
                    outcome.candidate_decisions
                )
        totals = tuple(
            item.model_copy(
                update={
                    "decisions": item.decisions + increments[item.stratum],
                }
            )
            for item in self._lineage_state.stratum_decisions
        )
        self._lineage_state = self._lineage_state.model_copy(
            update={
                "league_state": successor,
                "stratum_decisions": totals,
            }
        )
        return self._lineage_state

    def commit_quota(
        self,
        window: BoundOpponentQuotaWindow,
        outcomes: Sequence[StatelessGameOutcome],
    ) -> OpponentPoolCheckpointState:
        """Commit only PFSP leases materialized from an aggregate plan."""
        opponent_outcomes = tuple(
            OpponentOutcome(
                assignment_index=window.curriculum_assignment_cells[
                    outcome.curriculum_assignment_id
                ],
                status=outcome.status,
                candidate_decisions=outcome.candidate_decisions,
                candidate_score=outcome.candidate_score,
            )
            for outcome in outcomes
            if outcome.curriculum_assignment_id in window.curriculum_assignment_cells
        )
        checkpoint_state: (
            StatelessOpponentPoolLineageState
            | StatelessOpponentPoolAdaptiveState
            | None
        ) = self._adaptive_state or self._lineage_state
        if checkpoint_state is None or (
            window.lineage_state_fingerprint != checkpoint_state.fingerprint
        ):
            raise ValueError("lineage opponent-pool quota window state is stale")
        successor = self.controller.commit_quota(window.plan, opponent_outcomes)
        cells = {item.cell_index: item for item in window.plan.cells}
        if self._adaptive_state is not None:
            if self.config.behavior_version in {4, 5}:
                return self._commit_role_budget_quota(
                    window,
                    opponent_outcomes,
                    successor=successor,
                )
            return self._commit_adaptive_quota(
                window,
                opponent_outcomes,
                successor=successor,
            )
        if self._lineage_state is None:
            raise RuntimeError("lineage opponent-pool state is missing")
        increments: dict[QuotaStratum, int] = defaultdict(int)
        for outcome in opponent_outcomes:
            if outcome.status in {"engine_terminal", "window_cutoff", "step_limit"}:
                increments[cells[outcome.assignment_index].stratum] += (
                    outcome.candidate_decisions
                )
        totals = tuple(
            item.model_copy(
                update={"decisions": item.decisions + increments[item.stratum]}
            )
            for item in self._lineage_state.stratum_decisions
        )
        self._lineage_state = self._lineage_state.model_copy(
            update={
                "league_state": successor,
                "stratum_decisions": totals,
            }
        )
        return self._lineage_state

    def _commit_adaptive_quota(
        self,
        window: BoundOpponentQuotaWindow,
        outcomes: Sequence[OpponentOutcome],
        *,
        successor: LeagueState,
    ) -> StatelessOpponentPoolAdaptiveState:
        """Commit exact evidence and the target snapshot at one settled boundary."""
        if self._adaptive_state is None:
            raise RuntimeError("adaptive opponent-pool state is missing")
        allocation = self.config.adaptive_allocation
        snapshot = window.allocation_snapshot
        if allocation is None or snapshot is None:
            raise ValueError("adaptive quota window omitted its target snapshot")
        prepared = self._prepared_adaptive
        if prepared is None or prepared.snapshot.fingerprint != snapshot.fingerprint:
            raise ValueError("adaptive quota window lost its scored allocation")
        if snapshot.window_sequence != window.plan.window_sequence:
            raise ValueError("adaptive target snapshot clock differs from its plan")
        predecessor_state_fingerprint = self._adaptive_state.fingerprint
        previous_targets = {
            item.matchup_key: item.share
            for item in self._adaptive_state.last_target_weights
        }
        previous_candidate_targets = dict(self._adaptive_state.candidate_target_shares)
        typed_cells = {item.cell_index: item for item in window.plan.cells}
        routes = {item.route_id: item for item in successor.revision.routes}
        evidence = {
            item.identity.evidence_key: item for item in self._adaptive_state.evidence
        }
        portfolio_decisions = {
            name: self._adaptive_state.portfolio_decisions.get(name, 0)
            for name in snapshot.portfolio_target_mass
        }
        accepted_decisions = 0
        for outcome in outcomes:
            if outcome.status not in {"engine_terminal", "window_cutoff", "step_limit"}:
                continue
            cell = typed_cells[outcome.assignment_index]
            route = routes[cell.route_id]
            identity = AdaptiveMatchupIdentity(
                candidate_deck_digest=cell.candidate_deck_digest,
                artifact_id=cell.artifact_id,
                route_id=cell.route_id,
                opponent_deck_digest=route.exact_deck_digest,
                candidate_seat=cell.candidate_seat,
            )
            evidence[identity.evidence_key] = observe_evidence(
                evidence.get(identity.evidence_key),
                identity=identity,
                window_sequence=window.plan.window_sequence,
                candidate_score=outcome.candidate_score,
                candidate_decisions=outcome.candidate_decisions,
                config=allocation,
            )
            portfolio = window.portfolio_by_matchup.get(cell.matchup_key)
            if portfolio is None:
                raise ValueError("adaptive quota cell omitted its allocation reason")
            portfolio_decisions[portfolio] = (
                portfolio_decisions.get(portfolio, 0) + outcome.candidate_decisions
            )
            accepted_decisions += outcome.candidate_decisions
        self._adaptive_state = self._adaptive_state.model_copy(
            update={
                "league_state": successor,
                "evidence": tuple(
                    sorted(
                        evidence.values(), key=lambda item: item.identity.evidence_key
                    )
                ),
                "allocation_decision_clock": (
                    self._adaptive_state.allocation_decision_clock + accepted_decisions
                ),
                "last_target_weights": snapshot.target_weights,
                "candidate_target_shares": snapshot.candidate_target_shares,
                "portfolio_target_mass": snapshot.portfolio_target_mass,
                "portfolio_decisions": portfolio_decisions,
                "last_target_fingerprint": snapshot.fingerprint,
            }
        )
        self._last_adaptive_report = build_adaptive_allocation_report(
            predecessor_state_fingerprint=predecessor_state_fingerprint,
            committed_state=successor,
            committed_state_fingerprint=self._adaptive_state.fingerprint,
            snapshot=snapshot,
            scores=prepared.scores,
            previous_targets=previous_targets,
            previous_candidate_targets=previous_candidate_targets,
            matchup_decision_mass=prepared.matchup_decision_mass,
            base_candidate_shares=self.deck_target_shares,
            plan=window.plan,
            outcomes=outcomes,
        )
        self._prepared_adaptive = None
        self._prepared_role_budget = None
        self._selection_cache = None
        return self._adaptive_state

    def _commit_role_budget_quota(
        self,
        window: BoundOpponentQuotaWindow,
        outcomes: Sequence[OpponentOutcome],
        *,
        successor: LeagueState,
    ) -> StatelessOpponentPoolAdaptiveState:
        """Commit role evidence and actual decisions under explicit cell roles."""
        if self._adaptive_state is None or (
            self._adaptive_state.schema_version != self.config.behavior_version
        ):
            raise RuntimeError("role-budget opponent-pool state is missing")
        allocation = self.config.role_budget_allocation
        snapshot = window.role_budget_snapshot
        prepared = self._prepared_role_budget
        if allocation is None or snapshot is None or prepared is None:
            raise ValueError("role-budget quota window omitted its scored targets")
        if prepared.snapshot.fingerprint != snapshot.fingerprint:
            raise ValueError("role-budget quota window lost its scored allocation")
        if snapshot.window_sequence != window.plan.window_sequence:
            raise ValueError("role-budget target clock differs from its plan")

        predecessor_state_fingerprint = self._adaptive_state.fingerprint
        typed_cells = {item.cell_index: item for item in window.plan.cells}
        routes = {item.route_id: item for item in successor.revision.routes}
        evidence = {
            item.identity.evidence_key: item for item in self._adaptive_state.evidence
        }
        role_decisions = {
            role: self._adaptive_state.role_decisions.get(role, 0)
            for role in ROLE_ORDER
        }
        accepted_decisions = 0
        for outcome in outcomes:
            if outcome.status not in {
                "engine_terminal",
                "window_cutoff",
                "step_limit",
            }:
                continue
            cell = typed_cells[outcome.assignment_index]
            route = routes[cell.route_id]
            identity = AdaptiveMatchupIdentity(
                candidate_deck_digest=cell.candidate_deck_digest,
                artifact_id=cell.artifact_id,
                route_id=cell.route_id,
                opponent_deck_digest=route.exact_deck_digest,
                candidate_seat=cell.candidate_seat,
            )
            evidence[identity.evidence_key] = observe_evidence(
                evidence.get(identity.evidence_key),
                identity=identity,
                window_sequence=window.plan.window_sequence,
                candidate_score=outcome.candidate_score,
                candidate_decisions=outcome.candidate_decisions,
                config=allocation,
            )
            role = role_for_stratum(cell.stratum)
            role_decisions[role] += outcome.candidate_decisions
            accepted_decisions += outcome.candidate_decisions

        self._adaptive_state = self._adaptive_state.model_copy(
            update={
                "league_state": successor,
                "evidence": tuple(
                    sorted(
                        evidence.values(),
                        key=lambda item: item.identity.evidence_key,
                    )
                ),
                "allocation_decision_clock": (
                    self._adaptive_state.allocation_decision_clock + accepted_decisions
                ),
                "last_target_weights": snapshot.target_weights,
                "candidate_target_shares": snapshot.candidate_target_shares,
                "role_target_mass": snapshot.role_target_mass,
                "role_decisions": role_decisions,
                "last_target_fingerprint": snapshot.fingerprint,
            }
        )
        self._last_adaptive_report = build_role_budget_allocation_report(
            predecessor_state_fingerprint=predecessor_state_fingerprint,
            committed_state=successor,
            committed_state_fingerprint=self._adaptive_state.fingerprint,
            snapshot=snapshot,
            scores=prepared.scores,
            matchup_decision_mass=prepared.matchup_decision_mass,
            base_candidate_shares=self.deck_target_shares,
            plan=window.plan,
            outcomes=outcomes,
        )
        self._prepared_role_budget = None
        self._selection_cache = None
        return self._adaptive_state

    def abort(
        self,
        window: BoundOpponentPoolWindow | BoundOpponentQuotaWindow | WindowPlan,
    ) -> None:
        """Discard one pending plan without granting exposure credit."""
        plan = (
            window.plan
            if isinstance(window, (BoundOpponentPoolWindow, BoundOpponentQuotaWindow))
            else window
        )
        pending = self.controller.pending_plan
        if pending is not None and pending.plan_id == plan.plan_id:
            self.controller.abort()


def assign_stateless_games_v2(
    *,
    games: int,
    opponent_pool: StatelessOpponentPoolV2,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
    active_decks: Mapping[str, CanonicalDeck],
    opponent_decks: Mapping[str, CanonicalDeck],
    mirror_policy_fingerprint: str,
) -> PlannedStatelessAssignments:
    """Issue a full lane portfolio whose PFSP rows come from one V2 plan."""
    if games <= 0:
        raise ValueError("opponent-pool V2 assignment count must be positive")
    opponent_pool.set_deck_target_shares(deck_balance.target_share_mapping)
    if opponent_pool.config.behavior_version == 2:
        return _assign_stateless_games_lineage(
            games=games,
            opponent_pool=opponent_pool,
            curriculum=curriculum,
            deck_balance=deck_balance,
            active_decks=active_decks,
            opponent_decks=opponent_decks,
            mirror_policy_fingerprint=mirror_policy_fingerprint,
        )
    lanes = curriculum.preview_lanes(games)
    pfsp_games = sum(lane == "pfsp" for lane in lanes)
    plan = opponent_pool.begin_window(pfsp_games=pfsp_games)
    planned = iter(plan.assignments)
    requested_cells: list[tuple[str, Literal[0, 1]] | None] = []
    planned_member_ids: list[str | None] = []
    for lane in lanes:
        if lane != "pfsp":
            requested_cells.append(None)
            planned_member_ids.append(None)
            continue
        assignment = next(planned)
        requested_cells.append(
            (assignment.candidate_deck_digest, assignment.candidate_seat)
        )
        planned_member_ids.append(
            opponent_pool.member_id_for_route(assignment.route_id)
        )
    try:
        next(planned)
    except StopIteration:
        pass
    else:
        opponent_pool.abort(plan)
        raise RuntimeError("opponent-pool V2 plan exceeded PFSP lane count")

    balances = deck_balance.assign_mixed(tuple(requested_cells))
    mirror_decks = tuple(active_decks.values())
    try:
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
            ),
            pfsp_member_ids=planned_member_ids,
        )
    except BaseException:
        for balance in balances:
            deck_balance.cancel(balance.assignment_id)
        opponent_pool.abort(plan)
        raise
    assignments = tuple(
        StatelessAssignedGame(balance=balance, curriculum=assigned)
        for balance, assigned in zip(
            balances,
            curriculum_assignments,
            strict=True,
        )
    )
    if any(
        item.curriculum.opponent_deck_digest not in opponent_decks
        for item in assignments
    ):
        from ptcg_rl.rl.stateless_collection import cancel_stateless_assignments

        cancel_stateless_assignments(
            assignments,
            curriculum=curriculum,
            deck_balance=deck_balance,
        )
        opponent_pool.abort(plan)
        raise KeyError("opponent-pool V2 deck is not registered")
    pfsp_assignment_ids = tuple(
        item.curriculum.assignment_id
        for item in assignments
        if item.curriculum.lane == "pfsp"
    )
    return PlannedStatelessAssignments(
        assignments=assignments,
        opponent_pool_window=BoundOpponentPoolWindow(
            plan=plan,
            curriculum_assignment_ids=pfsp_assignment_ids,
        ),
    )


def _assign_stateless_games_lineage(
    *,
    games: int,
    opponent_pool: StatelessOpponentPoolV2,
    curriculum: StatelessCurriculumController,
    deck_balance: StatelessDeckBalanceSampler,
    active_decks: Mapping[str, CanonicalDeck],
    opponent_decks: Mapping[str, CanonicalDeck],
    mirror_policy_fingerprint: str,
) -> PlannedStatelessAssignments:
    """Bind weighted candidate cells before planning their PFSP opponents."""
    lanes = curriculum.preview_lanes(games)
    balances = deck_balance.assign_many(games)
    candidate_cells = tuple(
        (balance.deck_digest, balance.seat)
        for lane, balance in zip(lanes, balances, strict=True)
        if lane == "pfsp"
    )
    try:
        plan = opponent_pool.begin_window(candidate_cells=candidate_cells)
    except BaseException:
        for balance in balances:
            deck_balance.cancel(balance.assignment_id)
        raise
    planned = iter(plan.assignments)
    planned_member_ids: list[str | None] = []
    for lane, balance in zip(lanes, balances, strict=True):
        if lane != "pfsp":
            planned_member_ids.append(None)
            continue
        assignment = next(planned)
        if (assignment.candidate_deck_digest, assignment.candidate_seat) != (
            balance.deck_digest,
            balance.seat,
        ):
            opponent_pool.abort(plan)
            for leased in balances:
                deck_balance.cancel(leased.assignment_id)
            raise RuntimeError("opponent pool changed a deck-balance candidate cell")
        planned_member_ids.append(
            opponent_pool.member_id_for_route(assignment.route_id)
        )
    try:
        next(planned)
    except StopIteration:
        pass
    else:
        opponent_pool.abort(plan)
        for balance in balances:
            deck_balance.cancel(balance.assignment_id)
        raise RuntimeError("lineage opponent plan exceeded PFSP lane count")
    mirror_decks = tuple(active_decks.values())
    try:
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
            ),
            pfsp_member_ids=planned_member_ids,
        )
    except BaseException:
        for balance in balances:
            deck_balance.cancel(balance.assignment_id)
        opponent_pool.abort(plan)
        raise
    assignments = tuple(
        StatelessAssignedGame(balance=balance, curriculum=assigned)
        for balance, assigned in zip(
            balances,
            curriculum_assignments,
            strict=True,
        )
    )
    if any(
        item.curriculum.opponent_deck_digest not in opponent_decks
        for item in assignments
    ):
        from ptcg_rl.rl.stateless_collection import cancel_stateless_assignments

        cancel_stateless_assignments(
            assignments,
            curriculum=curriculum,
            deck_balance=deck_balance,
        )
        opponent_pool.abort(plan)
        raise KeyError("lineage opponent-pool deck is not registered")
    pfsp_assignment_ids = tuple(
        item.curriculum.assignment_id
        for item in assignments
        if item.curriculum.lane == "pfsp"
    )
    state = opponent_pool.state
    if not isinstance(state, StatelessOpponentPoolLineageState):
        raise RuntimeError("lineage assignment lost its exact-resume state")
    return PlannedStatelessAssignments(
        assignments=assignments,
        opponent_pool_window=BoundOpponentPoolWindow(
            plan=plan,
            curriculum_assignment_ids=pfsp_assignment_ids,
            lineage_state_fingerprint=state.fingerprint,
        ),
    )


__all__ = [
    "BoundOpponentPoolWindow",
    "PlannedStatelessAssignments",
    "STATELESS_OPPONENT_POOL_V2_COMPATIBILITY_FINGERPRINT",
    "OpponentPoolCheckpointState",
    "StatelessOpponentPoolAdaptiveState",
    "StatelessOpponentPoolLineageState",
    "StatelessOpponentPoolV2",
    "assign_stateless_games_v2",
]
