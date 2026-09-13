"""Metric-driven, resumable promotion controller for a bounded PFSP league."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.rl.curriculum import (
    CurriculumOutcome,
    CurriculumSampler,
    fingerprint_checkpoint,
)
from ptcg_rl.rl.frozen_league import (
    FrozenLeagueCandidate,
    FrozenLeagueConfig,
    read_frozen_league_candidate,
)

PromotionRoute = Literal["champion", "coverage", "timeout", "rejected"]
MAX_PROMOTION_DECISION_HISTORY = 128


class OutcomeTally(BaseModel):
    """Raw W/D/L evidence with draws worth half a point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @field_validator("wins", "draws", "losses")
    @classmethod
    def valid_non_negative(cls, value: int) -> int:
        """Reject negative outcome counts."""
        if value < 0:
            raise ValueError("probe outcome counts must be non-negative")
        return value

    @property
    def games(self) -> int:
        """Return the number of completed games."""
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Return accumulated score points."""
        return float(self.wins) + 0.5 * float(self.draws)

    @property
    def mean(self) -> float:
        """Return empirical score, using 0.5 before any observation."""
        if self.games == 0:
            return 0.5
        return self.score / float(self.games)

    def observe(self, reward: float) -> OutcomeTally:
        """Return an updated immutable tally for one terminal reward."""
        if reward > 0.0:
            return self.model_copy(update={"wins": self.wins + 1})
        if reward < 0.0:
            return self.model_copy(update={"losses": self.losses + 1})
        return self.model_copy(update={"draws": self.draws + 1})


class MemberProbeStats(BaseModel):
    """Contemporaneous promotion evidence for one frozen member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: OutcomeTally = Field(default_factory=OutcomeTally)
    by_opponent_deck: dict[str, OutcomeTally] = Field(default_factory=dict)

    def observe(self, *, reward: float, opponent_deck_label: str) -> MemberProbeStats:
        """Return statistics updated by one frozen game."""
        label = opponent_deck_label.strip() or "unknown"
        by_deck = dict(self.by_opponent_deck)
        by_deck[label] = by_deck.get(label, OutcomeTally()).observe(reward)
        return self.model_copy(
            update={
                "total": self.total.observe(reward),
                "by_opponent_deck": by_deck,
            }
        )


class PromotionProbe(BaseModel):
    """One balanced evidence window shared by a challenger and incumbents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    challenger_id: str
    challenger_version: int
    members: dict[str, MemberProbeStats]


class PromotionDecision(BaseModel):
    """One durable challenger decision and its posterior evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    challenger_id: str
    challenger_version: int
    route: PromotionRoute
    promoted: bool
    evicted_id: str | None = None
    champion_id_before: str | None = None
    champion_probability: float | None = None
    coverage_gain_mean: float | None = None
    coverage_gain_probability: float | None = None
    probe_games: dict[str, int] = Field(default_factory=dict)
    reason: str


class LeaguePromotionState(BaseModel):
    """Persistent role assignment and in-flight promotion evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    anchor_ids: tuple[str, ...] = ()
    champion_id: str | None = None
    exploiter_ids: tuple[str, ...] = ()
    challenger: FrozenLeagueCandidate | None = None
    probe: PromotionProbe | None = None
    last_candidate_version: int = 0
    last_promotion_version: int = 0
    promotion_count: int = 0
    protected_until_promotion: dict[str, int] = Field(default_factory=dict)
    decisions: tuple[PromotionDecision, ...] = ()

    @field_validator(
        "schema_version",
        "last_candidate_version",
        "last_promotion_version",
        "promotion_count",
    )
    @classmethod
    def valid_non_negative(cls, value: int) -> int:
        """Reject invalid persistent counters."""
        if value < 0:
            raise ValueError("league promotion counters must be non-negative")
        return value


class LeaguePromotionController:
    """Own challenger probing and bounded frozen-pool mutation."""

    def __init__(
        self,
        config: FrozenLeagueConfig,
        *,
        sampler: CurriculumSampler,
        pool_state_path: Path,
        promotion_state_path: Path,
        candidates_dir: Path,
    ) -> None:
        """Load or initialize one central curriculum promotion controller."""
        self.config = config
        self.sampler = sampler
        self.pool_state_path = pool_state_path
        self.promotion_state_path = promotion_state_path
        self.candidates_dir = candidates_dir
        self.state = self._load_or_initialize_state()
        self._reconcile_sampler_with_state()
        self._validate_state()
        self._refresh_probe_schedule()

    def poll_candidates(self) -> bool:
        """Stage the newest eligible durable candidate when the slot is free."""
        if not self.config.enabled or not self.candidates_dir.exists():
            return False
        candidates = sorted(
            (
                (path, read_frozen_league_candidate(path))
                for path in self.candidates_dir.glob("league_v*.json")
            ),
            key=lambda item: item[1].policy_version,
        )
        changed = False
        eligible: list[tuple[Path, FrozenLeagueCandidate]] = []
        for path, candidate in candidates:
            if candidate.policy_version <= self.state.last_candidate_version:
                path.unlink(missing_ok=True)
                changed = True
                continue
            eligible.append((path, candidate))
        if self.state.challenger is not None or not eligible:
            if changed:
                self.save_state()
            return changed
        _, candidate = eligible[-1]
        existing_ids = {member.opponent_id for member in self.sampler.frozen_members}
        if (
            candidate.opponent_id not in existing_ids
            and len(existing_ids) >= self.config.max_opponents
        ):
            return changed
        self._verify_candidate(candidate)
        self.sampler.add_frozen_member(
            opponent_id=candidate.opponent_id,
            checkpoint_path=candidate.checkpoint_path,
            winrate_ema=0.5,
            pinned=False,
            recurrent=candidate.recurrent,
        )
        member_ids = tuple(member.opponent_id for member in self.sampler.frozen_members)
        probe = PromotionProbe(
            challenger_id=candidate.opponent_id,
            challenger_version=candidate.policy_version,
            members={member_id: MemberProbeStats() for member_id in member_ids},
        )
        self.state = self.state.model_copy(
            update={
                "challenger": candidate,
                "probe": probe,
                "last_candidate_version": candidate.policy_version,
            }
        )
        self._persist_pool_and_state()
        for path, superseded in eligible:
            if superseded.policy_version <= candidate.policy_version:
                path.unlink(missing_ok=True)
        self._refresh_probe_schedule()
        return True

    def observe(self, outcome: CurriculumOutcome) -> bool:
        """Accumulate one checkpoint-identical frozen outcome for an active probe."""
        probe = self.state.probe
        if probe is None or outcome.opponent_kind != "frozen":
            return False
        if outcome.candidate_policy_version != probe.challenger_version:
            return False
        member = probe.members.get(outcome.opponent_id)
        if member is None:
            return False
        members = dict(probe.members)
        members[outcome.opponent_id] = member.observe(
            reward=outcome.candidate_reward,
            opponent_deck_label=outcome.opponent_deck_label,
        )
        self.state = self.state.model_copy(
            update={"probe": probe.model_copy(update={"members": members})}
        )
        self._refresh_probe_schedule()
        return True

    def maybe_decide(self) -> PromotionDecision | None:
        """Promote or reject a challenger once every member has enough evidence."""
        if not self._probe_ready():
            return None
        probe = self.state.probe
        challenger = self.state.challenger
        if probe is None or challenger is None:
            raise RuntimeError("ready league probe has no challenger")

        champion_probability = self._champion_probability(probe)
        coverage = self._best_coverage_replacement(probe, protect_champion=True)
        timed_out = (
            challenger.policy_version - self.state.last_promotion_version
            >= self.config.max_versions_without_promotion
        )
        champion_promoted = (
            champion_probability is not None
            and champion_probability >= self.config.promotion_probability
        )
        coverage_promoted = (
            coverage is not None
            and coverage[1] >= self.config.minimum_coverage_gain
            and coverage[2] >= self.config.promotion_probability
        )

        if champion_promoted:
            route: PromotionRoute = "champion"
            promoted = True
            replacement = self._best_coverage_replacement(
                probe,
                protect_champion=False,
            )
            reason = "challenger is probably harder than the current champion"
        elif coverage_promoted:
            route = "coverage"
            promoted = True
            replacement = coverage
            reason = "challenger increases posterior matchup coverage"
        elif timed_out:
            route = "timeout"
            promoted = True
            replacement = (
                coverage
                or self._best_coverage_replacement(
                    probe,
                    protect_champion=False,
                )
                or self._fallback_replacement(
                    probe,
                    protect_champion=False,
                )
            )
            reason = "maximum versions without a promotion reached"
        else:
            route = "rejected"
            promoted = False
            replacement = coverage
            reason = "challenger did not improve strength or matchup coverage"

        evicted_id = replacement[0] if promoted and replacement is not None else None
        if promoted and evicted_id is None:
            raise RuntimeError("promotion has no eligible bounded-pool replacement")
        decision = PromotionDecision(
            challenger_id=challenger.opponent_id,
            challenger_version=challenger.policy_version,
            route=route,
            promoted=promoted,
            evicted_id=evicted_id,
            champion_id_before=self.state.champion_id,
            champion_probability=champion_probability,
            coverage_gain_mean=None if replacement is None else replacement[1],
            coverage_gain_probability=(
                None if replacement is None else replacement[2]
            ),
            probe_games={
                member_id: stats.total.games
                for member_id, stats in sorted(probe.members.items())
            },
            reason=reason,
        )
        self._apply_decision(decision)
        return decision

    def save_state(self) -> None:
        """Atomically persist promotion state."""
        write_league_promotion_state(self.promotion_state_path, self.state)

    def summary(self) -> dict[str, object]:
        """Return bounded JSON-friendly controller diagnostics."""
        probe = self.state.probe
        latest_decision = self.state.decisions[-1] if self.state.decisions else None
        return {
            "enabled": self.config.enabled,
            "state_path": str(self.promotion_state_path),
            "roles": {
                "anchors": list(self.state.anchor_ids),
                "champion": self.state.champion_id,
                "exploiters": list(self.state.exploiter_ids),
                "challenger": (
                    None
                    if self.state.challenger is None
                    else self.state.challenger.opponent_id
                ),
            },
            "probe_games": (
                {}
                if probe is None
                else {
                    member_id: stats.total.games
                    for member_id, stats in sorted(probe.members.items())
                }
            ),
            "last_candidate_version": self.state.last_candidate_version,
            "last_promotion_version": self.state.last_promotion_version,
            "promotion_count": self.state.promotion_count,
            "latest_decision": (
                None if latest_decision is None else latest_decision.model_dump(mode="json")
            ),
        }

    def _load_or_initialize_state(self) -> LeaguePromotionState:
        if self.promotion_state_path.exists():
            raw = json.loads(self.promotion_state_path.read_text(encoding="utf-8"))
            return LeaguePromotionState.model_validate(raw)
        members = self.sampler.frozen_members
        anchors = tuple(member.opponent_id for member in members if member.pinned)
        non_pinned = [
            member for member in members if not member.pinned and not member.retired
        ]
        champion_id = self.config.initial_champion_id
        if champion_id is None and non_pinned:
            champion_id = max(non_pinned, key=lambda member: member.added_order).opponent_id
        exploiters = tuple(
            member.opponent_id
            for member in non_pinned
            if member.opponent_id != champion_id
        )
        champion_version = _opponent_policy_version(champion_id)
        state = LeaguePromotionState(
            anchor_ids=anchors,
            champion_id=champion_id,
            exploiter_ids=exploiters,
            last_promotion_version=champion_version,
        )
        write_league_promotion_state(self.promotion_state_path, state)
        return state

    def _reconcile_sampler_with_state(self) -> None:
        """Repair a crash between the two atomic controller state writes."""
        desired = set(self.state.anchor_ids) | set(self.state.exploiter_ids)
        if self.state.champion_id is not None:
            desired.add(self.state.champion_id)
        challenger = self.state.challenger
        if challenger is not None:
            desired.add(challenger.opponent_id)
        members = {
            member.opponent_id: member for member in self.sampler.frozen_members
        }
        if challenger is not None and challenger.opponent_id not in members:
            self._verify_candidate(challenger)
            self.sampler.add_frozen_member(
                opponent_id=challenger.opponent_id,
                checkpoint_path=challenger.checkpoint_path,
                winrate_ema=0.5,
                pinned=False,
                recurrent=challenger.recurrent,
            )
            members = {
                member.opponent_id: member for member in self.sampler.frozen_members
            }
        extras = [
            member_id
            for member_id, member in members.items()
            if member_id not in desired and not member.pinned
        ]
        for member_id in extras:
            self.sampler.remove_frozen_member(member_id)
        if challenger is not None or extras:
            self.sampler.save_state(self.pool_state_path)

    def _validate_state(self) -> None:
        members = {member.opponent_id: member for member in self.sampler.frozen_members}
        if len(members) > self.config.max_opponents:
            raise ValueError("frozen pool exceeds configured total league capacity")
        unknown = (
            set(self.state.anchor_ids)
            | set(self.state.exploiter_ids)
            | ({self.state.champion_id} if self.state.champion_id else set())
            | (
                {self.state.challenger.opponent_id}
                if self.state.challenger is not None
                else set()
            )
        ) - members.keys()
        if unknown:
            raise ValueError(f"league promotion state references unknown members: {unknown}")
        referenced = (
            set(self.state.anchor_ids)
            | set(self.state.exploiter_ids)
            | ({self.state.champion_id} if self.state.champion_id else set())
            | (
                {self.state.challenger.opponent_id}
                if self.state.challenger is not None
                else set()
            )
        )
        retired_roles = {
            opponent_id for opponent_id in referenced if members[opponent_id].retired
        }
        if retired_roles:
            raise ValueError(
                f"league roles reference retired members: {sorted(retired_roles)}"
            )
        if any(not members[opponent_id].pinned for opponent_id in self.state.anchor_ids):
            raise ValueError("league anchor roles must reference pinned members")
        if self.config.initial_champion_id is not None and self.state.champion_id is None:
            raise ValueError("configured initial champion is absent from league state")
        if (
            self.config.initial_champion_id is not None
            and self.config.initial_champion_id not in members
        ):
            raise ValueError("initial_champion_id is absent from the frozen pool")

    def _verify_candidate(self, candidate: FrozenLeagueCandidate) -> None:
        fingerprint = fingerprint_checkpoint(candidate.checkpoint_path)
        if fingerprint.size_bytes != candidate.checkpoint_size_bytes:
            raise ValueError("frozen league candidate checkpoint size changed")
        if fingerprint.sha256 != candidate.checkpoint_sha256:
            raise ValueError("frozen league candidate checkpoint digest changed")

    def _probe_ready(self) -> bool:
        probe = self.state.probe
        if probe is None:
            return False
        target = self.config.probe_games_per_member
        return bool(probe.members) and all(
            stats.total.games >= target for stats in probe.members.values()
        )

    def _refresh_probe_schedule(self) -> None:
        probe = self.state.probe
        if probe is None:
            self.sampler.clear_frozen_probe_schedule()
            return
        target = self.config.probe_games_per_member
        labels = self.sampler.opponent_deck_labels
        member_weights: dict[str, float] = {}
        deck_weights: dict[str, dict[str, float]] = {}
        for member_id, stats in probe.members.items():
            member_weights[member_id] = float(max(0, target - stats.total.games))
            if labels:
                per_deck_target = max(1, target // len(labels))
                deck_weights[member_id] = {
                    label: float(
                        max(
                            0,
                            per_deck_target
                            - stats.by_opponent_deck.get(label, OutcomeTally()).games,
                        )
                    )
                    for label in labels
                }
        self.sampler.set_frozen_probe_schedule(
            member_weights=member_weights,
            deck_weights=deck_weights,
            fraction=self.config.probe_sampling_fraction,
        )

    def _champion_probability(self, probe: PromotionProbe) -> float | None:
        champion_id = self.state.champion_id
        if champion_id is None or champion_id not in probe.members:
            return None
        challenger_stats = probe.members[probe.challenger_id].total
        champion_stats = probe.members[champion_id].total
        rng = random.Random(_decision_seed(probe.challenger_version, "champion"))
        harder = 0
        for _ in range(self.config.posterior_samples):
            challenger_score = _sample_score_posterior(rng, challenger_stats)
            champion_score = _sample_score_posterior(rng, champion_stats)
            harder += int(challenger_score < champion_score)
        return harder / float(self.config.posterior_samples)

    def _best_coverage_replacement(
        self,
        probe: PromotionProbe,
        *,
        protect_champion: bool,
    ) -> tuple[str, float, float] | None:
        challenger_id = probe.challenger_id
        active_ids = [member_id for member_id in probe.members if member_id != challenger_id]
        protected = set(self.state.anchor_ids)
        protected.update(
            member_id
            for member_id, until in self.state.protected_until_promotion.items()
            if until >= self.state.promotion_count
        )
        if protect_champion and self.state.champion_id is not None:
            protected.add(self.state.champion_id)
        evictable = [member_id for member_id in active_ids if member_id not in protected]
        if not evictable:
            return None

        labels = self.sampler.opponent_deck_labels
        if not labels:
            return None
        weights = self.sampler.opponent_deck_base_probabilities()
        rng = random.Random(_decision_seed(probe.challenger_version, "coverage"))
        deltas: dict[str, list[float]] = {
            member_id: [] for member_id in evictable
        }
        for _ in range(self.config.posterior_samples):
            samples = {
                member_id: {
                    label: _sample_deck_score_posterior(
                        rng,
                        stats,
                        label,
                    )
                    for label in labels
                }
                for member_id, stats in probe.members.items()
            }
            baseline = _coverage_score(samples, active_ids, weights)
            for evicted_id in evictable:
                promoted_ids = [
                    member_id for member_id in active_ids if member_id != evicted_id
                ]
                promoted_ids.append(challenger_id)
                promoted = _coverage_score(samples, promoted_ids, weights)
                deltas[evicted_id].append(promoted - baseline)

        results = []
        for member_id, values in deltas.items():
            mean = sum(values) / float(len(values))
            probability = sum(value > 0.0 for value in values) / float(len(values))
            results.append((member_id, mean, probability))
        return max(results, key=lambda result: (result[1], result[2], result[0]))

    def _fallback_replacement(
        self,
        probe: PromotionProbe,
        *,
        protect_champion: bool = True,
    ) -> tuple[str, float, float] | None:
        protected = set(self.state.anchor_ids)
        if protect_champion and self.state.champion_id is not None:
            protected.add(self.state.champion_id)
        candidates = [
            (member_id, stats.total.mean)
            for member_id, stats in probe.members.items()
            if member_id != probe.challenger_id and member_id not in protected
        ]
        if not candidates:
            return None
        # A higher current-policy score identifies the easier fallback opponent.
        member_id, _ = max(candidates, key=lambda item: (item[1], item[0]))
        return (member_id, 0.0, 0.0)

    def _apply_decision(self, decision: PromotionDecision) -> None:
        challenger = self.state.challenger
        if challenger is None:
            raise RuntimeError("cannot apply a decision without a challenger")
        decisions = (*self.state.decisions, decision)[
            -MAX_PROMOTION_DECISION_HISTORY:
        ]
        if not decision.promoted:
            self.sampler.retire_frozen_member(challenger.opponent_id)
            self.state = self.state.model_copy(
                update={"challenger": None, "probe": None, "decisions": decisions}
            )
            self._persist_pool_and_state()
            self._refresh_probe_schedule()
            return

        evicted_id = decision.evicted_id
        if evicted_id is None:
            raise RuntimeError("promoted decision is missing an eviction")
        self.sampler.retire_frozen_member(evicted_id)
        promotion_count = self.state.promotion_count + 1
        champion_id: str | None
        if decision.route == "champion":
            champion_id = challenger.opponent_id
        else:
            champion_id = self.state.champion_id
        active_non_anchors = [
            member.opponent_id
            for member in self.sampler.frozen_members
            if not member.pinned
            and not member.retired
            and member.opponent_id != champion_id
        ]
        protected = {
            member_id: until
            for member_id, until in self.state.protected_until_promotion.items()
            if member_id != evicted_id and until >= promotion_count
        }
        protected[challenger.opponent_id] = promotion_count
        self.state = self.state.model_copy(
            update={
                "champion_id": champion_id,
                "exploiter_ids": tuple(active_non_anchors),
                "challenger": None,
                "probe": None,
                "last_promotion_version": challenger.policy_version,
                "promotion_count": promotion_count,
                "protected_until_promotion": protected,
                "decisions": decisions,
            }
        )
        self._persist_pool_and_state()
        self._refresh_probe_schedule()

    def _persist_pool_and_state(self) -> None:
        # Promotion state is the role source of truth. If the process dies
        # between these atomic replacements, startup reconciles the pool to it.
        self.save_state()
        self.sampler.save_state(self.pool_state_path)


def _sample_score_posterior(rng: random.Random, tally: OutcomeTally) -> float:
    return rng.betavariate(1.0 + tally.score, 1.0 + tally.games - tally.score)


def _sample_deck_score_posterior(
    rng: random.Random,
    stats: MemberProbeStats,
    label: str,
) -> float:
    tally = stats.by_opponent_deck.get(label, OutcomeTally())
    prior_strength = 4.0
    prior_mean = stats.total.mean
    return rng.betavariate(
        1.0 + prior_strength * prior_mean + tally.score,
        1.0 + prior_strength * (1.0 - prior_mean) + tally.games - tally.score,
    )


def _coverage_score(
    samples: Mapping[str, Mapping[str, float]],
    member_ids: list[str],
    weights: Mapping[str, float],
) -> float:
    return sum(
        weight
        * (1.0 - min(samples[member_id][label] for member_id in member_ids))
        for label, weight in weights.items()
    )


def _decision_seed(policy_version: int, domain: str) -> int:
    digest = hashlib.sha256(
        f"ptcg-rl/league-promotion/v1/{domain}/{policy_version}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _opponent_policy_version(opponent_id: str | None) -> int:
    if opponent_id is None:
        return 0
    marker = opponent_id.rsplit("_v", 1)
    if len(marker) != 2:
        return 0
    try:
        return int(marker[1])
    except ValueError:
        return 0


def write_league_promotion_state(
    path: Path,
    state: LeaguePromotionState,
) -> None:
    """Atomically persist one validated promotion-role state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with pending.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        with suppress(OSError):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        pending.unlink(missing_ok=True)
