"""Identity and evidence preparation for the two-deck portfolio proxy."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ptcg_rl.dashboard.public_environment_models import (
    EnvironmentMetaDeck,
    EnvironmentRosterStanding,
    PublicEnvironmentPayload,
)
from ptcg_rl.dashboard.training_deck_strength_models import (
    TrainingDeckMatrixPayload,
    TrainingDeckStanding,
)
from ptcg_rl.evaluation.posterior import MatchupOutcome


@dataclass(frozen=True, slots=True)
class EligibleCandidate:
    """One exact deck that passes route, training, and public evidence gates."""

    standing: TrainingDeckStanding
    public: EnvironmentRosterStanding

    @property
    def digest(self) -> str:
        """Return the full internal routing identity."""
        assert self.standing.deck_digest is not None
        return self.standing.deck_digest


@dataclass(frozen=True, slots=True)
class MetaEvidence:
    """Expanded target meta plus explicit residual-mass diagnostics."""

    weights: dict[str, float]
    modeled_prior_mass: float
    unexpanded_explicit_mass: float
    rare_unknown_mass: float
    warning: str | None


def select_eligible_candidates(
    standings: Sequence[TrainingDeckStanding],
    *,
    public: PublicEnvironmentPayload,
) -> tuple[tuple[EligibleCandidate, ...], tuple[str, ...]]:
    """Apply exact identity and evidence gates without inventing deck hashes."""
    public_rows_by_digest: defaultdict[str, list[EnvironmentRosterStanding]] = (
        defaultdict(list)
    )
    for row in public.roster_standings:
        public_rows_by_digest[row.deck_digest].append(row)
    public_by_digest = {
        digest: rows[0]
        for digest, rows in public_rows_by_digest.items()
        if len(rows) == 1
    }
    standing_digests_by_hash: defaultdict[str, set[str]] = defaultdict(set)
    for standing in standings:
        if standing.deck_digest is not None:
            standing_digests_by_hash[standing.deck_hash].add(standing.deck_digest)
    ambiguous_hashes = {
        deck_hash
        for deck_hash, digests in standing_digests_by_hash.items()
        if len(digests) > 1
    }
    output: list[EligibleCandidate] = []
    reasons: defaultdict[str, int] = defaultdict(int)
    seen_digests: set[str] = set()
    for standing in standings:
        posterior = standing.posterior
        if standing.deck_digest is None:
            reasons["exact digest 无法解析"] += 1
            continue
        if standing.deck_hash in ambiguous_hashes:
            reasons["权威 deck_hash 不唯一"] += 1
            continue
        if standing.route_compatible is not True:
            reasons["checkpoint exact route 不兼容"] += 1
            continue
        if (
            posterior.evidence_state != "ready"
            or posterior.posterior_mean is None
            or posterior.credible_low is None
            or posterior.credible_high is None
            or standing.seat_scores[0].games <= 0
            or standing.seat_scores[1].games <= 0
        ):
            reasons["训练证据或双 seat 不完整"] += 1
            continue
        if len(public_rows_by_digest[standing.deck_digest]) > 1:
            reasons["Kaggle roster full digest 重复"] += 1
            continue
        public_row = public_by_digest.get(standing.deck_digest)
        if public_row is None:
            reasons["Kaggle roster 缺少相同 full digest"] += 1
            continue
        if public_row.deck_hash != standing.deck_hash:
            reasons["跨源 deck_hash 冲突"] += 1
            continue
        if (
            public_row.evidence_status != "eligible"
            or public_row.rank is None
            or public_row.valid_games <= 0
            or public_row.first_games <= 0
            or public_row.second_games <= 0
        ):
            reasons["Kaggle Daily 公开证据未通过 eligibility 门禁"] += 1
            continue
        if standing.deck_digest in seen_digests:
            reasons["重复 full deck digest"] += 1
            continue
        seen_digests.add(standing.deck_digest)
        output.append(EligibleCandidate(standing=standing, public=public_row))
    warnings = tuple(
        f"{reason}：排除 {count} 套" for reason, count in sorted(reasons.items())
    )
    return tuple(output), warnings


def build_meta_evidence(public: PublicEnvironmentPayload) -> MetaEvidence:
    """Keep expanded, omitted-exact, and rare public meta mass distinct."""
    if not public.available:
        return MetaEvidence(
            weights={},
            modeled_prior_mass=1.0,
            unexpanded_explicit_mass=0.0,
            rare_unknown_mass=1.0,
            warning=None,
        )
    weights: dict[str, float] = {}
    for row in public.meta_decks:
        if row.share <= 0.0:
            continue
        if row.deck_digest in weights:
            return _invalid_meta("Kaggle meta 含重复 full deck digest")
        weights[row.deck_digest] = row.share
    expanded_mass = sum(weights.values())
    if expanded_mass > 1.0 + 1e-6:
        return _invalid_meta("Kaggle meta 质量超过 100%")
    if expanded_mass > 1.0:
        weights = {digest: weight / expanded_mass for digest, weight in weights.items()}
        expanded_mass = 1.0
    explicit_mass = public.quality.explicit_meta_mass
    rare_unknown_mass = public.quality.unknown_tail_mass
    if not math.isclose(
        explicit_mass + rare_unknown_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return _invalid_meta("Kaggle explicit meta 与 rare/unknown tail 质量不闭合")
    if expanded_mass > explicit_mass + 1e-6:
        return _invalid_meta("展开的 Kaggle meta 质量超过 explicit meta 质量")
    unexpanded_explicit_mass = max(0.0, explicit_mass - expanded_mass)
    modeled_prior_mass = max(0.0, 1.0 - expanded_mass)
    if not math.isclose(
        unexpanded_explicit_mass + rare_unknown_mass,
        modeled_prior_mass,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return _invalid_meta("Kaggle 未展开 exact 与 rare/unknown 质量不闭合")
    return MetaEvidence(
        weights=weights,
        modeled_prior_mass=modeled_prior_mass,
        unexpanded_explicit_mass=unexpanded_explicit_mass,
        rare_unknown_mass=rare_unknown_mass,
        warning=None,
    )


def build_training_outcomes(
    matrices: Mapping[int, TrainingDeckMatrixPayload],
    *,
    candidates: Sequence[EligibleCandidate],
    meta_decks: Sequence[EnvironmentMetaDeck],
    meta_weights: Mapping[str, float],
) -> tuple[
    tuple[MatchupOutcome, ...],
    dict[tuple[str, str, int], tuple[int, int, int]],
]:
    """Resolve exact public opponents and preserve candidate seat strata."""
    candidate_by_label = {
        candidate.standing.deck_label: candidate.digest for candidate in candidates
    }
    candidate_hash_by_label = {
        candidate.standing.deck_label: candidate.standing.deck_hash
        for candidate in candidates
    }
    opponent_digests_by_label: defaultdict[str, set[str]] = defaultdict(set)
    opponent_hashes_by_label: defaultdict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        label = candidate.standing.deck_label
        opponent_digests_by_label[label].add(candidate.digest)
        opponent_hashes_by_label[label].add(candidate.standing.deck_hash)
    for row in meta_decks:
        if row.deck_label is None or row.deck_digest not in meta_weights:
            continue
        opponent_digests_by_label[row.deck_label].add(row.deck_digest)
        if row.deck_hash is not None:
            opponent_hashes_by_label[row.deck_label].add(row.deck_hash)
    opponent_by_label = {
        label: (
            next(iter(digests)),
            next(iter(opponent_hashes_by_label[label]))
            if len(opponent_hashes_by_label[label]) == 1
            else None,
        )
        for label, digests in opponent_digests_by_label.items()
        if len(digests) == 1 and len(opponent_hashes_by_label[label]) <= 1
    }
    aggregated: defaultdict[tuple[str, str, int], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    for seat, matrix in matrices.items():
        for cell in matrix.cells:
            candidate_digest = candidate_by_label.get(cell.candidate_deck_label)
            if (
                candidate_digest is None
                or cell.candidate_deck_hash
                != candidate_hash_by_label.get(cell.candidate_deck_label)
            ):
                continue
            opponent_identity = opponent_by_label.get(cell.opponent_deck_label)
            if opponent_identity is None:
                continue
            opponent_digest, expected_hash = opponent_identity
            if expected_hash is not None and cell.opponent_deck_hash != expected_hash:
                continue
            if opponent_digest not in meta_weights:
                continue
            observed = cell.posterior.observed
            if observed.games <= 0:
                continue
            counts = aggregated[(candidate_digest, opponent_digest, seat)]
            counts[0] += observed.wins
            counts[1] += observed.draws
            counts[2] += observed.losses
    outcomes = tuple(
        MatchupOutcome(
            candidate_id=candidate_digest,
            opponent_id=seat_opponent_id(opponent_digest, seat),
            wins=counts[0],
            draws=counts[1],
            losses=counts[2],
        )
        for (
            candidate_digest,
            opponent_digest,
            seat,
        ), counts in sorted(aggregated.items())
    )
    observed_cells = {
        key: (counts[0], counts[1], counts[2]) for key, counts in aggregated.items()
    }
    return outcomes, observed_cells


def seat_meta_weights(meta_weights: Mapping[str, float]) -> dict[str, float]:
    """Split each exact public deck's mass evenly across candidate seats."""
    return {
        seat_opponent_id(digest, seat): weight * 0.5
        for digest, weight in meta_weights.items()
        for seat in (0, 1)
    }


def seat_opponent_id(deck_digest: str, seat: int) -> str:
    """Build an internal context identity without changing the deck digest."""
    return f"{deck_digest}#candidate-seat={seat}"


def _invalid_meta(reason: str) -> MetaEvidence:
    return MetaEvidence(
        weights={},
        modeled_prior_mass=1.0,
        unexpanded_explicit_mass=0.0,
        rare_unknown_mass=1.0,
        warning=reason,
    )


__all__ = [
    "EligibleCandidate",
    "MetaEvidence",
    "build_meta_evidence",
    "build_training_outcomes",
    "seat_meta_weights",
    "seat_opponent_id",
    "select_eligible_candidates",
]
