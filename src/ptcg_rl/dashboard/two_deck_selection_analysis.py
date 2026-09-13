"""Statistical composition for the two-deck dashboard recommendation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from ptcg_rl.dashboard.public_environment_models import (
    EnvironmentWindow,
    PublicEnvironmentPayload,
)
from ptcg_rl.dashboard.training_deck_strength_models import (
    TrainingDeckMatrixPayload,
    TrainingDeckStrengthPayload,
    TrainingEvidenceRange,
)
from ptcg_rl.dashboard.two_deck_selection_evidence import (
    EligibleCandidate,
    build_meta_evidence,
    build_training_outcomes,
    seat_meta_weights,
    seat_opponent_id,
    select_eligible_candidates,
)
from ptcg_rl.dashboard.two_deck_selection_models import (
    TwoDeckCandidateEvidence,
    TwoDeckPairRecommendation,
    TwoDeckSelectionPayload,
    TwoDeckSelectionQuality,
)
from ptcg_rl.dashboard.workbench_models import CheckpointInfo
from ptcg_rl.evaluation.posterior import (
    BundlePortfolioSummary,
    BundlePosteriorSummary,
    MatchupOutcome,
    PosteriorEvaluationConfig,
    evaluate_bundle_posteriors,
)

_POSTERIOR_SAMPLES = 4_096
_MAX_RETURNED_PAIRS = 20


def build_two_deck_selection(
    *,
    training: TrainingDeckStrengthPayload,
    matrices: Sequence[TrainingDeckMatrixPayload],
    public: PublicEnvironmentPayload,
    checkpoint: CheckpointInfo | None,
    training_range: TrainingEvidenceRange,
    window_days: EnvironmentWindow,
    seed_material: str,
) -> TwoDeckSelectionPayload:
    """Build a portfolio proxy from synthetic or repository-backed payloads."""
    selected = training.ranges[training_range]
    warnings: list[str] = [
        "该结果是组合强度代理，不是 Kaggle rating μ、最终排名预测或提交授权。",
        "Kaggle Daily 决定共同 meta 场景；训练 matchup 按先后手 50/50 估计当前驾驶表现。",
        "两套牌先分别完成整体现状加权，再取较高分；不会按已知对手逐局换牌。",
    ]
    matrix_by_seat = {
        matrix.candidate_seat: matrix
        for matrix in matrices
        if matrix.candidate_seat in (0, 1)
    }
    matrices_complete = len(matrices) == 2 and set(matrix_by_seat) == {0, 1}
    identities_match = bool(
        training.run_id == public.run_id
        and matrices_complete
        and all(matrix.run_id == training.run_id for matrix in matrices)
    )
    if not identities_match:
        warnings.append("训练、双 seat matchup 与 Kaggle 快照身份不完整或不一致")
    if checkpoint is None:
        warnings.append("未选择 immutable checkpoint pair")
    elif not (
        checkpoint.run_id == training.run_id
        and training.checkpoint_version == checkpoint.version
        and training.checkpoint_pair_manifest_sha256 == checkpoint.pair_manifest_sha256
        and all(matrix.checkpoint_version == checkpoint.version for matrix in matrices)
    ):
        warnings.append("训练证据与所选 immutable checkpoint pair 身份不一致")
        identities_match = False
    if not all(matrix.range == training_range for matrix in matrices):
        warnings.append("训练双 seat matchup range 与请求不一致")
        identities_match = False
    if training_range != "checkpoint":
        warnings.append(
            "所选训练范围可能跨策略版本；checkpoint 仅用于 exact route 兼容门禁"
        )
    if public.window_days != window_days or public.snapshot_fingerprint is None:
        warnings.append("Kaggle Daily 窗口或 immutable snapshot identity 不一致")
        identities_match = False
    if not selected.metadata.available:
        warnings.append(
            selected.metadata.unavailable_reason or "所选训练证据范围不可用"
        )
    if not public.available:
        warnings.append(public.unavailable_reason or "Kaggle Daily 快照不可用")
    if public.score_semantics.submission_data_used:
        warnings.append("当前快照含非预期的 submission 数据，已拒绝组合计算")
    public_ready = public.available and not public.score_semantics.submission_data_used
    meta = build_meta_evidence(public)
    if meta.warning is not None:
        warnings.append(meta.warning)
    elif meta.modeled_prior_mass > 0.0:
        warnings.append(
            f"保留 {meta.modeled_prior_mass:.1%} 未建模 Kaggle meta 质量"
            f"（{meta.unexpanded_explicit_mass:.1%} 未展开 exact + "
            f"{meta.rare_unknown_mass:.1%} rare/unknown tail）"
        )

    candidates, gate_warnings = select_eligible_candidates(
        selected.standings,
        public=public,
    )
    warnings.extend(gate_warnings)
    training_ready = bool(
        selected.metadata.available
        and matrices_complete
        and all(matrix.available for matrix in matrices)
    )
    outcomes: tuple[MatchupOutcome, ...] = ()
    observed_cells: dict[tuple[str, str, int], tuple[int, int, int]] = {}
    if matrices_complete and meta.weights:
        outcomes, observed_cells = build_training_outcomes(
            matrix_by_seat,
            candidates=candidates,
            meta_decks=public.meta_decks,
            meta_weights=meta.weights,
        )
        mapped_candidates = {
            candidate_digest
            for candidate_digest, opponent_digest, _seat in observed_cells
            if candidate_digest != opponent_digest
        }
        unmapped_count = sum(
            candidate.digest not in mapped_candidates for candidate in candidates
        )
        if unmapped_count:
            warnings.append(
                "训练 matchup 无法按 full digest 映射至已展开 Daily meta："
                f"排除 {unmapped_count} 套"
            )
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.digest in mapped_candidates
        )
        candidate_digests = {candidate.digest for candidate in candidates}
        outcomes = tuple(
            outcome for outcome in outcomes if outcome.candidate_id in candidate_digests
        )
    ready = bool(
        checkpoint is not None
        and identities_match
        and training_ready
        and public_ready
        and meta.weights
        and meta.warning is None
        and outcomes
        and len(candidates) >= 2
    )
    if len(candidates) < 2:
        warnings.append("通过 exact route、双 seat 与跨源身份门禁的卡组不足两套")
    quality = TwoDeckSelectionQuality(
        ready=ready,
        training_available=training_ready,
        public_environment_available=public_ready,
        active_candidates=len(selected.standings),
        eligible_candidates=len(candidates),
        candidate_pairs=len(candidates) * max(0, len(candidates) - 1) // 2,
        known_meta_mass=sum(meta.weights.values()),
        unknown_meta_mass=meta.modeled_prior_mass,
        unexpanded_explicit_meta_mass=meta.unexpanded_explicit_mass,
        rare_unknown_meta_mass=meta.rare_unknown_mass,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    if not ready:
        return _payload(
            training=training,
            public=public,
            checkpoint=checkpoint,
            training_range=training_range,
            window_days=window_days,
            quality=quality,
        )

    resolved_seat_meta_weights = seat_meta_weights(meta.weights)
    fixed_matchup_scores = {
        (candidate.digest, seat_opponent_id(candidate.digest, seat)): 0.5
        for candidate in candidates
        if candidate.digest in meta.weights
        for seat in (0, 1)
    }
    posterior = evaluate_bundle_posteriors(
        candidate_ids=tuple(candidate.digest for candidate in candidates),
        outcomes=outcomes,
        meta_weights=resolved_seat_meta_weights,
        unknown_mass=meta.modeled_prior_mass,
        config=PosteriorEvaluationConfig(
            sample_count=_POSTERIOR_SAMPLES,
            sample_batch_size=1_024,
            seed=_stable_seed(seed_material),
        ),
        fixed_matchup_scores=fixed_matchup_scores,
        include_portfolios=True,
    )
    candidate_by_digest = {candidate.digest: candidate for candidate in candidates}
    proxy_by_digest = {row.candidate_id: row for row in posterior.candidates}
    candidate_rows = tuple(
        _candidate_evidence(candidate, proxy_by_digest[candidate.digest])
        for candidate in sorted(
            candidates,
            key=lambda item: proxy_by_digest[item.digest].rank,
        )
    )
    pair_rows = tuple(
        _pair_recommendation(
            row,
            candidates=candidate_by_digest,
            observed_cells=observed_cells,
            meta_weights=meta.weights,
        )
        for row in posterior.portfolios
    )
    recommendation = pair_rows[0] if pair_rows else None
    result_quality = quality.model_copy(update={"ready": recommendation is not None})
    if recommendation is not None:
        prior_by_digest = {
            item.deck_digest: item.prior_only_meta_mass for item in candidate_rows
        }
        highest_prior = max(
            prior_by_digest[recommendation.deck_a_digest],
            prior_by_digest[recommendation.deck_b_digest],
        )
        if highest_prior > 0.0:
            warnings.append(
                f"推荐组合仍保留最高 {highest_prior:.1%} 的 prior-only meta 质量"
            )
    result_quality = result_quality.model_copy(
        update={"warnings": tuple(dict.fromkeys(warnings))}
    )
    return _payload(
        training=training,
        public=public,
        checkpoint=checkpoint,
        training_range=training_range,
        window_days=window_days,
        quality=result_quality,
        recommendation=recommendation,
        pairs=pair_rows[:_MAX_RETURNED_PAIRS],
        candidates=candidate_rows,
    )


def _candidate_evidence(
    candidate: EligibleCandidate,
    proxy: BundlePosteriorSummary,
) -> TwoDeckCandidateEvidence:
    standing = candidate.standing
    posterior = standing.posterior
    assert posterior.posterior_mean is not None
    assert posterior.credible_low is not None
    assert posterior.credible_high is not None
    return TwoDeckCandidateEvidence(
        deck_label=standing.deck_label,
        deck_hash=standing.deck_hash,
        deck_digest=candidate.digest,
        display_name=standing.display_name,
        route_compatible=True,
        training_games=posterior.observed.games,
        training_first_games=standing.seat_scores[0].games,
        training_second_games=standing.seat_scores[1].games,
        training_mean=posterior.posterior_mean,
        training_credible_low=posterior.credible_low,
        training_credible_high=posterior.credible_high,
        public_evidence_status=candidate.public.evidence_status,
        public_games=candidate.public.valid_games,
        public_deploy_mean=candidate.public.deploy_mean,
        public_deploy_lcb=candidate.public.deploy_lcb,
        proxy_rank=proxy.rank,
        proxy_mean=proxy.deploy_mean,
        proxy_credible_low=proxy.deploy_credible_low,
        proxy_credible_high=proxy.deploy_credible_high,
        proxy_lcb=proxy.deploy_lcb,
        observed_meta_mass=proxy.observed_meta_mass,
        prior_only_meta_mass=proxy.prior_only_meta_mass,
    )


def _pair_recommendation(
    portfolio: BundlePortfolioSummary,
    *,
    candidates: Mapping[str, EligibleCandidate],
    observed_cells: Mapping[tuple[str, str, int], tuple[int, int, int]],
    meta_weights: Mapping[str, float],
) -> TwoDeckPairRecommendation:
    left_digest, right_digest = portfolio.candidate_ids
    left = candidates[left_digest].standing
    right = candidates[right_digest].standing
    shared_mass = 0.0
    weak_mass = 0.0
    for opponent_digest, weight in meta_weights.items():
        for seat in (0, 1):
            left_counts = observed_cells.get((left_digest, opponent_digest, seat))
            right_counts = observed_cells.get((right_digest, opponent_digest, seat))
            left_mean = (
                0.5
                if left_digest == opponent_digest
                else None
                if left_counts is None
                else _cell_posterior_mean(left_counts)
            )
            right_mean = (
                0.5
                if right_digest == opponent_digest
                else None
                if right_counts is None
                else _cell_posterior_mean(right_counts)
            )
            if left_mean is None or right_mean is None:
                continue
            seat_weight = weight * 0.5
            shared_mass += seat_weight
            if left_mean < 0.5 and right_mean < 0.5:
                weak_mass += seat_weight
    return TwoDeckPairRecommendation(
        rank=portfolio.rank,
        deck_a_label=left.deck_label,
        deck_a_hash=left.deck_hash,
        deck_a_digest=left_digest,
        deck_a_display_name=left.display_name,
        deck_b_label=right.deck_label,
        deck_b_hash=right.deck_hash,
        deck_b_digest=right_digest,
        deck_b_display_name=right.display_name,
        expected_best_score=portfolio.expected_best_score,
        credible_low=portfolio.credible_low,
        credible_high=portfolio.credible_high,
        best_score_lcb=portfolio.best_score_lcb,
        probability_at_least_one_above_even=(
            portfolio.probability_at_least_one_above_even
        ),
        joint_downside_probability=portfolio.joint_downside_probability,
        diversification_gain=portfolio.diversification_gain,
        score_correlation=portfolio.score_correlation,
        expected_regret=portfolio.expected_regret,
        common_weak_meta_mass=min(1.0, weak_mass),
        shared_observed_meta_mass=min(1.0, shared_mass),
    )


def _cell_posterior_mean(counts: tuple[int, int, int]) -> float:
    wins, draws, losses = counts
    return (1.0 + wins + 0.5 * draws) / (2.0 + wins + draws + losses)


def _payload(
    *,
    training: TrainingDeckStrengthPayload,
    public: PublicEnvironmentPayload,
    checkpoint: CheckpointInfo | None,
    training_range: TrainingEvidenceRange,
    window_days: EnvironmentWindow,
    quality: TwoDeckSelectionQuality,
    recommendation: TwoDeckPairRecommendation | None = None,
    pairs: tuple[TwoDeckPairRecommendation, ...] = (),
    candidates: tuple[TwoDeckCandidateEvidence, ...] = (),
) -> TwoDeckSelectionPayload:
    metadata = training.ranges[training_range].metadata
    return TwoDeckSelectionPayload(
        available=quality.ready,
        run_id=training.run_id,
        checkpoint_version=None if checkpoint is None else checkpoint.version,
        checkpoint_pair_manifest_sha256=(
            None if checkpoint is None else checkpoint.pair_manifest_sha256
        ),
        training_range=training_range,
        training_started_at_utc=metadata.started_at_utc,
        training_ended_at_utc=metadata.ended_at_utc,
        public_window_days=window_days,
        public_as_of_date=public.as_of_date,
        public_snapshot_fingerprint=public.snapshot_fingerprint,
        quality=quality,
        recommendation=recommendation,
        pairs=pairs,
        candidates=candidates,
        warnings=quality.warnings,
    )


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


__all__ = ["build_two_deck_selection"]
