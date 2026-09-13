"""Statistical evidence and alert derivation for the training workbench."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ptcg_rl.dashboard.models import MatchupRow, OutcomeStats
from ptcg_rl.dashboard.workbench_models import PosteriorStats, WorkbenchAlert

_POSTERIOR_SAMPLES = 20_000


def build_posterior(
    rows: Sequence[MatchupRow],
    *,
    fallback: OutcomeStats,
    seed_key: str,
    exact_available: bool,
) -> PosteriorStats:
    """Build a deterministic, equally seat-weighted W/D/L posterior."""
    if not exact_available:
        return PosteriorStats(
            observed=fallback,
            seat_games={},
            seat_balanced=False,
            evidence_state="unavailable",
        )
    seat_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    for row in rows:
        counts = seat_counts.setdefault(row.candidate_seat, Counter())
        counts["wins"] += row.outcomes.wins
        counts["draws"] += row.outcomes.draws
        counts["losses"] += row.outcomes.losses
    observed = _outcome_stats(
        sum(counts["wins"] for counts in seat_counts.values()),
        sum(counts["draws"] for counts in seat_counts.values()),
        sum(counts["losses"] for counts in seat_counts.values()),
    )
    seat_games = {seat: sum(counts.values()) for seat, counts in seat_counts.items()}
    if any(seat_games.get(seat, 0) == 0 for seat in (0, 1)):
        return PosteriorStats(
            observed=observed if observed.games else fallback,
            seat_games=seat_games,
            seat_balanced=False,
            evidence_state="missing_seat",
        )
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    score_samples: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for seat in (0, 1):
        counts = seat_counts[seat]
        samples = rng.dirichlet(
            np.asarray(
                (
                    counts["wins"] + 0.5,
                    counts["draws"] + 0.5,
                    counts["losses"] + 0.5,
                ),
                dtype=np.float64,
            ),
            size=_POSTERIOR_SAMPLES,
        )
        score_samples.append(samples[:, 0] + 0.5 * samples[:, 1])
    balanced = 0.5 * (score_samples[0] + score_samples[1])
    return PosteriorStats(
        observed=observed,
        seat_games=seat_games,
        seat_balanced=True,
        posterior_mean=float(np.mean(balanced)),
        credible_low=float(np.quantile(balanced, 0.025)),
        credible_high=float(np.quantile(balanced, 0.975)),
        probability_above_half=float(np.mean(balanced > 0.5)),
        evidence_state="ready",
    )


def build_alerts(
    status: Mapping[str, object],
    data_state: str,
    warnings: Iterable[str],
) -> tuple[WorkbenchAlert, ...]:
    """Derive a small actionable alert set from current health evidence."""
    alerts: list[WorkbenchAlert] = []
    if data_state != "ready":
        alerts.append(
            WorkbenchAlert(
                code="data_freshness",
                severity="critical" if data_state == "invalid" else "warning",
                title="训练镜像不是 ready",
                detail=f"当前数据状态为 {data_state}。",
            )
        )
    for index, warning in enumerate(warnings):
        alerts.append(
            WorkbenchAlert(
                code=f"history_warning_{index}",
                severity="warning",
                title="历史证据不完整",
                detail=warning,
            )
        )
    learner = _mapping(status.get("learner_status"))
    latest = _mapping(learner.get("latest_update"))
    timing = _mapping(learner.get("latest_timing"))
    metric_writer = _mapping(learner.get("learner_metric_history"))
    if bool(latest.get("target_kl_exceeded")):
        alerts.append(
            WorkbenchAlert(
                code="target_kl_exceeded",
                severity="warning",
                title="PPO target KL 已触发",
                detail="最近一次 learner update 超过 target KL。",
            )
        )
    stale = _optional_int(latest.get("fragments_stale")) or 0
    if stale > 0:
        alerts.append(
            WorkbenchAlert(
                code="stale_fragments",
                severity="warning",
                title="存在 stale fragments",
                detail=f"最近一次 learner update 排除了 {stale:,} 个 fragments。",
            )
        )
    retries = _optional_int(timing.get("cuda_checkpoint_allocation_retries")) or 0
    ooms = _optional_int(timing.get("cuda_checkpoint_ooms")) or 0
    if retries or ooms:
        alerts.append(
            WorkbenchAlert(
                code="cuda_allocator",
                severity="critical" if ooms else "warning",
                title="CUDA allocator 异常",
                detail=f"allocation retries={retries}，OOM={ooms}。",
            )
        )
    dropped = _optional_int(metric_writer.get("dropped_records")) or 0
    writer_error = _optional_text(metric_writer.get("last_error"))
    if dropped or writer_error:
        alerts.append(
            WorkbenchAlert(
                code="learner_metric_history",
                severity="warning",
                title="Learner 指标历史存在缺口",
                detail=(
                    f"dropped={dropped}" + (f"，{writer_error}" if writer_error else "")
                ),
            )
        )
    if not alerts:
        alerts.append(
            WorkbenchAlert(
                code="healthy",
                severity="info",
                title="未发现需要介入的异常",
                detail="数据新鲜，当前已观测门禁均正常。",
            )
        )
    return tuple(alerts)


def rows_outcomes(rows: Sequence[MatchupRow]) -> OutcomeStats:
    """Sum outcomes across exact matchup cells."""
    return _outcome_stats(
        sum(row.outcomes.wins for row in rows),
        sum(row.outcomes.draws for row in rows),
        sum(row.outcomes.losses for row in rows),
    )


def _outcome_stats(wins: int, draws: int, losses: int) -> OutcomeStats:
    games = wins + draws + losses
    return OutcomeStats(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=None if games == 0 else wins / games,
        score_rate=None if games == 0 else (wins + 0.5 * draws) / games,
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return int(value) if isinstance(value, (int, float)) else None
