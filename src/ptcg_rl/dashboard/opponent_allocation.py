"""Read-only aggregation for supported opponent-allocation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import orjson

from ptcg_rl.dashboard.opponent_allocation_models import (
    AllocationMode,
    AllocationSort,
    OpponentAllocationArtifactView,
    OpponentAllocationCandidateView,
    OpponentAllocationExecutionView,
    OpponentAllocationMatchupPage,
    OpponentAllocationMatchupView,
    OpponentAllocationSummaryPayload,
)
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.rl.opponent_pool.adaptive import PortfolioName
from ptcg_rl.rl.opponent_pool.adaptive_report import (
    AdaptiveMatchupAllocationReport,
    AdaptiveOpponentAllocationReport,
)
from ptcg_rl.rl.opponent_pool.role_budget import RoleBudgetName
from ptcg_rl.rl.opponent_pool.role_budget_report import (
    RoleBudgetMatchupReport,
    RoleBudgetOpponentAllocationReport,
)
from ptcg_rl.rl.stateless_curriculum_codec import decode_compact_mapping

_PathSignature = tuple[int, int]
_AllocationReport = (
    AdaptiveOpponentAllocationReport | RoleBudgetOpponentAllocationReport
)
_MatchupReport = AdaptiveMatchupAllocationReport | RoleBudgetMatchupReport


class OpponentAllocationService:
    """Expose compact summaries and bounded pages without touching checkpoints."""

    def __init__(self, repository: DashboardRepository) -> None:
        self.repository = repository
        self._report_cache: dict[
            str,
            tuple[_PathSignature, str, _AllocationReport],
        ] = {}
        self._deck_cache: dict[
            str,
            tuple[_PathSignature | None, dict[str, str]],
        ] = {}

    def summary(self, run: str) -> OpponentAllocationSummaryPayload:
        """Return current candidate, portfolio, and artifact allocation."""
        run_id = self.repository.resolve_run_id(run)
        execution = self._execution(run_id)
        loaded = self._load(run_id)
        if loaded is None:
            return OpponentAllocationSummaryPayload(
                available=False,
                run_id=run_id,
                detail="等待首个对手分配窗口结算",
                execution=execution,
            )
        recorded_at_utc, report = loaded
        allocation_mode = _allocation_mode(report)
        candidates = tuple(
            OpponentAllocationCandidateView.model_validate(
                {
                    **item.model_dump(mode="python"),
                    **self._display_fields(
                        run_id,
                        item.candidate_deck_digest,
                        prefix="candidate",
                    ),
                }
            )
            for item in report.candidates
        )
        return OpponentAllocationSummaryPayload(
            available=True,
            run_id=run_id,
            allocation_mode=allocation_mode,
            recorded_at_utc=recorded_at_utc,
            window_sequence=report.window_sequence,
            plan_id=report.plan_id,
            target_fingerprint=report.target_fingerprint,
            predecessor_state_fingerprint=report.predecessor_state_fingerprint,
            committed_state_fingerprint=report.committed_state_fingerprint,
            revision_fingerprint=report.revision_fingerprint,
            evidence_cells=report.evidence_cells,
            low_evidence_cells=report.low_evidence_cells,
            matchup_count=len(report.matchups),
            execution=execution,
            candidates=candidates,
            portfolios=(
                report.portfolios
                if isinstance(report, AdaptiveOpponentAllocationReport)
                else ()
            ),
            roles=(
                report.roles
                if isinstance(report, RoleBudgetOpponentAllocationReport)
                else ()
            ),
            artifacts=tuple(
                OpponentAllocationArtifactView.model_validate(
                    item.model_dump(mode="python")
                )
                for item in report.artifacts
            ),
        )

    def matchups(
        self,
        run: str,
        *,
        offset: int,
        limit: int,
        sort: AllocationSort,
        candidate_deck_digest: str | None,
        artifact_id: str | None,
        portfolio: PortfolioName | None,
        role: RoleBudgetName | None,
        candidate_seat: int | None,
    ) -> OpponentAllocationMatchupPage:
        """Filter and page exact cells in the dashboard process."""
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("opponent allocation page bounds are invalid")
        if candidate_seat not in {None, 0, 1}:
            raise ValueError("opponent allocation candidate seat is invalid")
        run_id = self.repository.resolve_run_id(run)
        loaded = self._load(run_id)
        if loaded is None:
            return OpponentAllocationMatchupPage(
                available=False,
                run_id=run_id,
                total=0,
                offset=offset,
                limit=limit,
                sort=sort,
                candidate_deck_digest=candidate_deck_digest,
                artifact_id=artifact_id,
                portfolio=portfolio,
                role=role,
                candidate_seat=cast(Any, candidate_seat),
            )
        _recorded_at_utc, report = loaded
        allocation_mode = _allocation_mode(report)
        if allocation_mode == "role_budget" and sort in {"debt", "change"}:
            raise ValueError(
                f"allocation sort {sort} is unavailable for role-budget reports"
            )
        selected = [
            row
            for row in report.matchups
            if (
                candidate_deck_digest is None
                or row.candidate_deck_digest == candidate_deck_digest
            )
            and (artifact_id is None or row.artifact_id == artifact_id)
            and (
                portfolio is None
                or (
                    isinstance(row, AdaptiveMatchupAllocationReport)
                    and row.portfolio == portfolio
                )
            )
            and (
                role is None
                or (isinstance(row, RoleBudgetMatchupReport) and row.role == role)
            )
            and (candidate_seat is None or row.candidate_seat == candidate_seat)
        ]
        selected.sort(key=lambda row: _sort_key(row, sort))
        page = selected[offset : offset + limit]
        artifacts = {
            item.artifact_id: OpponentAllocationArtifactView.model_validate(
                item.model_dump(mode="python")
            )
            for item in report.artifacts
        }
        rows = tuple(
            self._matchup_view(run_id, row, artifact=artifacts[row.artifact_id])
            for row in page
        )
        return OpponentAllocationMatchupPage(
            available=True,
            run_id=run_id,
            allocation_mode=allocation_mode,
            window_sequence=report.window_sequence,
            target_fingerprint=report.target_fingerprint,
            total=len(selected),
            offset=offset,
            limit=limit,
            sort=sort,
            candidate_deck_digest=candidate_deck_digest,
            artifact_id=artifact_id,
            portfolio=portfolio,
            role=role,
            candidate_seat=cast(Any, candidate_seat),
            rows=rows,
        )

    def _matchup_view(
        self,
        run_id: str,
        row: _MatchupReport,
        *,
        artifact: OpponentAllocationArtifactView,
    ) -> OpponentAllocationMatchupView:
        artifact_fields: dict[str, object] = {}
        if isinstance(row, RoleBudgetMatchupReport):
            artifact_fields = {
                "source_fingerprint": artifact.source_fingerprint,
                "source_policy_version": artifact.source_policy_version,
                "stratum": artifact.stratum,
            }
        return OpponentAllocationMatchupView.model_validate(
            {
                **row.model_dump(mode="python"),
                **artifact_fields,
                **self._display_fields(
                    run_id,
                    row.candidate_deck_digest,
                    prefix="candidate",
                ),
                **self._display_fields(
                    run_id,
                    row.opponent_deck_digest,
                    prefix="opponent",
                ),
            }
        )

    def _display_fields(
        self,
        run_id: str,
        deck_digest: str,
        *,
        prefix: str,
    ) -> dict[str, str | None]:
        label = self._deck_labels(run_id).get(deck_digest)
        deck_hash = None if label is None else self.repository.exact_deck_hash(label)
        display_name = (
            "Deck identifier unavailable"
            if label is None
            else self.repository.deck_display_name(label)
        )
        return {
            f"{prefix}_deck_label": label,
            f"{prefix}_deck_hash": deck_hash,
            f"{prefix}_display_name": display_name,
        }

    def _execution(self, run_id: str) -> OpponentAllocationExecutionView | None:
        """Read the small live quota projection without scanning fragments."""
        path = (
            self.repository.run_directory(run_id)
            / "control"
            / "native_distributed_status.json"
        )
        if not path.is_file():
            return None
        payload = _read_object(path)
        coordinator = payload.get("coordinator")
        quota = payload.get("quota_execution")
        if not isinstance(coordinator, dict) or not isinstance(quota, dict):
            return None
        return OpponentAllocationExecutionView.model_validate(
            {
                "window_id": coordinator.get("window_id"),
                "window_sequence": coordinator.get("window_sequence_id"),
                "window_state": coordinator.get("window_state"),
                "exposure_cohort_games": quota.get("exposure_cohort_games"),
                "initial_wave_workers_issued": quota.get("initial_wave_workers_issued"),
                "initial_wave_workers_total": quota.get("initial_wave_workers_total"),
                "assignment_pool_issued": payload.get("assignment_pool_issued"),
                "assignment_pool_total": payload.get("assignment_pool_total"),
                "target_decisions": coordinator.get("target_decisions"),
                "accepted_decisions": coordinator.get("accepted_decisions"),
                "provisional_decisions": coordinator.get("provisional_decisions"),
                "inflight_decision_credit": coordinator.get("inflight_decision_credit"),
                "shards_issued": coordinator.get("shards_issued"),
                "shards_completed": coordinator.get("shards_completed"),
            }
        )

    def _deck_labels(self, run_id: str) -> dict[str, str]:
        path = self.repository.run_directory(run_id) / "resolved_config.json"
        signature = _path_signature(path)
        cached = self._deck_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        payload = _read_object(path)
        raw_routes = payload.get("active_deck_routes", [])
        if not isinstance(raw_routes, list):
            raise ValueError("resolved active deck routes must be a list")
        labels: dict[str, str] = {}
        for raw in raw_routes:
            if not isinstance(raw, dict):
                raise ValueError("resolved active deck route must be an object")
            digest = str(raw.get("deck_digest", ""))
            label = str(raw.get("label", ""))
            if len(digest) != 64 or not label:
                raise ValueError("resolved active deck route identity is incomplete")
            previous = labels.setdefault(digest, label)
            if previous != label:
                raise ValueError("one exact deck digest has ambiguous labels")
        self._deck_cache[run_id] = (signature, labels)
        return labels

    def _load(
        self,
        run_id: str,
    ) -> tuple[str, _AllocationReport] | None:
        path = (
            self.repository.run_directory(run_id)
            / "control"
            / "opponent_allocation.json"
        )
        signature = _path_signature(path)
        if signature is None:
            return None
        cached = self._report_cache.get(run_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        payload = _read_object(path)
        envelope_format = payload.get("format")
        if envelope_format not in {
            "adaptive-opponent-allocation-envelope-v1",
            "role-budget-opponent-allocation-envelope-v1",
        }:
            raise ValueError("unsupported opponent allocation envelope")
        if str(payload.get("run_version")) != run_id:
            raise ValueError("opponent allocation envelope belongs to another run")
        recorded_at_utc = str(payload.get("recorded_at_utc", "")).strip()
        if not recorded_at_utc:
            raise ValueError("opponent allocation envelope omitted its timestamp")
        report_payload = _report_payload(path.parent, payload.get("report"))
        if envelope_format == "adaptive-opponent-allocation-envelope-v1":
            report: _AllocationReport = AdaptiveOpponentAllocationReport.model_validate(
                report_payload
            )
        else:
            report = RoleBudgetOpponentAllocationReport.model_validate(report_payload)
        self._report_cache[run_id] = (signature, recorded_at_utc, report)
        return recorded_at_utc, report


def _sort_key(
    row: _MatchupReport,
    sort: AllocationSort,
) -> tuple[float, tuple[str, str, int]]:
    if sort == "weakness":
        return row.posterior_score, row.matchup_key
    if sort == "debt":
        if not isinstance(row, AdaptiveMatchupAllocationReport):
            raise ValueError("decision debt is unavailable for role-budget reports")
        value = -row.decision_debt_before
    elif sort == "uncertainty":
        value = -row.posterior_stddev
    elif sort == "target":
        value = -row.global_target_share
    elif sort == "change":
        if not isinstance(row, AdaptiveMatchupAllocationReport):
            raise ValueError("target change is unavailable for role-budget reports")
        value = -abs(row.target_delta or 0.0)
    elif sort == "planned_games":
        value = -float(row.planned_games)
    else:  # pragma: no cover - the API and Literal contract reject this first.
        raise ValueError(f"unsupported adaptive allocation sort: {sort}")
    return value, row.matchup_key


def _allocation_mode(report: _AllocationReport) -> AllocationMode:
    if isinstance(report, RoleBudgetOpponentAllocationReport):
        return "role_budget"
    return "adaptive"


def _read_object(path: Path) -> dict[str, Any]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _report_payload(parent: Path, value: Any) -> Any:
    """Read legacy inline reports or one verified compact detail pointer."""
    if not isinstance(value, dict) or value.get("format") != (
        "compact-mapping-msgpack-zlib-v1"
    ):
        return value
    relative = Path(str(value.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError("opponent allocation detail path must be relative")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("opponent allocation detail escaped the control directory")
    payload = path.read_bytes()
    if len(payload) != int(value.get("size_bytes", -1)) or (
        hashlib.sha256(payload).hexdigest() != str(value.get("sha256", ""))
    ):
        raise ValueError("opponent allocation detail identity mismatch")
    return decode_compact_mapping(payload)


def _path_signature(path: Path) -> _PathSignature | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


__all__ = ["OpponentAllocationService"]
