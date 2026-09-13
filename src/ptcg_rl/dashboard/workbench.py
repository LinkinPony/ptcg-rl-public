"""Decision-oriented aggregation over existing dashboard artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptcg_rl.dashboard.models import MatchupRow, WindowName
from ptcg_rl.dashboard.opponent_allocation import OpponentAllocationService
from ptcg_rl.dashboard.opponent_allocation_models import (
    AllocationSort,
    OpponentAllocationMatchupPage,
    OpponentAllocationSummaryPayload,
)
from ptcg_rl.dashboard.public_environment import PublicEnvironmentService
from ptcg_rl.dashboard.public_environment_models import (
    EnvironmentWindow,
    PublicEnvironmentMatchupsPayload,
    PublicEnvironmentMatrixPayload,
    PublicEnvironmentPayload,
)
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.dashboard.training_deck_strength import TrainingDeckStrengthService
from ptcg_rl.dashboard.training_deck_strength_models import (
    OpponentSet,
    TrainingController,
    TrainingDeckMatchupsPayload,
    TrainingDeckMatrixPayload,
    TrainingDeckSeriesPayload,
    TrainingDeckStrengthPayload,
    TrainingEvidenceRange,
)
from ptcg_rl.dashboard.two_deck_selection import TwoDeckSelectionService
from ptcg_rl.dashboard.two_deck_selection_models import TwoDeckSelectionPayload
from ptcg_rl.dashboard.workbench_analysis import (
    build_alerts,
    build_posterior,
    rows_outcomes,
)
from ptcg_rl.dashboard.workbench_models import (
    CheckpointInfo,
    ComparisonPayload,
    ComparisonRequest,
    ComparisonRow,
    CurrentLearnerSnapshot,
    DeckEvidence,
    DeckEvidencePayload,
    DeckMatchupPayload,
    LearnerSeriesPayload,
    MatchupEvidence,
    ProgressSummary,
    WorkbenchSummary,
)
from ptcg_rl.rl.learner_metric_history import (
    LearnerMetricHistory,
    LearnerMetricRecord,
)
from ptcg_rl.rl.opponent_pool.adaptive import PortfolioName
from ptcg_rl.rl.opponent_pool.role_budget import RoleBudgetName
from ptcg_rl.rl.training_game_statistics import (
    TrainingGameStatistics,
    TrainingGameStatisticsCollector,
)


class WorkbenchService:
    """Expose normalized, read-only training and evidence views."""

    def __init__(self, repository: DashboardRepository) -> None:
        self.repository = repository
        self.training_game_collector = TrainingGameStatisticsCollector(
            repository.run_root,
            known_run_ids=tuple(
                segment
                for lineage in repository.config.lineages.values()
                for segment in lineage.segments
            ),
        )
        self.training_strength = TrainingDeckStrengthService(repository)
        self.public_environment = PublicEnvironmentService(repository)
        self.two_deck_selection = TwoDeckSelectionService(
            repository,
            self.training_strength,
            self.public_environment,
        )
        self.opponent_allocation = OpponentAllocationService(repository)
        self._checkpoint_cache: dict[
            str,
            tuple[tuple[object, ...], tuple[CheckpointInfo, ...]],
        ] = {}
        self._learner_record_cache: dict[
            str,
            tuple[tuple[int, int] | None, tuple[LearnerMetricRecord, ...]],
        ] = {}
        self._deck_evidence_cache: dict[
            tuple[str, WindowName],
            tuple[tuple[object, ...], DeckEvidencePayload],
        ] = {}

    def training_game_statistics(self, run: str) -> TrainingGameStatistics:
        """Return cross-segment terminal games with the selected run separated."""
        run_id = self.repository.resolve_run_id(run)
        return self.training_game_collector.collect(selected_run=run_id)

    def opponent_allocation_summary(
        self,
        run: str,
    ) -> OpponentAllocationSummaryPayload:
        """Return the compact latest opponent-allocation summary."""
        return self.opponent_allocation.summary(run)

    def opponent_allocation_matchups(
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
        """Return one filtered page of exact allocation cells."""
        return self.opponent_allocation.matchups(
            run,
            offset=offset,
            limit=limit,
            sort=sort,
            candidate_deck_digest=candidate_deck_digest,
            artifact_id=artifact_id,
            portfolio=portfolio,
            role=role,
            candidate_seat=candidate_seat,
        )

    def public_environment_summary(
        self,
        run: str,
        *,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentPayload:
        """Return cached Kaggle Daily environment evidence without doing work."""
        self._checkpoint(run, checkpoint_version)
        return self.public_environment.summary(
            run,
            window_days=window_days,
            checkpoint_version=checkpoint_version,
        )

    def public_environment_matrix(
        self,
        run: str,
        *,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentMatrixPayload:
        """Return the cached roster-by-public-meta matchup matrix."""
        self._checkpoint(run, checkpoint_version)
        return self.public_environment.matrix(
            run,
            window_days=window_days,
            checkpoint_version=checkpoint_version,
        )

    def public_environment_matchups(
        self,
        run: str,
        *,
        deck_hash: str,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentMatchupsPayload:
        """Return cached public matchups for one authoritative compact identity."""
        self._checkpoint(run, checkpoint_version)
        return self.public_environment.matchups(
            run,
            deck_hash=deck_hash,
            window_days=window_days,
            checkpoint_version=checkpoint_version,
        )

    def training_deck_strength(
        self,
        run: str,
        *,
        checkpoint_version: int | None,
    ) -> TrainingDeckStrengthPayload:
        """Return all observed-training evidence ranges without starting work."""
        checkpoint = self._checkpoint(run, checkpoint_version)
        return self.training_strength.standings(run, checkpoint=checkpoint)

    def two_deck_recommendation(
        self,
        run: str,
        *,
        checkpoint_version: int | None,
        training_range: TrainingEvidenceRange,
        window_days: EnvironmentWindow,
    ) -> TwoDeckSelectionPayload:
        """Recommend two tracked slots from one artifact-bound evidence view."""
        checkpoint = self._checkpoint(run, checkpoint_version)
        return self.two_deck_selection.recommend(
            run,
            checkpoint=checkpoint,
            training_range=training_range,
            window_days=window_days,
        )

    def training_deck_strength_series(
        self,
        run: str,
        *,
        range_name: TrainingEvidenceRange,
        checkpoint_version: int | None,
        controller: TrainingController,
    ) -> TrainingDeckSeriesPayload:
        """Return one independently filtered observed-training trend."""
        self._checkpoint(run, checkpoint_version)
        return self.training_strength.series(
            run,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
        )

    def training_deck_strength_matrix(
        self,
        run: str,
        *,
        range_name: TrainingEvidenceRange,
        checkpoint_version: int | None,
        controller: TrainingController,
        candidate_seat: int | None,
        opponent_set: OpponentSet,
    ) -> TrainingDeckMatrixPayload:
        """Return one candidate-deck by opponent-deck training matrix."""
        self._checkpoint(run, checkpoint_version)
        return self.training_strength.matrix(
            run,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
            candidate_seat=candidate_seat,
            opponent_set=opponent_set,
        )

    def training_deck_strength_matchups(
        self,
        run: str,
        *,
        deck_label: str,
        range_name: TrainingEvidenceRange,
        checkpoint_version: int | None,
        controller: TrainingController,
        candidate_seat: int | None,
        opponent_set: OpponentSet,
    ) -> TrainingDeckMatchupsPayload:
        """Return exact pilot and seat evidence for one active deck."""
        self._checkpoint(run, checkpoint_version)
        return self.training_strength.matchups(
            run,
            deck_label=deck_label,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
            candidate_seat=candidate_seat,
            opponent_set=opponent_set,
        )

    def checkpoints(self, run: str) -> tuple[CheckpointInfo, ...]:
        """List validated checkpoint manifests without loading tensor payloads."""
        run_id = self.repository.resolve_run_id(run)
        run_dir = self.repository.run_directory(run_id)
        revision, pair_paths = _checkpoint_revision(run_dir)
        cached = self._checkpoint_cache.get(run_id)
        if cached is not None and cached[0] == revision:
            return cached[1]
        metric_versions = {
            record.checkpoint_version for record in self._learner_records(run_id)
        }
        latest = _read_json(run_dir / "weights" / "latest.json")
        latest_version = _optional_int(latest.get("version"))
        checkpoints: list[CheckpointInfo] = []
        for path in pair_paths:
            payload = _read_json(path)
            version = int(payload["version"])
            policy = _mapping(payload.get("policy"))
            training_state = _mapping(payload.get("training_state"))
            metadata = _mapping(payload.get("metadata"))
            identity = _mapping(metadata.get("simple_stateless_identity"))
            checkpoints.append(
                CheckpointInfo(
                    run_id=run_id,
                    version=version,
                    pair_manifest_sha256=_sha256(path),
                    policy_sha256=str(policy["sha256"]),
                    learner_state_sha256=str(training_state["sha256"]),
                    model_config_fingerprint=_optional_text(
                        identity.get("model_config_fingerprint")
                    ),
                    training_roster_fingerprint=_optional_text(
                        identity.get("training_roster_fingerprint")
                    ),
                    exact_registry_fingerprint=_optional_text(
                        identity.get("exact_registry_fingerprint")
                    ),
                    active_exact_deck_digests=tuple(
                        str(item)
                        for item in _sequence(identity.get("active_exact_deck_digests"))
                        if isinstance(item, str)
                    ),
                    updated_at_utc=datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=UTC,
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                    metric_available=version in metric_versions,
                    current=version == latest_version,
                )
            )
        checkpoints.sort(key=lambda checkpoint: checkpoint.version, reverse=True)
        result = tuple(checkpoints)
        self._checkpoint_cache[run_id] = (revision, result)
        return result

    def learner_series(self, run: str) -> LearnerSeriesPayload:
        """Load artifact-bound metrics and surface gaps as data, not guesses."""
        run_id = self.repository.resolve_run_id(run)
        try:
            records = self._learner_records(run_id)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return LearnerSeriesPayload(
                run_id=run_id,
                records=(),
                complete=False,
                warning=f"{type(error).__name__}: {error}",
            )
        checkpoints = self.checkpoints(run_id)
        expected = {
            checkpoint.version for checkpoint in checkpoints if checkpoint.version > 0
        }
        present = {record.checkpoint_version for record in records}
        missing = sorted(expected - present)
        warning = (
            None
            if not missing
            else "历史 checkpoint 缺少 learner 指标："
            + ", ".join(f"v{version}" for version in missing)
        )
        return LearnerSeriesPayload(
            run_id=run_id,
            records=records,
            complete=not missing,
            warning=warning,
        )

    def deck_evidence(
        self,
        run: str,
        *,
        window: WindowName = "cumulative",
    ) -> DeckEvidencePayload:
        """Build seat-balanced posterior evidence from exact W/D/L cells."""
        run_id = self.repository.resolve_run_id(run)
        history_revision = self.repository.training_history_revision(run_id)
        revision: tuple[object, ...] = (
            history_revision
            if window == "cumulative"
            else (history_revision[0], history_revision[1])
        )
        cache_key = (run_id, window)
        cached = self._deck_evidence_cache.get(cache_key)
        if cached is not None and cached[0] == revision:
            return cached[1]
        table = self.repository.performance_table(run_id, window=window)
        matchups = self.repository.matchups(run_id, window=window)
        rows_by_deck: dict[str, list[MatchupRow]] = defaultdict(list)
        for matchup in matchups:
            rows_by_deck[matchup.candidate_deck_label].append(matchup)
        decks: list[DeckEvidence] = []
        for deck in table.decks:
            deck_rows = rows_by_deck.get(deck.deck_label, [])
            opponents = {
                (
                    row.opponent_kind,
                    row.opponent_deck_label,
                    row.opponent_id,
                )
                for row in deck_rows
            }
            decks.append(
                DeckEvidence(
                    deck_label=deck.deck_label,
                    deck_hash=deck.deck_hash,
                    display_name=deck.display_name,
                    posterior=build_posterior(
                        deck_rows,
                        fallback=deck.slices["all"],
                        seed_key=deck.deck_label,
                        exact_available=table.exact_dimensions_available,
                    ),
                    controller_scores=deck.slices,
                    opponent_count=len(opponents),
                )
            )
        decks.sort(
            key=lambda deck: (
                deck.posterior.posterior_mean
                if deck.posterior.posterior_mean is not None
                else deck.posterior.observed.score_rate
                if deck.posterior.observed.score_rate is not None
                else -1.0
            ),
            reverse=True,
        )
        result = DeckEvidencePayload(
            run_id=run_id,
            window=window,
            semantics="training_pool_wdl_posterior_v1",
            overall=build_posterior(
                matchups,
                fallback=table.overall["all"],
                seed_key=f"{run_id}/overall",
                exact_available=table.exact_dimensions_available,
            ),
            decks=tuple(decks),
            exact_dimensions_available=table.exact_dimensions_available,
        )
        self._deck_evidence_cache[cache_key] = (revision, result)
        return result

    def summary(
        self,
        run: str,
        *,
        window: WindowName = "15m",
    ) -> WorkbenchSummary:
        """Return one-screen health, progress, evidence, and alerts."""
        run_id = self.repository.resolve_run_id(run)
        run_info = self.repository.run_info(run_id)
        health = self.repository.health(run_id)
        checkpoints = self.checkpoints(run_id)
        evidence = self.deck_evidence(run_id, window=window).overall
        learner = _mapping(health.status.get("learner_status"))
        throughput = _mapping(learner.get("throughput"))
        latest = _mapping(learner.get("latest_update"))
        timing = _mapping(learner.get("latest_timing"))
        return WorkbenchSummary(
            run=run_info,
            data_age_seconds=health.data_age_seconds,
            data_state=health.data_state,
            warnings=health.warnings,
            checkpoint=checkpoints[0] if checkpoints else None,
            training_games=self.training_game_collector.collect(
                selected_run=run_id,
            ),
            progress=ProgressSummary(
                update_index=_optional_nonnegative_int(learner.get("update_index")),
                target_updates=_optional_nonnegative_int(learner.get("target_updates")),
                optimizer_step_index=_optional_nonnegative_int(
                    learner.get("optimizer_step_index")
                ),
                decisions_seen=_optional_nonnegative_int(
                    learner.get("fresh_decisions_seen")
                ),
                target_decisions=_optional_nonnegative_int(
                    learner.get("target_decisions")
                ),
                kept_decisions_per_second=_optional_nonnegative_float(
                    throughput.get("kept_decisions_per_second")
                ),
            ),
            latest_learner=CurrentLearnerSnapshot(
                loss=_optional_float(latest.get("loss")),
                policy_loss=_optional_float(latest.get("policy_loss")),
                value_loss=_optional_float(latest.get("value_loss")),
                belief_loss=_optional_float(latest.get("belief_loss")),
                entropy=_optional_float(latest.get("entropy")),
                approximate_kl=_optional_float(latest.get("approximate_kl")),
                clip_fraction=_optional_nonnegative_float(latest.get("clip_fraction")),
                gradient_norm=_optional_nonnegative_float(latest.get("gradient_norm")),
                learning_rate=_optional_nonnegative_float(latest.get("learning_rate")),
                collection_seconds=_optional_nonnegative_float(
                    timing.get("collection_seconds")
                ),
                learner_seconds=_optional_nonnegative_float(
                    timing.get("learner_seconds")
                ),
                checkpoint_seconds=_optional_nonnegative_float(
                    timing.get("checkpoint_seconds")
                ),
                cuda_peak_allocated_bytes=_optional_nonnegative_int(
                    timing.get("cuda_checkpoint_peak_allocated_bytes")
                ),
                cuda_peak_reserved_bytes=_optional_nonnegative_int(
                    timing.get("cuda_checkpoint_peak_reserved_bytes")
                ),
                fragments_stale=_optional_nonnegative_int(
                    latest.get("fragments_stale")
                ),
            ),
            evidence=evidence,
            alerts=build_alerts(
                health.status,
                health.data_state,
                health.warnings,
            ),
        )

    def deck_matchups(
        self,
        run: str,
        *,
        deck_label: str,
        window: WindowName = "cumulative",
    ) -> DeckMatchupPayload:
        """Return focused opponent identities without materializing a giant matrix."""
        run_id = self.repository.resolve_run_id(run)
        selected = [
            row
            for row in self.repository.matchups(run_id, window=window)
            if row.candidate_deck_label == deck_label
        ]
        grouped: dict[tuple[str, str, str], list[MatchupRow]] = defaultdict(list)
        names: dict[tuple[str, str, str], str] = {}
        for row in selected:
            key = (row.opponent_kind, row.opponent_deck_label, row.opponent_id)
            grouped[key].append(row)
            names[key] = row.opponent_display_name
        evidence = [
            MatchupEvidence(
                opponent_kind=key[0],
                opponent_deck_label=key[1],
                opponent_display_name=names[key],
                opponent_id=key[2],
                posterior=build_posterior(
                    rows,
                    fallback=rows_outcomes(rows),
                    seed_key=f"{run_id}/{deck_label}/{key}",
                    exact_available=True,
                ),
            )
            for key, rows in grouped.items()
        ]
        evidence.sort(
            key=lambda item: (
                item.posterior.posterior_mean
                if item.posterior.posterior_mean is not None
                else item.posterior.observed.score_rate
                if item.posterior.observed.score_rate is not None
                else 2.0
            )
        )
        return DeckMatchupPayload(
            run_id=run_id,
            candidate_deck_label=deck_label,
            window=window,
            matchups=tuple(evidence),
        )

    def compare(self, request: ComparisonRequest) -> ComparisonPayload:
        """Compare bounded run/checkpoint references under explicit identities."""
        rows: list[ComparisonRow] = []
        for reference in request.references:
            run_id = self.repository.resolve_run_id(reference.run_id)
            checkpoints = self.checkpoints(run_id)
            checkpoint = _select_checkpoint(
                checkpoints,
                version=reference.checkpoint_version,
            )
            records = {
                record.checkpoint_version: record
                for record in self._learner_records(run_id)
            }
            metric = None if checkpoint is None else records.get(checkpoint.version)
            warnings: list[str] = []
            if checkpoint is None:
                warnings.append("未找到可比较的 immutable checkpoint pair")
            elif metric is None:
                warnings.append("该 checkpoint 没有 artifact-bound learner 指标")
            group = (
                None
                if checkpoint is None
                else ":".join(
                    (
                        checkpoint.model_config_fingerprint or "unknown-model",
                        checkpoint.training_roster_fingerprint or "unknown-roster",
                    )
                )
            )
            rows.append(
                ComparisonRow(
                    reference=reference.model_copy(update={"run_id": run_id}),
                    checkpoint=checkpoint,
                    learner_metric=metric,
                    training_pool=self.deck_evidence(
                        run_id,
                        window="cumulative",
                    ).overall,
                    compatible_group=group,
                    warnings=tuple(warnings),
                )
            )
        groups = {row.compatible_group for row in rows}
        return ComparisonPayload(
            rows=tuple(rows),
            all_compatible=len(groups) == 1 and None not in groups,
        )

    def _learner_records(self, run: str) -> tuple[LearnerMetricRecord, ...]:
        run_dir = self.repository.run_directory(run)
        manifest = run_dir / "performance" / "learner_metrics_manifest.json"
        signature = _path_signature(manifest)
        cached = self._learner_record_cache.get(run)
        if cached is not None and cached[0] == signature:
            return cached[1]
        records = LearnerMetricHistory(run_dir).load()
        self._learner_record_cache[run] = (signature, records)
        return records

    def _checkpoint(
        self,
        run: str,
        version: int | None,
    ) -> CheckpointInfo | None:
        if version is None:
            return None
        checkpoint = _select_checkpoint(self.checkpoints(run), version=version)
        if checkpoint is None:
            raise KeyError(f"unknown checkpoint version for {run}: {version}")
        return checkpoint


def _select_checkpoint(
    checkpoints: Sequence[CheckpointInfo],
    *,
    version: int | None,
) -> CheckpointInfo | None:
    if version is None:
        return checkpoints[0] if checkpoints else None
    return next(
        (checkpoint for checkpoint in checkpoints if checkpoint.version == version),
        None,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return int(value) if isinstance(value, (int, float)) else None


def _optional_nonnegative_int(value: object) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _optional_nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if parsed >= 0.0 else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_revision(
    run_dir: Path,
) -> tuple[tuple[object, ...], tuple[Path, ...]]:
    weights = run_dir / "weights"
    pair_paths = tuple(sorted(weights.glob("checkpoint_pair_v*.json")))
    revision: tuple[object, ...] = (
        _path_signature(weights / "latest.json"),
        _path_signature(run_dir / "performance" / "learner_metrics_manifest.json"),
        tuple((path.name, _path_signature(path)) for path in pair_paths),
    )
    return revision, pair_paths


def _path_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size
