"""Observed training-distribution deck strength projections."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any, TypeVar, cast

from ptcg_rl.dashboard.deck_routes import DeckRouteIdentity
from ptcg_rl.dashboard.models import OutcomeStats
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    SelectedTrainingWindows as _SelectedWindows,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    TrainingHistory as _History,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    active_opponent as _active_opponent,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    cell_matches as _cell_matches,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    controller_matches as _controller_matches,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    int_or_none as _int_or_none,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import mapping as _mapping
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    opponent_deck_label as _opponent_deck_label,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    outcome_stats as _outcome_stats,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    posterior as _posterior,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    select_training_windows as _select_windows,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    selected_cells as _selected_cells,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import sequence as _sequence
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    sortable_score as _sortable_score,
)
from ptcg_rl.dashboard.training_deck_strength_analysis import (
    window_cells as _window_cells,
)
from ptcg_rl.dashboard.training_deck_strength_models import (
    OpponentSet,
    TrainingController,
    TrainingDeckFamily,
    TrainingDeckMatchupsPayload,
    TrainingDeckMatrixPayload,
    TrainingDeckSeries,
    TrainingDeckSeriesPayload,
    TrainingDeckStanding,
    TrainingDeckStrengthPayload,
    TrainingEvidenceRange,
    TrainingEvidenceRangeStandings,
    TrainingMatchupDetail,
    TrainingMatchupSeatSummary,
    TrainingMatchupStratumSummary,
    TrainingMatchupSummary,
    TrainingMatrixCell,
    TrainingOpponentMatchupSummary,
    TrainingSeriesPoint,
)
from ptcg_rl.dashboard.workbench_models import CheckpointInfo
from ptcg_rl.rl.performance_state import (
    OutcomeCounts,
    normalize_outcome_cell,
    resolve_opponent_stratum,
)

_RANGES: tuple[TrainingEvidenceRange, ...] = (
    "checkpoint",
    "recent_15m",
    "recent_60m",
    "cumulative",
)
_CONTROLLERS: tuple[TrainingController, ...] = (
    "all",
    "self_play",
    "sentinel",
    "adaptive_history",
    "scripted",
    "legacy_unknown",
    "frozen",
    "stationary",
)

_CacheKey = TypeVar("_CacheKey", bound=Hashable)
_CacheValue = TypeVar("_CacheValue")


class TrainingDeckStrengthService:
    """Build read-only strength views from existing performance history."""

    def __init__(self, repository: DashboardRepository) -> None:
        self.repository = repository
        self._standings_cache: dict[
            str,
            tuple[tuple[object, ...], TrainingDeckStrengthPayload],
        ] = {}
        self._series_cache: dict[
            tuple[str, TrainingEvidenceRange, int | None, TrainingController],
            tuple[tuple[object, ...], TrainingDeckSeriesPayload],
        ] = {}
        self._matrix_cache: dict[
            tuple[
                str,
                TrainingEvidenceRange,
                int | None,
                TrainingController,
                int | None,
                OpponentSet,
            ],
            tuple[tuple[object, ...], TrainingDeckMatrixPayload],
        ] = {}
        self._matchups_cache: dict[
            tuple[
                str,
                str,
                TrainingEvidenceRange,
                int | None,
                TrainingController,
                int | None,
                OpponentSet,
            ],
            tuple[tuple[object, ...], TrainingDeckMatchupsPayload],
        ] = {}

    def standings(
        self,
        run: str,
        *,
        checkpoint: CheckpointInfo | None,
    ) -> TrainingDeckStrengthPayload:
        """Return four independent observed-training rankings."""
        revision = (
            *self.repository.training_history_revision(run),
            None
            if checkpoint is None
            else (checkpoint.version, checkpoint.pair_manifest_sha256),
        )
        run_id = str(revision[0])
        cached = self._standings_cache.get(run_id)
        if cached is not None and cached[0] == revision:
            return cached[1]
        history = self._history(run)
        route_metadata = self.repository.deck_route_metadata(
            history.run_id,
            deck_labels=history.targets,
        )
        family_labels: dict[str, list[str]] = defaultdict(list)
        for label in history.targets:
            identity = route_metadata.identities.get(label)
            if identity is not None and identity.family_id is not None:
                family_labels[identity.family_id].append(label)
        family_names = {
            family_id: self.repository.deck_family_display_name(
                family_id,
                deck_labels=labels,
            )
            for family_id, labels in family_labels.items()
        }
        route_compatibility = self.repository.checkpoint_route_compatibility(
            history.run_id,
            deck_labels=history.targets,
            active_deck_digests=(
                () if checkpoint is None else checkpoint.active_exact_deck_digests
            ),
            exact_registry_fingerprint=(
                None if checkpoint is None else checkpoint.exact_registry_fingerprint
            ),
            route_identities=route_metadata.identities,
        )
        ranges: dict[TrainingEvidenceRange, TrainingEvidenceRangeStandings] = {}
        for range_name in _RANGES:
            selected = _select_windows(
                history,
                range_name=range_name,
                checkpoint_version=None if checkpoint is None else checkpoint.version,
            )
            ranges[range_name] = TrainingEvidenceRangeStandings(
                metadata=selected.metadata,
                standings=self._standings(
                    history,
                    selected,
                    route_compatibility=route_compatibility,
                    route_identities=route_metadata.identities,
                    family_names=family_names,
                ),
            )
        warnings = list(history.warnings)
        if route_metadata.family_routes_declared and route_metadata.unresolved_labels:
            warnings.append(
                "family route identity could not be resolved for: "
                + ", ".join(route_metadata.unresolved_labels)
            )
        result = TrainingDeckStrengthPayload(
            run_id=history.run_id,
            checkpoint_version=None if checkpoint is None else checkpoint.version,
            checkpoint_pair_manifest_sha256=(
                None if checkpoint is None else checkpoint.pair_manifest_sha256
            ),
            semantics="observed_training_distribution_wdl_posterior_v1",
            source_warning=(
                "这是训练真实采样分布下的观测强度，不是部署 bundle 后验，"
                "也没有对 controller 或 opponent 做等权重算。Frozen 与 Stationary "
                "仅为兼容旧消费者的 deprecated 聚合；诊断应优先使用 opponent stratum。"
            ),
            exact_dimensions_available=history.exact_available,
            active_deck_labels=history.targets,
            families=tuple(
                TrainingDeckFamily(
                    family_id=family_id,
                    display_name=family_names[family_id],
                    deck_labels=tuple(labels),
                )
                for family_id, labels in sorted(
                    family_labels.items(),
                    key=lambda item: (family_names[item[0]], item[0]),
                )
            ),
            stationary_opponent_kinds=tuple(sorted(history.stationary_kinds)),
            ranges=ranges,
            warnings=tuple(warnings),
        )
        self._standings_cache[run_id] = (revision, result)
        return result

    def series(
        self,
        run: str,
        *,
        range_name: TrainingEvidenceRange,
        checkpoint_version: int | None,
        controller: TrainingController,
    ) -> TrainingDeckSeriesPayload:
        """Return per-window points under one independently chosen range."""
        revision = self.repository.training_history_revision(run)
        cache_key = (revision[0], range_name, checkpoint_version, controller)
        cached = self._series_cache.get(cache_key)
        if cached is not None and cached[0] == revision:
            return cached[1]
        history = self._history(run)
        selected = _select_windows(
            history,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
        )
        series: list[TrainingDeckSeries] = []
        for deck_label in history.targets:
            points: list[TrainingSeriesPoint] = []
            for fallback_index, window in enumerate(selected.windows, start=1):
                counts = OutcomeCounts()
                for cell in _window_cells(window):
                    if str(cell.get("candidate_deck_label", "")) != deck_label:
                        continue
                    if not _controller_matches(
                        str(cell.get("opponent_kind", "")),
                        controller,
                        history.stationary_kinds,
                        opponent_stratum=_optional_text(
                            cell.get("opponent_stratum")
                        ),
                    ):
                        continue
                    counts.merge(OutcomeCounts.from_mapping(cell))
                points.append(
                    TrainingSeriesPoint(
                        minute_index=int(window.get("minute_index") or fallback_index),
                        ended_at_utc=str(window.get("ended_at_utc", "")),
                        games=counts.games,
                        score_rate=counts.score,
                    )
                )
            series.append(
                TrainingDeckSeries(
                    deck_label=deck_label,
                    deck_hash=self.repository.deck_hash(deck_label),
                    display_name=self.repository.deck_display_name(deck_label),
                    points=tuple(points),
                )
            )
        result = TrainingDeckSeriesPayload(
            run_id=history.run_id,
            range=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
            available=selected.metadata.available,
            unavailable_reason=selected.metadata.unavailable_reason,
            series=tuple(series),
        )
        _store_bounded(self._series_cache, cache_key, (revision, result))
        return result

    def matrix(
        self,
        run: str,
        *,
        range_name: TrainingEvidenceRange,
        checkpoint_version: int | None,
        controller: TrainingController,
        candidate_seat: int | None,
        opponent_set: OpponentSet,
    ) -> TrainingDeckMatrixPayload:
        """Aggregate candidate deck by opponent deck without pilot reweighting."""
        revision = self.repository.training_history_revision(run)
        cache_key = (
            revision[0],
            range_name,
            checkpoint_version,
            controller,
            candidate_seat,
            opponent_set,
        )
        cached = self._matrix_cache.get(cache_key)
        if cached is not None and cached[0] == revision:
            return cached[1]
        history = self._history(run)
        selected = _select_windows(
            history,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
        )
        target_hashes = {
            self.repository.exact_deck_hash(label) for label in history.targets
        }
        grouped: dict[tuple[str, str], OutcomeCounts] = {}
        pilots: dict[tuple[str, str], set[str]] = defaultdict(set)
        for cell in _selected_cells(selected.windows):
            candidate = str(cell.get("candidate_deck_label", ""))
            if candidate not in history.targets:
                continue
            if not _cell_matches(
                cell,
                controller=controller,
                candidate_seat=candidate_seat,
                stationary_kinds=history.stationary_kinds,
            ):
                continue
            opponent = _opponent_deck_label(cell)
            if opponent_set == "active" and not _active_opponent(
                opponent,
                targets=history.targets,
                target_hashes=target_hashes,
                repository=self.repository,
            ):
                continue
            key = (candidate, opponent)
            grouped.setdefault(key, OutcomeCounts()).merge(
                OutcomeCounts.from_mapping(cell)
            )
            pilots[key].add(str(cell.get("opponent_id", "")))
        cells = tuple(
            TrainingMatrixCell(
                candidate_deck_label=candidate,
                candidate_deck_hash=self.repository.deck_hash(candidate),
                candidate_display_name=self.repository.deck_display_name(candidate),
                opponent_deck_label=opponent,
                opponent_deck_hash=self.repository.exact_deck_hash(opponent),
                opponent_display_name=self.repository.deck_display_name(opponent),
                pilot_count=len(pilots[(candidate, opponent)]),
                posterior=_posterior(counts),
            )
            for (candidate, opponent), counts in sorted(grouped.items())
        )
        result = TrainingDeckMatrixPayload(
            run_id=history.run_id,
            range=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
            candidate_seat=candidate_seat,
            opponent_set=opponent_set,
            available=selected.metadata.available,
            unavailable_reason=selected.metadata.unavailable_reason,
            cells=cells,
        )
        _store_bounded(self._matrix_cache, cache_key, (revision, result))
        return result

    def matchups(
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
        """Retain opponent pilot and candidate-seat identity for drill-down."""
        revision = self.repository.training_history_revision(run)
        cache_key = (
            revision[0],
            deck_label,
            range_name,
            checkpoint_version,
            controller,
            candidate_seat,
            opponent_set,
        )
        cached = self._matchups_cache.get(cache_key)
        if cached is not None and cached[0] == revision:
            return cached[1]
        history = self._history(run)
        if deck_label not in history.targets:
            raise KeyError(f"unknown active training deck: {deck_label}")
        selected = _select_windows(
            history,
            range_name=range_name,
            checkpoint_version=checkpoint_version,
        )
        target_hashes = {
            self.repository.exact_deck_hash(label) for label in history.targets
        }
        grouped: dict[
            tuple[str, str, str, str, int],
            OutcomeCounts,
        ] = {}
        overall_counts = OutcomeCounts()
        pilot_ids: set[str] = set()
        seat_counts: dict[int, OutcomeCounts] = {}
        stratum_counts: dict[tuple[str, str], OutcomeCounts] = {}
        stratum_pilots: dict[tuple[str, str], set[str]] = defaultdict(set)
        opponent_counts: dict[str, OutcomeCounts] = {}
        opponent_pilots: dict[str, set[str]] = defaultdict(set)
        opponent_seat_counts: dict[str, dict[int, OutcomeCounts]] = {}
        opponent_stratum_counts: dict[
            str,
            dict[tuple[str, str], OutcomeCounts],
        ] = {}
        opponent_stratum_pilots: dict[
            str,
            dict[tuple[str, str], set[str]],
        ] = {}
        for cell in _selected_cells(selected.windows):
            if str(cell.get("candidate_deck_label", "")) != deck_label:
                continue
            if not _cell_matches(
                cell,
                controller=controller,
                candidate_seat=candidate_seat,
                stationary_kinds=history.stationary_kinds,
            ):
                continue
            seat = _int_or_none(cell.get("candidate_seat"))
            if seat not in (0, 1):
                continue
            opponent = _opponent_deck_label(cell)
            if opponent_set == "active" and not _active_opponent(
                opponent,
                targets=history.targets,
                target_hashes=target_hashes,
                repository=self.repository,
            ):
                continue
            normalized_cell = normalize_outcome_cell(cell)
            counts = OutcomeCounts.from_mapping(cell)
            opponent_kind = str(normalized_cell.get("opponent_kind", ""))
            opponent_stratum = str(normalized_cell["opponent_stratum"])
            opponent_id = str(cell.get("opponent_id", ""))
            stratum_key = (opponent_kind, opponent_stratum)
            key = (
                opponent_kind,
                opponent_stratum,
                opponent,
                opponent_id,
                seat,
            )
            grouped.setdefault(key, OutcomeCounts()).merge(counts)
            overall_counts.merge(counts)
            seat_counts.setdefault(seat, OutcomeCounts()).merge(counts)
            stratum_counts.setdefault(stratum_key, OutcomeCounts()).merge(counts)
            opponent_counts.setdefault(opponent, OutcomeCounts()).merge(counts)
            opponent_seat_counts.setdefault(opponent, {}).setdefault(
                seat,
                OutcomeCounts(),
            ).merge(counts)
            opponent_stratum_counts.setdefault(opponent, {}).setdefault(
                stratum_key,
                OutcomeCounts(),
            ).merge(counts)
            if opponent_id:
                pilot_ids.add(opponent_id)
                stratum_pilots[stratum_key].add(opponent_id)
                opponent_pilots[opponent].add(opponent_id)
                opponent_stratum_pilots.setdefault(opponent, {}).setdefault(
                    stratum_key,
                    set(),
                ).add(opponent_id)
        matchups = [
            TrainingMatchupDetail(
                opponent_kind=key[0],
                opponent_stratum=key[1],
                opponent_deck_label=key[2],
                opponent_deck_hash=self.repository.exact_deck_hash(key[2]),
                opponent_display_name=self.repository.opponent_display_name(
                    deck_label=key[2],
                    opponent_id=key[3],
                ),
                opponent_id=key[3],
                candidate_seat=key[4],
                posterior=_posterior(counts),
            )
            for key, counts in grouped.items()
        ]
        matchups.sort(
            key=lambda item: (
                _sortable_score(item.posterior.posterior_mean),
                item.opponent_deck_label,
                item.opponent_id,
                item.candidate_seat,
            )
        )
        opponents = [
            TrainingOpponentMatchupSummary(
                opponent_deck_label=opponent,
                opponent_deck_hash=self.repository.exact_deck_hash(opponent),
                opponent_display_name=self.repository.deck_display_name(opponent),
                pilot_count=len(opponent_pilots[opponent]),
                posterior=_posterior(counts),
                seat_breakdown=tuple(
                    TrainingMatchupSeatSummary(
                        candidate_seat=seat_key,
                        posterior=_posterior(seat_value),
                    )
                    for seat_key, seat_value in sorted(
                        opponent_seat_counts[opponent].items()
                    )
                ),
                stratum_breakdown=tuple(
                    TrainingMatchupStratumSummary(
                        opponent_kind=stratum_key[0],
                        opponent_stratum=stratum_key[1],
                        pilot_count=len(
                            opponent_stratum_pilots.get(opponent, {}).get(
                                stratum_key,
                                set(),
                            )
                        ),
                        posterior=_posterior(stratum_value),
                    )
                    for stratum_key, stratum_value in sorted(
                        opponent_stratum_counts[opponent].items()
                    )
                ),
            )
            for opponent, counts in opponent_counts.items()
        ]
        opponents.sort(
            key=lambda item: (
                _sortable_score(item.posterior.posterior_mean),
                item.opponent_display_name,
                item.opponent_deck_label,
            )
        )
        result = TrainingDeckMatchupsPayload(
            run_id=history.run_id,
            candidate_deck_label=deck_label,
            range=range_name,
            checkpoint_version=checkpoint_version,
            controller=controller,
            candidate_seat=candidate_seat,
            opponent_set=opponent_set,
            available=selected.metadata.available,
            unavailable_reason=selected.metadata.unavailable_reason,
            overall=_posterior(overall_counts),
            opponent_deck_count=len(opponents),
            pilot_count=len(pilot_ids),
            seat_breakdown=tuple(
                TrainingMatchupSeatSummary(
                    candidate_seat=seat_key,
                    posterior=_posterior(counts),
                )
                for seat_key, counts in sorted(seat_counts.items())
            ),
            stratum_breakdown=tuple(
                TrainingMatchupStratumSummary(
                    opponent_kind=stratum_key[0],
                    opponent_stratum=stratum_key[1],
                    pilot_count=len(stratum_pilots[stratum_key]),
                    posterior=_posterior(counts),
                )
                for stratum_key, counts in sorted(stratum_counts.items())
            ),
            opponents=tuple(opponents),
            matchups=tuple(matchups),
        )
        _store_bounded(self._matchups_cache, cache_key, (revision, result))
        return result

    def _history(self, run: str) -> _History:
        run_id, payload, windows, warnings = self.repository.training_history(run)
        semantics = _mapping(payload.get("semantics"))
        configured_targets = tuple(
            str(item)
            for item in _sequence(semantics.get("target_deck_labels"))
            if isinstance(item, str) and item
        )
        targets = configured_targets or tuple(
            dict.fromkeys(
                str(cell.get("candidate_deck_label", ""))
                for window in windows
                for cell in _window_cells(window)
                if str(cell.get("candidate_deck_label", ""))
            )
        )
        stationary = frozenset(
            str(item)
            for item in _sequence(semantics.get("stationary_opponent_kinds"))
            if isinstance(item, str)
        )
        return _History(
            run_id=run_id,
            payload=payload,
            windows=windows,
            warnings=warnings,
            targets=targets,
            stationary_kinds=stationary,
            exact_available=int(payload.get("schema_version", 0)) >= 2,
        )

    def _standings(
        self,
        history: _History,
        selected: _SelectedWindows,
        *,
        route_compatibility: Mapping[str, bool | None],
        route_identities: Mapping[str, DeckRouteIdentity],
        family_names: Mapping[str, str],
    ) -> tuple[TrainingDeckStanding, ...]:
        all_counts = {label: OutcomeCounts() for label in history.targets}
        controller_counts = {
            label: {controller: OutcomeCounts() for controller in _CONTROLLERS}
            for label in history.targets
        }
        seat_counts = {
            label: {seat: OutcomeCounts() for seat in (0, 1)}
            for label in history.targets
        }
        matchup_counts: dict[str, dict[str, OutcomeCounts]] = {
            label: {} for label in history.targets
        }
        targets = set(history.targets)
        for window in selected.windows:
            for cell in _window_cells(window):
                deck_label = str(cell.get("candidate_deck_label", ""))
                if deck_label not in targets:
                    continue
                counts = OutcomeCounts.from_mapping(cell)
                all_counts[deck_label].merge(counts)
                deck_controllers = controller_counts[deck_label]
                deck_controllers["all"].merge(counts)
                opponent_kind = str(cell.get("opponent_kind", ""))
                stratum = resolve_opponent_stratum(
                    opponent_kind=opponent_kind,
                    opponent_stratum=_optional_text(cell.get("opponent_stratum")),
                )
                deck_controllers[cast(TrainingController, stratum)].merge(counts)
                if opponent_kind == "frozen":
                    deck_controllers["frozen"].merge(counts)
                if opponent_kind in history.stationary_kinds:
                    deck_controllers["stationary"].merge(counts)
                seat = _int_or_none(cell.get("candidate_seat"))
                if seat in (0, 1):
                    seat_counts[deck_label][seat].merge(counts)
                opponent = _opponent_deck_label(cell)
                matchup_counts[deck_label].setdefault(
                    opponent,
                    OutcomeCounts(),
                ).merge(counts)

        rows: list[TrainingDeckStanding] = []
        for deck_label in history.targets:
            route_identity = route_identities.get(deck_label)
            deck_digest = None if route_identity is None else route_identity.deck_digest
            family_id = None if route_identity is None else route_identity.family_id
            controllers: dict[str, OutcomeStats] = {
                controller: _outcome_stats(controller_counts[deck_label][controller])
                for controller in _CONTROLLERS
            }
            seat_scores = {
                seat: _outcome_stats(seat_counts[deck_label][seat])
                for seat in (0, 1)
            }
            matchup_summaries = self._matchup_summaries_from_counts(
                matchup_counts[deck_label]
            )
            rows.append(
                TrainingDeckStanding(
                    deck_label=deck_label,
                    deck_hash=self.repository.deck_hash(deck_label),
                    deck_digest=deck_digest,
                    display_name=self.repository.deck_display_name(deck_label),
                    family_id=family_id,
                    family_display_name=(
                        None if family_id is None else family_names.get(family_id)
                    ),
                    route_compatible=route_compatibility.get(deck_label),
                    posterior=_posterior(all_counts[deck_label]),
                    controller_scores=controllers,
                    seat_scores=seat_scores,
                    opponent_count=len(matchup_summaries),
                    strongest_matchup=(
                        max(
                            matchup_summaries,
                            key=lambda item: _sortable_score(
                                item.outcomes.score_rate,
                                missing=-1.0,
                            ),
                        )
                        if matchup_summaries
                        else None
                    ),
                    weakest_matchup=(
                        min(
                            matchup_summaries,
                            key=lambda item: _sortable_score(item.outcomes.score_rate),
                        )
                        if matchup_summaries
                        else None
                    ),
                )
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                -_sortable_score(
                    row.posterior.posterior_mean,
                    missing=-1.0,
                ),
                row.display_name,
            ),
        )
        output: list[TrainingDeckStanding] = []
        rank = 0
        for row in ranked:
            if row.posterior.posterior_mean is not None:
                rank += 1
                output.append(row.model_copy(update={"rank": rank}))
            else:
                output.append(row)
        return tuple(output)

    def _matchup_summaries_from_counts(
        self,
        grouped: Mapping[str, OutcomeCounts],
    ) -> tuple[TrainingMatchupSummary, ...]:
        return tuple(
            TrainingMatchupSummary(
                opponent_deck_label=opponent,
                opponent_deck_hash=self.repository.exact_deck_hash(opponent),
                opponent_display_name=self.repository.deck_display_name(opponent),
                outcomes=_outcome_stats(counts),
            )
            for opponent, counts in sorted(grouped.items())
        )


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _store_bounded(
    cache: dict[_CacheKey, _CacheValue],
    key: _CacheKey,
    value: _CacheValue,
    *,
    maximum: int = 64,
) -> None:
    """Keep interactive filter caches bounded without a second cache package."""
    if key not in cache and len(cache) >= maximum:
        cache.pop(next(iter(cache)))
    cache[key] = value


__all__ = ["TrainingDeckStrengthService"]
