"""Build hierarchical Kaggle metagame distributions from deck observations.

The builder deliberately keeps archetype prevalence separate from the exact
variant distribution inside an archetype.  This prevents an archetype with
many near-duplicate candidate lists from receiving extra mass merely because
it has more variants in an evaluation pool.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TypeAlias, cast

from ptcg_rl.evaluation.meta_distribution_models import (
    ArchetypeWeight,
    CoverageDiagnostics,
    MetaDistribution,
    MetaDistributionConfig,
    RatingBandConfig,
    VariantWeight,
)

__all__ = [
    "ArchetypeWeight",
    "CoverageDiagnostics",
    "MetaDistribution",
    "MetaDistributionConfig",
    "RatingBandConfig",
    "VariantWeight",
    "build_meta_distribution",
]

ObservationSource: TypeAlias = str | Path | Iterable[Mapping[str, object]]


@dataclass
class _Bucket:
    """Aggregated rows sharing all fields used by the weighting model."""

    raw_rows: int = 0
    rating_weight: float = 0.0


@dataclass
class _ScanDiagnostics:
    """Mutable counters populated while streaming source observations."""

    total_rows: int = 0
    rating_positive_weight_rows: int = 0
    dated_rows: int = 0
    missing_date_rows: int = 0
    rating_in_band_rows: int = 0
    rating_out_of_band_rows: int = 0
    missing_rating_rows: int = 0
    rows_with_team_day: int = 0
    rows_without_team_day: int = 0


_BucketKey: TypeAlias = tuple[date | None, str | None, str, str, str | None]


def build_meta_distribution(
    source: ObservationSource,
    *,
    config: MetaDistributionConfig | None = None,
    signature_archetypes: Mapping[str, str] | None = None,
    tracked_signatures: Collection[str] | None = None,
    tracked_archetypes: Collection[str] | None = None,
) -> MetaDistribution:
    """Build a recency- and sampling-adjusted hierarchical meta distribution.

    Args:
        source: A side-observation Parquet/CSV path or a streaming row iterable.
        config: Column names and weighting controls.
        signature_archetypes: Optional exact-signature to archetype overrides.
        tracked_signatures: Optional exact variants covered by the evaluator.
            Observations outside this set retain explicit ``other`` mass.
        tracked_archetypes: Optional archetypes covered by the evaluator.

    Returns:
        A normalized archetype mixture with conditional exact-variant weights.

    Raises:
        ValueError: If no rows, or no row has positive effective weight.
    """
    resolved = config or MetaDistributionConfig()
    signature_map = {
        str(signature).strip(): str(archetype).strip()
        for signature, archetype in (signature_archetypes or {}).items()
        if str(signature).strip() and str(archetype).strip()
    }
    tracked_signature_set = _clean_collection(tracked_signatures)
    tracked_archetype_set = _clean_collection(tracked_archetypes)
    buckets: dict[_BucketKey, _Bucket] = defaultdict(_Bucket)
    columns_seen: set[str] = set()
    scan = _ScanDiagnostics()
    observed_dates: set[date] = set()

    for row in _iter_observations(source, resolved):
        scan.total_rows += 1
        columns_seen.update(row)
        day = _parse_date(_row_value(row, resolved.date_column))
        if day is None:
            scan.missing_date_rows += 1
        else:
            scan.dated_rows += 1
            observed_dates.add(day)

        rating_weight = _rating_weight(row, resolved, scan)
        if rating_weight <= 0.0:
            continue
        scan.rating_positive_weight_rows += 1

        team = _clean_text(_row_value(row, resolved.team_column)) or None
        if team is not None and day is not None:
            scan.rows_with_team_day += 1
        else:
            scan.rows_without_team_day += 1

        archetype, signature, other_reason = _classify_observation(
            row,
            config=resolved,
            signature_map=signature_map,
            tracked_signatures=tracked_signature_set,
            tracked_archetypes=tracked_archetype_set,
        )
        key = (day, team, archetype, signature, other_reason)
        bucket = buckets[key]
        bucket.raw_rows += 1
        bucket.rating_weight += rating_weight

    if scan.total_rows == 0:
        raise ValueError("cannot build a meta distribution from zero rows")

    reference_date = resolved.reference_date
    if reference_date is None and observed_dates:
        reference_date = max(observed_dates)
    return _finalize_distribution(
        buckets,
        config=resolved,
        scan=scan,
        columns_seen=columns_seen,
        reference_date=reference_date,
    )


def _finalize_distribution(
    buckets: Mapping[_BucketKey, _Bucket],
    *,
    config: MetaDistributionConfig,
    scan: _ScanDiagnostics,
    columns_seen: set[str],
    reference_date: date | None,
) -> MetaDistribution:
    team_day_rows: dict[tuple[date, str], int] = defaultdict(int)
    for (day, team, _archetype, _signature, _reason), bucket in buckets.items():
        if config.equalize_team_days and day is not None and team is not None:
            team_day_rows[(day, team)] += bucket.raw_rows

    variant_effective: dict[tuple[str, str], float] = defaultdict(float)
    variant_raw: dict[tuple[str, str], int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    contributing_rows = 0
    future_date_rows = 0
    observed_signatures: set[str] = set()
    tracked_signatures: set[str] = set()
    other_signatures: set[str] = set()
    for (day, team, archetype, signature, reason), bucket in buckets.items():
        base_weight = bucket.rating_weight
        if config.equalize_team_days and day is not None and team is not None:
            denominator = team_day_rows[(day, team)]
            base_weight = base_weight / denominator if denominator > 0.0 else 0.0
        date_weight, is_future = _date_weight(day, reference_date, config)
        effective_weight = base_weight * date_weight
        if effective_weight <= 0.0:
            continue
        if is_future:
            future_date_rows += bucket.raw_rows
        contributing_rows += bucket.raw_rows
        variant_key = (archetype, signature)
        variant_effective[variant_key] += effective_weight
        variant_raw[variant_key] += bucket.raw_rows
        if reason is None:
            observed_signatures.add(signature)
            tracked_signatures.add(signature)
        else:
            reason_counts[reason] += bucket.raw_rows
            if signature != config.other_name:
                other_signatures.add(signature)

    total_effective = sum(variant_effective.values())
    if total_effective <= 0.0:
        raise ValueError("no observation has positive effective meta weight")

    archetype_effective: dict[str, float] = defaultdict(float)
    archetype_raw: dict[str, int] = defaultdict(int)
    for (archetype, signature), effective in variant_effective.items():
        archetype_effective[archetype] += effective
        archetype_raw[archetype] += variant_raw[(archetype, signature)]
    archetype_effective.setdefault(config.other_name, 0.0)
    archetype_raw.setdefault(config.other_name, 0)

    archetypes = tuple(
        _archetype_result(
            archetype,
            total_effective=total_effective,
            archetype_effective=archetype_effective,
            archetype_raw=archetype_raw,
            variant_effective=variant_effective,
            variant_raw=variant_raw,
            other_name=config.other_name,
        )
        for archetype in sorted(
            archetype_effective,
            key=lambda name: (
                name == config.other_name,
                -archetype_effective[name],
                name,
            ),
        )
    )
    other_effective = archetype_effective[config.other_name]
    tracked_rows = contributing_rows - archetype_raw[config.other_name]
    effective_coverage = 1.0 - (other_effective / total_effective)
    diagnostics = CoverageDiagnostics(
        total_rows=scan.total_rows,
        rating_positive_weight_rows=scan.rating_positive_weight_rows,
        contributing_rows=contributing_rows,
        zero_weight_rows=scan.total_rows - contributing_rows,
        tracked_rows=tracked_rows,
        other_rows=archetype_raw[config.other_name],
        raw_coverage=_safe_ratio(tracked_rows, contributing_rows),
        effective_coverage=effective_coverage,
        observed_signature_count=len(observed_signatures | other_signatures),
        tracked_signature_count=len(tracked_signatures),
        other_observed_signature_count=len(other_signatures),
        dated_rows=scan.dated_rows,
        missing_or_invalid_date_rows=scan.missing_date_rows,
        future_date_rows=future_date_rows,
        rating_in_band_rows=scan.rating_in_band_rows,
        rating_out_of_band_rows=scan.rating_out_of_band_rows,
        missing_or_invalid_rating_rows=scan.missing_rating_rows,
        rows_with_team_day=scan.rows_with_team_day,
        rows_without_team_day=scan.rows_without_team_day,
        unique_team_days=len(team_day_rows),
        date_column_available=_column_available(config.date_column, columns_seen),
        team_column_available=_column_available(config.team_column, columns_seen),
        rating_column_available=_column_available(
            config.rating_column,
            columns_seen,
        ),
        recency_applied=(
            config.recency_half_life_days is not None and scan.dated_rows > 0
        ),
        team_day_equalization_applied=(
            config.equalize_team_days and bool(team_day_rows)
        ),
        rating_band_applied=(
            config.rating_band is not None
            and scan.rating_in_band_rows + scan.rating_out_of_band_rows > 0
        ),
        other_reason_counts=dict(sorted(reason_counts.items())),
    )
    return MetaDistribution(
        reference_date=reference_date,
        total_effective_observations=total_effective,
        archetypes=archetypes,
        other_name=config.other_name,
        other_mass=other_effective / total_effective,
        diagnostics=diagnostics,
    )


def _archetype_result(
    archetype: str,
    *,
    total_effective: float,
    archetype_effective: Mapping[str, float],
    archetype_raw: Mapping[str, int],
    variant_effective: Mapping[tuple[str, str], float],
    variant_raw: Mapping[tuple[str, str], int],
    other_name: str,
) -> ArchetypeWeight:
    effective = archetype_effective[archetype]
    variant_names = {
        signature
        for variant_archetype, signature in variant_effective
        if variant_archetype == archetype
    }
    if archetype == other_name and not variant_names:
        variant_names.add(other_name)
    variants = tuple(
        VariantWeight(
            signature=signature,
            archetype=archetype,
            raw_observations=variant_raw.get((archetype, signature), 0),
            effective_observations=variant_effective.get(
                (archetype, signature),
                0.0,
            ),
            conditional_weight=_safe_ratio(
                variant_effective.get((archetype, signature), 0.0),
                effective,
            ),
            meta_weight=(
                variant_effective.get((archetype, signature), 0.0) / total_effective
            ),
        )
        for signature in sorted(
            variant_names,
            key=lambda name: (-variant_effective.get((archetype, name), 0.0), name),
        )
    )
    return ArchetypeWeight(
        archetype=archetype,
        raw_observations=archetype_raw[archetype],
        effective_observations=effective,
        meta_weight=effective / total_effective,
        variants=variants,
    )


def _classify_observation(
    row: Mapping[str, object],
    *,
    config: MetaDistributionConfig,
    signature_map: Mapping[str, str],
    tracked_signatures: set[str] | None,
    tracked_archetypes: set[str] | None,
) -> tuple[str, str, str | None]:
    signature = _clean_text(row.get(config.signature_column))
    if not signature:
        return config.other_name, config.other_name, "missing_signature"
    if tracked_signatures is not None and signature not in tracked_signatures:
        return config.other_name, signature, "untracked_signature"
    archetype = signature_map.get(signature)
    if archetype is None:
        archetype = _clean_text(row.get(config.archetype_column))
    if not archetype:
        return config.other_name, signature, "missing_archetype"
    if archetype == config.other_name:
        return config.other_name, signature, "reserved_other_archetype"
    if tracked_archetypes is not None and archetype not in tracked_archetypes:
        return config.other_name, signature, "untracked_archetype"
    return archetype, signature, None


def _rating_weight(
    row: Mapping[str, object],
    config: MetaDistributionConfig,
    scan: _ScanDiagnostics,
) -> float:
    band = config.rating_band
    if band is None:
        return 1.0
    rating = _parse_float(_row_value(row, config.rating_column))
    if rating is None:
        scan.missing_rating_rows += 1
        return band.missing_weight
    in_band = (band.minimum is None or rating >= band.minimum) and (
        band.maximum is None or rating <= band.maximum
    )
    if in_band:
        scan.rating_in_band_rows += 1
        return 1.0
    scan.rating_out_of_band_rows += 1
    return band.outside_weight


def _date_weight(
    day: date | None,
    reference_date: date | None,
    config: MetaDistributionConfig,
) -> tuple[float, bool]:
    if config.recency_half_life_days is None:
        return 1.0, False
    if day is None or reference_date is None:
        return config.missing_date_weight, False
    age_days = (reference_date - day).days
    if age_days < 0:
        return 1.0, True
    return 0.5 ** (age_days / config.recency_half_life_days), False


def _iter_observations(
    source: ObservationSource,
    config: MetaDistributionConfig,
) -> Iterator[Mapping[str, object]]:
    if not isinstance(source, (str, Path)):
        yield from source
        return
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"meta observation source not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        yield from _iter_parquet_rows(path, config)
        return
    if suffix == ".csv":
        yield from _iter_csv_rows(path)
        return
    raise ValueError(f"unsupported meta observation format: {path.suffix}")


def _iter_parquet_rows(
    path: Path,
    config: MetaDistributionConfig,
) -> Iterator[Mapping[str, object]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    requested = _requested_columns(config)
    selected = [name for name in requested if name in available]
    if not selected and available:
        selected = [next(iter(available))]
    for batch in parquet_file.iter_batches(
        batch_size=config.parquet_batch_size,
        columns=selected,
    ):
        rows = cast(list[dict[str, object]], batch.to_pylist())
        yield from rows


def _iter_csv_rows(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        yield from csv.DictReader(file_obj)


def _requested_columns(config: MetaDistributionConfig) -> list[str]:
    return list(
        dict.fromkeys(
            name
            for name in (
                config.signature_column,
                config.archetype_column,
                config.date_column,
                config.team_column,
                config.rating_column,
            )
            if name is not None
        )
    )


def _clean_collection(values: Collection[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def _row_value(row: Mapping[str, object], column: str | None) -> object:
    return row.get(column) if column is not None else None


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _column_available(column: str | None, columns_seen: set[str]) -> bool:
    return column is not None and column in columns_seen


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
