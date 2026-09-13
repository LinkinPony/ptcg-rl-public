"""Build one immutable public exact-deck catalog from full replay history."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.belief.identity import file_sha256
from ptcg_rl.belief.public_catalog import (
    build_public_deck_catalog_from_summary,
    estimate_unknown_prior_mass_from_summary,
    write_public_deck_catalog,
)
from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS
from ptcg_rl.data.kaggle_deck.records import fast_episode_identity_and_decks
from ptcg_rl.decks.identity import canonical_signature, parse_canonical_signature

REPO_ROOT = Path(__file__).resolve().parents[4]


class ArchiveReplaySource(BaseModel):
    """Validated daily tar+zstd replay source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    dates: tuple[str, ...]
    zstd_binary: str = "zstd"

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique ISO dates."""
        return _validated_dates(values)


class DirectoryReplaySource(BaseModel):
    """Daily replay directories already present in data/external."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    dates: tuple[str, ...]

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique ISO dates."""
        return _validated_dates(values)


class SupplementalParquetSource(BaseModel):
    """Fixed public snapshot that supplements incomplete current-day data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    dates: tuple[str, ...]
    require_exact_replay: bool = True
    coverage: Literal["complete", "partial"] = "partial"

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique ISO dates."""
        return _validated_dates(values)


class DiscoveryReplaySource(BaseModel):
    """Recursive public replay corpus used only to discover missing decks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    snapshot_date: str

    @field_validator("snapshot_date")
    @classmethod
    def valid_snapshot_date(cls, value: str) -> str:
        """Require a reproducible evidence snapshot date."""
        datetime.strptime(value, "%Y-%m-%d")
        return value


class FullHistoryPublicCatalogConfig(BaseModel):
    """Hydra-resolved inputs for one full-history public catalog asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_summary_path: Path
    base_start_date: str
    base_end_date: str
    base_excluded_support_fields: tuple[str, ...] = ()
    archive_sources: tuple[ArchiveReplaySource, ...] = ()
    directory_sources: tuple[DirectoryReplaySource, ...] = ()
    supplemental_parquet_sources: tuple[SupplementalParquetSource, ...] = ()
    discovery_replay_sources: tuple[DiscoveryReplaySource, ...] = ()
    output_dir: Path
    runtime_output_dir: Path | None = None
    card_catalog_fingerprint: str
    card_vocab_size: int = DEFAULT_NUM_CARD_IDS
    presence_floor: int = Field(default=1, gt=0)
    prefix_bytes: int = Field(default=65_536, gt=0)

    @field_validator("base_start_date", "base_end_date")
    @classmethod
    def valid_base_date(cls, value: str) -> str:
        """Require ISO dates for the aggregate base interval."""
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("base_excluded_support_fields")
    @classmethod
    def valid_base_excluded_support_fields(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require explicit unique CSV columns for incremental subtraction."""
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("base excluded support fields cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("base excluded support fields must be unique")
        return normalized

    @field_validator("card_catalog_fingerprint")
    @classmethod
    def valid_card_catalog_fingerprint(cls, value: str) -> str:
        """Require one explicit card vocabulary identity."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("card catalog fingerprint must contain 64 hex digits")
        return normalized

    @model_validator(mode="after")
    def disjoint_date_coverage(self) -> FullHistoryPublicCatalogConfig:
        """Prevent double counting between the aggregate and replay sources."""
        if self.base_start_date > self.base_end_date:
            raise ValueError("base history start date must not exceed end date")
        claimed: dict[str, str] = {}
        for kind, sources in (
            ("archive", self.archive_sources),
            ("directory", self.directory_sources),
            ("supplemental", self.supplemental_parquet_sources),
        ):
            for source in sources:
                for date in source.dates:
                    if self.base_start_date <= date <= self.base_end_date:
                        raise ValueError(f"{kind} date overlaps aggregate base: {date}")
                    previous = claimed.setdefault(date, kind)
                    if previous != kind:
                        raise ValueError(
                            f"date {date} is claimed by both {previous} and {kind}"
                        )
        return self


@dataclass(slots=True)
class _Aggregate:
    support: int = 0
    base_support: int = 0
    replay_support: int = 0
    supplemental_support: int = 0
    discovery_observations: int = 0
    first_date: str = ""
    last_date: str = ""

    def add(self, support: int, date: str, source_kind: str) -> None:
        if support <= 0:
            raise ValueError("public exact-deck support must be positive")
        self.support += support
        if source_kind == "base":
            self.base_support += support
        elif source_kind == "replay":
            self.replay_support += support
        elif source_kind == "supplemental":
            self.supplemental_support += support
        else:  # pragma: no cover - internal programming error.
            raise ValueError(f"unsupported source kind: {source_kind}")
        self.first_date = min(filter(None, (self.first_date, date)), default=date)
        self.last_date = max(self.last_date, date)

    def discover(self, date: str) -> None:
        """Record overlap-safe public evidence without changing prior support."""
        self.discovery_observations += 1
        self.first_date = min(filter(None, (self.first_date, date)), default=date)
        self.last_date = max(self.last_date, date)


@dataclass(frozen=True, slots=True)
class _EpisodeDecks:
    episode_id: int
    signatures: tuple[str, ...]


def run(config: FullHistoryPublicCatalogConfig) -> dict[str, Any]:
    """Merge disjoint public history sources and publish immutable assets."""
    base_summary_path = _repo_path(config.base_summary_path)
    output_dir = _repo_path(config.output_dir)
    aggregates: dict[str, _Aggregate] = {}
    _load_base_summary(base_summary_path, aggregates, config=config)

    seen_episodes: dict[int, tuple[str, ...]] = {}
    source_reports: list[dict[str, Any]] = []
    for archive_source in config.archive_sources:
        for date in archive_source.dates:
            report = _scan_archive_date(
                archive_source,
                date=date,
                aggregates=aggregates,
                seen_episodes=seen_episodes,
                config=config,
            )
            source_reports.append(report)
    for directory_source in config.directory_sources:
        for date in directory_source.dates:
            report = _scan_directory_date(
                directory_source,
                date=date,
                aggregates=aggregates,
                seen_episodes=seen_episodes,
                config=config,
            )
            source_reports.append(report)
    for supplemental_source in config.supplemental_parquet_sources:
        source_reports.append(
            _scan_supplemental_parquet(
                supplemental_source,
                aggregates=aggregates,
                seen_episodes=seen_episodes,
                config=config,
            )
        )
    for discovery_source in config.discovery_replay_sources:
        source_reports.append(
            _scan_discovery_replays(
                discovery_source,
                aggregates=aggregates,
                seen_episodes=seen_episodes,
                config=config,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "deck_signature_summary.csv"
    _write_summary(summary_path, aggregates)
    unknown_prior_mass = estimate_unknown_prior_mass_from_summary(summary_path)
    catalog = build_public_deck_catalog_from_summary(
        summary_path,
        unknown_prior_mass=unknown_prior_mass,
        card_catalog_fingerprint=config.card_catalog_fingerprint,
        card_vocab_size=config.card_vocab_size,
        presence_floor=config.presence_floor,
    )
    manifest = write_public_deck_catalog(
        catalog,
        artifact_path=output_dir / "catalog.npz",
        manifest_path=output_dir / "manifest.json",
    )
    coverage_dates = sorted(
        {
            config.base_start_date,
            config.base_end_date,
            *(
                date
                for report in source_reports
                for date in report.get("dates", ())
            ),
        }
    )
    partial_dates = {
        date
        for source in config.supplemental_parquet_sources
        if source.coverage == "partial"
        for date in source.dates
    }
    provenance = {
        "schema": "public-deck-catalog-provenance-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "coverage": {
            "start_date": min(coverage_dates),
            "end_date": max(coverage_dates),
            "complete_through": max(
                date
                for date in coverage_dates
                if date not in partial_dates
            ),
            "supplemental_partial_dates": sorted(partial_dates),
        },
        "base_summary": {
            "path": _display_path(base_summary_path),
            "sha256": file_sha256(base_summary_path),
            "start_date": config.base_start_date,
            "end_date": config.base_end_date,
        },
        "sources": source_reports,
        "deduplication": {
            "scope": "post-base replay and supplemental sources",
            "key": "episode_id",
            "unique_episodes": len(seen_episodes),
        },
        "result": {
            "deck_count": len(aggregates),
            "public_support": sum(item.support for item in aggregates.values()),
            "unknown_prior_mass": unknown_prior_mass,
            "summary_sha256": file_sha256(summary_path),
            "catalog_fingerprint": catalog.fingerprint,
            "catalog_artifact_sha256": manifest.artifact_sha256,
            "runtime_output_dir": (
                None
                if config.runtime_output_dir is None
                else _display_path(_repo_path(config.runtime_output_dir))
            ),
        },
        "config": config.model_dump(mode="json"),
    }
    _atomic_write_json(output_dir / "provenance.json", provenance)
    if config.runtime_output_dir is not None:
        _mirror_catalog(output_dir, _repo_path(config.runtime_output_dir))
    return provenance


def _load_base_summary(
    path: Path,
    aggregates: dict[str, _Aggregate],
    *,
    config: FullHistoryPublicCatalogConfig,
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            signature = str(row.get("deck_signature", "")).strip()
            if not signature:
                continue
            parse_canonical_signature(signature, max_card_id=config.card_vocab_size)
            raw_support = row.get("games")
            if raw_support in {None, ""}:
                raw_support = row.get("public_support", "0")
            support = int(str(raw_support))
            excluded_support = 0
            for field in config.base_excluded_support_fields:
                if field not in row:
                    raise ValueError(
                        f"base summary is missing excluded support field: {field}"
                    )
                excluded_support += int(str(row[field] or "0"))
            support -= excluded_support
            if support < 0:
                raise ValueError(
                    "base excluded support exceeds public support for one deck"
                )
            first_date = str(row.get("first_date", config.base_start_date))
            last_date = min(
                str(row.get("last_date", config.base_end_date)),
                config.base_end_date,
            )
            aggregate = aggregates.setdefault(signature, _Aggregate())
            if support > 0:
                aggregate.add(support, first_date, "base")
                aggregate.last_date = max(aggregate.last_date, last_date)
            aggregate.discovery_observations += int(
                str(row.get("discovery_observations", "0") or "0")
            )


def _scan_archive_date(
    source: ArchiveReplaySource,
    *,
    date: str,
    aggregates: dict[str, _Aggregate],
    seen_episodes: dict[int, tuple[str, ...]],
    config: FullHistoryPublicCatalogConfig,
) -> dict[str, Any]:
    root = _repo_path(source.root)
    archive_path = root / f"{date}.tar.zst"
    manifest_path = root / f"{date}.manifest.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("date") != date:
        raise ValueError(f"archive manifest date mismatch: {manifest_path}")
    if int(stored.get("archive_bytes", -1)) != archive_path.stat().st_size:
        raise ValueError(f"archive size differs from manifest: {archive_path}")

    process = subprocess.Popen(
        [source.zstd_binary, "--quiet", "--decompress", "--stdout", str(archive_path)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:  # pragma: no cover - Popen contract guard.
        raise RuntimeError("failed to open zstd output pipe")
    counters: Counter[str] = Counter()
    source_digest = hashlib.sha256()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                counters["members"] += 1
                if not member.isreg() or Path(member.name).name != member.name:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
                member_file = archive.extractfile(member)
                if member_file is None:
                    raise RuntimeError(f"could not read archive member: {member.name}")
                evidence = _read_episode(member_file, prefix_bytes=config.prefix_bytes)
                if evidence is None:
                    counters["non_replay_or_missing_registration"] += 1
                    continue
                source_digest.update(_episode_digest_payload(evidence))
                _merge_episode(
                    evidence,
                    date=date,
                    source_kind="replay",
                    aggregates=aggregates,
                    seen_episodes=seen_episodes,
                    counters=counters,
                )
        process.stdout.close()
        return_code = process.wait()
        expected_members = int(stored["source"]["file_count"])
        if counters["members"] != expected_members:
            raise RuntimeError(
                f"archive member count differs from manifest: {archive_path}"
            )
        if return_code not in {0, -signal.SIGPIPE}:
            raise subprocess.CalledProcessError(return_code, process.args)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    return {
        "kind": "daily_archive",
        "dates": [date],
        "archive_path": _display_path(archive_path),
        "archive_sha256": stored["archive_sha256"],
        "archive_manifest_path": _display_path(manifest_path),
        "archive_manifest_sha256": file_sha256(manifest_path),
        "episode_deck_digest": source_digest.hexdigest(),
        "counts": dict(sorted(counters.items())),
    }


def _scan_directory_date(
    source: DirectoryReplaySource,
    *,
    date: str,
    aggregates: dict[str, _Aggregate],
    seen_episodes: dict[int, tuple[str, ...]],
    config: FullHistoryPublicCatalogConfig,
) -> dict[str, Any]:
    date_dir = _repo_path(source.root) / date
    paths = sorted(date_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"daily replay directory contains no JSON files: {date_dir}")
    counters: Counter[str] = Counter()
    member_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    for path in paths:
        stat_result = path.stat()
        member_digest.update(f"{path.name}\0{stat_result.st_size}\n".encode())
        with path.open("rb") as handle:
            evidence = _read_episode(handle, prefix_bytes=config.prefix_bytes)
        if evidence is None:
            counters["non_replay_or_missing_registration"] += 1
            continue
        source_digest.update(_episode_digest_payload(evidence))
        _merge_episode(
            evidence,
            date=date,
            source_kind="replay",
            aggregates=aggregates,
            seen_episodes=seen_episodes,
            counters=counters,
        )
    return {
        "kind": "daily_directory",
        "dates": [date],
        "path": _display_path(date_dir),
        "member_index_sha256": member_digest.hexdigest(),
        "episode_deck_digest": source_digest.hexdigest(),
        "counts": dict(sorted(counters.items())),
    }


def _scan_supplemental_parquet(
    source: SupplementalParquetSource,
    *,
    aggregates: dict[str, _Aggregate],
    seen_episodes: dict[int, tuple[str, ...]],
    config: FullHistoryPublicCatalogConfig,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = _repo_path(source.path)
    columns = ["date", "episode_id", "player_index", "deck_signature"]
    if source.require_exact_replay:
        columns.append("exact_replay")
    table = pq.read_table(path, columns=columns)
    table = table.filter(
        pc.is_in(table["date"], value_set=pa.array(source.dates))
    )
    if source.require_exact_replay:
        table = table.filter(pc.equal(table["exact_replay"], True))
    rows = table.to_pylist()
    by_episode: dict[int, dict[int, str]] = {}
    episode_dates: dict[int, str] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        date = str(row["date"])
        previous_date = episode_dates.setdefault(episode_id, date)
        if previous_date != date:
            raise ValueError(f"supplemental episode {episode_id} spans dates")
        player_index = int(row["player_index"])
        signature = str(row["deck_signature"])
        parse_canonical_signature(signature, max_card_id=config.card_vocab_size)
        previous = by_episode.setdefault(episode_id, {}).setdefault(
            player_index, signature
        )
        if previous != signature:
            raise ValueError(
                f"supplemental episode {episode_id} has conflicting exact decks"
            )
    counters: Counter[str] = Counter(rows=len(rows))
    source_digest = hashlib.sha256()
    for episode_id in sorted(by_episode):
        signatures = tuple(
            by_episode[episode_id][index] for index in sorted(by_episode[episode_id])
        )
        evidence = _EpisodeDecks(episode_id=episode_id, signatures=signatures)
        source_digest.update(_episode_digest_payload(evidence))
        _merge_episode(
            evidence,
            date=episode_dates[episode_id],
            source_kind="supplemental",
            aggregates=aggregates,
            seen_episodes=seen_episodes,
            counters=counters,
        )
    return {
        "kind": "supplemental_parquet",
        "dates": list(source.dates),
        "path": _display_path(path),
        "sha256": file_sha256(path),
        "require_exact_replay": source.require_exact_replay,
        "coverage": source.coverage,
        "episode_deck_digest": source_digest.hexdigest(),
        "counts": dict(sorted(counters.items())),
    }


def _scan_discovery_replays(
    source: DiscoveryReplaySource,
    *,
    aggregates: dict[str, _Aggregate],
    seen_episodes: dict[int, tuple[str, ...]],
    config: FullHistoryPublicCatalogConfig,
) -> dict[str, Any]:
    root = _repo_path(source.root)
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError(f"discovery replay corpus contains no JSON files: {root}")
    counters: Counter[str] = Counter()
    member_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    for path in paths:
        stat_result = path.stat()
        relative_path = path.relative_to(root)
        member_digest.update(
            f"{relative_path}\0{stat_result.st_size}\n".encode()
        )
        with path.open("rb") as handle:
            evidence = _read_episode(handle, prefix_bytes=config.prefix_bytes)
        if evidence is None:
            counters["non_replay_or_missing_registration"] += 1
            continue
        source_digest.update(_episode_digest_payload(evidence))
        previous = seen_episodes.get(evidence.episode_id)
        if previous is not None:
            if not _compatible_episode_signatures(previous, evidence.signatures):
                raise ValueError(
                    f"episode {evidence.episode_id} has conflicting deck registrations"
                )
            if len(evidence.signatures) > len(previous):
                seen_episodes[evidence.episode_id] = evidence.signatures
            for signature in (
                Counter(evidence.signatures) - Counter(previous)
            ).elements():
                was_known = signature in aggregates
                aggregates.setdefault(signature, _Aggregate()).discover(
                    source.snapshot_date
                )
                counters[
                    "known_deck_sides" if was_known else "new_deck_sides"
                ] += 1
            counters["duplicate_episodes"] += 1
            continue
        seen_episodes[evidence.episode_id] = evidence.signatures
        counters["unique_discovery_episodes"] += 1
        counters["exact_deck_sides"] += len(evidence.signatures)
        for signature in evidence.signatures:
            was_known = signature in aggregates
            aggregates.setdefault(signature, _Aggregate()).discover(
                source.snapshot_date
            )
            counters["known_deck_sides" if was_known else "new_deck_sides"] += 1
    return {
        "kind": "discovery_replay_directory",
        "snapshot_date": source.snapshot_date,
        "path": _display_path(root),
        "member_index_sha256": member_digest.hexdigest(),
        "episode_deck_digest": source_digest.hexdigest(),
        "support_policy": "discovery_only_zero_support_to_avoid_base_overlap",
        "counts": dict(sorted(counters.items())),
    }


def _read_episode(handle: IO[bytes], *, prefix_bytes: int) -> _EpisodeDecks | None:
    prefix = handle.read(prefix_bytes)
    parsed = fast_episode_identity_and_decks(prefix)
    if parsed is None:
        remaining = handle.read()
        if not remaining:
            return None
        parsed = fast_episode_identity_and_decks(prefix + remaining)
    if parsed is None:
        return None
    episode_id, decks = parsed
    signatures = tuple(
        canonical_signature(decks[player_index]) for player_index in sorted(decks)
    )
    return _EpisodeDecks(episode_id=episode_id, signatures=signatures)


def _merge_episode(
    evidence: _EpisodeDecks,
    *,
    date: str,
    source_kind: str,
    aggregates: dict[str, _Aggregate],
    seen_episodes: dict[int, tuple[str, ...]],
    counters: Counter[str],
) -> None:
    previous = seen_episodes.get(evidence.episode_id)
    if previous is not None:
        if previous != evidence.signatures:
            raise ValueError(
                f"episode {evidence.episode_id} has conflicting deck registrations"
            )
        counters["duplicate_episodes"] += 1
        return
    seen_episodes[evidence.episode_id] = evidence.signatures
    counters["unique_episodes"] += 1
    counters["exact_deck_sides"] += len(evidence.signatures)
    for signature in evidence.signatures:
        aggregates.setdefault(signature, _Aggregate()).add(1, date, source_kind)


def _write_summary(path: Path, aggregates: dict[str, _Aggregate]) -> None:
    fields = (
        "deck_signature",
        "public_support",
        "first_date",
        "last_date",
        "base_support",
        "replay_support",
        "supplemental_support",
        "discovery_observations",
    )
    rows = sorted(aggregates.items(), key=lambda item: (-item[1].support, item[0]))
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for signature, aggregate in rows:
                writer.writerow(
                    {
                        "deck_signature": signature,
                        "public_support": aggregate.support,
                        "first_date": aggregate.first_date,
                        "last_date": aggregate.last_date,
                        "base_support": aggregate.base_support,
                        "replay_support": aggregate.replay_support,
                        "supplemental_support": aggregate.supplemental_support,
                        "discovery_observations": aggregate.discovery_observations,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mirror_catalog(source_dir: Path, target_dir: Path) -> None:
    """Publish materialized runtime copies under the immutable outputs link."""
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("runtime catalog mirror must differ from tracked asset")
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "catalog.npz",
        "deck_signature_summary.csv",
        "manifest.json",
        "provenance.json",
    ):
        source = source_dir / filename
        target = target_dir / filename
        temporary = _temporary_path(target)
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    temporary_root = REPO_ROOT / "tmp" / "full_history_public_catalog"
    temporary_root.mkdir(parents=True, exist_ok=True)
    return temporary_root / f"{path.name}.{uuid.uuid4().hex}.tmp"


def _episode_digest_payload(evidence: _EpisodeDecks) -> bytes:
    return (
        f"{evidence.episode_id}\0{'|'.join(evidence.signatures)}\n".encode()
    )


def _compatible_episode_signatures(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    """Allow a one-sided attributed row to match its two-sided raw replay."""
    left_counts = Counter(left)
    right_counts = Counter(right)
    return left_counts <= right_counts or right_counts <= left_counts


def _validated_dates(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values or len(set(values)) != len(values):
        raise ValueError("source dates must be non-empty and unique")
    for value in values:
        datetime.strptime(value, "%Y-%m-%d")
    return values


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


__all__ = ["FullHistoryPublicCatalogConfig", "run"]
