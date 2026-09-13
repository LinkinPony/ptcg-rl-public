"""Stream Kaggle Daily packages into compact, immutable Parquet partitions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ptcg_rl.data.kaggle import episode_archive
from ptcg_rl.data.kaggle.public_environment_models import PublicEnvironmentConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.rl.performance_state import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[4]
PARTITION_SCHEMA_VERSION = 1
_INDEX_STATE = "public_environment_fetch_state.json"
_CORPUS_STATE = "corpus_state.json"

SIDE_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("episode_id", pa.int64()),
        ("player_index", pa.int8()),
        ("first_player_index", pa.int8()),
        ("went_first", pa.bool_()),
        ("pilot_key", pa.string()),
        ("reward", pa.float32()),
        ("status", pa.string()),
        ("opponent_status", pa.string()),
        ("result", pa.string()),
        ("terminal_valid", pa.bool_()),
        ("deck_digest", pa.string()),
        ("deck_signature", pa.string()),
        ("deck_hash", pa.string()),
        ("deck_label", pa.string()),
        ("opponent_deck_digest", pa.string()),
        ("opponent_deck_signature", pa.string()),
        ("opponent_deck_hash", pa.string()),
        ("opponent_deck_label", pa.string()),
    ]
)


def refresh_index(config: PublicEnvironmentConfig) -> tuple[Path, bool]:
    """Refresh the small Kaggle index only after the configured cooldown."""
    index_dir = repo_path(config.index_dir)
    manifest = index_dir / "manifest.csv"
    state_path = repo_path(config.processed_root) / _INDEX_STATE
    state = _read_json(state_path)
    last_checked = _parse_datetime(state.get("checked_at_utc"))
    cutoff = datetime.now(UTC) - timedelta(hours=config.network_min_interval_hours)
    due = not manifest.is_file() or last_checked is None or last_checked < cutoff
    if not due:
        return manifest, False
    if not config.network_enabled:
        if not manifest.is_file():
            raise FileNotFoundError(
                "Kaggle index is unavailable and network is disabled"
            )
        return manifest, False
    if shutil.which(config.kaggle_binary) is None:
        raise RuntimeError(f"Kaggle CLI binary not found: {config.kaggle_binary}")

    stage = index_dir / ".public_environment_staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    command = [
        config.kaggle_binary,
        "datasets",
        "download",
        config.source_index,
        "-f",
        "manifest.csv",
        "-p",
        str(stage),
        "--unzip",
    ]
    logger.info("refreshing compact public-environment index")
    subprocess.run(command, check=True)
    downloaded = stage / "manifest.csv"
    rows = read_index(downloaded)
    if not rows:
        raise ValueError("downloaded Kaggle index is empty")
    index_dir.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, manifest)
    shutil.rmtree(stage)
    atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "checked_at_utc": _utc_now(),
            "source_index": config.source_index,
            "manifest_sha256": sha256_file(manifest),
            "latest_date": rows[-1]["date"],
        },
    )
    return manifest, True


def read_index(path: Path) -> list[dict[str, str]]:
    """Read and validate the public Daily index manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "date",
        "daily_dataset_slug",
        "episode_count",
        "total_bytes",
    }
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Kaggle index is missing columns: {sorted(required)}")
    rows.sort(key=lambda row: row["date"])
    for row in rows:
        datetime.strptime(row["date"], "%Y-%m-%d")
        int(row["episode_count"])
        int(row["total_bytes"])
    return rows


def compact_required_dates(
    config: PublicEnvironmentConfig,
    rows: list[dict[str, str]],
    *,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Ensure every Daily partition since the immutable bootstrap date exists."""
    rows = _complete_index_rows(rows)
    if not rows:
        raise ValueError("Kaggle Daily index contains no complete dates")
    processed_root = repo_path(config.processed_root)
    state_path = processed_root / _CORPUS_STATE
    state = _read_json(state_path)
    bootstrap_start = state.get("bootstrap_start_date")
    if not isinstance(bootstrap_start, str):
        selected = rows[-config.bootstrap_days :]
        if len(selected) < config.bootstrap_days:
            raise ValueError("Kaggle index cannot satisfy the bootstrap window")
        bootstrap_start = selected[0]["date"]
    required_days = max(config.windows)
    if len(rows) < required_days:
        raise ValueError("Kaggle index cannot satisfy the largest analysis window")
    required_start = rows[-required_days]["date"]
    target_start = min(bootstrap_start, required_start)
    targets = [row for row in rows if row["date"] >= target_start]
    card_meta = records.load_card_meta(repo_path(config.card_data_csv))
    actions: list[dict[str, Any]] = []
    total = len(targets)
    for index, row in enumerate(targets):
        day = row["date"]
        if progress is not None:
            progress(index, total, day, "processing")
        action = ensure_partition(config, row, card_meta=card_meta)
        actions.append(action)
        if progress is not None:
            progress(index + 1, total, day, str(action["status"]))
    atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "bootstrap_start_date": bootstrap_start,
            "latest_compacted_date": targets[-1]["date"],
            "updated_at_utc": _utc_now(),
        },
    )
    return {
        "bootstrap_start_date": bootstrap_start,
        "latest_date": targets[-1]["date"],
        "actions": actions,
    }


def _complete_index_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Exclude the current UTC date even if an upstream row appears early."""
    today = datetime.now(UTC).date()
    return [
        row for row in rows if datetime.strptime(row["date"], "%Y-%m-%d").date() < today
    ]


def ensure_partition(
    config: PublicEnvironmentConfig,
    index_row: Mapping[str, str],
    *,
    card_meta: dict[int, records.CardMeta],
) -> dict[str, Any]:
    """Build one date partition or reuse its validated immutable manifest."""
    date = index_row["date"]
    partition_dir = partition_directory(config, date)
    parquet_path = partition_dir / "sides.parquet"
    manifest_path = partition_dir / "manifest.json"
    existing = _read_json(manifest_path)
    if (
        parquet_path.is_file()
        and existing.get("schema_version") == PARTITION_SCHEMA_VERSION
        and existing.get("dataset_ref") == _dataset_ref(index_row)
        and existing.get("expected_episode_count") == int(index_row["episode_count"])
        and existing.get("expected_total_bytes") == int(index_row["total_bytes"])
    ):
        pq.read_schema(parquet_path)
        return {"date": date, "status": "reused", "manifest": str(manifest_path)}

    source = _resolve_source(config, index_row)
    partition_dir.mkdir(parents=True, exist_ok=True)
    temp_path = partition_dir / f".sides.{os.getpid()}.parquet"
    if temp_path.exists():
        temp_path.unlink()
    quality = _write_partition(
        source,
        temp_path,
        date=date,
        card_meta=card_meta,
        batch_rows=config.parquet_batch_rows,
        prefix_bytes=config.prefix_bytes,
        max_prefix_bytes=config.max_prefix_bytes,
    )
    expected_episodes = int(index_row["episode_count"])
    expected_bytes = int(index_row["total_bytes"])
    quality["source_missing_episodes"] = max(0, expected_episodes - quality["episodes"])
    quality["source_missing_bytes"] = max(
        0, expected_bytes - quality["source_uncompressed_bytes"]
    )
    quality["source_unexpected_episodes"] = max(
        0, quality["episodes"] - expected_episodes
    )
    if quality["source_missing_episodes"]:
        logger.warning(
            "Daily source omits indexed episodes date={} missing={} available={}",
            date,
            quality["source_missing_episodes"],
            quality["episodes"],
        )
    pq.read_schema(temp_path)
    os.replace(temp_path, parquet_path)
    manifest = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "date": date,
        "created_at_utc": _utc_now(),
        "dataset_ref": _dataset_ref(index_row),
        "expected_episode_count": expected_episodes,
        "expected_total_bytes": expected_bytes,
        "source_kind": source.kind,
        "source_path": display_path(source.path),
        "source_sha256": source.sha256,
        "parquet_path": display_path(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        **quality,
    }
    atomic_write_json(manifest_path, manifest)
    return {"date": date, "status": "compacted", "manifest": str(manifest_path)}


class _ReplaySource:
    """One immutable source for a Daily dataset."""

    def __init__(self, *, kind: str, path: Path, sha256: str) -> None:
        self.kind = kind
        self.path = path
        self.sha256 = sha256


def _resolve_source(
    config: PublicEnvironmentConfig,
    index_row: Mapping[str, str],
) -> _ReplaySource:
    date = index_row["date"]
    package_path = repo_path(config.package_root) / date / "episodes.zip"
    if package_path.is_file():
        return _ReplaySource(
            kind="zip", path=package_path, sha256=sha256_file(package_path)
        )
    archive_path = episode_archive.archive_path(repo_path(config.archive_root), date)
    archive_manifest = episode_archive.manifest_path(
        repo_path(config.archive_root), date
    )
    if archive_path.is_file() and archive_manifest.is_file():
        payload = _read_json(archive_manifest)
        digest = str(payload.get("archive_sha256") or sha256_file(archive_path))
        return _ReplaySource(kind="tar_zst", path=archive_path, sha256=digest)
    raw_dir = repo_path(config.replay_root) / date
    if raw_dir.is_dir() and any(raw_dir.glob("*.json")):
        return _ReplaySource(
            kind="raw", path=raw_dir, sha256=_raw_member_fingerprint(raw_dir)
        )
    if not config.network_enabled:
        raise FileNotFoundError(f"no compactable source for Daily {date}")
    return _download_package(config, index_row, package_path)


def _download_package(
    config: PublicEnvironmentConfig,
    index_row: Mapping[str, str],
    output_path: Path,
) -> _ReplaySource:
    if shutil.which(config.kaggle_binary) is None:
        raise RuntimeError(f"Kaggle CLI binary not found: {config.kaggle_binary}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = output_path.parent / ".download_staging"
    stage.mkdir(exist_ok=True)
    dataset_ref = _dataset_ref(index_row)
    logger.info(
        "downloading compressed Daily date={} dataset={}",
        index_row["date"],
        dataset_ref,
    )
    subprocess.run(
        [
            config.kaggle_binary,
            "datasets",
            "download",
            dataset_ref,
            "-p",
            str(stage),
        ],
        check=True,
    )
    archives = list(stage.glob("*.zip"))
    if len(archives) != 1 or not zipfile.is_zipfile(archives[0]):
        raise ValueError(f"Kaggle download did not produce one ZIP: {stage}")
    os.replace(archives[0], output_path)
    shutil.rmtree(stage)
    digest = sha256_file(output_path)
    atomic_write_json(
        output_path.parent / "package_manifest.json",
        {
            "schema_version": 1,
            "date": index_row["date"],
            "dataset_ref": dataset_ref,
            "downloaded_at_utc": _utc_now(),
            "sha256": digest,
            "size_bytes": output_path.stat().st_size,
        },
    )
    return _ReplaySource(kind="zip", path=output_path, sha256=digest)


def _write_partition(
    source: _ReplaySource,
    output_path: Path,
    *,
    date: str,
    card_meta: dict[int, records.CardMeta],
    batch_rows: int,
    prefix_bytes: int,
    max_prefix_bytes: int,
) -> dict[str, int]:
    writer = pq.ParquetWriter(output_path, SIDE_SCHEMA, compression="zstd")
    buffer: list[dict[str, Any]] = []
    episodes = 0
    valid_episodes = 0
    unresolved_episodes = 0
    source_bytes = 0
    try:
        for name, size_bytes, replay in _iter_replays(
            source,
            date=date,
            card_meta=card_meta,
            prefix_bytes=prefix_bytes,
            max_prefix_bytes=max_prefix_bytes,
        ):
            del name
            compact = _compact_rows(replay)
            if len(compact) != 2:
                raise ValueError(
                    f"episode did not contain two registered decks: {date}"
                )
            episodes += 1
            source_bytes += size_bytes
            if all(bool(row["terminal_valid"]) for row in compact):
                valid_episodes += 1
            else:
                unresolved_episodes += 1
            buffer.extend(compact)
            if len(buffer) >= batch_rows:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SIDE_SCHEMA))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SIDE_SCHEMA))
    finally:
        writer.close()
    return {
        "episodes": episodes,
        "sides": episodes * 2,
        "valid_episodes": valid_episodes,
        "unresolved_episodes": unresolved_episodes,
        "source_uncompressed_bytes": source_bytes,
    }


def _iter_replays(
    source: _ReplaySource,
    *,
    date: str,
    card_meta: dict[int, records.CardMeta],
    prefix_bytes: int,
    max_prefix_bytes: int,
) -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    if source.kind == "raw":
        for path in sorted(source.path.glob("*.json")):
            yield (
                path.name,
                path.stat().st_size,
                _parse_file(
                    path,
                    date=date,
                    card_meta=card_meta,
                    prefix_bytes=prefix_bytes,
                    max_prefix_bytes=max_prefix_bytes,
                ),
            )
        return
    if source.kind == "zip":
        with zipfile.ZipFile(source.path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or not info.filename.endswith(".json"):
                    continue
                with archive.open(info) as stream:
                    prefix = stream.read(max_prefix_bytes)
                yield (
                    info.filename,
                    info.file_size,
                    _parse_member(
                        prefix,
                        full_reader=lambda info=info: _read_zip_member(archive, info),
                        replay_path=Path(date) / Path(info.filename).name,
                        size_bytes=info.file_size,
                        card_meta=card_meta,
                        prefix_bytes=prefix_bytes,
                        max_prefix_bytes=max_prefix_bytes,
                    ),
                )
        return
    yield from _iter_tar_zst(
        source.path,
        date=date,
        card_meta=card_meta,
        prefix_bytes=prefix_bytes,
        max_prefix_bytes=max_prefix_bytes,
    )


def _iter_tar_zst(
    path: Path,
    *,
    date: str,
    card_meta: dict[int, records.CardMeta],
    prefix_bytes: int,
    max_prefix_bytes: int,
) -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("failed to open zstd archive stream")
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                data = extracted.read()
                yield (
                    member.name,
                    member.size,
                    _parse_member(
                        data[:max_prefix_bytes],
                        full_reader=lambda data=data: data,
                        replay_path=Path(date) / Path(member.name).name,
                        size_bytes=member.size,
                        card_meta=card_meta,
                        prefix_bytes=prefix_bytes,
                        max_prefix_bytes=max_prefix_bytes,
                    ),
                )
    finally:
        process.stdout.close()
    stderr = b"" if process.stderr is None else process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd failed for {path}: {stderr.decode(errors='replace')}")


def _parse_file(
    path: Path,
    *,
    date: str,
    card_meta: dict[int, records.CardMeta],
    prefix_bytes: int,
    max_prefix_bytes: int,
) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        prefix = stream.read(max_prefix_bytes)
    return _parse_member(
        prefix,
        full_reader=path.read_bytes,
        replay_path=Path(date) / path.name,
        size_bytes=path.stat().st_size,
        card_meta=card_meta,
        prefix_bytes=prefix_bytes,
        max_prefix_bytes=max_prefix_bytes,
    )


def _parse_member(
    prefix: bytes,
    *,
    full_reader: Any,
    replay_path: Path,
    size_bytes: int,
    card_meta: dict[int, records.CardMeta],
    prefix_bytes: int,
    max_prefix_bytes: int,
) -> list[dict[str, Any]]:
    limit = prefix_bytes
    while limit <= max_prefix_bytes:
        rows = records.fast_episode_side_rows_from_bytes(
            data=prefix[:limit],
            replay_path=replay_path,
            size_bytes=size_bytes,
            card_meta=card_meta,
            known_decks={},
            include_step_count=False,
            require_first_player=True,
        )
        if rows is not None:
            return rows
        limit *= 2
    payload = orjson.loads(full_reader())
    if not isinstance(payload, dict):
        raise ValueError(f"replay member is not a JSON object: {replay_path}")
    return records.episode_side_rows_from_data(
        replay_path=replay_path,
        replay=payload,
        size_bytes=size_bytes,
        card_meta=card_meta,
        known_decks={},
    )


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_player = {int(row["player_index"]): row for row in rows}
    both_done = len(by_player) == 2 and all(
        str(row.get("status")) == "DONE" for row in by_player.values()
    )
    first_players = {row.get("first_player_index") for row in by_player.values()}
    first_player = next(iter(first_players)) if len(first_players) == 1 else None
    seat_valid = type(first_player) is int and first_player in (0, 1)
    output: list[dict[str, Any]] = []
    for player_index in sorted(by_player):
        row = by_player[player_index]
        opponent = by_player.get(1 - player_index, {})
        signature = str(row["deck_signature"])
        opponent_signature = str(row["opponent_deck_signature"])
        terminal_valid = bool(
            both_done and seat_valid and row.get("result") in {"win", "draw", "loss"}
        )
        output.append(
            {
                "date": str(row["date"]),
                "episode_id": int(row["episode_id"]),
                "player_index": player_index,
                "first_player_index": first_player if seat_valid else None,
                "went_first": player_index == first_player if seat_valid else None,
                "pilot_key": _pilot_key(str(row.get("team_name", ""))),
                "reward": row.get("reward"),
                "status": str(row.get("status", "")),
                "opponent_status": str(opponent.get("status", "")),
                "result": str(row["result"]) if terminal_valid else "unresolved",
                "terminal_valid": terminal_valid,
                "deck_digest": parse_canonical_signature(signature).deck_digest,
                "deck_signature": signature,
                "deck_hash": str(row["deck_hash"]),
                "deck_label": str(row["deck_label"]),
                "opponent_deck_digest": parse_canonical_signature(
                    opponent_signature
                ).deck_digest,
                "opponent_deck_signature": opponent_signature,
                "opponent_deck_hash": str(row["opponent_deck_hash"]),
                "opponent_deck_label": str(row["opponent_deck_label"]),
            }
        )
    return output


def partition_directory(config: PublicEnvironmentConfig, date: str) -> Path:
    """Return one stable Hive-style compact partition directory."""
    return repo_path(config.processed_root) / "daily" / f"date={date}"


def partition_manifest(config: PublicEnvironmentConfig, date: str) -> Path:
    """Return one compact partition manifest path."""
    return partition_directory(config, date) / "manifest.json"


def _dataset_ref(row: Mapping[str, str]) -> str:
    slug = row["daily_dataset_slug"]
    return slug if "/" in slug else f"kaggle/{slug}"


def _read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with archive.open(info) as stream:
        return stream.read()


def _pilot_key(value: str) -> str:
    normalized = " ".join(value.strip().casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _raw_member_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.glob("*.json")):
        stat = member.stat()
        digest.update(f"{member.name}\0{stat.st_size}\0".encode())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> Path:
    """Resolve a repository-relative operational path."""
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "SIDE_SCHEMA",
    "compact_required_dates",
    "partition_directory",
    "partition_manifest",
    "read_index",
    "refresh_index",
    "repo_path",
]
