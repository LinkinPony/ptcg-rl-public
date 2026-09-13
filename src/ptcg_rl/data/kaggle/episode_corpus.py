"""Build an immutable all-side replay inventory from daily Kaggle datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle import episode_archive

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_SCHEMA_VERSION = 2
REPLAY_COLUMNS = (
    "date",
    "episode_id",
    "relative_path",
    "size_bytes",
    "sha256",
    "split",
)
ReplaySplit = Literal["train", "validation", "test"]


class EpisodeCorpusConfig(BaseModel):
    """Validated source-inventory configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    archive_root: Path = Path("data/external/kaggle_top_episodes_archives")
    index_manifest: Path = Path(
        "data/external/kaggle_top_episodes_index/latest/manifest.csv"
    )
    output_dir: Path
    dates: tuple[str, ...] = Field(min_length=1)
    validation_dates: tuple[str, ...] = ()
    test_dates: tuple[str, ...] = ()
    selection: Literal["all_sides"] = "all_sides"
    require_archives: bool = True
    resume: bool = True

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique, ordered complete UTC dates."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        if tuple(sorted(set(values))) != values:
            raise ValueError("episode corpus dates must be unique and ordered")
        return values

    @model_validator(mode="after")
    def coherent_split_dates(self) -> EpisodeCorpusConfig:
        """Keep chronological held-out dates disjoint and inside the corpus."""
        validation = set(self.validation_dates)
        test = set(self.test_dates)
        if (
            len(validation) != len(self.validation_dates)
            or len(test) != len(self.test_dates)
            or validation & test
            or not validation.issubset(self.dates)
            or not test.issubset(self.dates)
        ):
            raise ValueError(
                "episode corpus validation/test dates must be unique, disjoint, "
                "and selected from dates"
            )
        return self


def run(config: EpisodeCorpusConfig) -> dict[str, Any]:
    """Validate daily raw/archive identities and publish one replay inventory."""
    replay_root = _repo_path(config.replay_root)
    archive_root = _repo_path(config.archive_root)
    index_manifest = _repo_path(config.index_manifest)
    output_dir = _repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = _indexed_rows(index_manifest)
    inventories = tuple(
        _daily_inventory(
            date,
            split=_split_for_date(date, config=config),
            replay_root=replay_root,
            archive_root=archive_root,
            index_row=_required_index_row(index_rows, date),
            require_archive=config.require_archives,
        )
        for date in config.dates
    )
    replay_manifest_path = output_dir / "replays.csv"
    replay_temp_path = output_dir / f".replays.{uuid.uuid4().hex}.csv"
    try:
        _write_replay_manifest(replay_temp_path, inventories)
        replay_manifest_sha256 = _sha256_file(replay_temp_path)
        _publish_or_validate(
            replay_temp_path,
            replay_manifest_path,
            resume=config.resume,
        )
    finally:
        replay_temp_path.unlink(missing_ok=True)

    source_manifest_path = output_dir / "manifest.json"
    payload = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": _display_path(replay_root),
        "selection": {
            "mode": config.selection,
            "leaderboard_cohort_filter": False,
        },
        "dates": list(config.dates),
        "index_manifest_path": _display_path(index_manifest),
        "index_manifest_sha256": _sha256_file(index_manifest),
        "outputs": {
            "replays_manifest_path": _display_path(replay_manifest_path),
            "replays_manifest_sha256": replay_manifest_sha256,
        },
        "per_date": [dict(inventory["identity"]) for inventory in inventories],
        "summary": {
            "source_replays": sum(
                int(inventory["identity"]["replay_count"]) for inventory in inventories
            ),
            "source_bytes": sum(
                int(inventory["identity"]["replay_bytes"]) for inventory in inventories
            ),
            "source_sides": "all_available",
            "leaderboard_cohort_filter": False,
        },
    }
    if config.validation_dates or config.test_dates:
        payload["split_assignment"] = {
            "unit": "complete_episode",
            "validation_dates": list(config.validation_dates),
            "test_dates": list(config.test_dates),
        }
    if source_manifest_path.is_file():
        if not config.resume:
            raise FileExistsError(source_manifest_path)
        existing = _read_json_object(source_manifest_path)
        _validate_existing_manifest(existing, payload)
        payload = existing
    else:
        _write_json_atomic(source_manifest_path, payload)
    summary = _mapping(payload.get("summary"))
    report = {
        "source_manifest_path": _display_path(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "replay_manifest_path": _display_path(replay_manifest_path),
        "replay_manifest_sha256": replay_manifest_sha256,
        "dates": list(config.dates),
        "source_replays": int(summary["source_replays"]),
        "source_bytes": int(summary["source_bytes"]),
        "selection": config.selection,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _daily_inventory(
    date: str,
    *,
    split: ReplaySplit,
    replay_root: Path,
    archive_root: Path,
    index_row: Mapping[str, str],
    require_archive: bool,
) -> dict[str, Any]:
    date_dir = replay_root / date
    daily_manifest_path = date_dir / "manifest.csv"
    if not daily_manifest_path.is_file():
        raise FileNotFoundError(f"daily replay manifest does not exist: {date}")
    rows = _read_csv(daily_manifest_path)
    expected_count = int(index_row["episode_count"])
    if len(rows) != expected_count:
        raise ValueError(
            f"daily replay count differs from index for {date}: "
            f"{len(rows)} != {expected_count}"
        )
    by_episode: dict[int, tuple[Path, int]] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id in by_episode:
            raise ValueError(f"duplicate episode ID {episode_id} on {date}")
        path = date_dir / f"{episode_id}.json"
        size_bytes = int(row["size_bytes"])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != size_bytes:
            raise ValueError(f"daily replay size differs from manifest: {path}")
        by_episode[episode_id] = (path, size_bytes)
    json_names = {
        path.name
        for path in date_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    }
    expected_names = {path.name for path, _size in by_episode.values()}
    if json_names != expected_names:
        raise ValueError(f"daily replay files differ from manifest for {date}")
    replay_bytes = sum(size for _path, size in by_episode.values())
    if replay_bytes != int(index_row["total_bytes"]):
        raise ValueError(
            f"daily replay bytes differ from index for {date}: "
            f"{replay_bytes} != {index_row['total_bytes']}"
        )

    archive_identity: dict[str, Any] = {}
    archive_path = episode_archive.archive_path(archive_root, date)
    archive_manifest_path = episode_archive.manifest_path(archive_root, date)
    if require_archive and (
        not archive_path.is_file() or not archive_manifest_path.is_file()
    ):
        raise FileNotFoundError(f"complete replay archive does not exist for {date}")
    if archive_path.is_file() or archive_manifest_path.is_file():
        if not archive_path.is_file() or not archive_manifest_path.is_file():
            raise FileNotFoundError(f"partial replay archive exists for {date}")
        archive_manifest = _read_json_object(archive_manifest_path)
        archive_sha256 = _sha256_file(archive_path)
        if archive_sha256 != str(archive_manifest.get("archive_sha256", "")):
            raise ValueError(f"replay archive SHA-256 differs for {date}")
        source = _mapping(archive_manifest.get("source"))
        expected_files = len(rows) + 1
        expected_bytes = replay_bytes + daily_manifest_path.stat().st_size
        expected_member_index_sha256 = _member_index_sha256(
            (
                ("manifest.csv", daily_manifest_path.stat().st_size),
                *(
                    (path.name, size_bytes)
                    for path, size_bytes in by_episode.values()
                ),
            )
        )
        if (
            int(source.get("file_count", -1)) != expected_files
            or int(source.get("total_bytes", -1)) != expected_bytes
            or str(source.get("member_index_sha256", ""))
            != expected_member_index_sha256
        ):
            raise ValueError(f"replay archive source inventory differs for {date}")
        archive_identity = {
            "archive_path": _display_path(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "archive_manifest_path": _display_path(archive_manifest_path),
            "archive_manifest_sha256": _sha256_file(archive_manifest_path),
            "member_index_sha256": str(source.get("member_index_sha256", "")),
        }

    return {
        "identity": {
            "date": date,
            "daily_dataset_slug": index_row["daily_dataset_slug"],
            "replay_count": len(rows),
            "replay_bytes": replay_bytes,
            "daily_manifest_path": _display_path(daily_manifest_path),
            "daily_manifest_sha256": _sha256_file(daily_manifest_path),
            **archive_identity,
        },
        "replays": tuple(
            {
                "date": date,
                "episode_id": episode_id,
                "relative_path": path.relative_to(replay_root).as_posix(),
                "size_bytes": size_bytes,
                "sha256": "",
                "split": split,
            }
            for episode_id, (path, size_bytes) in sorted(by_episode.items())
        ),
    }


def _write_replay_manifest(
    path: Path,
    inventories: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=REPLAY_COLUMNS)
        writer.writeheader()
        for inventory in inventories:
            for row in cast(Sequence[Mapping[str, object]], inventory["replays"]):
                writer.writerow(row)
        file_obj.flush()
        os.fsync(file_obj.fileno())


def _publish_or_validate(source: Path, target: Path, *, resume: bool) -> None:
    if target.is_file():
        if not resume:
            raise FileExistsError(target)
        if _sha256_file(source) != _sha256_file(target):
            raise ValueError("existing replay inventory differs from validated sources")
        return
    os.replace(source, target)
    _fsync_directory(target.parent)


def _validate_existing_manifest(
    existing: Mapping[str, Any],
    proposed: Mapping[str, Any],
) -> None:
    for field in (
        "schema_version",
        "source_root",
        "selection",
        "dates",
        "index_manifest_sha256",
        "outputs",
        "per_date",
        "summary",
    ):
        if existing.get(field) != proposed.get(field):
            raise ValueError(f"existing source manifest differs at {field}")
    if existing.get("split_assignment") != proposed.get("split_assignment"):
        raise ValueError("existing source manifest differs at split_assignment")


def _split_for_date(
    date: str,
    *,
    config: EpisodeCorpusConfig,
) -> ReplaySplit:
    """Assign every replay on one UTC date to the same durable split."""
    if date in config.validation_dates:
        return "validation"
    if date in config.test_dates:
        return "test"
    return "train"


def _indexed_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    required = {
        "date",
        "daily_dataset_slug",
        "episode_count",
        "total_bytes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Kaggle episode index is missing required fields")
    by_date = {row["date"]: row for row in rows}
    if len(by_date) != len(rows):
        raise ValueError("Kaggle episode index contains duplicate dates")
    return by_date


def _required_index_row(
    rows: Mapping[str, dict[str, str]],
    date: str,
) -> dict[str, str]:
    try:
        return rows[date]
    except KeyError as error:
        raise ValueError(f"date is absent from Kaggle episode index: {date}") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _member_index_sha256(members: Sequence[tuple[str, int]]) -> str:
    """Hash the exact replay member names and sizes like the archive contract."""
    digest = hashlib.sha256()
    for name, size_bytes in sorted(members):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size_bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_episode_corpus_cold_bc18_20260730",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    run(EpisodeCorpusConfig.model_validate(cast(dict[str, Any], raw_config)))


if __name__ == "__main__":
    main()
