"""Archive and restore immutable Kaggle episode replay datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[4]
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
ARCHIVE_SUFFIX = ".tar.zst"
MANIFEST_SUFFIX = ".manifest.json"
FORMAT_VERSION = 1


class EpisodeArchiveConfig(BaseModel):
    """Configuration for replay archival and restoration."""

    model_config = ConfigDict(extra="forbid")

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    archive_root: Path = Path("data/external/kaggle_top_episodes_archives")
    index_manifest: Path | None = Path(
        "data/external/kaggle_top_episodes_index/latest/manifest.csv"
    )
    mode: Literal["archive", "restore"] = "archive"
    dates: list[str] = Field(default_factory=list)
    keep_raw_dates: int = 2
    remove_source: bool = True
    compression_level: int = 1
    zstd_binary: str = "zstd"
    dry_run: bool = False

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("keep_raw_dates")
    @classmethod
    def valid_keep_raw_dates(cls, value: int) -> int:
        """Reject a negative retention count."""
        if value < 0:
            raise ValueError("keep_raw_dates must be non-negative")
        return value

    @field_validator("compression_level")
    @classmethod
    def valid_compression_level(cls, value: int) -> int:
        """Restrict compression to regular zstd levels."""
        if value < 1 or value > 19:
            raise ValueError("compression_level must be in [1, 19]")
        return value

    @model_validator(mode="after")
    def restore_requires_dates(self) -> EpisodeArchiveConfig:
        """Prevent an accidental restoration of the entire archive."""
        if self.mode == "restore" and not self.dates:
            raise ValueError("restore mode requires at least one explicit date")
        return self


class _SourceSnapshot(BaseModel):
    """Stable metadata for one raw date directory."""

    model_config = ConfigDict(frozen=True)

    file_count: int
    total_bytes: int
    member_index_sha256: str
    file_state: tuple[tuple[str, int, int, int], ...] = Field(exclude=True)


def run(config: EpisodeArchiveConfig) -> dict[str, Any]:
    """Archive or restore selected replay dates."""
    start = time.perf_counter()
    if not config.dry_run and shutil.which(config.zstd_binary) is None:
        raise RuntimeError(f"zstd binary not found: {config.zstd_binary}")

    if config.mode == "restore":
        actions = [_restore_date(date, config) for date in config.dates]
    else:
        actions = [
            _archive_date(date_dir, config)
            for date_dir in _archive_candidates(config)
        ]

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "actions": actions,
        "summary": _summary(actions),
        "elapsed_seconds": time.perf_counter() - start,
    }
    logger.info(
        "finished Kaggle replay {} actions={} source_bytes={} archive_bytes={} "
        "seconds={:.2f}",
        config.mode,
        len(actions),
        report["summary"]["source_bytes"],
        report["summary"]["archive_bytes"],
        report["elapsed_seconds"],
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return report


def archive_is_available(archive_root: Path, date: str) -> bool:
    """Return whether an archive and its integrity manifest both exist."""
    resolved_root = repo_path(archive_root)
    return archive_path(resolved_root, date).is_file() and manifest_path(
        resolved_root, date
    ).is_file()


def archive_path(archive_root: Path, date: str) -> Path:
    """Return the archive path for a date."""
    return archive_root / f"{date}{ARCHIVE_SUFFIX}"


def manifest_path(archive_root: Path, date: str) -> Path:
    """Return the archive manifest path for a date."""
    return archive_root / f"{date}{MANIFEST_SUFFIX}"


def _archive_candidates(config: EpisodeArchiveConfig) -> list[Path]:
    replay_root = repo_path(config.replay_root)
    if not replay_root.exists():
        raise FileNotFoundError(f"replay_root does not exist: {replay_root}")
    date_dirs = sorted(
        path
        for path in replay_root.iterdir()
        if path.is_dir() and DATE_PATTERN.fullmatch(path.name)
    )
    if config.dates:
        by_date = {path.name: path for path in date_dirs}
        missing = [date for date in config.dates if date not in by_date]
        if missing:
            raise FileNotFoundError(f"raw replay dates do not exist: {missing}")
        return [by_date[date] for date in config.dates]
    if config.keep_raw_dates == 0:
        return date_dirs
    return date_dirs[: -config.keep_raw_dates]


def _archive_date(
    date_dir: Path,
    config: EpisodeArchiveConfig,
) -> dict[str, Any]:
    date = date_dir.name
    source = _source_snapshot(date_dir)
    resolved_archive_root = repo_path(config.archive_root)
    output_path = archive_path(resolved_archive_root, date)
    output_manifest_path = manifest_path(resolved_archive_root, date)
    logger.info(
        "archiving Kaggle replay date={} files={} source_bytes={} archive={}",
        date,
        source.file_count,
        source.total_bytes,
        display_path(output_path),
    )
    if config.dry_run:
        return _action(
            date=date,
            status="planned_archive",
            source=source,
            output_path=output_path,
        )

    resolved_archive_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        archive_details = _reuse_existing_archive(
            output_path,
            output_manifest_path,
            source,
            config,
        )
        status = "reused_archive"
    else:
        if output_manifest_path.exists():
            raise RuntimeError(
                f"archive manifest exists without archive: {output_manifest_path}"
            )
        archive_details = _build_archive(
            date_dir=date_dir,
            output_path=output_path,
            source=source,
            config=config,
        )
        archive_manifest = _archive_manifest(
            date=date,
            output_path=output_path,
            source=source,
            archive_details=archive_details,
            config=config,
        )
        _write_json_atomic(output_manifest_path, archive_manifest)
        status = "archived"

    if _source_snapshot(date_dir) != source:
        raise RuntimeError(f"raw replay files changed during archival: {date_dir}")
    if config.remove_source:
        shutil.rmtree(date_dir)
        status += "_source_removed"
    else:
        status += "_source_kept"
    logger.info(
        "archived Kaggle replay date={} status={} archive_bytes={} sha256={}",
        date,
        status,
        archive_details["archive_bytes"],
        archive_details["archive_sha256"],
    )
    return _action(
        date=date,
        status=status,
        source=source,
        output_path=output_path,
        archive_details=archive_details,
    )


def _build_archive(
    *,
    date_dir: Path,
    output_path: Path,
    source: _SourceSnapshot,
    config: EpisodeArchiveConfig,
) -> dict[str, Any]:
    file_paths = _source_files(date_dir)
    file_descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{date_dir.name}.",
        suffix=f"{ARCHIVE_SUFFIX}.tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(raw_temp_path)
    process: subprocess.Popen[bytes] | None = None
    try:
        with temp_path.open("wb") as output_file:
            process = subprocess.Popen(
                [
                    config.zstd_binary,
                    f"-{config.compression_level}",
                    "--quiet",
                    "--check",
                    "--stdout",
                ],
                stdin=subprocess.PIPE,
                stdout=output_file,
            )
            if process.stdin is None:
                raise RuntimeError("failed to open zstd input pipe")
            with tarfile.open(
                fileobj=process.stdin,
                mode="w|",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for file_path in file_paths:
                    archive.add(file_path, arcname=file_path.name, recursive=False)
            process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)
            output_file.flush()
            os.fsync(output_file.fileno())

        archive_details = _inspect_archive(temp_path, config.zstd_binary)
        _require_matching_source(source, archive_details, temp_path)
        archive_details["archive_sha256"] = _sha256_file(temp_path)
        archive_details["archive_bytes"] = temp_path.stat().st_size
        os.replace(temp_path, output_path)
        _fsync_directory(output_path.parent)
        return archive_details
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        temp_path.unlink(missing_ok=True)
        raise


def _reuse_existing_archive(
    output_path: Path,
    output_manifest_path: Path,
    source: _SourceSnapshot,
    config: EpisodeArchiveConfig,
) -> dict[str, Any]:
    archive_details = _inspect_archive(output_path, config.zstd_binary)
    _require_matching_source(source, archive_details, output_path)
    archive_details["archive_sha256"] = _sha256_file(output_path)
    archive_details["archive_bytes"] = output_path.stat().st_size
    if output_manifest_path.exists():
        stored = _read_json_object(output_manifest_path)
        _require_matching_manifest(stored, archive_details, source)
    else:
        recovered_manifest = _archive_manifest(
            date=output_path.name.removesuffix(ARCHIVE_SUFFIX),
            output_path=output_path,
            source=source,
            archive_details=archive_details,
            config=config,
        )
        _write_json_atomic(output_manifest_path, recovered_manifest)
    return archive_details


def _restore_date(date: str, config: EpisodeArchiveConfig) -> dict[str, Any]:
    replay_root = repo_path(config.replay_root)
    resolved_archive_root = repo_path(config.archive_root)
    input_path = archive_path(resolved_archive_root, date)
    input_manifest_path = manifest_path(resolved_archive_root, date)
    target_dir = replay_root / date
    if target_dir.exists():
        source = _source_snapshot(target_dir)
        return _action(
            date=date,
            status="skipped_existing_raw",
            source=source,
            output_path=input_path,
        )
    if not input_path.is_file() or not input_manifest_path.is_file():
        raise FileNotFoundError(f"complete archive does not exist for date {date}")
    stored_manifest = _read_json_object(input_manifest_path)
    expected_source = _source_from_manifest(stored_manifest)
    if config.dry_run:
        return _action(
            date=date,
            status="planned_restore",
            source=expected_source,
            output_path=input_path,
        )

    expected_sha256 = str(stored_manifest.get("archive_sha256", ""))
    actual_sha256 = _sha256_file(input_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"archive SHA-256 mismatch for {input_path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    replay_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{date}.restore.", dir=replay_root))
    try:
        _extract_archive(input_path, temp_dir, config.zstd_binary)
        restored = _source_snapshot(temp_dir, require_date_name=False)
        if not _same_source_content(restored, expected_source):
            raise RuntimeError(f"restored replay metadata mismatch for date {date}")
        os.replace(temp_dir, target_dir)
        _fsync_directory(replay_root)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    logger.info(
        "restored Kaggle replay date={} files={} bytes={} target={}",
        date,
        expected_source.file_count,
        expected_source.total_bytes,
        display_path(target_dir),
    )
    return _action(
        date=date,
        status="restored",
        source=expected_source,
        output_path=input_path,
    )


def _extract_archive(input_path: Path, output_dir: Path, zstd_binary: str) -> None:
    process = subprocess.Popen(
        [zstd_binary, "--quiet", "--decompress", "--stdout", str(input_path)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("failed to open zstd output pipe")
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                _validate_member(member, input_path)
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise RuntimeError(f"could not read archive member {member.name}")
                destination = output_dir / member.name
                with destination.open("wb") as output_file:
                    shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise


def _inspect_archive(input_path: Path, zstd_binary: str) -> dict[str, Any]:
    process = subprocess.Popen(
        [zstd_binary, "--quiet", "--decompress", "--stdout", str(input_path)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("failed to open zstd output pipe")
    members: list[tuple[str, int]] = []
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                _validate_member(member, input_path)
                members.append((member.name, member.size))
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    return {
        "file_count": len(members),
        "total_bytes": sum(size for _, size in members),
        "member_index_sha256": _member_index_sha256(members),
    }


def _validate_member(member: tarfile.TarInfo, input_path: Path) -> None:
    member_path = Path(member.name)
    if (
        not member.isreg()
        or member_path.name != member.name
        or not _is_expected_member_name(member.name)
    ):
        raise RuntimeError(f"unsafe archive member in {input_path}: {member.name}")


def _source_snapshot(
    date_dir: Path,
    *,
    require_date_name: bool = True,
) -> _SourceSnapshot:
    files = _source_files(date_dir, require_date_name=require_date_name)
    members: list[tuple[str, int]] = []
    file_state: list[tuple[str, int, int, int]] = []
    for file_path in files:
        stat_result = file_path.stat()
        members.append((file_path.name, stat_result.st_size))
        file_state.append(
            (
                file_path.name,
                stat_result.st_size,
                stat_result.st_mtime_ns,
                stat_result.st_ino,
            )
        )
    return _SourceSnapshot(
        file_count=len(files),
        total_bytes=sum(size for _, size in members),
        member_index_sha256=_member_index_sha256(members),
        file_state=tuple(file_state),
    )


def _source_files(
    date_dir: Path,
    *,
    require_date_name: bool = True,
) -> list[Path]:
    if not date_dir.is_dir() or (
        require_date_name and not DATE_PATTERN.fullmatch(date_dir.name)
    ):
        raise ValueError(f"invalid replay date directory: {date_dir}")
    entries = sorted(date_dir.iterdir())
    invalid = [
        path.name
        for path in entries
        if not path.is_file()
        or path.is_symlink()
        or not _is_expected_member_name(path.name)
    ]
    if invalid:
        raise RuntimeError(
            f"refusing to archive date with unexpected entries {date_dir}: {invalid[:10]}"
        )
    if not entries:
        raise RuntimeError(f"replay date contains no JSON files: {date_dir}")
    return entries


def _is_expected_member_name(name: str) -> bool:
    return name.endswith(".json") or name == "manifest.csv"


def _member_index_sha256(members: list[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for name, size in sorted(members):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_manifest(
    *,
    date: str,
    output_path: Path,
    source: _SourceSnapshot,
    archive_details: Mapping[str, Any],
    config: EpisodeArchiveConfig,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "date": date,
        "archive_format": "tar+zstd",
        "archive_path": display_path(output_path),
        "archive_bytes": int(archive_details["archive_bytes"]),
        "archive_sha256": str(archive_details["archive_sha256"]),
        "compression_level": config.compression_level,
        "source": {
            "replay_root": display_path(repo_path(config.replay_root)),
            "file_count": source.file_count,
            "total_bytes": source.total_bytes,
            "member_index_sha256": source.member_index_sha256,
            "dataset": _dataset_metadata(config.index_manifest, date),
        },
    }


def _dataset_metadata(index_manifest: Path | None, date: str) -> dict[str, str]:
    if index_manifest is None:
        return {}
    resolved_path = repo_path(index_manifest)
    if not resolved_path.is_file():
        return {}
    with resolved_path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("date") == date:
                return dict(row)
    return {}


def _require_matching_source(
    source: _SourceSnapshot,
    archive_details: Mapping[str, Any],
    path: Path,
) -> None:
    if (
        int(archive_details["file_count"]) != source.file_count
        or int(archive_details["total_bytes"]) != source.total_bytes
        or str(archive_details["member_index_sha256"])
        != source.member_index_sha256
    ):
        raise RuntimeError(f"archive contents do not match raw replay files: {path}")


def _require_matching_manifest(
    stored: Mapping[str, Any],
    archive_details: Mapping[str, Any],
    source: _SourceSnapshot,
) -> None:
    expected_source = _source_from_manifest(stored)
    if not _same_source_content(expected_source, source):
        raise RuntimeError("existing archive manifest does not match raw replay files")
    if (
        int(stored.get("archive_bytes", -1))
        != int(archive_details["archive_bytes"])
        or str(stored.get("archive_sha256", ""))
        != str(archive_details["archive_sha256"])
    ):
        raise RuntimeError("existing archive does not match its integrity manifest")


def _source_from_manifest(stored: Mapping[str, Any]) -> _SourceSnapshot:
    raw_source = stored.get("source")
    if not isinstance(raw_source, Mapping):
        raise ValueError("archive manifest has no source metadata")
    return _SourceSnapshot(
        file_count=int(raw_source["file_count"]),
        total_bytes=int(raw_source["total_bytes"]),
        member_index_sha256=str(raw_source["member_index_sha256"]),
        file_state=(),
    )


def _same_source_content(left: _SourceSnapshot, right: _SourceSnapshot) -> bool:
    return (
        left.file_count == right.file_count
        and left.total_bytes == right.total_bytes
        and left.member_index_sha256 == right.member_index_sha256
    )


def _action(
    *,
    date: str,
    status: str,
    source: _SourceSnapshot,
    output_path: Path,
    archive_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    details = archive_details or {}
    return {
        "date": date,
        "status": status,
        "source_files": source.file_count,
        "source_bytes": source.total_bytes,
        "archive_path": display_path(output_path),
        "archive_bytes": int(details.get("archive_bytes", 0)),
        "archive_sha256": str(details.get("archive_sha256", "")),
    }


def _summary(actions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "dates": len(actions),
        "source_files": sum(int(action["source_files"]) for action in actions),
        "source_bytes": sum(int(action["source_bytes"]) for action in actions),
        "archive_bytes": sum(int(action["archive_bytes"]) for action in actions),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    file_descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actions": report["actions"],
        "summary": report["summary"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def repo_path(path: Path) -> Path:
    """Resolve a repository-relative path."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path) -> str:
    """Render a path relative to the repository when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_episode_archive",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = EpisodeArchiveConfig.model_validate(cast(dict[str, Any], raw_config))
    run(config)


if __name__ == "__main__":
    main()
