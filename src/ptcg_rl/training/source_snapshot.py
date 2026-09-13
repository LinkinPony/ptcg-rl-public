"""Materialize immutable no-.git source trees for formal training processes."""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ptcg_rl.training.source_identity import (
    TRAINING_RUNTIME_ARTIFACT_PATHS,
    TRAINING_SOURCE_MANIFEST,
    TRAINING_SOURCE_PATHS,
    TRAINING_SOURCE_SUBMODULE_PATHS,
    TrainingSourceIdentity,
    TrainingSourceSubmodule,
    build_filesystem_source_identity,
    build_filesystem_submodule_identity,
    load_training_source_identity,
    verify_training_source_inventory,
    write_training_source_identity,
)


class MaterializedTrainingSource(BaseModel):
    """One verified immutable source tree ready for process execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    manifest_path: Path
    identity: TrainingSourceIdentity


def materialize_training_source(
    repo_root: Path,
    *,
    revision: str,
) -> MaterializedTrainingSource:
    """Materialize and verify a reusable source tree under repository tmp/."""
    resolved_root = repo_root.resolve()
    cache_root = resolved_root / "tmp" / "training_source_snapshots"
    normalized_revision = revision.strip().lower()
    if _is_full_git_identity(normalized_revision):
        cached = _load_cached_training_source(
            cache_root / normalized_revision,
            expected_commit=normalized_revision,
        )
        if cached is not None:
            return cached
    commit = _git_text(
        ("rev-parse", f"{revision}^{{commit}}"),
        repo_root=resolved_root,
    ).strip()
    target = cache_root / commit
    cached = _load_cached_training_source(target, expected_commit=commit)
    if cached is not None:
        return cached

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = cache_root / f".{commit}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    snapshot_root = staging / "root"
    archive_path = staging / "source.tar"
    try:
        snapshot_root.mkdir(parents=True)
        _git_archive(resolved_root, commit=commit, destination=archive_path)
        with tarfile.open(archive_path, mode="r:") as archive:
            archive.extractall(snapshot_root, filter="data")
        relative_paths = _git_paths(
            (
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                commit,
                "--",
                *TRAINING_SOURCE_PATHS,
            ),
            repo_root=resolved_root,
        )
        submodules = _materialize_training_submodules(
            resolved_root,
            snapshot_root=snapshot_root,
            superproject_commit=commit,
            staging_root=staging,
            cache_root=cache_root,
        )
        identity = build_filesystem_source_identity(
            snapshot_root,
            commit=commit,
            relative_paths=relative_paths,
            submodules=submodules,
        )
        write_training_source_identity(
            snapshot_root / TRAINING_SOURCE_MANIFEST,
            identity,
        )
        _copy_runtime_artifacts(snapshot_root, resolved_root)
        _link_runtime_artifacts(snapshot_root, resolved_root)
        archive_path.unlink()
        try:
            os.replace(snapshot_root, target)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            existing = load_training_source_identity(target / TRAINING_SOURCE_MANIFEST)
            if existing != identity:
                raise RuntimeError("concurrent training source cache differs") from None
        verify_training_source_inventory(target, identity)
        return MaterializedTrainingSource(
            root=target,
            manifest_path=target / TRAINING_SOURCE_MANIFEST,
            identity=identity,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_cached_training_source(
    target: Path,
    *,
    expected_commit: str,
) -> MaterializedTrainingSource | None:
    """Load a verified snapshot without requiring its Git object to remain local."""
    manifest_path = target / TRAINING_SOURCE_MANIFEST
    if not manifest_path.is_file():
        return None
    identity = load_training_source_identity(manifest_path)
    if identity.source_git_commit != expected_commit:
        raise RuntimeError("cached training source commit differs")
    verify_training_source_inventory(target, identity)
    return MaterializedTrainingSource(
        root=target,
        manifest_path=manifest_path,
        identity=identity,
    )


def _is_full_git_identity(value: str) -> bool:
    """Return whether a revision is an unambiguous SHA-1 or SHA-256 identity."""
    return len(value) in (40, 64) and all(
        character in "0123456789abcdef" for character in value
    )


def require_clean_main_checkout(repo_root: Path) -> None:
    """Require the authoritative clean local main before a formal launch."""
    resolved_root = repo_root.resolve()
    branch = _git_text(
        ("branch", "--show-current"),
        repo_root=resolved_root,
    ).strip()
    if branch != "main":
        raise RuntimeError("formal training must launch from the local main branch")
    status_output = _git_text(
        ("status", "--porcelain", "--untracked-files=all"),
        repo_root=resolved_root,
    )
    if status_output:
        raise RuntimeError("formal training requires a clean working tree")


def _link_runtime_artifacts(snapshot_root: Path, live_root: Path) -> None:
    for name in ("outputs", "tmp"):
        target = snapshot_root / name
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        target.symlink_to(live_root / name, target_is_directory=True)
    live_data = live_root / "data"
    snapshot_data = snapshot_root / "data"
    if not live_data.is_dir() or not snapshot_data.is_dir():
        return
    for child in live_data.iterdir():
        target = snapshot_data / child.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())


def _copy_runtime_artifacts(snapshot_root: Path, live_root: Path) -> None:
    for relative_path in TRAINING_RUNTIME_ARTIFACT_PATHS:
        source = live_root / relative_path
        if not source.is_file():
            continue
        target = snapshot_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _git_archive(repo_root: Path, *, commit: str, destination: Path) -> None:
    completed = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "archive",
            "--format=tar",
            f"--output={destination}",
            commit,
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "training source archive failed: " + completed.stderr.strip()
        )


def _git_paths(arguments: tuple[str, ...], *, repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "-c", f"safe.directory={repo_root}", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "training source Git inventory failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return tuple(
        sorted(item.decode("utf-8") for item in completed.stdout.split(b"\x00") if item)
    )


def _git_text(arguments: tuple[str, ...], *, repo_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-c", f"safe.directory={repo_root}", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "training source Git identity failed: " + completed.stderr.strip()
        )
    return completed.stdout


def _materialize_training_submodules(
    repo_root: Path,
    *,
    snapshot_root: Path,
    superproject_commit: str,
    staging_root: Path,
    cache_root: Path,
) -> tuple[TrainingSourceSubmodule, ...]:
    identities: list[TrainingSourceSubmodule] = []
    for index, relative_path in enumerate(TRAINING_SOURCE_SUBMODULE_PATHS):
        pinned = _gitlink_commit(
            repo_root,
            revision=superproject_commit,
            relative_path=relative_path,
        )
        if pinned is None:
            continue
        live_module_root = (repo_root / relative_path).resolve()
        module_root = snapshot_root / relative_path
        module_root.mkdir(parents=True, exist_ok=True)
        try:
            relative_paths = _export_submodule_tree(
                live_module_root,
                snapshot_root=module_root,
                commit=pinned,
                staging_root=staging_root,
                archive_key=f"{index}",
            )
        except RuntimeError as export_error:
            cached = _restore_cached_submodule(
                cache_root,
                snapshot_root=module_root,
                relative_path=relative_path,
                commit=pinned,
            )
            if cached is None:
                raise export_error
            relative_paths = cached
        identities.append(
            build_filesystem_submodule_identity(
                module_root,
                relative_path=relative_path,
                commit=pinned,
                relative_paths=relative_paths,
            )
        )
    return tuple(identities)


def _restore_cached_submodule(
    cache_root: Path,
    *,
    snapshot_root: Path,
    relative_path: str,
    commit: str,
) -> tuple[str, ...] | None:
    """Restore one verified pinned submodule when its local Git object is absent."""
    if not cache_root.is_dir():
        return None
    for candidate in sorted(cache_root.iterdir(), reverse=True):
        if not candidate.is_dir() or not _is_full_git_identity(candidate.name):
            continue
        manifest_path = candidate / TRAINING_SOURCE_MANIFEST
        if not manifest_path.is_file():
            continue
        identity = load_training_source_identity(manifest_path)
        cached_submodule = next(
            (
                item
                for item in identity.submodules
                if item.path == relative_path and item.source_git_commit == commit
            ),
            None,
        )
        if cached_submodule is None:
            continue
        verify_training_source_inventory(candidate, identity)
        source_root = candidate / relative_path
        relative_paths = tuple(item.path for item in cached_submodule.files)
        _copy_source_files(
            source_root,
            snapshot_root,
            relative_paths=relative_paths,
        )
        restored = build_filesystem_submodule_identity(
            snapshot_root,
            relative_path=relative_path,
            commit=commit,
            relative_paths=relative_paths,
        )
        if restored != cached_submodule:
            raise RuntimeError("cached training submodule inventory differs")
        return relative_paths
    return None


def _copy_source_files(
    source_root: Path,
    target_root: Path,
    *,
    relative_paths: tuple[str, ...],
) -> None:
    """Copy only manifest-declared files while preserving modes and symlinks."""
    for relative_path in relative_paths:
        source = source_root / relative_path
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)


def _export_submodule_tree(
    live_root: Path,
    *,
    snapshot_root: Path,
    commit: str,
    staging_root: Path,
    archive_key: str,
) -> tuple[str, ...]:
    archive_path = staging_root / f"submodule-{archive_key}.tar"
    _git_archive(
        live_root,
        commit=commit,
        destination=archive_path,
    )
    with tarfile.open(archive_path, mode="r:") as archive:
        archive.extractall(snapshot_root, filter="data")
    archive_path.unlink()
    files, _nested = _submodule_tree_inventory(live_root, commit=commit)
    return files


def _submodule_tree_inventory(
    repo_root: Path,
    *,
    commit: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    completed = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "ls-tree",
            "-r",
            "-z",
            commit,
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "training submodule inventory failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    files: list[str] = []
    nested: list[tuple[str, str]] = []
    for raw_record in completed.stdout.split(b"\x00"):
        if not raw_record:
            continue
        metadata, raw_path = raw_record.split(b"\t", maxsplit=1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if object_type == "blob" and mode in {"100644", "100755", "120000"}:
            files.append(path)
        elif object_type == "commit" and mode == "160000":
            nested.append((path, object_id))
        else:
            raise RuntimeError(f"unsupported training submodule entry: {path}")
    return tuple(sorted(files)), tuple(sorted(nested))


def _gitlink_commit(
    repo_root: Path,
    *,
    revision: str,
    relative_path: str,
) -> str | None:
    output = _git_text(
        ("ls-tree", revision, "--", relative_path),
        repo_root=repo_root,
    ).strip()
    if not output:
        return None
    metadata, observed_path = output.split("\t", maxsplit=1)
    mode, object_type, commit = metadata.split()
    if observed_path != relative_path or mode != "160000" or object_type != "commit":
        raise RuntimeError(
            f"training dependency is not a Git submodule: {relative_path}"
        )
    return commit


__all__ = [
    "MaterializedTrainingSource",
    "materialize_training_source",
    "require_clean_main_checkout",
]
