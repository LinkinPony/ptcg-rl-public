"""Content-addressed identity and immutable binding for formal training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_FINGERPRINT_DOMAIN = b"ptcg-rl/training-source-content/v1\x00"
TRAINING_SOURCE_PATHS = (
    "src",
    "data/sample_submission",
    "train.py",
    "pyproject.toml",
)
TRAINING_SOURCE_SUBMODULE_PATHS = (
    "third_party/pokemon-tcg-ai-battle",
)
TRAINING_RUNTIME_ARTIFACT_PATHS = (
    "src/native/cg_train/libcg_train.so",
)
TRAINING_SOURCE_MANIFEST = ".ptcg_training_source.json"
_RUN_BINDING_PATH = Path("control/training_source_identity.json")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_SOURCE_MANIFEST_ENV = "PTCG_RL_TRAINING_SOURCE_MANIFEST"
_ADOPT_SOURCE_ENV = "PTCG_RL_ADOPT_TRAINING_SOURCE_IDENTITY"


class TrainingSourceFile(BaseModel):
    """One tracked executable-source file in a portable inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    mode: Literal["100644", "100755", "120000"]
    sha256: str

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        """Require a normalized repository-relative path."""
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("training source paths must be normalized and relative")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256 content identity."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("training source file identity must be SHA-256")
        return value


class TrainingSourceSubmodule(BaseModel):
    """Pinned, content-addressed source exported from one Git submodule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    source_git_commit: str
    content_fingerprint: str
    files: tuple[TrainingSourceFile, ...]

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        """Require one normalized repository-relative submodule path."""
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("training submodule paths must be normalized and relative")
        return value

    @field_validator("source_git_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        """Require the full Git object pinned by the superproject."""
        normalized = value.strip().lower()
        if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
            raise ValueError("training submodule commit must be a full Git identity")
        return normalized

    @field_validator("content_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require one lowercase SHA-256 tree identity."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("training submodule fingerprint must be SHA-256")
        return value

    @model_validator(mode="after")
    def coherent_inventory(self) -> Self:
        """Bind the submodule fingerprint to its sorted file inventory."""
        paths = tuple(item.path for item in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("training submodule inventory must be sorted and unique")
        if _inventory_fingerprint(self.files) != self.content_fingerprint:
            raise ValueError("training submodule inventory fingerprint differs")
        return self


class TrainingSourceIdentity(BaseModel):
    """Provenance commit plus commit-independent executable compatibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["ptcg-training-source-v1"] = "ptcg-training-source-v1"
    source_git_commit: str
    training_source_fingerprint: str
    files: tuple[TrainingSourceFile, ...]
    submodules: tuple[TrainingSourceSubmodule, ...] = ()

    @field_validator("source_git_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        """Require one full Git object identity."""
        normalized = value.strip().lower()
        if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
            raise ValueError("training source commit must be a full Git identity")
        return normalized

    @field_validator("training_source_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require a lowercase SHA-256 compatibility identity."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("training source fingerprint must be SHA-256")
        return value

    @model_validator(mode="after")
    def coherent_inventory(self) -> Self:
        """Bind the declared fingerprint to a sorted unique inventory."""
        paths = tuple(item.path for item in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError("training source inventory must be sorted and unique")
        submodule_paths = tuple(item.path for item in self.submodules)
        if submodule_paths != tuple(sorted(set(submodule_paths))):
            raise ValueError("training submodule paths must be sorted and unique")
        if (
            _inventory_fingerprint(self.files, submodules=self.submodules)
            != self.training_source_fingerprint
        ):
            raise ValueError("training source inventory fingerprint differs")
        return self


def resolve_training_source_identity(
    repo_root: Path,
) -> TrainingSourceIdentity:
    """Resolve source identity from a verified capsule or a clean Git checkout."""
    manifest_value = os.environ.get(_SOURCE_MANIFEST_ENV, "").strip()
    if manifest_value:
        manifest_path = Path(manifest_value).resolve()
        identity = load_training_source_identity(manifest_path)
        verify_training_source_inventory(repo_root, identity)
        return identity
    return build_checkout_source_identity(repo_root)


def build_checkout_source_identity(
    repo_root: Path,
) -> TrainingSourceIdentity:
    """Build a portable source identity from the clean checked-out revision."""
    resolved_root = repo_root.resolve()
    commit = _git_text(
        ("rev-parse", "HEAD^{commit}"),
        repo_root=resolved_root,
    ).strip()
    status_output = _git_text(
        (
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *TRAINING_SOURCE_PATHS,
        ),
        repo_root=resolved_root,
    )
    if status_output:
        raise RuntimeError("formal training requires clean executable source paths")
    relative_paths = _git_paths(
        ("ls-files", "-z", "--", *TRAINING_SOURCE_PATHS),
        repo_root=resolved_root,
    )
    submodules = tuple(
        _checkout_submodule_identity(
            resolved_root,
            superproject_commit=commit,
            relative_path=relative_path,
        )
        for relative_path in TRAINING_SOURCE_SUBMODULE_PATHS
        if _pinned_submodule_commit(
            resolved_root,
            revision=commit,
            relative_path=relative_path,
        )
        is not None
    )
    return build_filesystem_source_identity(
        resolved_root,
        commit=commit,
        relative_paths=relative_paths,
        submodules=submodules,
    )


def bind_run_source_identity(
    output_dir: Path,
    *,
    identity: TrainingSourceIdentity,
    exact_resume: bool,
    allow_legacy_adoption: bool,
) -> Path:
    """Create or validate the immutable source binding for one run directory."""
    path = output_dir.resolve() / _RUN_BINDING_PATH
    if path.is_file():
        bound = load_training_source_identity(path)
        if bound.training_source_fingerprint != identity.training_source_fingerprint:
            raise RuntimeError(
                "training source differs from the immutable run source binding"
            )
        return path
    if exact_resume and not allow_legacy_adoption:
        raise RuntimeError(
            "legacy exact resume lacks a training source binding; rerun once with "
            "the audited source-adoption launcher flag"
        )
    write_training_source_identity(path, identity, immutable=True)
    return path


def load_training_source_identity(path: Path) -> TrainingSourceIdentity:
    """Load one strict source identity manifest."""
    return TrainingSourceIdentity.model_validate_json(path.read_bytes())


def source_manifest_environment_name() -> str:
    """Return the environment variable consumed by runtime source checks."""
    return _SOURCE_MANIFEST_ENV


def source_adoption_environment_name() -> str:
    """Return the explicit legacy-adoption environment variable name."""
    return _ADOPT_SOURCE_ENV


def source_adoption_requested() -> bool:
    """Return whether the launcher explicitly authorized legacy binding."""
    return os.environ.get(_ADOPT_SOURCE_ENV) == "1"


def run_source_binding_path(output_dir: Path) -> Path:
    """Return the run-level immutable source identity path."""
    return output_dir.resolve() / _RUN_BINDING_PATH


def build_filesystem_source_identity(
    root: Path,
    *,
    commit: str,
    relative_paths: tuple[str, ...],
    submodules: tuple[TrainingSourceSubmodule, ...] = (),
) -> TrainingSourceIdentity:
    files = tuple(_source_file(root, path) for path in sorted(relative_paths))
    return TrainingSourceIdentity(
        source_git_commit=commit,
        training_source_fingerprint=_inventory_fingerprint(
            files,
            submodules=submodules,
        ),
        files=files,
        submodules=submodules,
    )


def build_filesystem_submodule_identity(
    root: Path,
    *,
    relative_path: str,
    commit: str,
    relative_paths: tuple[str, ...],
) -> TrainingSourceSubmodule:
    """Build one portable identity from an exported submodule tree."""
    files = tuple(_source_file(root, path) for path in sorted(relative_paths))
    return TrainingSourceSubmodule(
        path=relative_path,
        source_git_commit=commit,
        content_fingerprint=_inventory_fingerprint(files),
        files=files,
    )


def _source_file(root: Path, relative_path: str) -> TrainingSourceFile:
    path = root / relative_path
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        mode: Literal["100644", "100755", "120000"] = "120000"
        payload = os.readlink(path).encode("utf-8")
    elif stat.S_ISREG(metadata.st_mode):
        mode = "100755" if metadata.st_mode & 0o111 else "100644"
        payload = path.read_bytes()
    else:
        raise RuntimeError(f"unsupported training source file type: {relative_path}")
    return TrainingSourceFile(
        path=relative_path,
        mode=mode,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _inventory_fingerprint(
    files: tuple[TrainingSourceFile, ...],
    *,
    submodules: tuple[TrainingSourceSubmodule, ...] = (),
) -> str:
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_DOMAIN)
    for item in files:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item.mode.encode("ascii"))
        digest.update(b"\x00")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\x00")
    for submodule in submodules:
        digest.update(b"submodule\x00")
        digest.update(submodule.path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(submodule.content_fingerprint.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def verify_training_source_inventory(
    root: Path,
    identity: TrainingSourceIdentity,
) -> None:
    expected_paths = frozenset(item.path for item in identity.files)
    unexpected_paths = sorted(_source_paths_on_disk(root) - expected_paths)
    if unexpected_paths:
        raise RuntimeError(
            "materialized training source contains unexpected files: "
            + ", ".join(unexpected_paths[:5])
        )
    actual = build_filesystem_source_identity(
        root,
        commit=identity.source_git_commit,
        relative_paths=tuple(sorted(expected_paths)),
        submodules=identity.submodules,
    )
    if actual != identity:
        raise RuntimeError("materialized training source inventory differs")
    for submodule in identity.submodules:
        module_root = root / submodule.path
        expected_module_paths = frozenset(item.path for item in submodule.files)
        unexpected_module_paths = sorted(
            _all_source_paths_on_disk(module_root) - expected_module_paths
        )
        if unexpected_module_paths:
            raise RuntimeError(
                "materialized training submodule contains unexpected files: "
                + ", ".join(unexpected_module_paths[:5])
            )
        actual_submodule = build_filesystem_submodule_identity(
            module_root,
            relative_path=submodule.path,
            commit=submodule.source_git_commit,
            relative_paths=tuple(sorted(expected_module_paths)),
        )
        if actual_submodule != submodule:
            raise RuntimeError("materialized training submodule inventory differs")


def _source_paths_on_disk(root: Path) -> frozenset[str]:
    paths: set[str] = set()
    for pathspec in TRAINING_SOURCE_PATHS:
        target = root / pathspec
        if target.is_file() or target.is_symlink():
            paths.add(pathspec)
            continue
        if not target.is_dir():
            continue
        for directory, directories, filenames in os.walk(
            target,
            followlinks=False,
        ):
            directories[:] = [name for name in directories if name != "__pycache__"]
            base = Path(directory)
            for filename in filenames:
                if filename.endswith((".pyc", ".pyo")):
                    continue
                paths.add((base / filename).relative_to(root).as_posix())
    return frozenset(paths - set(TRAINING_RUNTIME_ARTIFACT_PATHS))


def _all_source_paths_on_disk(root: Path) -> frozenset[str]:
    paths: set[str] = set()
    if not root.is_dir():
        return frozenset()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name != "__pycache__"]
        base = Path(directory)
        for filename in filenames:
            if filename.endswith((".pyc", ".pyo")):
                continue
            paths.add((base / filename).relative_to(root).as_posix())
    return frozenset(paths)


def _checkout_submodule_identity(
    repo_root: Path,
    *,
    superproject_commit: str,
    relative_path: str,
) -> TrainingSourceSubmodule:
    pinned = _pinned_submodule_commit(
        repo_root,
        revision=superproject_commit,
        relative_path=relative_path,
    )
    if pinned is None:
        raise RuntimeError(f"training submodule is not pinned: {relative_path}")
    module_root = (repo_root / relative_path).resolve()
    relative_paths = _checkout_submodule_file_paths(
        module_root,
        commit=pinned,
        identity_path=relative_path,
    )
    return build_filesystem_submodule_identity(
        module_root,
        relative_path=relative_path,
        commit=pinned,
        relative_paths=relative_paths,
    )


def _checkout_submodule_file_paths(
    module_root: Path,
    *,
    commit: str,
    identity_path: str,
) -> tuple[str, ...]:
    head = _submodule_git_text(
        module_root,
        ("rev-parse", "HEAD^{commit}"),
    ).strip()
    if head != commit:
        raise RuntimeError(f"training submodule checkout differs: {identity_path}")
    status_output = _submodule_git_text(
        module_root,
        ("status", "--porcelain", "--untracked-files=no"),
    )
    if status_output:
        raise RuntimeError(f"training submodule has tracked changes: {identity_path}")
    files, _nested = _submodule_tree_inventory(
        module_root,
        commit=commit,
    )
    return files


def _pinned_submodule_commit(
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
    if (
        observed_path != relative_path
        or mode != "160000"
        or object_type != "commit"
    ):
        raise RuntimeError(f"training dependency is not a Git submodule: {relative_path}")
    return commit


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


def _submodule_git_text(
    repo_root: Path,
    arguments: tuple[str, ...],
) -> str:
    completed = subprocess.run(
        ("git", "-c", f"safe.directory={repo_root}", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "training submodule identity failed: " + completed.stderr.strip()
        )
    return completed.stdout


def _git_paths(arguments: tuple[str, ...], *, repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", *arguments),
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
        sorted(
            item.decode("utf-8")
            for item in completed.stdout.split(b"\x00")
            if item
        )
    )


def _git_text(arguments: tuple[str, ...], *, repo_root: Path) -> str:
    completed = subprocess.run(
        ("git", *arguments),
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


def write_training_source_identity(
    path: Path,
    identity: TrainingSourceIdentity,
    *,
    immutable: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.model_dump(mode="json")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    pending = path.parent / f".{path.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with pending.open("xb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        if immutable:
            try:
                os.link(pending, path)
            except FileExistsError:
                existing = load_training_source_identity(path)
                if existing != identity:
                    raise RuntimeError(
                        "immutable training source binding differs"
                    ) from None
            pending.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        else:
            os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TRAINING_SOURCE_MANIFEST",
    "TRAINING_SOURCE_PATHS",
    "TRAINING_SOURCE_SUBMODULE_PATHS",
    "TRAINING_RUNTIME_ARTIFACT_PATHS",
    "TrainingSourceIdentity",
    "TrainingSourceSubmodule",
    "bind_run_source_identity",
    "build_checkout_source_identity",
    "build_filesystem_source_identity",
    "build_filesystem_submodule_identity",
    "load_training_source_identity",
    "resolve_training_source_identity",
    "run_source_binding_path",
    "source_adoption_environment_name",
    "source_adoption_requested",
    "source_manifest_environment_name",
    "verify_training_source_inventory",
    "write_training_source_identity",
]
