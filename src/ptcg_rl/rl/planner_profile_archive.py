"""Safe lifecycle for one manifest-bound planner submission archive."""

from __future__ import annotations

import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_package_models import (
    PlannerPackageAssetManifestEntry,
)


@dataclass(slots=True)
class ExtractedPlannerPackage:
    """Owned extraction of one exactly validated package asset."""

    agent_dir: Path
    _temporary: tempfile.TemporaryDirectory[str]
    _closed: bool = False
    _read_only: bool = False

    @property
    def root(self) -> Path:
        """Return the extraction workspace root outside the agent directory."""
        return Path(self._temporary.name)

    @property
    def read_only(self) -> bool:
        """Return whether the extracted package has been sealed."""
        return self._read_only

    def close(self) -> None:
        """Remove the private extraction exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._read_only:
            _make_tree_owner_writable(self.agent_dir)
        self._temporary.cleanup()

    def seal_read_only(self) -> None:
        """Prevent shared replay children from mutating extracted bytes."""
        if self._closed:
            raise RuntimeError("cannot seal a closed planner package")
        if self._read_only:
            raise RuntimeError("planner package is already read-only")
        self._read_only = True
        paths = tuple(self.agent_dir.rglob("*"))
        for path in paths:
            if path.is_file():
                path.chmod(path.stat().st_mode & ~0o222)
        directories = tuple(path for path in paths if path.is_dir())
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            path.chmod(path.stat().st_mode & ~0o222)
        self.agent_dir.chmod(self.agent_dir.stat().st_mode & ~0o222)
        self.require_read_only()

    def require_read_only(self) -> None:
        """Reject any writable mode bit in a sealed package tree."""
        if not self._read_only:
            raise RuntimeError("planner package was not sealed read-only")
        writable = tuple(
            path
            for path in (self.agent_dir, *self.agent_dir.rglob("*"))
            if path.stat().st_mode & 0o222
        )
        if writable:
            raise RuntimeError("planner package read-only seal changed")


def extract_validated_planner_package(
    asset: PlannerPackageAssetManifestEntry,
) -> ExtractedPlannerPackage:
    """Verify and extract the exact archive/spec named by a manifest entry."""
    workspace = extract_verified_package_archive(
        asset.submission_archive_path,
        expected_sha256=asset.submission_archive_sha256,
    )
    try:
        runtime_path = workspace.agent_dir / "planner_runtime.json"
        if file_sha256(runtime_path) != asset.planner_runtime_sha256:
            raise ValueError("extracted planner runtime differs from its manifest")
    except BaseException:
        workspace.close()
        raise
    return workspace


def extract_verified_package_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
) -> ExtractedPlannerPackage:
    """Verify and safely extract one immutable submission archive."""
    if file_sha256(archive_path) != expected_sha256:
        raise ValueError("package archive differs from its expected fingerprint")
    temporary = tempfile.TemporaryDirectory(prefix="ptcg-profile-archive-")
    agent_dir = Path(temporary.name) / "agent"
    agent_dir.mkdir()
    try:
        _safe_extract(archive_path, agent_dir)
    except BaseException:
        temporary.cleanup()
        raise
    return ExtractedPlannerPackage(
        agent_dir=agent_dir,
        _temporary=temporary,
    )


def _safe_extract(archive_path: Path, extract_dir: Path) -> None:
    root = extract_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target_name = PurePosixPath(member.name)
            if (
                target_name.is_absolute()
                or ".." in target_name.parts
                or str(target_name) in {"", "."}
            ):
                raise ValueError(f"unsafe package archive target: {member.name}")
            target = (extract_dir / target_name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"package archive target escapes root: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"package archive contains a link: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read package archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _make_tree_owner_writable(root: Path) -> None:
    """Restore owner permissions only for private temporary cleanup."""
    paths = (root, *root.rglob("*"))
    for path in paths:
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        else:
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)


__all__ = [
    "ExtractedPlannerPackage",
    "extract_validated_planner_package",
    "extract_verified_package_archive",
]
