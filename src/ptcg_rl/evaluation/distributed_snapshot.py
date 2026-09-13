"""Content-addressed source and artifact snapshots for remote evaluation."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.distributed_hosts import HostAllocation
from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.submission.release_assets import (
    load_release_bundle_for_native_execution,
)

_STATIC_FEATURES_PATH = Path(
    "outputs/cards/static_features/card_static_features.npy"
)


class SnapshotFile(BaseModel):
    """One repository-relative file frozen into a remote snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str
    size_bytes: int


class EvaluationSnapshot(BaseModel):
    """Complete source and immutable input identity synchronized to every host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str = "RELEASE-H2H-SNAPSHOT-v1"
    snapshot_fingerprint: str
    files: tuple[SnapshotFile, ...]


def build_evaluation_snapshot(
    config: ReleaseH2HConfig,
    *,
    tool_paths: Sequence[Path],
) -> EvaluationSnapshot:
    """Hash only the source and release assets required by remote workers."""
    repo_root = records.repo_path(Path(".")).resolve()
    paths: set[Path] = set()
    for source_root in (
        repo_root / "src" / "ptcg_rl",
        repo_root / "data" / "sample_submission",
    ):
        paths.update(_tree_files(source_root))
    paths.update(records.repo_path(path).resolve() for path in tool_paths)
    # Checkpoint reconstruction consumes this repository-level model input even
    # though release manifests do not repeat its path. Freeze it into the same
    # content-addressed snapshot as the participant artifacts.
    paths.add(records.repo_path(_STATIC_FEATURES_PATH).resolve())
    for participant in (config.candidate, config.opponent):
        manifest_path = records.repo_path(participant.manifest_path).resolve()
        bundle = load_release_bundle_for_native_execution(manifest_path)
        paths.update(
            {
                manifest_path,
                records.repo_path(bundle.checkpoint_path).resolve(),
                records.repo_path(bundle.deck_path).resolve(),
                records.repo_path(bundle.runtime_archive_path).resolve(),
            }
        )
        if bundle.belief_path is not None:
            paths.add(records.repo_path(bundle.belief_path).resolve())

    files: list[SnapshotFile] = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"evaluation snapshot input is missing: {path}")
        if not path.is_relative_to(repo_root):
            raise ValueError(f"remote evaluation input is outside repository: {path}")
        relative = path.relative_to(repo_root)
        if "third_party" in relative.parts:
            raise ValueError(f"remote evaluation cannot synchronize third_party: {path}")
        files.append(
            SnapshotFile(
                path=relative,
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )
    payload = [item.model_dump(mode="json") for item in files]
    return EvaluationSnapshot(
        snapshot_fingerprint=fingerprint_payload(
            {"protocol": "RELEASE-H2H-SNAPSHOT-v1", "files": payload}
        ),
        files=tuple(files),
    )


def write_snapshot_manifest(path: Path, snapshot: EvaluationSnapshot) -> None:
    """Atomically persist the exact file inventory used for remote sync."""
    write_identity_atomic(path, snapshot.model_dump(mode="json"))


def remote_snapshot_root(
    allocation: HostAllocation,
    *,
    remote_root: Path,
    snapshot_fingerprint: str,
) -> Path:
    """Return the absolute content-addressed snapshot path on one host."""
    return allocation.home / remote_root / "snapshots" / snapshot_fingerprint


def sync_snapshot_to_host(
    snapshot: EvaluationSnapshot,
    *,
    manifest_path: Path,
    allocation: HostAllocation,
    remote_root: Path,
    connect_timeout_seconds: int,
) -> Path:
    """Upload, then independently hash-check, a snapshot on one host."""
    repo_root = records.repo_path(Path(".")).resolve()
    destination = remote_snapshot_root(
        allocation,
        remote_root=remote_root,
        snapshot_fingerprint=snapshot.snapshot_fingerprint,
    )
    link_destination = _latest_remote_snapshot(
        allocation.target,
        destination=destination,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    _run_checked(
        _ssh_command(
            allocation.target,
            f"mkdir -p {shlex.quote(str(destination))}",
            timeout=connect_timeout_seconds,
        )
    )
    relative_paths = b"\0".join(
        str(item.path).encode("utf-8") for item in snapshot.files
    ) + b"\0"
    rsync_command = [
        "rsync",
        "-az",
        "--relative",
        "--from0",
        "--files-from=-",
    ]
    if link_destination is not None:
        rsync_command.extend(("--link-dest", str(link_destination)))
    rsync_command.extend(("./", f"{allocation.target}:{destination}/"))
    _run_checked(
        tuple(rsync_command),
        cwd=repo_root,
        input_bytes=relative_paths,
    )
    remote_manifest = destination / ".snapshot_manifest.json"
    _run_checked(
        (
            "rsync",
            "-az",
            str(manifest_path),
            f"{allocation.target}:{remote_manifest}",
        )
    )
    _verify_remote_snapshot(
        allocation.target,
        snapshot=snapshot,
        snapshot_root=destination,
        manifest_path=remote_manifest,
        local_manifest_path=manifest_path,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    return destination


def _latest_remote_snapshot(
    target: str,
    *,
    destination: Path,
    connect_timeout_seconds: int,
) -> Path | None:
    """Select one verified prior tree as an rsync hard-link base."""
    script = f'''import pathlib

destination = pathlib.Path({str(destination)!r})
root = destination.parent
candidates = []
if root.is_dir():
    for path in root.iterdir():
        if path == destination or not path.is_dir():
            continue
        manifest = path / ".snapshot_manifest.json"
        if manifest.is_file():
            candidates.append((manifest.stat().st_mtime_ns, path))
if candidates:
    print(max(candidates, key=lambda item: item[0])[1])'''
    completed = subprocess.run(
        _ssh_command(
            target,
            f"python3 -c {shlex.quote(script)}",
            timeout=connect_timeout_seconds,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=max(30, connect_timeout_seconds * 3),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote snapshot base discovery failed for {target}: "
            f"{completed.stderr.strip()[-1000:]}"
        )
    raw = completed.stdout.strip()
    return Path(raw) if raw else None


def _tree_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if root.name == "ptcg_rl" and relative.parts[:2] == (
            "dashboard",
            "frontend",
        ):
            continue
        yield path.resolve()


def _verify_remote_snapshot(
    target: str,
    *,
    snapshot: EvaluationSnapshot,
    snapshot_root: Path,
    manifest_path: Path,
    local_manifest_path: Path,
    connect_timeout_seconds: int,
) -> None:
    expected_manifest_sha256 = file_sha256(local_manifest_path)
    script = _verification_script(
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
    )
    completed = subprocess.run(
        _ssh_command(
            target,
            f"python3 -c {shlex.quote(script)}",
            timeout=connect_timeout_seconds,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=600.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote snapshot verification failed for {target}: "
            f"{completed.stderr.strip()[-2000:]}"
        )
    raw = json.loads(completed.stdout)
    expected = {
        "manifest_sha256": expected_manifest_sha256,
        "snapshot_fingerprint": snapshot.snapshot_fingerprint,
        "files": len(snapshot.files),
    }
    if raw != expected:
        raise ValueError(
            f"remote snapshot identity mismatch for {target}: {raw} != {expected}"
        )


def _verification_script(*, snapshot_root: Path, manifest_path: Path) -> str:
    return f'''import hashlib
import json
import pathlib

root = pathlib.Path({str(snapshot_root)!r}).resolve(strict=True)
manifest_path = pathlib.Path({str(manifest_path)!r}).resolve(strict=True)
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
for item in manifest["files"]:
    path = (root / item["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError("snapshot path escaped its root")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"] or path.stat().st_size != item["size_bytes"]:
        raise RuntimeError("snapshot file identity mismatch: " + item["path"])
print(json.dumps({{
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "snapshot_fingerprint": manifest["snapshot_fingerprint"],
    "files": len(manifest["files"]),
}}, sort_keys=True))'''


def _ssh_command(target: str, command: str, *, timeout: int) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        target,
        command,
    )


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
) -> None:
    subprocess.run(
        tuple(command),
        cwd=cwd,
        input=input_bytes,
        check=True,
        timeout=1200.0,
    )


def manifest_sha256(snapshot: EvaluationSnapshot) -> str:
    """Hash a snapshot model without writing it, for nested identities."""
    encoded = snapshot.model_dump_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EvaluationSnapshot",
    "SnapshotFile",
    "build_evaluation_snapshot",
    "manifest_sha256",
    "remote_snapshot_root",
    "sync_snapshot_to_host",
    "write_snapshot_manifest",
]
