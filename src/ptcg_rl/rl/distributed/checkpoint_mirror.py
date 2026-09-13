"""Mirror complete checkpoint and exact-resume pairs from a remote learner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_SCHEMA_VERSION = 1
_PAIR_PATTERN = re.compile(r"^(?:policy|training_state)_v([0-9]+)\.pt$")

# This runs on the learner host. Relative pointer paths are resolved to the
# versioned artifact beside their latest.json, matching WeightPublisher and
# publish_training_state semantics without depending on the SSH login cwd.
_REMOTE_PROBE = r"""
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
probe_mode = sys.argv[2]
with_hashes = probe_mode == "1"
with_diagnostics = probe_mode in {"1", "d"}
diagnostics = json.loads(sys.argv[3])
diagnostic_limit = int(sys.argv[4])
require_policy_binding = sys.argv[5] == "1"
diagnostic_manifests = json.loads(sys.argv[6])

def fingerprint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns
    )
    if not stable:
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()

def pointer(directory, version_key, filename_prefix):
    latest = run_dir / directory / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError(f"missing pointer: {latest}")
    record = json.loads(latest.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"pointer must be an object: {latest}")
    version = int(record[version_key])
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"pointer path must be a non-empty string: {latest}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = latest.parent / path.name
    expected = run_dir / directory / f"{filename_prefix}_v{version}.pt"
    if path.resolve() != expected.resolve():
        raise ValueError(
            f"pointer artifact does not match its run/version: {path} != {expected}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    path = path.resolve(strict=True)
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": fingerprint(path) if with_hashes else None,
    }
    return version, record, result

try:
    weight_version, weight_record, weight = pointer(
        "weights", "version", "policy"
    )
    resume_version, resume_record, resume = pointer(
        "resume", "policy_version", "training_state"
    )
    if weight_version != resume_version:
        print(json.dumps({
            "ready": False,
            "reason": "version_mismatch",
            "weight_version": weight_version,
            "resume_version": resume_version,
        }))
        raise SystemExit(0)
    bound_policy_size = resume_record.get("policy_size_bytes")
    bound_policy_sha256 = resume_record.get("policy_sha256")
    if require_policy_binding and (
        not isinstance(bound_policy_size, int)
        or not isinstance(bound_policy_sha256, str)
        or len(bound_policy_sha256) != 64
    ):
        print(json.dumps({
            "ready": False,
            "reason": "missing_policy_binding",
            "version": weight_version,
        }))
        raise SystemExit(0)
    if require_policy_binding and with_hashes and (
        bound_policy_size != weight["size_bytes"]
        or bound_policy_sha256 != weight["sha256"]
    ):
        print(json.dumps({
            "ready": False,
            "reason": "policy_binding_mismatch",
            "version": weight_version,
            "bound_policy_size_bytes": bound_policy_size,
            "actual_policy_size_bytes": weight["size_bytes"],
            "bound_policy_sha256": bound_policy_sha256,
            "actual_policy_sha256": weight["sha256"],
        }))
        raise SystemExit(0)
    diagnostic_records = []
    if with_diagnostics:
        seen_diagnostics = set()
        for raw_relative in diagnostics:
            relative = Path(raw_relative)
            path = run_dir / relative
            if not path.is_file() or path.stat().st_size > diagnostic_limit:
                continue
            path = path.resolve(strict=True)
            diagnostic_records.append({
                "relative_path": raw_relative,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": fingerprint(path),
            })
            seen_diagnostics.add(raw_relative)
        for raw_manifest_relative in diagnostic_manifests:
            try:
                manifest_relative = Path(raw_manifest_relative)
                manifest_path = run_dir / manifest_relative
                if (
                    not manifest_path.is_file()
                    or manifest_path.stat().st_size > diagnostic_limit
                ):
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = manifest.get("files", [])
                if not isinstance(files, list):
                    continue
                for record in files:
                    if not isinstance(record, dict):
                        continue
                    raw_relative = record.get("relative_path")
                    if not isinstance(raw_relative, str) or raw_relative in seen_diagnostics:
                        continue
                    relative = Path(raw_relative)
                    if relative.is_absolute() or ".." in relative.parts:
                        continue
                    path = (run_dir / relative).resolve(strict=True)
                    if not path.is_relative_to(run_dir.resolve(strict=True)):
                        continue
                    expected_size = record.get("size_bytes")
                    expected_sha256 = record.get("sha256")
                    if (
                        not path.is_file()
                        or path.stat().st_size > diagnostic_limit
                        or not isinstance(expected_size, int)
                        or not isinstance(expected_sha256, str)
                        or path.stat().st_size != expected_size
                    ):
                        continue
                    actual_sha256 = fingerprint(path)
                    if actual_sha256 != expected_sha256:
                        continue
                    diagnostic_records.append({
                        "relative_path": raw_relative,
                        "path": str(path),
                        "size_bytes": expected_size,
                        "sha256": actual_sha256,
                    })
                    seen_diagnostics.add(raw_relative)
                if raw_manifest_relative not in seen_diagnostics:
                    manifest_path = manifest_path.resolve(strict=True)
                    diagnostic_records.append({
                        "relative_path": raw_manifest_relative,
                        "path": str(manifest_path),
                        "size_bytes": manifest_path.stat().st_size,
                        "sha256": fingerprint(manifest_path),
                    })
                    seen_diagnostics.add(raw_manifest_relative)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                continue
    print(json.dumps({
        "ready": True,
        "reason": None,
        "version": weight_version,
        "weight_record": weight_record,
        "resume_record": resume_record,
        "weight": weight,
        "resume": resume,
        "diagnostics": diagnostic_records,
    }))
except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
    print(json.dumps({"ready": False, "reason": str(error)}))
"""


class CheckpointMirrorConfig(BaseModel):
    """Configuration for a local canonical mirror of a remote RL run."""

    model_config = ConfigDict(extra="forbid")

    remote_host: str
    remote_run_dir: Path
    local_run_dir: Path
    watch: bool = False
    poll_interval_seconds: float = 15.0
    diagnostics: tuple[Path, ...] = ()
    diagnostic_manifests: tuple[Path, ...] = ()
    diagnostic_max_bytes: int = 8 * 1024 * 1024
    keep_last_pairs: int | None = None
    retain_every_versions: int | None = None
    ssh_executable: str = "ssh"
    ssh_options: tuple[str, ...] = ()
    rsync_executable: str = "rsync"
    command_timeout_seconds: float = 3600.0
    require_policy_binding: bool = False

    @field_validator("remote_host", "ssh_executable", "rsync_executable")
    @classmethod
    def valid_command_segment(cls, value: str) -> str:
        """Reject empty command and host values or option-like SSH targets."""
        cleaned = value.strip()
        if not cleaned or cleaned.startswith("-"):
            raise ValueError("command and remote host values must be non-empty")
        return cleaned

    @field_validator("poll_interval_seconds", "command_timeout_seconds")
    @classmethod
    def valid_positive_seconds(cls, value: float) -> float:
        """Reject non-positive timing values."""
        if value <= 0:
            raise ValueError("timing values must be positive")
        return value

    @field_validator("diagnostic_max_bytes", "keep_last_pairs", "retain_every_versions")
    @classmethod
    def valid_positive_optional_int(cls, value: int | None) -> int | None:
        """Reject non-positive size and retention values."""
        if value is not None and value <= 0:
            raise ValueError("size and retention values must be positive")
        return value

    @field_validator("diagnostics", "diagnostic_manifests")
    @classmethod
    def valid_diagnostics(cls, values: tuple[Path, ...]) -> tuple[Path, ...]:
        """Keep optional diagnostics confined to non-checkpoint relative paths."""
        for value in values:
            if value.is_absolute() or ".." in value.parts or not value.parts:
                raise ValueError("diagnostic paths must be safe relative paths")
            if value.parts[0] in {"weights", "resume", ".checkpoint_mirror"}:
                raise ValueError("diagnostics cannot overlap canonical mirror files")
        return values

    @model_validator(mode="after")
    def valid_retention(self) -> CheckpointMirrorConfig:
        """Require rolling retention when milestone retention is requested."""
        if self.retain_every_versions is not None and self.keep_last_pairs is None:
            raise ValueError("retain_every_versions requires keep_last_pairs")
        return self


class _RemoteArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int
    sha256: str | None


class _RemoteDiagnostic(_RemoteArtifact):
    relative_path: Path


class _RemoteSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    ready: bool
    reason: str | None = None
    version: int | None = None
    weight_record: dict[str, Any] | None = None
    resume_record: dict[str, Any] | None = None
    weight: _RemoteArtifact | None = None
    resume: _RemoteArtifact | None = None
    diagnostics: tuple[_RemoteDiagnostic, ...] = ()


@dataclass(frozen=True)
class MirrorResult:
    """Outcome of one remote pointer poll."""

    status: Literal["mirrored", "current", "not_ready", "remote_older"]
    version: int | None
    detail: str | None = None
    diagnostics: tuple[str, ...] = ()
    pruned_versions: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


class CheckpointMirror:
    """Poll and transactionally mirror remote checkpoint/resume pairs."""

    def __init__(
        self,
        config: CheckpointMirrorConfig,
        *,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self._run_command = command_runner or _run_command
        self.local_run_dir = config.local_run_dir.expanduser().resolve()
        self._verified_local: dict[Path, tuple[int, int, int, str]] = {}

    def mirror_once(self) -> MirrorResult:
        """Poll once and mirror the latest complete matching pair, if new."""
        snapshot = self._probe_remote(with_hashes=False)
        if not snapshot.ready:
            return MirrorResult("not_ready", None, snapshot.reason)
        version = _required_snapshot_version(snapshot)
        current_version = self._local_current_version()
        if current_version is not None and version < current_version:
            return MirrorResult("remote_older", version)
        if current_version == version and self._current_files_are_present():
            diagnostic_snapshot = self._probe_remote(
                with_hashes=False,
                diagnostics=True,
            )
            diagnostics = (
                self._mirror_diagnostics(diagnostic_snapshot.diagnostics)
                if diagnostic_snapshot.ready
                else ()
            )
            return MirrorResult("current", version, diagnostics=diagnostics)

        snapshot = self._probe_remote(with_hashes=True)
        if not snapshot.ready:
            return MirrorResult("not_ready", None, snapshot.reason)
        version = _required_snapshot_version(snapshot)
        current_version = self._local_current_version()
        if current_version is not None and version < current_version:
            return MirrorResult("remote_older", version)
        if current_version == version and self._current_files_are_present():
            return MirrorResult("current", version)

        weight = _required_artifact(snapshot.weight, "weight")
        resume = _required_artifact(snapshot.resume, "resume")
        weight_final = self.local_run_dir / "weights" / Path(weight.path).name
        resume_final = self.local_run_dir / "resume" / Path(resume.path).name
        weight_partial = self._stage_checkpoint(weight, weight_final)
        resume_partial = self._stage_checkpoint(resume, resume_final)

        # Both partials have passed size and SHA256 verification before either
        # new final path becomes visible. A crash between renames leaves no new
        # latest pointer and is safely completed by the next invocation.
        if weight_partial is not None:
            os.rename(weight_partial, weight_final)
        if resume_partial is not None:
            os.rename(resume_partial, resume_final)

        self._publish_local_pointers(
            snapshot,
            weight_path=weight_final,
            resume_path=resume_final,
        )
        self._write_manifest(
            snapshot,
            weight_path=weight_final,
            resume_path=resume_final,
        )
        mirrored_diagnostics = self._mirror_diagnostics(snapshot.diagnostics)
        pruned = self._prune_old_pairs(current_version=version)
        return MirrorResult(
            "mirrored",
            version,
            diagnostics=mirrored_diagnostics,
            pruned_versions=pruned,
        )

    def watch(self, report: Callable[[MirrorResult], None]) -> None:
        """Poll forever, reporting successful polls to the caller."""
        while True:
            report(self.mirror_once())
            time.sleep(self.config.poll_interval_seconds)

    def _probe_remote(
        self,
        *,
        with_hashes: bool,
        diagnostics: bool = False,
    ) -> _RemoteSnapshot:
        probe_mode = "1" if with_hashes else ("d" if diagnostics else "0")
        remote_command = shlex.join(
            (
                "python3",
                "-c",
                _REMOTE_PROBE,
                str(self.config.remote_run_dir),
                probe_mode,
                json.dumps([str(path) for path in self.config.diagnostics]),
                str(self.config.diagnostic_max_bytes),
                "1" if self.config.require_policy_binding else "0",
                json.dumps(
                    [str(path) for path in self.config.diagnostic_manifests]
                ),
            )
        )
        command = (
            self.config.ssh_executable,
            *self.config.ssh_options,
            self.config.remote_host,
            remote_command,
        )
        completed = self._run_command(command, self.config.command_timeout_seconds)
        return _RemoteSnapshot.model_validate_json(completed.stdout)

    def _stage_checkpoint(
        self,
        remote: _RemoteArtifact,
        final_path: Path,
    ) -> Path | None:
        expected = _required_fingerprint(remote)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            if _fingerprint(final_path) != expected:
                raise FileExistsError(
                    f"conflicting immutable local artifact: {final_path}"
                )
            self._remember_verified(final_path, expected[1])
            return None
        partial = final_path.with_name(f"{final_path.name}.partial")
        self._rsync(remote.path, partial)
        actual = _fingerprint(partial)
        if actual != expected:
            raise ValueError(
                f"remote/local fingerprint mismatch for {remote.path}: "
                f"expected={expected}, actual={actual}"
            )
        self._remember_verified(partial, actual[1])
        return partial

    def _rsync(self, remote_path: str, local_path: Path) -> None:
        ssh_transport = shlex.join(
            (self.config.ssh_executable, *self.config.ssh_options)
        )
        command = (
            self.config.rsync_executable,
            "--archive",
            "--partial",
            "--protect-args",
            "--rsh",
            ssh_transport,
            "--",
            f"{self.config.remote_host}:{remote_path}",
            str(local_path),
        )
        self._run_command(command, self.config.command_timeout_seconds)

    def _publish_local_pointers(
        self,
        snapshot: _RemoteSnapshot,
        *,
        weight_path: Path,
        resume_path: Path,
    ) -> None:
        weight_record = dict(_required_record(snapshot.weight_record, "weight"))
        resume_record = dict(_required_record(snapshot.resume_record, "resume"))
        replacements = {
            _required_artifact(snapshot.weight, "weight").path: str(weight_path),
            _required_artifact(snapshot.resume, "resume").path: str(resume_path),
        }
        weight_record = cast(
            dict[str, Any], _replace_exact_paths(weight_record, replacements)
        )
        resume_record = cast(
            dict[str, Any], _replace_exact_paths(resume_record, replacements)
        )
        weight_record["path"] = str(weight_path)
        resume_record["path"] = str(resume_path)
        weight_metadata = weight_record.get("metadata")
        if isinstance(weight_metadata, dict) and "resume_state_path" in weight_metadata:
            weight_metadata["resume_state_path"] = str(resume_path)

        resume_tmp = _write_json_partial(
            self.local_run_dir / "resume" / "latest.json", resume_record
        )
        weight_tmp = _write_json_partial(
            self.local_run_dir / "weights" / "latest.json", weight_record
        )
        os.replace(resume_tmp, self.local_run_dir / "resume" / "latest.json")
        os.replace(weight_tmp, self.local_run_dir / "weights" / "latest.json")

    def _write_manifest(
        self,
        snapshot: _RemoteSnapshot,
        *,
        weight_path: Path,
        resume_path: Path,
    ) -> None:
        weight = _required_artifact(snapshot.weight, "weight")
        resume = _required_artifact(snapshot.resume, "resume")
        record = {
            "schema_version": _SCHEMA_VERSION,
            "version": _required_snapshot_version(snapshot),
            "mirrored_at": datetime.now(UTC).isoformat(),
            "source": {
                "host": self.config.remote_host,
                "run_dir": str(self.config.remote_run_dir),
            },
            "weight": {
                "path": str(weight_path),
                "remote_path": weight.path,
                "size_bytes": weight.size_bytes,
                "sha256": weight.sha256,
            },
            "resume": {
                "path": str(resume_path),
                "remote_path": resume.path,
                "size_bytes": resume.size_bytes,
                "sha256": resume.sha256,
            },
            "policy_binding_verified": self.config.require_policy_binding,
        }
        path = self.local_run_dir / ".checkpoint_mirror" / "latest.json"
        os.replace(_write_json_partial(path, record), path)

    def _mirror_diagnostics(
        self, diagnostics: Sequence[_RemoteDiagnostic]
    ) -> tuple[str, ...]:
        mirrored: list[str] = []
        for diagnostic in diagnostics:
            try:
                destination = self.local_run_dir / diagnostic.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected = _required_fingerprint(diagnostic)
                if (
                    destination.is_file()
                    and self._local_fingerprint(destination) == expected
                ):
                    continue
                partial = destination.with_name(f"{destination.name}.partial")
                self._rsync(diagnostic.path, partial)
                actual = _fingerprint(partial)
                if actual != expected:
                    continue
                os.replace(partial, destination)
                self._remember_verified(destination, actual[1])
                mirrored.append(str(diagnostic.relative_path))
            except Exception:
                # Mutable diagnostics are deliberately best effort and may
                # change between the remote hash and rsync. They never gate a
                # verified canonical checkpoint/resume pair.
                continue
        return tuple(mirrored)

    def _local_current_version(self) -> int | None:
        manifest = _read_json(self.local_run_dir / ".checkpoint_mirror" / "latest.json")
        if manifest is None:
            return None
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            return None
        if source.get("host") != self.config.remote_host or source.get(
            "run_dir"
        ) != str(self.config.remote_run_dir):
            return None
        return int(manifest["version"])

    def _current_files_are_present(self) -> bool:
        manifest = _read_json(self.local_run_dir / ".checkpoint_mirror" / "latest.json")
        if manifest is None:
            return False
        for key in ("weight", "resume"):
            artifact = manifest.get(key)
            if not isinstance(artifact, Mapping):
                return False
            path = artifact.get("path")
            size = artifact.get("size_bytes")
            sha256 = artifact.get("sha256")
            if not isinstance(path, str) or not Path(path).is_file():
                return False
            if not isinstance(size, int) or not isinstance(sha256, str):
                return False
            if self._local_fingerprint(Path(path)) != (size, sha256):
                return False
        return True

    def _local_fingerprint(self, path: Path) -> tuple[int, str]:
        stat = path.stat()
        cached = self._verified_local.get(path)
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if cached is not None and cached[:3] == signature:
            return cached[0], cached[3]
        fingerprint = _fingerprint(path)
        self._verified_local[path] = (*signature, fingerprint[1])
        return fingerprint

    def _remember_verified(self, path: Path, sha256: str) -> None:
        stat = path.stat()
        self._verified_local[path] = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            sha256,
        )

    def _prune_old_pairs(self, *, current_version: int) -> tuple[int, ...]:
        keep_last = self.config.keep_last_pairs
        if keep_last is None:
            return ()
        weights = _versioned_files(self.local_run_dir / "weights", "policy")
        resumes = _versioned_files(self.local_run_dir / "resume", "training_state")
        paired_versions = sorted(set(weights).intersection(resumes), reverse=True)
        protected = set(paired_versions[:keep_last])
        protected.add(current_version)
        interval = self.config.retain_every_versions
        if interval is not None:
            protected.update(
                version for version in paired_versions if version % interval == 0
            )
        pruned: list[int] = []
        for version in paired_versions:
            if version in protected:
                continue
            weights[version].unlink()
            resumes[version].unlink()
            pruned.append(version)
        return tuple(sorted(pruned))


def _run_command(
    command: Sequence[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _required_fingerprint(artifact: _RemoteArtifact) -> tuple[int, str]:
    if artifact.sha256 is None:
        raise ValueError(f"remote fingerprint is missing for {artifact.path}")
    return artifact.size_bytes, artifact.sha256


def _required_snapshot_version(snapshot: _RemoteSnapshot) -> int:
    if snapshot.version is None:
        raise ValueError("ready remote snapshot is missing its version")
    return snapshot.version


def _required_artifact(artifact: _RemoteArtifact | None, name: str) -> _RemoteArtifact:
    if artifact is None:
        raise ValueError(f"ready remote snapshot is missing its {name} artifact")
    return artifact


def _required_record(record: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if record is None:
        raise ValueError(f"ready remote snapshot is missing its {name} pointer")
    return record


def _replace_exact_paths(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_exact_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_exact_paths(item, replacements) for key, item in value.items()
        }
    return value


def _write_json_partial(path: Path, record: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    return partial


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _versioned_files(directory: Path, prefix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob(f"{prefix}_v*.pt"):
        if path.is_symlink() or not path.is_file():
            continue
        match = _PAIR_PATTERN.fullmatch(path.name)
        if match is not None:
            result[int(match.group(1))] = path
    return result
