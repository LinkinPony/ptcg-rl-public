"""Immutable release-archive materialization and isolated episode agents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.submission.release_assets import (
    load_release_bundle_for_native_execution,
    submission_fingerprint,
)
from ptcg_rl.submission.release_assets.models import ReleaseBundleIdentityLike
from ptcg_rl.training.arena_utils import field_value

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_BRIDGE_PATH = Path(__file__).with_name("release_agent_bridge.py").resolve()
_INTERNAL_CHECKPOINT = Path("agent_checkpoint.pt")
_INTERNAL_DECK = Path("deck.csv")
_INTERNAL_BELIEF = Path("belief_prior.csv")
_INTERNAL_MAIN = Path("main.py")
_INTERNAL_ENGINE = Path("cg/libcg.so")


class ReleaseAgentError(RuntimeError):
    """A packaged participant failed startup, protocol, or inference."""


@dataclass(frozen=True)
class ReleaseCapsule:
    """Verified executable identity for one immutable release archive."""

    label: str
    bundle_id: str
    protocol: str
    manifest_path: Path
    manifest_sha256: str
    submission_fingerprint: str
    runtime_archive_path: Path
    runtime_sha256: str
    runtime_config_sha256: str
    checkpoint_sha256: str
    deck_path: Path
    deck_sha256: str
    belief_sha256: str | None
    extract_dir: Path
    main_sha256: str
    packaged_engine_sha256: str
    capsule_fingerprint: str

    def identity(self) -> dict[str, Any]:
        """Return the semantic identity persisted with evaluation rows."""
        return {
            "label": self.label,
            "bundle_id": self.bundle_id,
            "protocol": self.protocol,
            "manifest_path": records.display_path(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "submission_fingerprint": self.submission_fingerprint,
            "runtime_archive_path": records.display_path(self.runtime_archive_path),
            "runtime_sha256": self.runtime_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "deck_path": records.display_path(self.deck_path),
            "deck_sha256": self.deck_sha256,
            "belief_sha256": self.belief_sha256,
            "main_sha256": self.main_sha256,
            "packaged_engine_sha256": self.packaged_engine_sha256,
            "capsule_fingerprint": self.capsule_fingerprint,
            "environment_provenance": "native-reconstructed",
        }


def materialize_release_capsule(
    *,
    label: str,
    manifest_path: Path,
    cache_root: Path,
) -> ReleaseCapsule:
    """Verify one release manifest and safely extract its exact runtime."""
    manifest = load_release_bundle_for_native_execution(
        records.repo_path(manifest_path)
    )
    archive_path = records.repo_path(manifest.runtime_archive_path).resolve()
    extract_dir = cache_root.resolve() / manifest.runtime_sha256
    _extract_archive_once(
        archive_path,
        extract_dir=extract_dir,
        expected_sha256=manifest.runtime_sha256,
    )
    _verify_internal_assets(extract_dir, manifest)
    main_sha256 = file_sha256(extract_dir / _INTERNAL_MAIN)
    packaged_engine_sha256 = file_sha256(extract_dir / _INTERNAL_ENGINE)
    identity_payload = {
        "manifest_sha256": manifest.source_manifest_sha256,
        "submission_fingerprint": submission_fingerprint(manifest),
        "runtime_sha256": manifest.runtime_sha256,
        "runtime_config_sha256": manifest.runtime_config_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "deck_sha256": manifest.deck_sha256,
        "belief_sha256": manifest.belief_sha256,
        "main_sha256": main_sha256,
        "packaged_engine_sha256": packaged_engine_sha256,
        "execution_mode": "fresh-process-per-seat-episode",
        "environment_provenance": "native-reconstructed",
    }
    return ReleaseCapsule(
        label=label,
        bundle_id=manifest.bundle_id,
        protocol=manifest.protocol,
        manifest_path=records.repo_path(manifest.source_manifest_path).resolve(),
        manifest_sha256=manifest.source_manifest_sha256,
        submission_fingerprint=submission_fingerprint(manifest),
        runtime_archive_path=archive_path,
        runtime_sha256=manifest.runtime_sha256,
        runtime_config_sha256=manifest.runtime_config_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        deck_path=records.repo_path(manifest.deck_path).resolve(),
        deck_sha256=manifest.deck_sha256,
        belief_sha256=manifest.belief_sha256,
        extract_dir=extract_dir,
        main_sha256=main_sha256,
        packaged_engine_sha256=packaged_engine_sha256,
        capsule_fingerprint=fingerprint_payload(identity_payload),
    )


def bridge_path() -> Path:
    """Return the stdlib child bridge used by every release worker."""
    return _BRIDGE_PATH


def cleanup_release_cache(capsules: Sequence[ReleaseCapsule]) -> None:
    """Remove only the verified extraction directories owned by this run."""
    for capsule in capsules:
        marker = capsule.extract_dir / ".release-extract.json"
        if not marker.is_file():
            continue
        raw = json.loads(marker.read_text(encoding="utf-8"))
        if raw.get("runtime_sha256") != capsule.runtime_sha256:
            raise ValueError(
                f"refusing to clean mismatched release cache: {capsule.extract_dir}"
            )
        shutil.rmtree(capsule.extract_dir)


class IsolatedReleaseAgent:
    """Arena agent backed by one fresh packaged-submission subprocess."""

    def __init__(
        self,
        capsule: ReleaseCapsule,
        *,
        log_path: Path,
        work_dir: Path,
        startup_timeout_seconds: float,
        action_timeout_seconds: float,
    ) -> None:
        self.name = capsule.label
        self.capsule = capsule
        self._action_timeout_seconds = action_timeout_seconds
        self._channel: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: TextIO | None = None
        self._last_telemetry: Mapping[str, Any] = {}
        self.runtime_status: Mapping[str, Any] = {}
        self.execution_status: Mapping[str, Any] = {}
        self.policy_loaded_observed = False
        self.startup_seconds = 0.0
        self.process_id = -1
        self.log_path = log_path
        self._start(
            work_dir=work_dir,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    def _start(self, *, work_dir: Path, startup_timeout_seconds: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        parent_channel, child_channel = socket.socketpair()
        self._channel = parent_channel
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        command = (
            sys.executable,
            "-I",
            "-u",
            str(_BRIDGE_PATH),
            "--agent-root",
            str(self.capsule.extract_dir),
            "--work-dir",
            str(work_dir),
            "--protocol-fd",
            str(child_channel.fileno()),
        )
        started_at = time.perf_counter()
        try:
            self._process = subprocess.Popen(
                command,
                cwd=work_dir,
                env=_sanitized_environment(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                pass_fds=(child_channel.fileno(),),
                start_new_session=True,
            )
            self.process_id = self._process.pid
        finally:
            child_channel.close()
        try:
            response = self._receive(timeout_seconds=startup_timeout_seconds)
            self.startup_seconds = time.perf_counter() - started_at
            self._validate_ready(response)
        except BaseException:
            self.close()
            raise

    def _validate_ready(self, response: Mapping[str, Any]) -> None:
        if response.get("type") != "ready":
            raise ReleaseAgentError(_response_error("release startup failed", response))
        for field_name in ("main_path", "package_path"):
            raw_path = response.get(field_name)
            if not isinstance(raw_path, str):
                raise ReleaseAgentError(f"release worker omitted {field_name}")
            resolved = Path(raw_path).resolve()
            if not resolved.is_relative_to(self.capsule.extract_dir.resolve()):
                raise ReleaseAgentError(
                    f"release worker imported {field_name} outside archive: {resolved}"
                )
        self.runtime_status = _validated_runtime_status(response.get("runtime_status"))
        raw_execution_status = response.get("execution_status")
        self.execution_status = (
            cast(Mapping[str, Any], raw_execution_status)
            if isinstance(raw_execution_status, Mapping)
            else {}
        )
        self.policy_loaded_observed = self.runtime_status.get("policy_loaded") is True

    def act(self, observation: Any) -> Sequence[int]:
        """Invoke the packaged ``main.agent`` and reject any silent fallback."""
        self._send(
            {
                "type": "act",
                "observation": observation,
                "configuration": None,
            }
        )
        response = self._receive(timeout_seconds=self._action_timeout_seconds)
        if response.get("type") != "action":
            raise ReleaseAgentError(_response_error("release act failed", response))
        raw_action = response.get("action")
        if not isinstance(raw_action, list) or not all(
            type(item) is int for item in raw_action
        ):
            raise ReleaseAgentError("packaged agent returned a non-integer action list")
        self.runtime_status = _validated_runtime_status(response.get("runtime_status"))
        raw_execution_status = response.get("execution_status")
        if isinstance(raw_execution_status, Mapping):
            self.execution_status = cast(Mapping[str, Any], raw_execution_status)
        self.policy_loaded_observed = (
            self.policy_loaded_observed
            or self.runtime_status.get("policy_loaded") is True
        )
        telemetry = response.get("telemetry")
        self._last_telemetry = (
            cast(Mapping[str, Any], telemetry) if isinstance(telemetry, Mapping) else {}
        )
        action = tuple(cast(list[int], raw_action))
        if not is_legal_action(field_value(observation, "select"), action):
            raise ReleaseAgentError(f"packaged agent returned illegal action: {action}")
        return action

    def last_act_telemetry(self) -> Mapping[str, Any]:
        """Expose optional packaged telemetry to the existing arena accumulator."""
        return self._last_telemetry

    def close(self) -> None:
        """Close the bridge and kill the complete episode process group if needed."""
        process = self._process
        channel = self._channel
        if process is not None and process.poll() is None and channel is not None:
            try:
                self._send({"type": "close"})
                self._receive(timeout_seconds=5.0)
            except (OSError, EOFError, TimeoutError, ReleaseAgentError):
                pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
        if channel is not None:
            channel.close()
            self._channel = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _send(self, payload: Mapping[str, Any]) -> None:
        channel = self._channel
        if channel is None:
            raise ReleaseAgentError("release-agent channel is closed")
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ReleaseAgentError("release-agent request exceeds protocol limit")
        try:
            channel.sendall(_HEADER.pack(len(encoded)) + encoded)
        except OSError as exc:
            raise ReleaseAgentError(_dead_process_message(self._process, exc)) from exc

    def _receive(self, *, timeout_seconds: float) -> dict[str, Any]:
        channel = self._channel
        if channel is None:
            raise ReleaseAgentError("release-agent channel is closed")
        channel.settimeout(timeout_seconds)
        try:
            size = _HEADER.unpack(_read_exact(channel, _HEADER.size))[0]
            if size > _MAX_MESSAGE_BYTES:
                raise ReleaseAgentError(
                    f"release-agent response exceeds protocol limit: {size}"
                )
            value = json.loads(_read_exact(channel, size))
        except TimeoutError as exc:
            raise TimeoutError(
                f"release agent exceeded {timeout_seconds:.3f}s RPC timeout"
            ) from exc
        except (EOFError, OSError) as exc:
            raise ReleaseAgentError(_dead_process_message(self._process, exc)) from exc
        if not isinstance(value, dict):
            raise ReleaseAgentError("release-agent response must be a JSON object")
        return cast(dict[str, Any], value)


def _extract_archive_once(
    archive_path: Path,
    *,
    extract_dir: Path,
    expected_sha256: str,
) -> None:
    marker = extract_dir / ".release-extract.json"
    if extract_dir.exists():
        if not marker.is_file():
            raise ValueError(f"unverified release cache already exists: {extract_dir}")
        raw = json.loads(marker.read_text(encoding="utf-8"))
        if raw.get("runtime_sha256") != expected_sha256:
            raise ValueError(f"release cache fingerprint mismatch: {extract_dir}")
        return
    if file_sha256(archive_path) != expected_sha256:
        raise ValueError(f"release archive hash changed: {archive_path}")
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = extract_dir.parent / f".{extract_dir.name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        root = temporary.resolve()
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"release archive must not contain links: {member.name}"
                    )
                target = (temporary / member.name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(
                        f"release archive member escapes extraction root: {member.name}"
                    )
                archive.extract(member, temporary)
        write_identity_atomic(
            temporary / marker.name,
            {
                "runtime_archive_path": records.display_path(archive_path),
                "runtime_sha256": expected_sha256,
            },
        )
        temporary.replace(extract_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_internal_assets(
    extract_dir: Path,
    manifest: ReleaseBundleIdentityLike,
) -> None:
    expected: tuple[tuple[Path, str | None], ...] = (
        (_INTERNAL_CHECKPOINT, manifest.checkpoint_sha256),
        (_INTERNAL_DECK, manifest.deck_sha256),
        (_INTERNAL_MAIN, None),
        (_INTERNAL_ENGINE, None),
    )
    if manifest.belief_sha256 is not None:
        expected += ((_INTERNAL_BELIEF, manifest.belief_sha256),)
    for relative_path, expected_sha256 in expected:
        path = extract_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"release archive is missing required member: {relative_path}"
            )
        if expected_sha256 is not None and file_sha256(path) != expected_sha256:
            raise ValueError(f"release archive member hash mismatch: {relative_path}")


def _validated_runtime_status(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseAgentError("packaged runtime did not expose runtime_status()")
    status = cast(Mapping[str, Any], value)
    if status.get("checkpoint_path") is None:
        raise ReleaseAgentError("packaged runtime did not resolve its checkpoint")
    for error_name in ("prewarm_error", "engine_prewarm_error", "policy_error"):
        if status.get(error_name) is not None:
            raise ReleaseAgentError(
                f"packaged runtime reported {error_name}: {status[error_name]!r}"
            )
    if status.get("used_random_fallback") is not False:
        raise ReleaseAgentError(
            "packaged runtime used or may have used random fallback"
        )
    return status


def _read_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("release-agent protocol closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _response_error(prefix: str, response: Mapping[str, Any]) -> str:
    return (
        f"{prefix}: {response.get('error_type', response.get('type'))}: "
        f"{response.get('error_message', '')}\n{response.get('traceback', '')}"
    ).strip()


def _dead_process_message(
    process: subprocess.Popen[bytes] | None,
    exc: BaseException,
) -> str:
    returncode = process.poll() if process is not None else None
    return f"release-agent protocol failed (returncode={returncode}): {exc}"


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _sanitized_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PTCG_RL_") and key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def sha256_bytes(value: bytes) -> str:
    """Hash small protocol evidence without writing another artifact."""
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "IsolatedReleaseAgent",
    "ReleaseAgentError",
    "ReleaseCapsule",
    "bridge_path",
    "cleanup_release_cache",
    "materialize_release_capsule",
]
