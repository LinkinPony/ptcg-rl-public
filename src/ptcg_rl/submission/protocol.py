"""Build and validate multi-file Kaggle submission archives."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.submission.dispatch import (
    SubmissionDispatchRequest,
    dispatch_submission_once,
    submission_archive_fingerprint,
    tagged_submission_message,
)
from ptcg_rl.submission.release_assets import (
    ReleaseBundleIdentityLike,
    ReleaseBundleIdentityV2,
    ReleaseBundleIdentityV3,
    ReleaseBundleIdentityV4,
    load_release_bundle,
    submission_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "configs"
DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
KAGGLE_SUBMISSION_MAX_BYTES = 200_000_000
DEFAULT_DISPATCH_DIR = Path("outputs/submission/dispatches")
DEFAULT_HISTORY_PATH = Path("configs/submission/submitted_bundles.json")
VALIDATION_CODE = r"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

agent_dir = Path(sys.argv[1]).resolve()
startup_observation = json.loads(sys.argv[2])
expected_action_length = json.loads(sys.argv[3])
require_checkpoint_policy = json.loads(sys.argv[4])
require_engine = json.loads(sys.argv[5])
require_belief_prior = json.loads(sys.argv[6])

# Mimic the Kaggle agent loader: the archive directory is appended to sys.path
# but cwd never points inside it, neither at import time nor at act time.
os.chdir(tempfile.mkdtemp(prefix="ptcg_rl_submission_validate_cwd_"))
sys.path.insert(0, str(agent_dir))
main = importlib.import_module("main")

sys.path = [
    item for item in sys.path
    if Path(item or ".").resolve() != agent_dir
]

action = main.agent(startup_observation, None)
if not isinstance(action, list):
    raise TypeError(f"agent() must return a list, got {type(action)!r}")
if not all(type(item) is int for item in action):
    raise TypeError("agent() action must contain only ints")
if expected_action_length is not None and len(action) != expected_action_length:
    raise ValueError(
        f"agent() action length {len(action)} != expected {expected_action_length}"
    )

result = {"action_length": len(action)}
if require_checkpoint_policy or require_engine or require_belief_prior:
    runtime = getattr(main, "_AGENT", None)
    runtime_status = getattr(runtime, "runtime_status", None)
    if not callable(runtime_status):
        raise TypeError("main must expose _AGENT.runtime_status() for validation")
    status = runtime_status()
    result["runtime_status"] = status
    if require_checkpoint_policy:
        if status["checkpoint_path"] is None:
            raise ValueError("checkpoint asset was not resolved by the runtime")
        if not status["policy_loaded"]:
            raise ValueError(
                "checkpoint policy failed to load: "
                f"prewarm_error={status['prewarm_error']}"
            )
        if status["prewarm_error"] is not None:
            raise ValueError(
                f"checkpoint policy prewarm failed: {status['prewarm_error']}"
            )
        if status["used_random_fallback"]:
            raise ValueError("runtime used the random fallback during validation")
    if require_engine and status.get("engine_prewarm_error") is not None:
        raise ValueError(
            f"engine prewarm failed: {status['engine_prewarm_error']}"
        )
    if require_belief_prior:
        if status.get("belief_summary_path") is None:
            raise ValueError("belief prior asset was not resolved by the runtime")
        if status.get("belief_error") is not None:
            raise ValueError(
                f"belief prior failed to load: {status['belief_error']}"
            )
        if int(status.get("belief_prior_decks") or 0) <= 0:
            raise ValueError("belief prior loaded zero archetype decks")
print(json.dumps(result))
"""


class ArchiveEntry(BaseModel):
    """One source path copied into a submission archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Path
    target: PurePosixPath
    required: bool = True

    @field_validator("target")
    @classmethod
    def valid_relative_target(cls, value: PurePosixPath) -> PurePosixPath:
        """Reject archive paths that can escape the extraction directory."""
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"archive target must be a relative path: {value}")
        if str(value) in {"", "."}:
            raise ValueError("archive target must not be empty")
        return value


class ValidationConfig(BaseModel):
    """Archive validation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    startup_observation: Mapping[str, Any] = Field(
        default_factory=lambda: {"select": None},
    )
    expected_action_length: int | None = 60
    expected_deck_sha256: str | None = None
    require_checkpoint_policy: bool = False
    require_engine: bool = False
    require_belief_prior: bool = False
    timeout_seconds: float = 30.0

    @field_validator("expected_action_length")
    @classmethod
    def valid_expected_action_length(cls, value: int | None) -> int | None:
        """Reject invalid expected action lengths."""
        if value is not None and value < 0:
            raise ValueError("expected_action_length must be non-negative")
        return value

    @field_validator("expected_deck_sha256")
    @classmethod
    def valid_expected_deck_sha256(cls, value: str | None) -> str | None:
        """Normalize and reject malformed expected deck checksums."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("expected_deck_sha256 must be a 64-character hex digest")
        return normalized

    @field_validator("timeout_seconds")
    @classmethod
    def valid_timeout(cls, value: float) -> float:
        """Reject non-positive validation timeouts."""
        if value <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        return value


class ReleaseAssetConfig(BaseModel):
    """Immutable deployment manifest bound into a submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: Path
    history_path: Path = DEFAULT_HISTORY_PATH
    checkpoint_target: PurePosixPath = PurePosixPath("agent_checkpoint.pt")
    deck_target: PurePosixPath = PurePosixPath("deck.csv")
    belief_target: PurePosixPath | None = PurePosixPath("belief_prior.csv")
    manifest_target: PurePosixPath = PurePosixPath("release_manifest.json")

    @field_validator("manifest_path", "history_path")
    @classmethod
    def immutable_source_path(cls, value: Path) -> Path:
        """Reject moving latest aliases in the deployment identity."""
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("release asset paths cannot contain 'latest'")
        return value

    @field_validator(
        "checkpoint_target",
        "deck_target",
        "belief_target",
        "manifest_target",
    )
    @classmethod
    def valid_target(
        cls,
        value: PurePosixPath | None,
    ) -> PurePosixPath | None:
        """Reject archive targets which can escape extraction."""
        if value is None:
            return None
        if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
            raise ValueError(f"release archive target must be relative: {value}")
        return value

    @model_validator(mode="after")
    def unique_targets(self) -> ReleaseAssetConfig:
        """Require each immutable asset to have a distinct archive target."""
        targets = [
            self.checkpoint_target,
            self.deck_target,
            self.manifest_target,
            *([self.belief_target] if self.belief_target is not None else []),
        ]
        if len(set(targets)) != len(targets):
            raise ValueError("release asset archive targets must be unique")
        return self


class SubmissionProfileConfig(BaseModel):
    """Hydra-loaded build profile for one Kaggle submission package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "random_baseline"
    competition: str = DEFAULT_COMPETITION
    draft: bool = True
    include_paths: tuple[ArchiveEntry, ...]
    required_files: tuple[PurePosixPath, ...] = (
        PurePosixPath("main.py"),
        PurePosixPath("deck.csv"),
    )
    exclude_globs: tuple[str, ...] = (
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.mypy_cache/**",
        "**/.pytest_cache/**",
        "**/.ruff_cache/**",
        "**/node_modules/**",
    )
    validation: ValidationConfig = ValidationConfig()
    release_asset: ReleaseAssetConfig | None = None

    @field_validator("include_paths")
    @classmethod
    def valid_include_paths(
        cls,
        value: tuple[ArchiveEntry, ...],
    ) -> tuple[ArchiveEntry, ...]:
        """Accept explicit entries; release profiles may use only their archive."""
        return value

    @field_validator("required_files")
    @classmethod
    def valid_required_files(
        cls,
        value: tuple[PurePosixPath, ...],
    ) -> tuple[PurePosixPath, ...]:
        """Reject unsafe required archive paths."""
        for path in value:
            if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
                raise ValueError(f"required file must be relative: {path}")
        return value

    @model_validator(mode="after")
    def required_files_are_included(self) -> SubmissionProfileConfig:
        """Ensure configured required files are covered by archive entries."""
        targets = {entry.target for entry in self.include_paths}
        if self.release_asset is not None:
            release_targets = _release_targets(self.release_asset)
            overlap = targets & release_targets
            if overlap:
                raise ValueError(
                    f"release asset duplicates include_paths targets: {sorted(overlap)}"
                )
            targets.update(release_targets)
            # The approved runtime archive supplies main.py, cg, model, deck,
            # belief, and runtime data. Their presence is verified after build.
            targets.update(self.required_files)
        elif not self.include_paths:
            raise ValueError("include_paths may be empty only with a release_asset")
        missing = [
            path
            for path in self.required_files
            if not _is_target_covered(path, targets)
        ]
        if missing:
            raise ValueError(
                f"required files are not covered by include_paths: {missing}"
            )
        return self


def load_submission_profile(profile: str) -> SubmissionProfileConfig:
    """Load one submission profile from ``configs/submission`` with Hydra."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONF_DIR.resolve()),
    ):
        hydra_config = compose(config_name=f"submission/{profile}")
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Submission profile must resolve to a dictionary.")
    return SubmissionProfileConfig.model_validate(cast(dict[str, Any], raw_config))


def build_submission_archive(
    config: SubmissionProfileConfig,
    output_path: Path,
) -> Path:
    """Build a tar.gz submission archive from a validated profile."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    release_bundle = _release_profile_bundle(config)
    release_entries: tuple[ArchiveEntry, ...] = ()
    release_targets: set[PurePosixPath] = set()
    if release_bundle is not None and config.release_asset is not None:
        release_entries = _release_archive_entries(
            config.release_asset,
            release_bundle,
        )
        release_targets = {entry.target for entry in release_entries}
    added_targets: set[PurePosixPath] = set()
    with tarfile.open(output_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        if release_bundle is not None:
            _add_release_runtime_archive(
                archive=archive,
                bundle=release_bundle,
                exclude_globs=config.exclude_globs,
                replaced_targets=release_targets,
                added_targets=added_targets,
            )
        for entry in config.include_paths:
            source_path = _resolve_repo_path(entry.source)
            if not source_path.exists():
                if entry.required:
                    raise FileNotFoundError(
                        f"required source path does not exist: {source_path}"
                    )
                continue
            _add_archive_entry(
                archive=archive,
                source_path=source_path,
                target_path=entry.target,
                exclude_globs=config.exclude_globs,
                added_targets=added_targets,
            )
        if release_entries:
            for entry in release_entries:
                _add_archive_entry(
                    archive=archive,
                    source_path=_resolve_repo_path(entry.source),
                    target_path=entry.target,
                    exclude_globs=config.exclude_globs,
                    added_targets=added_targets,
                )
    _verify_archive_required_files(output_path, config.required_files)
    _validate_submission_archive_size(output_path)
    return output_path


def _validate_submission_archive_size(path: Path) -> int:
    """Reject an archive above Kaggle's hard submission-size ceiling."""
    archive_bytes = path.stat().st_size
    if archive_bytes > KAGGLE_SUBMISSION_MAX_BYTES:
        raise ValueError(
            "submission archive exceeds the Kaggle limit: "
            f"{archive_bytes} > {KAGGLE_SUBMISSION_MAX_BYTES} bytes"
        )
    return archive_bytes


def validate_submission_archive(
    archive_path: Path,
    config: SubmissionProfileConfig,
) -> Mapping[str, Any]:
    """Import packaged ``main.py`` and call ``agent`` in an isolated cwd."""
    with tempfile.TemporaryDirectory(prefix="ptcg_rl_submission_validate_") as raw_dir:
        temp_dir = Path(raw_dir)
        extract_dir = temp_dir / "agent"
        extract_dir.mkdir()
        _safe_extract(archive_path, extract_dir)
        _verify_extracted_required_files(extract_dir, config.required_files)
        validation = dict(
            _run_validation_subprocess(
                extract_dir,
                config.validation.startup_observation,
                expected_action_length=config.validation.expected_action_length,
                require_checkpoint_policy=config.validation.require_checkpoint_policy,
                require_engine=config.validation.require_engine,
                require_belief_prior=config.validation.require_belief_prior,
                timeout_seconds=config.validation.timeout_seconds,
            )
        )
        if config.validation.expected_deck_sha256 is not None:
            validation["deck_sha256"] = _verify_expected_file_sha256(
                extract_dir / "deck.csv",
                config.validation.expected_deck_sha256,
            )
        release_bundle = _release_profile_bundle(config)
        if release_bundle is not None and config.release_asset is not None:
            validation["release_asset"] = _verify_release_archive_assets(
                extract_dir,
                release_asset=config.release_asset,
                bundle=release_bundle,
            )
            if isinstance(
                release_bundle,
                (
                    ReleaseBundleIdentityV2,
                    ReleaseBundleIdentityV3,
                    ReleaseBundleIdentityV4,
                ),
            ):
                _verify_conditioned_release_runtime(validation, release_bundle)
        return validation


def _verify_expected_file_sha256(path: Path, expected_sha256: str) -> str:
    """Verify one extracted submission file against an expected SHA256 digest."""
    if not path.is_file():
        raise FileNotFoundError(f"expected hashed file does not exist: {path}")
    digest = _file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"packaged {path.name} sha256 {digest} != expected {expected_sha256}"
        )
    return digest


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _submit_archive_unprotected(
    archive_path: Path,
    *,
    competition: str,
    message: str,
) -> None:
    """Low-level uploader; callers must hold a durable dispatch reservation."""
    if not message.strip():
        raise ValueError("submission message must not be empty")
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            str(archive_path),
            "-m",
            message,
        ],
        check=True,
    )


def _submit_archive_for_dispatch(
    archive_path: Path,
    competition: str,
    message: str,
) -> None:
    """Adapt the low-level uploader to the protected dispatcher."""
    _submit_archive_unprotected(
        archive_path,
        competition=competition,
        message=message,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the repository submission protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="random_baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--message")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--dispatch-dir", type=Path, default=DEFAULT_DISPATCH_DIR)
    parser.add_argument("--wait-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    config = load_submission_profile(str(args.profile))
    if config.draft and args.submit and not args.allow_draft:
        raise SystemExit(
            f"profile {config.name!r} is marked draft; pass --allow-draft to submit"
        )
    if (
        args.submit
        and config.validation.require_checkpoint_policy
        and config.release_asset is None
    ):
        raise SystemExit(
            "checkpoint-policy submissions require an immutable release_asset"
        )
    release_bundle = _release_profile_bundle(config)
    release_fingerprint = (
        submission_fingerprint(release_bundle) if release_bundle is not None else None
    )
    requested_output = (
        cast(Path, args.output)
        if args.output is not None
        else _default_submission_output(config.name, release_fingerprint)
    )
    archive_path = build_submission_archive(
        config,
        _resolve_repo_path(requested_output),
    )
    validation = validate_submission_archive(archive_path, config)
    result: dict[str, Any] = {
        "archive": str(archive_path),
        "profile": config.name,
        "validated": validation,
    }

    if args.submit:
        fingerprint = release_fingerprint or submission_archive_fingerprint(
            archive_path
        )
        message = tagged_submission_message(
            str(args.message or config.name),
            fingerprint,
        )
        history_path = (
            config.release_asset.history_path
            if config.release_asset is not None
            else DEFAULT_HISTORY_PATH
        )
        bundle_id = (
            release_bundle.bundle_id
            if release_bundle is not None
            else f"profile::{config.name}"
        )
        outcome = dispatch_submission_once(
            SubmissionDispatchRequest(
                submission_fingerprint=fingerprint,
                bundle_id=bundle_id,
                profile=config.name,
                competition=config.competition,
                archive_path=archive_path,
                archive_sha256=_file_sha256(archive_path),
                archive_size_bytes=archive_path.stat().st_size,
                message=message,
                dispatch_dir=_resolve_repo_path(cast(Path, args.dispatch_dir)),
                history_path=_resolve_repo_path(history_path),
            ),
            submitter=_submit_archive_for_dispatch,
            wait_seconds=float(args.wait_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
        )
        result["dispatch"] = outcome.model_dump(mode="json")
    print(json.dumps(result, sort_keys=True))
    return 0


def _default_submission_output(name: str, fingerprint: str | None) -> Path:
    """Derive a stable archive path for the one-command workflow."""
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in name
    ).strip("_")
    if not safe_name:
        raise ValueError("submission profile name has no safe filename characters")
    suffix = f"_{fingerprint[:16]}" if fingerprint is not None else ""
    return Path("dist") / f"{safe_name}{suffix}.tar.gz"


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _release_profile_bundle(
    config: SubmissionProfileConfig,
) -> ReleaseBundleIdentityLike | None:
    release_asset = config.release_asset
    if release_asset is None:
        return None
    return load_release_bundle(release_asset.manifest_path)


def ensure_unsubmitted_release(config: SubmissionProfileConfig) -> str | None:
    """Reject an effective deployment bundle already present in history."""
    release_asset = config.release_asset
    if release_asset is None:
        return None
    bundle = load_release_bundle(release_asset.manifest_path)
    fingerprint = submission_fingerprint(bundle)
    history_path = _resolve_repo_path(release_asset.history_path)
    history = _read_submission_history(history_path)
    duplicates = [
        str(entry.get("submission_ref", "unknown"))
        for entry in history
        if str(entry.get("submission_fingerprint", "")) == fingerprint
    ]
    if duplicates:
        raise ValueError(
            "effective release bundle was already submitted as: "
            + ", ".join(duplicates)
        )
    return fingerprint


def _read_submission_history(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"submission history does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("submission history must be a JSON object")
    if raw.get("protocol") != "SUBMITTED-BUNDLES-v1":
        raise ValueError("unsupported submission history protocol")
    entries = raw.get("submissions")
    if not isinstance(entries, list):
        raise ValueError("submission history submissions must be a list")
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise ValueError("submission history entries must be objects")
    return tuple(cast(Mapping[str, Any], entry) for entry in entries)


def _release_targets(release_asset: ReleaseAssetConfig) -> set[PurePosixPath]:
    targets = {
        release_asset.checkpoint_target,
        release_asset.deck_target,
        release_asset.manifest_target,
    }
    if release_asset.belief_target is not None:
        targets.add(release_asset.belief_target)
    return targets


def _release_archive_entries(
    release_asset: ReleaseAssetConfig,
    bundle: ReleaseBundleIdentityLike,
) -> tuple[ArchiveEntry, ...]:
    entries = [
        ArchiveEntry(
            source=bundle.checkpoint_path,
            target=release_asset.checkpoint_target,
        ),
        ArchiveEntry(source=bundle.deck_path, target=release_asset.deck_target),
        ArchiveEntry(
            source=release_asset.manifest_path,
            target=release_asset.manifest_target,
        ),
    ]
    if release_asset.belief_target is not None:
        if bundle.belief_path is None or bundle.belief_sha256 is None:
            raise ValueError(
                "release asset requires a belief target but the bundle has no belief"
            )
        entries.append(
            ArchiveEntry(
                source=bundle.belief_path,
                target=release_asset.belief_target,
            )
        )
    return tuple(entries)


def _add_release_runtime_archive(
    *,
    archive: tarfile.TarFile,
    bundle: ReleaseBundleIdentityLike,
    exclude_globs: Sequence[str],
    replaced_targets: set[PurePosixPath],
    added_targets: set[PurePosixPath],
) -> None:
    source_path = _resolve_repo_path(bundle.runtime_archive_path)
    if _file_sha256(source_path) != bundle.runtime_sha256:
        raise ValueError("approved runtime archive SHA256 changed")
    with tarfile.open(source_path, "r:*") as source_archive:
        for member in source_archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"approved runtime archive contains a non-file: {member.name}"
                )
            target = PurePosixPath(member.name)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(
                    f"approved runtime archive target is unsafe: {member.name}"
                )
            if _is_excluded(target, exclude_globs):
                continue
            if target in replaced_targets:
                continue
            if target in added_targets:
                raise ValueError(f"duplicate archive target: {target}")
            file_obj = source_archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"cannot read approved runtime file: {member.name}")
            archive.addfile(member, file_obj)
            added_targets.add(target)


def _verify_release_archive_assets(
    extract_dir: Path,
    *,
    release_asset: ReleaseAssetConfig,
    bundle: ReleaseBundleIdentityLike,
) -> Mapping[str, str | None]:
    manifest_sha256 = _verify_expected_file_sha256(
        extract_dir / release_asset.manifest_target,
        bundle.source_manifest_sha256,
    )
    checkpoint_sha256 = _verify_expected_file_sha256(
        extract_dir / release_asset.checkpoint_target,
        bundle.checkpoint_sha256,
    )
    deck_sha256 = _verify_expected_file_sha256(
        extract_dir / release_asset.deck_target,
        bundle.deck_sha256,
    )
    belief_sha256: str | None = None
    if release_asset.belief_target is not None and bundle.belief_sha256 is not None:
        belief_sha256 = _verify_expected_file_sha256(
            extract_dir / release_asset.belief_target,
            bundle.belief_sha256,
        )
    return {
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "submission_fingerprint": submission_fingerprint(bundle),
        "runtime_sha256": bundle.runtime_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "deck_sha256": deck_sha256,
        "belief_sha256": belief_sha256,
    }


def _verify_conditioned_release_runtime(
    validation: Mapping[str, Any],
    bundle: (
        ReleaseBundleIdentityV2 | ReleaseBundleIdentityV3 | ReleaseBundleIdentityV4
    ),
) -> None:
    """Match the isolated first callback to the conditioned deck identity."""
    status = validation.get("runtime_status")
    if not isinstance(status, Mapping):
        raise ValueError("conditioned release validation has no runtime status")
    expected = bundle.deck_conditioning
    expected_fields = {
        "deck_conditioning_enabled": True,
        "bound_deck_signature": expected.canonical_deck_signature,
        "selected_private_profile_module_key": (
            expected.selected_private_profile_module_key
        ),
        "packaged_private_profile_count": expected.packaged_private_profile_count,
    }
    for key, expected_value in expected_fields.items():
        if status.get(key) != expected_value:
            raise ValueError(f"conditioned release runtime mismatch: {key}")
    registry_sha256 = status.get("checkpoint_registry_sha256")
    if not isinstance(registry_sha256, str) or len(registry_sha256) != 64:
        raise ValueError("conditioned release runtime has no checkpoint registry")
    if (
        not expected.pruning_applied
        and registry_sha256 != expected.source_registry_sha256
    ):
        raise ValueError("conditioned release runtime registry mismatch")


def _add_archive_entry(
    *,
    archive: tarfile.TarFile,
    source_path: Path,
    target_path: PurePosixPath,
    exclude_globs: Sequence[str],
    added_targets: set[PurePosixPath],
) -> None:
    if source_path.is_file():
        _add_file(
            archive=archive,
            source_path=source_path,
            archive_path=target_path,
            exclude_globs=exclude_globs,
            added_targets=added_targets,
        )
        return
    if not source_path.is_dir():
        raise ValueError(f"source path is not a file or directory: {source_path}")
    for child in sorted(source_path.rglob("*")):
        if not child.is_file():
            continue
        relative_child = PurePosixPath(child.relative_to(source_path).as_posix())
        archive_child = target_path / relative_child
        _add_file(
            archive=archive,
            source_path=child,
            archive_path=archive_child,
            exclude_globs=exclude_globs,
            added_targets=added_targets,
        )


def _add_file(
    *,
    archive: tarfile.TarFile,
    source_path: Path,
    archive_path: PurePosixPath,
    exclude_globs: Sequence[str],
    added_targets: set[PurePosixPath],
) -> None:
    if _is_excluded(archive_path, exclude_globs):
        return
    if archive_path in added_targets:
        raise ValueError(f"duplicate archive target: {archive_path}")
    archive.add(source_path, arcname=archive_path.as_posix(), recursive=False)
    added_targets.add(archive_path)


def _is_excluded(path: PurePosixPath, patterns: Sequence[str]) -> bool:
    text = path.as_posix()
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        return True
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def _verify_archive_required_files(
    archive_path: Path,
    required_files: Sequence[PurePosixPath],
) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        names = {PurePosixPath(member.name) for member in archive.getmembers()}
    missing = [path for path in required_files if path not in names]
    if missing:
        raise FileNotFoundError(f"archive is missing required files: {missing}")


def _verify_extracted_required_files(
    extract_dir: Path,
    required_files: Sequence[PurePosixPath],
) -> None:
    missing = [path for path in required_files if not (extract_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"extracted archive is missing required files: {missing}"
        )


def _safe_extract(archive_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive must not contain links: {member.name}")
            target = (extract_dir / member.name).resolve()
            if not _is_relative_to(target, extract_root):
                raise ValueError(
                    f"archive member escapes extraction dir: {member.name}"
                )
            archive.extract(member, extract_dir)


def _run_validation_subprocess(
    extract_dir: Path,
    startup_observation: Mapping[str, Any],
    *,
    expected_action_length: int | None,
    require_checkpoint_policy: bool,
    require_engine: bool,
    require_belief_prior: bool,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PTCG_RL_CHECKPOINT_PATH", None)
    env.pop("PTCG_RL_DECK_PATH", None)
    env.pop("PTCG_RL_BELIEF_SUMMARY_PATH", None)
    env.pop("PTCG_RL_PLANNER_RUNTIME_PATH", None)
    env.pop("PTCG_RL_PLANNER_RUNTIME_SHA256", None)
    env.pop("PTCG_RL_PLANNER_PROFILE_ENABLED", None)
    env.pop("PTCG_RL_POLICY_TEMPERATURE", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            VALIDATION_CODE,
            str(extract_dir),
            json.dumps(startup_observation),
            json.dumps(expected_action_length),
            json.dumps(require_checkpoint_policy),
            json.dumps(require_engine),
            json.dumps(require_belief_prior),
        ],
        cwd=extract_dir.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    output = completed.stdout.strip().splitlines()
    if not output:
        return {}
    result = json.loads(output[-1])
    if not isinstance(result, dict):
        raise ValueError("validation subprocess returned non-object JSON")
    return cast(Mapping[str, Any], result)


def _is_target_covered(
    required_path: PurePosixPath,
    targets: set[PurePosixPath],
) -> bool:
    return any(
        required_path == target or required_path.is_relative_to(target)
        for target in targets
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
