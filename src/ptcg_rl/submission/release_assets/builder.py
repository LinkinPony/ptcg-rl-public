"""Build immutable FP16 runtime assets for deployment candidates."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.bundle_models import BundleGauntletConfig
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.submission.checkpoint_assets import (
    RuntimeCheckpointExportConfig,
    export_runtime_checkpoint,
)
from ptcg_rl.submission.protocol import (
    KAGGLE_SUBMISSION_MAX_BYTES,
    ArchiveEntry,
    SubmissionProfileConfig,
    ValidationConfig,
    build_submission_archive,
    validate_submission_archive,
)
from ptcg_rl.submission.release_assets.manifest import runtime_config_sha256
from ptcg_rl.submission.release_assets.models import (
    ReleaseBundleManifest,
    ReleaseBundleManifestLike,
    ReleaseBundleManifestV2,
    ReleaseBundleManifestV3,
    ReleaseBundleManifestV4,
    ReleaseBundleManifestV5,
    ReleaseDeckCompositionalManifest,
    ReleaseDeckConditioningManifest,
    ReleaseDeckDensePrivateManifest,
    ReleaseDeckLoRAMergeManifest,
)


class ReleaseCandidateAssetConfig(BaseModel):
    """Frozen inputs for one release-candidate deployment bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    checkpoint_tag: str
    raw_checkpoint_path: Path
    deck_path: Path
    belief_path: Path
    runtime_template_bundle_config_path: Path
    runtime_template_bundle_id: str | None = None
    static_features_path: Path = Path(
        "outputs/cards/static_features/card_static_features.npy"
    )
    output_dir: Path
    validation_timeout_seconds: float = 300.0
    private_profile_mode: Literal["full", "pruned", "merged", "fixed"] = "full"
    tensor_storage: Literal["native", "byte_shuffle_v1"] = "native"
    direct_policy_only: bool = False
    require_belief_prior: bool = True
    runtime_exclude_globs: tuple[str, ...] = ()
    max_archive_bytes: int = KAGGLE_SUBMISSION_MAX_BYTES

    @field_validator("bundle_id", "checkpoint_tag")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Reject blank or moving candidate labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("release candidate identity must be immutable")
        return normalized

    @field_validator(
        "raw_checkpoint_path",
        "deck_path",
        "belief_path",
        "runtime_template_bundle_config_path",
        "output_dir",
    )
    @classmethod
    def immutable_path(cls, value: Path) -> Path:
        """Reject moving latest aliases in evaluated release inputs."""
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("release candidate paths cannot contain 'latest'")
        return value

    @field_validator("max_archive_bytes")
    @classmethod
    def valid_archive_limit(cls, value: int) -> int:
        """Allow stricter local limits without weakening Kaggle's hard cap."""
        if value <= 0 or value > KAGGLE_SUBMISSION_MAX_BYTES:
            raise ValueError(
                "max_archive_bytes must be in "
                f"[1, {KAGGLE_SUBMISSION_MAX_BYTES}]"
            )
        return value


def prepare_release_candidate_asset(
    config: ReleaseCandidateAssetConfig,
) -> dict[str, Any]:
    """Export, package, validate, and atomically freeze one candidate."""
    output_dir = records.repo_path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable release asset already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / (
        f".{output_dir.name}.{time.time_ns()}.tmp"
    )
    temporary_dir.mkdir()
    try:
        summary = _prepare_in_directory(
            config,
            temporary_dir=temporary_dir,
            final_dir=output_dir,
        )
        temporary_dir.replace(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _prepare_in_directory(
    config: ReleaseCandidateAssetConfig,
    *,
    temporary_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    raw_checkpoint = _required_file(config.raw_checkpoint_path, "raw checkpoint")
    deck_path = _required_file(config.deck_path, "candidate deck")
    belief_path = _required_file(config.belief_path, "candidate belief")
    static_features = _required_file(config.static_features_path, "static features")
    source_bundle_path = _required_file(
        config.runtime_template_bundle_config_path,
        "runtime template bundle config",
    )
    source_bundle = BundleGauntletConfig.model_validate_json(
        source_bundle_path.read_text(encoding="utf-8")
    )
    template_agent = _runtime_template_agent(
        source_bundle,
        bundle_id=config.runtime_template_bundle_id,
    )
    fp16_relative = Path("checkpoint") / f"{config.checkpoint_tag}.pt"
    archive_relative = Path("runtime") / f"{config.checkpoint_tag}.tar.gz"
    fp16_temporary = temporary_dir / fp16_relative
    archive_temporary = temporary_dir / archive_relative
    fp16_final = final_dir / fp16_relative
    archive_final = final_dir / archive_relative
    export_summary = export_runtime_checkpoint(
        RuntimeCheckpointExportConfig(
            source_checkpoint=raw_checkpoint,
            output_checkpoint=fp16_temporary,
            precision="fp16",
            deck_path=deck_path,
            private_profile_mode=config.private_profile_mode,
            tensor_storage=config.tensor_storage,
            direct_policy_only=config.direct_policy_only,
        )
    )
    deck_sha256 = file_sha256(deck_path)
    profile = _candidate_submission_profile(
        config,
        checkpoint_path=fp16_temporary,
        deck_path=deck_path,
        deck_sha256=deck_sha256,
        belief_path=belief_path,
        static_features_path=static_features,
    )
    build_submission_archive(profile, archive_temporary)
    archive_bytes = _validate_archive_size(
        archive_temporary,
        max_bytes=config.max_archive_bytes,
    )
    validation = validate_submission_archive(archive_temporary, profile)
    raw_deck_conditioning = export_summary.get("deck_conditioning")
    deck_conditioning: ReleaseDeckConditioningManifest | None = None
    if raw_deck_conditioning is not None:
        if not isinstance(raw_deck_conditioning, Mapping):
            raise ValueError("checkpoint deck-conditioning identity must be a mapping")
        if raw_deck_conditioning.get("fixed_compositional") is True:
            deck_conditioning = ReleaseDeckCompositionalManifest.model_validate(
                raw_deck_conditioning
            )
        elif raw_deck_conditioning.get("fixed_strategy") is True:
            deck_conditioning = ReleaseDeckDensePrivateManifest.model_validate(
                raw_deck_conditioning
            )
        elif raw_deck_conditioning.get("lora_merged") is True:
            deck_conditioning = ReleaseDeckLoRAMergeManifest.model_validate(
                raw_deck_conditioning
            )
        else:
            deck_conditioning = ReleaseDeckConditioningManifest.model_validate(
                raw_deck_conditioning
            )
    if deck_conditioning is not None:
        _verify_validation_runtime_binding(validation, deck_conditioning)
    checkpoint_sha256 = file_sha256(fp16_temporary)
    belief_sha256 = file_sha256(belief_path)
    runtime_sha256 = file_sha256(archive_temporary)
    runtime_sha256_config = runtime_config_sha256(
        template_agent.model_dump(mode="json")
    )
    fingerprint_fields: dict[str, Any] = {
        "bundle_id": config.bundle_id,
        "checkpoint_sha256": checkpoint_sha256,
        "deck_sha256": deck_sha256,
        "belief_sha256": belief_sha256,
        "runtime_config_sha256": runtime_sha256_config,
    }
    if deck_conditioning is not None:
        fingerprint_fields["deck_conditioning"] = deck_conditioning.model_dump(
            mode="json"
        )
    bundle_fingerprint = fingerprint_payload(fingerprint_fields)
    manifest_fields: dict[str, Any] = {
        "bundle_id": config.bundle_id,
        "checkpoint_tag": config.checkpoint_tag,
        "checkpoint_path": Path(records.display_path(fp16_final)),
        "checkpoint_sha256": checkpoint_sha256,
        "deck_path": Path(records.display_path(deck_path)),
        "deck_sha256": deck_sha256,
        "belief_path": Path(records.display_path(belief_path)),
        "belief_sha256": belief_sha256,
        "runtime_archive_path": Path(records.display_path(archive_final)),
        "runtime_sha256": runtime_sha256,
        "runtime_config_sha256": runtime_sha256_config,
        "bundle_fingerprint": bundle_fingerprint,
        "storage_precision": "fp16",
        "compute_precision": "fp32",
    }
    manifest: ReleaseBundleManifestLike
    if isinstance(deck_conditioning, ReleaseDeckCompositionalManifest):
        manifest = ReleaseBundleManifestV5(
            **manifest_fields,
            deck_conditioning=deck_conditioning,
        )
    elif isinstance(deck_conditioning, ReleaseDeckDensePrivateManifest):
        manifest = ReleaseBundleManifestV4(
            **manifest_fields,
            deck_conditioning=deck_conditioning,
        )
    elif isinstance(deck_conditioning, ReleaseDeckLoRAMergeManifest):
        manifest = ReleaseBundleManifestV3(
            **manifest_fields,
            deck_conditioning=deck_conditioning,
        )
    elif deck_conditioning is None:
        manifest = ReleaseBundleManifest(**manifest_fields)
    else:
        manifest = ReleaseBundleManifestV2(
            **manifest_fields,
            deck_conditioning=deck_conditioning,
        )
    manifest_relative = Path("bundle_manifest.json")
    write_identity_atomic(
        temporary_dir / manifest_relative,
        manifest.model_dump(mode="json"),
    )
    build_manifest = {
        "protocol": (
            "RELEASE-CANDIDATE-ASSET-v5"
            if isinstance(deck_conditioning, ReleaseDeckCompositionalManifest)
            else (
                "RELEASE-CANDIDATE-ASSET-v4"
                if isinstance(deck_conditioning, ReleaseDeckDensePrivateManifest)
                else (
                    "RELEASE-CANDIDATE-ASSET-v3"
                    if isinstance(deck_conditioning, ReleaseDeckLoRAMergeManifest)
                    else (
                        "RELEASE-CANDIDATE-ASSET-v2"
                        if deck_conditioning is not None
                        else "RELEASE-CANDIDATE-ASSET-v1"
                    )
                )
            )
        ),
        "bundle_manifest_path": records.display_path(final_dir / manifest_relative),
        "bundle_fingerprint": bundle_fingerprint,
        "raw_checkpoint_path": records.display_path(raw_checkpoint),
        "raw_checkpoint_sha256": file_sha256(raw_checkpoint),
        "fp16_checkpoint_path": records.display_path(fp16_final),
        "fp16_checkpoint_sha256": checkpoint_sha256,
        "runtime_archive_path": records.display_path(archive_final),
        "runtime_archive_sha256": runtime_sha256,
        "runtime_archive_bytes": archive_bytes,
        "runtime_archive_max_bytes": config.max_archive_bytes,
        "runtime_template_bundle_config_path": records.display_path(
            source_bundle_path
        ),
        "runtime_template_bundle_config_sha256": file_sha256(source_bundle_path),
        "runtime_config_sha256": runtime_sha256_config,
        "export": dict(export_summary),
        "submission_validation": dict(validation),
        "git": _git_identity(),
    }
    write_identity_atomic(temporary_dir / "build_manifest.json", build_manifest)
    return build_manifest


def _validate_archive_size(path: Path, *, max_bytes: int) -> int:
    """Reject a runtime archive that Kaggle cannot accept."""
    archive_bytes = path.stat().st_size
    if archive_bytes > max_bytes:
        raise ValueError(
            "submission archive exceeds the configured Kaggle limit: "
            f"{archive_bytes} > {max_bytes} bytes"
        )
    return archive_bytes


def _candidate_submission_profile(
    config: ReleaseCandidateAssetConfig,
    *,
    checkpoint_path: Path,
    deck_path: Path,
    deck_sha256: str,
    belief_path: Path,
    static_features_path: Path,
) -> SubmissionProfileConfig:
    return SubmissionProfileConfig(
        name=f"release_candidate_{config.checkpoint_tag}",
        draft=True,
        include_paths=(
            ArchiveEntry(source=Path("src/main.py"), target=PurePosixPath("main.py")),
            ArchiveEntry(source=Path("src/ptcg_rl"), target=PurePosixPath("ptcg_rl")),
            ArchiveEntry(
                source=Path("data/sample_submission/cg"),
                target=PurePosixPath("cg"),
            ),
            ArchiveEntry(source=deck_path, target=PurePosixPath("deck.csv")),
            ArchiveEntry(
                source=checkpoint_path,
                target=PurePosixPath("agent_checkpoint.pt"),
            ),
            ArchiveEntry(
                source=belief_path,
                target=PurePosixPath("belief_prior.csv"),
            ),
            ArchiveEntry(
                source=static_features_path,
                target=PurePosixPath(
                    "outputs/cards/static_features/card_static_features.npy"
                ),
            ),
        ),
        required_files=(
            PurePosixPath("main.py"),
            PurePosixPath("deck.csv"),
            PurePosixPath("agent_checkpoint.pt"),
            PurePosixPath("belief_prior.csv"),
            PurePosixPath("cg/api.py"),
            PurePosixPath("outputs/cards/static_features/card_static_features.npy"),
        ),
        validation=ValidationConfig(
            expected_action_length=60,
            expected_deck_sha256=deck_sha256,
            require_checkpoint_policy=True,
            require_engine=True,
            require_belief_prior=config.require_belief_prior,
            timeout_seconds=config.validation_timeout_seconds,
        ),
        exclude_globs=(
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
            "**/node_modules/**",
            *config.runtime_exclude_globs,
        ),
    )


def _runtime_template_agent(
    source: BundleGauntletConfig,
    *,
    bundle_id: str | None,
) -> Any:
    candidates = (
        [item for item in source.candidates if item.bundle_id == bundle_id]
        if bundle_id is not None
        else list(source.candidates)
    )
    for candidate in candidates:
        if candidate.agent.kind == "runtime":
            return candidate.agent
    label = bundle_id if bundle_id is not None else "any runtime candidate"
    raise ValueError(f"runtime template bundle config has no {label}")


def _required_file(path: Path, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _verify_validation_runtime_binding(
    validation: Mapping[str, Any],
    expected: ReleaseDeckConditioningManifest,
) -> None:
    """Require the Kaggle-like first callback to load the selected module."""
    status = validation.get("runtime_status")
    if not isinstance(status, Mapping):
        raise ValueError("submission validation did not return runtime status")
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
            raise ValueError(f"submission runtime deck binding mismatch: {key}")


def _git_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def load_release_candidate_asset_config(path: Path) -> ReleaseCandidateAssetConfig:
    """Load a JSON config for non-Hydra callers and queue workers."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"release candidate config must be an object: {path}")
    return ReleaseCandidateAssetConfig.model_validate(raw)


__all__ = [
    "ReleaseCandidateAssetConfig",
    "load_release_candidate_asset_config",
    "prepare_release_candidate_asset",
]
