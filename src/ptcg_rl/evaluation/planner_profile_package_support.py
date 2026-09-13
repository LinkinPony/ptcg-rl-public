"""Shared package identity and archive helpers for planner profiling."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ptcg_rl.agent.packaged_planner import PackagedPlannerConfig
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context.belief import OpponentBeliefFeatureProducer
from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR,
    native_planning_session_abi_fingerprint,
    native_planning_session_schema_fingerprint,
)
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerPackageAssetId,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_package_models import (
    ASSET_IDS,
    REQUIRED_FILES,
    PlannerPackageAssetManifestEntry,
    PlannerPackageAssetsManifest,
    SharedPackageIdentity,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.submission.protocol import (
    REPO_ROOT,
    ArchiveEntry,
    SubmissionProfileConfig,
    load_submission_profile,
)


def validate_shared_package_identity(
    config: IntegratedPlannerProfileConfig,
) -> SharedPackageIdentity:
    """Validate model, engine, and belief bytes shared by all packages."""
    package = config.package_assets
    checkpoint_path = package.deployment_checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"deployment checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != package.expected_deployment_checkpoint_sha256:
        raise ValueError("deployment checkpoint fingerprint differs from config")
    _validate_export_lineage(config)
    checkpoint_policy = CheckpointPolicy(checkpoint_path, device="cpu")
    if checkpoint_policy.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("loaded deployment checkpoint fingerprint changed")
    if checkpoint_policy.policy_version != config.policy_version:
        raise ValueError("deployment checkpoint publication version differs")
    model_fingerprint = canonical_model_state_fingerprint(
        checkpoint_policy.planner_model
    )
    if model_fingerprint != package.expected_deployment_model_fingerprint:
        raise ValueError("deployment loaded-model fingerprint differs from config")
    del checkpoint_policy

    native_sha256 = file_sha256(config.native_library_path)
    if native_sha256 != config.expected_native_library_sha256:
        raise ValueError("package native library fingerprint differs from config")
    native_abi = native_planning_session_abi_fingerprint(
        NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR
    )
    native_schema = native_planning_session_schema_fingerprint()
    if native_abi != config.expected_native_abi_fingerprint or (
        native_schema != config.expected_native_schema_fingerprint
    ):
        raise ValueError("package native ABI/schema differs from config")

    workloads = tuple(runtime.belief for runtime in config.runtime_profiles.values())
    workload = workloads[0]
    prior_path = workload.sampler.prior_deck_signature_summary_path
    prior_expected = workload.sampler.prior_deck_signature_summary_sha256
    if prior_path is None or prior_expected is None or not prior_path.is_file():
        raise FileNotFoundError("package belief prior is not available")
    prior_sha256 = file_sha256(prior_path)
    if prior_sha256 != prior_expected:
        raise ValueError("package belief prior fingerprint differs from config")
    sampler = BeliefSampler(config=workload.sampler)
    producer = OpponentBeliefFeatureProducer.from_config(workload.producer)
    belief_fingerprint = belief_runtime_fingerprint(sampler, producer)
    configured_beliefs = {
        runtime.planner.scenario.belief_sampler_fingerprint
        for runtime in config.runtime_profiles.values()
    }
    if configured_beliefs != {belief_fingerprint}:
        raise ValueError("package belief runtime fingerprint differs from planner")
    return SharedPackageIdentity(
        checkpoint_sha256=checkpoint_sha256,
        model_fingerprint=model_fingerprint,
        native_sha256=native_sha256,
        native_abi_fingerprint=native_abi,
        native_schema_fingerprint=native_schema,
        belief_prior_path=prior_path,
        belief_prior_sha256=prior_sha256,
        belief_runtime_fingerprint=belief_fingerprint,
    )


def planner_runtime_payload(
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    *,
    shared: SharedPackageIdentity,
) -> tuple[dict[str, Any], str, str]:
    """Resolve one packaged planner schema and its composite identities."""
    package = config.package_assets
    workload = runtime.packaged
    if workload is None:
        raise ValueError("planner package payload requires a packaged workload")
    resolved = runtime.planner.resolve_for_lease(
        model_fingerprint=shared.model_fingerprint,
        policy_version=config.policy_version,
        proposal_version=config.proposal_version,
    )
    sampler = runtime.belief.sampler.model_copy(
        update={
            "prior_deck_signature_summary_path": Path(
                package.belief_prior_target.as_posix()
            )
        }
    )
    producer = runtime.belief.producer.model_copy(
        update={
            "deck_signature_summary_path": Path(package.belief_prior_target.as_posix())
        }
    )
    packaged = PackagedPlannerConfig(
        planner_enabled_by_default=workload.planner_enabled_by_default,
        source_checkpoint_sha256=package.source_checkpoint_sha256,
        checkpoint_sha256=shared.checkpoint_sha256,
        model_fingerprint=shared.model_fingerprint,
        policy_version=config.policy_version,
        proposal_version=config.proposal_version,
        expected_planner_fingerprint=resolved.static.planner_fingerprint,
        expected_runtime_fingerprint=resolved.runtime_fingerprint,
        native_library_sha256=shared.native_sha256,
        native_abi_fingerprint=shared.native_abi_fingerprint,
        native_schema_fingerprint=shared.native_schema_fingerprint,
        belief_prior_sha256=shared.belief_prior_sha256,
        belief_runtime_fingerprint=shared.belief_runtime_fingerprint,
        native_library_path=Path(package.native_library_target.as_posix()),
        planner=runtime.planner,
        belief_sampler=sampler,
        belief_producer=producer,
        stochastic_seed=runtime.belief.stochastic_seed,
    )
    return (
        packaged.model_dump(mode="json"),
        resolved.static.planner_fingerprint,
        resolved.runtime_fingerprint,
    )


def packaged_runtimes(
    config: IntegratedPlannerProfileConfig,
) -> dict[PlannerPackageAssetId, tuple[str, PlannerProfileRuntimeConfig]]:
    """Index the exact control/k4/k8/k12 runtime matrix."""
    result: dict[PlannerPackageAssetId, tuple[str, PlannerProfileRuntimeConfig]] = {}
    for runtime_id, runtime in config.runtime_profiles.items():
        packaged = runtime.packaged
        if packaged is None:
            continue
        asset_id = packaged.package_asset_id
        if asset_id in result:
            raise ValueError(f"multiple packaged runtimes claim {asset_id}")
        result[asset_id] = (runtime_id, runtime)
    if set(result) != set(ASSET_IDS):
        raise ValueError(
            "package build requires exactly control, k4, k8, and k12 runtimes"
        )
    return result


def validated_staging_profile(
    config: IntegratedPlannerProfileConfig,
    *,
    asset_id: PlannerPackageAssetId,
    profile: SubmissionProfileConfig,
    staged_runtime: Path,
) -> SubmissionProfileConfig:
    """Retarget only planner_runtime.json to a staged immutable build input."""
    validate_profile_contract(config, asset_id=asset_id, profile=profile)
    final_runtime = config.package_assets.runtime_path(asset_id)
    replaced = 0
    entries: list[ArchiveEntry] = []
    for entry in profile.include_paths:
        if entry.target == config.package_assets.planner_runtime_target:
            if entry.source != final_runtime:
                raise ValueError("submission profile names another planner runtime")
            entry = entry.model_copy(update={"source": staged_runtime.resolve()})
            replaced += 1
        entries.append(entry)
    if replaced != 1:
        raise ValueError("submission profile must include one planner runtime")
    return profile.model_copy(update={"include_paths": tuple(entries)})


def load_profile(profile_name: str) -> SubmissionProfileConfig:
    """Compose a submission profile inside or outside an active Hydra job."""
    if not GlobalHydra.instance().is_initialized():
        return load_submission_profile(profile_name)
    raw = OmegaConf.to_container(
        compose(config_name=f"submission/{profile_name}"),
        resolve=True,
    )
    if not isinstance(raw, dict):
        raise ValueError("submission profile must resolve to a mapping")
    return SubmissionProfileConfig.model_validate(cast(dict[str, Any], raw))


def validate_profile_contract(
    config: IntegratedPlannerProfileConfig,
    *,
    asset_id: PlannerPackageAssetId,
    profile: SubmissionProfileConfig,
) -> None:
    """Validate one exact source-to-archive deployment mapping."""
    package = config.package_assets
    expected_name = package.submission_profiles[asset_id]
    if profile.name != expected_name:
        raise ValueError("submission profile name differs from package config")
    if set(profile.required_files) != REQUIRED_FILES:
        raise ValueError("submission profile required-file contract differs")
    by_target = {entry.target: entry for entry in profile.include_paths}
    required_sources = {
        package.checkpoint_target: package.deployment_checkpoint_path,
        package.belief_prior_target: prior_path(config),
        package.native_library_target: config.native_library_path,
        package.planner_runtime_target: package.runtime_path(asset_id),
    }
    for target, expected_source in required_sources.items():
        entry = by_target.get(target)
        if entry is None or entry.source != expected_source:
            raise ValueError(f"submission profile changes package source for {target}")
    for directory in (PurePosixPath("ptcg_rl"), PurePosixPath("cg")):
        if directory not in by_target:
            raise ValueError(f"submission profile omits package tree {directory}")


def package_manifest(
    config: IntegratedPlannerProfileConfig,
    *,
    shared: SharedPackageIdentity,
    entries: Mapping[PlannerPackageAssetId, PlannerPackageAssetManifestEntry],
) -> PlannerPackageAssetsManifest:
    """Build the manifest payload shared by build and read-only validation."""
    return PlannerPackageAssetsManifest(
        campaign_id=config.campaign_id,
        source_checkpoint_path=config.checkpoint_path,
        source_checkpoint_sha256=config.package_assets.source_checkpoint_sha256,
        deployment_checkpoint_path=(config.package_assets.deployment_checkpoint_path),
        deployment_checkpoint_sha256=shared.checkpoint_sha256,
        deployment_model_fingerprint=shared.model_fingerprint,
        policy_version=config.policy_version,
        proposal_version=config.proposal_version,
        native_library_path=config.native_library_path,
        native_library_sha256=shared.native_sha256,
        native_abi_fingerprint=shared.native_abi_fingerprint,
        native_schema_fingerprint=shared.native_schema_fingerprint,
        belief_prior_path=shared.belief_prior_path,
        belief_prior_sha256=shared.belief_prior_sha256,
        belief_runtime_fingerprint=shared.belief_runtime_fingerprint,
        assets=dict(entries),
    )


def validate_manifest_shared_identity(
    config: IntegratedPlannerProfileConfig,
    manifest: PlannerPackageAssetsManifest,
    *,
    shared: SharedPackageIdentity,
) -> None:
    """Check manifest-wide identities while preserving point entries."""
    expected = package_manifest(config, shared=shared, entries=manifest.assets)
    actual_payload = manifest.model_dump(mode="json", exclude={"assets"})
    expected_payload = expected.model_dump(mode="json", exclude={"assets"})
    if actual_payload != expected_payload:
        raise ValueError("package manifest shared identity differs from config")


def require_fresh_package_outputs(config: IntegratedPlannerProfileConfig) -> None:
    """Refuse an implicit rebuild over immutable package outputs."""
    package = config.package_assets
    existing = [
        path
        for path in (
            package.manifest_path,
            package.manifest_sha256_path,
            *(package.root / asset_id for asset_id in ASSET_IDS),
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"immutable package assets already exist: {existing}")


def prior_path(config: IntegratedPlannerProfileConfig) -> Path:
    """Resolve the single belief prior shared by all runtimes."""
    paths = {
        runtime.belief.sampler.prior_deck_signature_summary_path
        for runtime in config.runtime_profiles.values()
    }
    if len(paths) != 1 or None in paths:
        raise ValueError("package runtimes do not share one prior path")
    return cast(Path, next(iter(paths)))


def required_files_fingerprint(paths: Sequence[PurePosixPath]) -> str:
    """Fingerprint the exact required-file surface."""
    payload = json.dumps(
        sorted(str(path) for path in paths),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/planner-package-required-files/v1\x00" + payload
    ).hexdigest()


def archive_contents_fingerprint(path: Path) -> str:
    """Fingerprint every logical file path and byte payload in an archive."""
    contents: dict[PurePosixPath, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"package archive contains a non-file: {member.name}")
            target = PurePosixPath(member.name)
            if target in contents:
                raise ValueError(f"package archive repeats target: {target}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read package archive member: {target}")
            contents[target] = _stream_sha256(source)
    return _logical_contents_fingerprint(contents)


def profile_source_fingerprint(profile: SubmissionProfileConfig) -> str:
    """Fingerprint current source bytes using archive target names."""
    if profile.release_asset is not None:
        raise ValueError("planner package profiles cannot use release archives")
    contents: dict[PurePosixPath, str] = {}
    for entry in profile.include_paths:
        source_path = _resolve_source(entry.source)
        if not source_path.exists():
            if entry.required:
                raise FileNotFoundError(
                    f"required package source does not exist: {source_path}"
                )
            continue
        if source_path.is_file():
            _add_source_fingerprint(
                contents,
                source_path=source_path,
                target=entry.target,
                exclude_globs=profile.exclude_globs,
            )
            continue
        if not source_path.is_dir():
            raise ValueError(
                f"package source is not a file or directory: {source_path}"
            )
        for child in sorted(source_path.rglob("*")):
            if not child.is_file():
                continue
            relative = PurePosixPath(child.relative_to(source_path).as_posix())
            _add_source_fingerprint(
                contents,
                source_path=child,
                target=entry.target / relative,
                exclude_globs=profile.exclude_globs,
            )
    return _logical_contents_fingerprint(contents)


def archive_member_sha256s(
    path: Path,
    targets: Sequence[PurePosixPath],
) -> dict[PurePosixPath, str]:
    """Hash selected immutable archive members."""
    expected = set(targets)
    result: dict[PurePosixPath, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = PurePosixPath(member.name)
            if target not in expected:
                continue
            if not member.isfile():
                raise ValueError(f"package archive member is not a file: {target}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read package archive member: {target}")
            result[target] = _stream_sha256(source)
    missing = expected.difference(result)
    if missing:
        raise FileNotFoundError(f"package archive omits immutable files: {missing}")
    return result


def validated_runtime_status(
    validation: Mapping[str, Any],
    *,
    planner_enabled_by_default: bool,
    planner_fingerprint: str,
    runtime_fingerprint: str,
) -> dict[str, Any]:
    """Check isolated startup used the expected planner and no fallback."""
    status = validation.get("runtime_status")
    if not isinstance(status, Mapping):
        raise ValueError("isolated package validation returned no runtime status")
    expected = {
        "policy_loaded": True,
        "planner_configured": planner_enabled_by_default,
        "planner_enabled": planner_enabled_by_default,
        "planner_fingerprint": (
            planner_fingerprint if planner_enabled_by_default else None
        ),
        "planner_runtime_fingerprint": (
            runtime_fingerprint if planner_enabled_by_default else None
        ),
        "used_random_fallback": False,
    }
    for name, expected_value in expected.items():
        if status.get(name) != expected_value:
            raise ValueError(f"isolated package validation differs at {name}")
    return expected


def validate_manifest_sidecar(path: Path, manifest_sha256: str) -> None:
    """Validate the immutable manifest checksum sidecar."""
    if not path.is_file():
        raise FileNotFoundError(f"package manifest SHA sidecar not found: {path}")
    expected = f"{manifest_sha256}  manifest.json\n"
    if path.read_text(encoding="utf-8") != expected:
        raise ValueError("package manifest SHA sidecar differs from manifest")


def write_text_atomic(path: Path, value: str) -> None:
    """Publish a small text file after flushing its complete contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_export_lineage(config: IntegratedPlannerProfileConfig) -> None:
    if file_sha256(config.checkpoint_path) != config.expected_checkpoint_sha256:
        raise ValueError("package source checkpoint fingerprint differs from config")
    checkpoint = torch.load(
        config.package_assets.deployment_checkpoint_path,
        map_location="cpu",
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("deployment checkpoint must be a mapping")
    export = checkpoint.get("export")
    if not isinstance(export, Mapping):
        raise ValueError("deployment checkpoint has no export identity")
    if export.get("storage_precision") != "fp16" or (
        export.get("compute_precision") != "fp32"
    ):
        raise ValueError("deployment checkpoint is not the FP16 runtime export")
    source_path = export.get("source_checkpoint")
    if not isinstance(source_path, str) or Path(source_path) != config.checkpoint_path:
        raise ValueError("deployment checkpoint source path differs from campaign")
    conditioning = export.get("deck_conditioning")
    if not isinstance(conditioning, Mapping) or (
        conditioning.get("source_checkpoint_sha256")
        != config.package_assets.source_checkpoint_sha256
    ):
        raise ValueError("deployment checkpoint source lineage differs from config")


def _resolve_source(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _add_source_fingerprint(
    contents: dict[PurePosixPath, str],
    *,
    source_path: Path,
    target: PurePosixPath,
    exclude_globs: Sequence[str],
) -> None:
    if _is_excluded(target, exclude_globs):
        return
    if target in contents:
        raise ValueError(f"duplicate package archive target: {target}")
    contents[target] = file_sha256(source_path)


def _is_excluded(path: PurePosixPath, patterns: Sequence[str]) -> bool:
    text = path.as_posix()
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        return True
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def _stream_sha256(source: Any) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _logical_contents_fingerprint(
    contents: Mapping[PurePosixPath, str],
) -> str:
    payload = json.dumps(
        [
            {"path": path.as_posix(), "sha256": contents[path]}
            for path in sorted(contents)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/planner-package-logical-contents/v1\x00" + payload
    ).hexdigest()


__all__ = [
    "archive_contents_fingerprint",
    "archive_member_sha256s",
    "load_profile",
    "package_manifest",
    "packaged_runtimes",
    "planner_runtime_payload",
    "prior_path",
    "profile_source_fingerprint",
    "required_files_fingerprint",
    "require_fresh_package_outputs",
    "validate_manifest_shared_identity",
    "validate_manifest_sidecar",
    "validate_profile_contract",
    "validate_shared_package_identity",
    "validated_runtime_status",
    "validated_staging_profile",
    "write_text_atomic",
]
