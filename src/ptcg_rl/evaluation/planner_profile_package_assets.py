"""Build and read-only validation of immutable planner package assets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from ptcg_rl.agent.packaged_planner import PackagedPlannerConfig
from ptcg_rl.evaluation.consequence_parity_artifact import (
    file_sha256,
    write_json_atomic,
)
from ptcg_rl.evaluation.planner_profile_config import IntegratedPlannerProfileConfig
from ptcg_rl.evaluation.planner_profile_package_models import (
    ASSET_IDS,
    PlannerPackageAssetManifestEntry,
    PlannerPackageAssetsManifest,
    PlannerProfilePackageAssetsValidation,
)
from ptcg_rl.evaluation.planner_profile_package_support import (
    archive_contents_fingerprint,
    archive_member_sha256s,
    load_profile,
    package_manifest,
    packaged_runtimes,
    planner_runtime_payload,
    profile_source_fingerprint,
    require_fresh_package_outputs,
    required_files_fingerprint,
    validate_manifest_shared_identity,
    validate_manifest_sidecar,
    validate_profile_contract,
    validate_shared_package_identity,
    validated_runtime_status,
    validated_staging_profile,
    write_text_atomic,
)
from ptcg_rl.submission.protocol import (
    KAGGLE_SUBMISSION_MAX_BYTES,
    build_submission_archive,
    validate_submission_archive,
)


def build_planner_profile_package_assets(
    config: IntegratedPlannerProfileConfig,
) -> PlannerProfilePackageAssetsValidation:
    """Build and atomically publish the four validated package assets."""
    package = config.package_assets
    require_fresh_package_outputs(config)
    shared = validate_shared_package_identity(config)
    runtimes = packaged_runtimes(config)
    package.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".package-assets-", dir=package.root))
    manifest_sha256: str | None = None
    try:
        entries = {}
        for asset_id in ASSET_IDS:
            runtime_id, runtime = runtimes[asset_id]
            workload = runtime.packaged
            if workload is None:
                raise AssertionError("package matrix contains a non-packaged runtime")
            staged_dir = staging / asset_id
            staged_runtime = staged_dir / package.planner_runtime_target
            staged_archive = staged_dir / package.archive_filename
            payload, planner_fingerprint, runtime_fingerprint = planner_runtime_payload(
                config, runtime, shared=shared
            )
            write_json_atomic(staged_runtime, payload)
            PackagedPlannerConfig.model_validate_json(
                staged_runtime.read_text(encoding="utf-8")
            )
            profile_name = package.submission_profiles[asset_id]
            profile = validated_staging_profile(
                config,
                asset_id=asset_id,
                profile=load_profile(profile_name),
                staged_runtime=staged_runtime,
            )
            build_submission_archive(profile, staged_archive)
            validation = validate_submission_archive(staged_archive, profile)
            isolated_validation = validated_runtime_status(
                validation,
                planner_enabled_by_default=workload.planner_enabled_by_default,
                planner_fingerprint=planner_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
            archive_contents = archive_contents_fingerprint(staged_archive)
            if profile_source_fingerprint(profile) != archive_contents:
                raise ValueError(
                    "built archive logical contents differ from profile sources"
                )
            member_sha256s = archive_member_sha256s(
                staged_archive,
                (
                    package.checkpoint_target,
                    package.belief_prior_target,
                    package.native_library_target,
                    package.planner_runtime_target,
                    PurePosixPath("deck.csv"),
                ),
            )
            expected_member_sha256s = {
                package.checkpoint_target: shared.checkpoint_sha256,
                package.belief_prior_target: shared.belief_prior_sha256,
                package.native_library_target: shared.native_sha256,
                package.planner_runtime_target: file_sha256(staged_runtime),
            }
            for target, expected_sha256 in expected_member_sha256s.items():
                if member_sha256s[target] != expected_sha256:
                    raise ValueError(f"packaged archive changes immutable {target}")
            entries[asset_id] = PlannerPackageAssetManifestEntry(
                asset_id=asset_id,
                runtime_id=runtime_id,
                submission_profile=profile.name,
                planner_enabled_by_default=workload.planner_enabled_by_default,
                planner_runtime_path=package.runtime_path(asset_id),
                planner_runtime_sha256=file_sha256(staged_runtime),
                submission_archive_path=package.archive_path(asset_id),
                submission_archive_sha256=file_sha256(staged_archive),
                submission_archive_bytes=staged_archive.stat().st_size,
                archive_contents_fingerprint=archive_contents,
                required_files=tuple(
                    sorted(str(path) for path in profile.required_files)
                ),
                required_files_fingerprint=required_files_fingerprint(
                    profile.required_files
                ),
                deck_sha256=member_sha256s[PurePosixPath("deck.csv")],
                planner_fingerprint=planner_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                isolated_validation=isolated_validation,
            )

        manifest = package_manifest(config, shared=shared, entries=entries)
        staged_manifest = staging / "manifest.json"
        staged_sha256 = staging / "manifest.sha256"
        write_json_atomic(staged_manifest, manifest.model_dump(mode="json"))
        manifest_sha256 = file_sha256(staged_manifest)
        configured_sha256 = package.expected_manifest_sha256
        if configured_sha256 is not None and manifest_sha256 != configured_sha256:
            raise ValueError("built package manifest differs from configured SHA-256")
        write_text_atomic(
            staged_sha256,
            f"{manifest_sha256}  manifest.json\n",
        )
        for asset_id in ASSET_IDS:
            os.replace(staging / asset_id, package.root / asset_id)
        os.replace(staged_manifest, package.manifest_path)
        os.replace(staged_sha256, package.manifest_sha256_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if manifest_sha256 is None:
        raise RuntimeError("package build did not produce a manifest")
    return validate_planner_profile_package_assets(
        config,
        expected_manifest_sha256=(package.expected_manifest_sha256 or manifest_sha256),
    )


def validate_planner_profile_package_assets(
    config: IntegratedPlannerProfileConfig,
    *,
    expected_manifest_sha256: str | None = None,
) -> PlannerProfilePackageAssetsValidation:
    """Read and fingerprint-check pre-existing specs and archives without writes."""
    package = config.package_assets
    expected = expected_manifest_sha256 or package.expected_manifest_sha256
    if expected is None:
        raise ValueError("package manifest SHA-256 is not configured")
    if not package.manifest_path.is_file():
        raise FileNotFoundError(f"package manifest not found: {package.manifest_path}")
    manifest_sha256 = file_sha256(package.manifest_path)
    if manifest_sha256 != expected:
        raise ValueError("package manifest fingerprint differs from config")
    validate_manifest_sidecar(package.manifest_sha256_path, manifest_sha256)
    manifest = PlannerPackageAssetsManifest.model_validate_json(
        package.manifest_path.read_text(encoding="utf-8")
    )
    shared = validate_shared_package_identity(config)
    validate_manifest_shared_identity(config, manifest, shared=shared)
    runtimes = packaged_runtimes(config)
    if set(manifest.assets) != set(ASSET_IDS):
        raise ValueError("package manifest does not contain exactly four assets")
    runtime_assets = {}
    for asset_id in ASSET_IDS:
        runtime_id, runtime = runtimes[asset_id]
        workload = runtime.packaged
        if workload is None:
            raise AssertionError("package matrix contains a non-packaged runtime")
        entry = manifest.assets[asset_id]
        if entry.asset_id != asset_id or entry.runtime_id != runtime_id:
            raise ValueError("package manifest runtime matrix differs from config")
        if entry.planner_enabled_by_default != workload.planner_enabled_by_default:
            raise ValueError("package manifest planner default differs from config")
        profile = load_profile(package.submission_profiles[asset_id])
        validate_profile_contract(config, asset_id=asset_id, profile=profile)
        if entry.submission_profile != profile.name:
            raise ValueError("package manifest submission profile differs from config")
        runtime_path = package.runtime_path(asset_id)
        archive_path = package.archive_path(asset_id)
        if entry.planner_runtime_path != runtime_path or (
            entry.submission_archive_path != archive_path
        ):
            raise ValueError("package manifest path layout differs from config")
        if file_sha256(runtime_path) != entry.planner_runtime_sha256:
            raise ValueError("packaged planner runtime fingerprint changed")
        if file_sha256(archive_path) != entry.submission_archive_sha256:
            raise ValueError("packaged submission archive fingerprint changed")
        if archive_path.stat().st_size != entry.submission_archive_bytes:
            raise ValueError("packaged submission archive size changed")
        if entry.submission_archive_bytes > KAGGLE_SUBMISSION_MAX_BYTES:
            raise ValueError("packaged submission archive exceeds Kaggle limit")
        archive_contents = archive_contents_fingerprint(archive_path)
        if archive_contents != entry.archive_contents_fingerprint:
            raise ValueError("packaged archive logical contents changed")
        if profile_source_fingerprint(profile) != archive_contents:
            raise ValueError(
                "packaged archive is stale relative to current profile sources"
            )
        if tuple(sorted(entry.required_files)) != tuple(
            sorted(str(path) for path in profile.required_files)
        ):
            raise ValueError("package manifest required files differ from profile")
        if entry.required_files_fingerprint != required_files_fingerprint(
            profile.required_files
        ):
            raise ValueError("package required-file fingerprint changed")
        payload, planner_fingerprint, runtime_fingerprint = planner_runtime_payload(
            config,
            runtime,
            shared=shared,
        )
        persisted = json.loads(runtime_path.read_text(encoding="utf-8"))
        if persisted != payload:
            raise ValueError("packaged planner runtime semantics differ from config")
        PackagedPlannerConfig.model_validate(persisted)
        if entry.planner_fingerprint != planner_fingerprint or (
            entry.runtime_fingerprint != runtime_fingerprint
        ):
            raise ValueError("package composite identity differs from config")
        isolated_validation = validated_runtime_status(
            {"runtime_status": entry.isolated_validation},
            planner_enabled_by_default=workload.planner_enabled_by_default,
            planner_fingerprint=planner_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )
        if dict(entry.isolated_validation) != isolated_validation:
            raise ValueError("package isolated validation evidence differs from config")
        member_sha256s = archive_member_sha256s(
            archive_path,
            (
                package.checkpoint_target,
                package.belief_prior_target,
                package.native_library_target,
                package.planner_runtime_target,
                PurePosixPath("deck.csv"),
            ),
        )
        expected_members = {
            package.checkpoint_target: shared.checkpoint_sha256,
            package.belief_prior_target: shared.belief_prior_sha256,
            package.native_library_target: shared.native_sha256,
            package.planner_runtime_target: entry.planner_runtime_sha256,
            PurePosixPath("deck.csv"): entry.deck_sha256,
        }
        if member_sha256s != expected_members:
            raise ValueError("package archive member identity differs from manifest")
        runtime_assets[runtime_id] = entry
    return PlannerProfilePackageAssetsValidation(
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        runtime_assets=runtime_assets,
    )


__all__ = [
    "PlannerPackageAssetManifestEntry",
    "PlannerPackageAssetsManifest",
    "PlannerProfilePackageAssetsValidation",
    "build_planner_profile_package_assets",
    "validate_planner_profile_package_assets",
]
