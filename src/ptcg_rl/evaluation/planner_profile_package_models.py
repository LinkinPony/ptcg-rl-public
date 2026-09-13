"""Manifest models for immutable integrated-profile package assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ptcg_rl.evaluation.planner_profile_package_config import PlannerPackageAssetId

ASSET_IDS: tuple[PlannerPackageAssetId, ...] = ("control", "k4", "k8", "k12")
REQUIRED_FILES = frozenset(
    {
        PurePosixPath("main.py"),
        PurePosixPath("ptcg_rl/agent/runtime.py"),
        PurePosixPath("cg/api.py"),
        PurePosixPath("deck.csv"),
        PurePosixPath("agent_checkpoint.pt"),
        PurePosixPath("belief_prior.csv"),
        PurePosixPath("outputs/cards/static_features/card_static_features.npy"),
        PurePosixPath("src/native/cg_probe/libcg_probe.so"),
        PurePosixPath("planner_runtime.json"),
    }
)


class PlannerPackageAssetManifestEntry(BaseModel):
    """One immutable point-specific spec and validated submission archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: PlannerPackageAssetId
    runtime_id: str
    submission_profile: str
    planner_enabled_by_default: bool
    planner_runtime_path: Path
    planner_runtime_sha256: str
    submission_archive_path: Path
    submission_archive_sha256: str
    submission_archive_bytes: int
    archive_contents_fingerprint: str
    required_files: tuple[str, ...]
    required_files_fingerprint: str
    deck_sha256: str
    planner_fingerprint: str
    runtime_fingerprint: str
    isolated_validation: Mapping[str, Any]


class PlannerPackageAssetsManifest(BaseModel):
    """Small commit record binding all deployment files and identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    campaign_id: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    deployment_checkpoint_path: Path
    deployment_checkpoint_sha256: str
    deployment_model_fingerprint: str
    policy_version: int
    proposal_version: int
    native_library_path: Path
    native_library_sha256: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str
    belief_prior_path: Path
    belief_prior_sha256: str
    belief_runtime_fingerprint: str
    assets: Mapping[PlannerPackageAssetId, PlannerPackageAssetManifestEntry]


@dataclass(frozen=True, slots=True)
class PlannerProfilePackageAssetsValidation:
    """Read-only validation result used by campaign and packaged backends."""

    manifest_sha256: str
    manifest: PlannerPackageAssetsManifest
    runtime_assets: Mapping[str, PlannerPackageAssetManifestEntry]

    def asset_for_runtime(self, runtime_id: str) -> PlannerPackageAssetManifestEntry:
        """Return one fingerprint-bound package, rejecting unknown runtimes."""
        try:
            return self.runtime_assets[runtime_id]
        except KeyError as exc:
            raise KeyError(f"no package asset for runtime: {runtime_id}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "deployment_checkpoint_sha256": (
                self.manifest.deployment_checkpoint_sha256
            ),
            "deployment_model_fingerprint": (
                self.manifest.deployment_model_fingerprint
            ),
            "assets": {
                asset_id: {
                    "runtime_id": entry.runtime_id,
                    "planner_enabled_by_default": (entry.planner_enabled_by_default),
                    "planner_runtime_sha256": entry.planner_runtime_sha256,
                    "submission_archive_sha256": entry.submission_archive_sha256,
                    "submission_archive_bytes": entry.submission_archive_bytes,
                    "archive_contents_fingerprint": (
                        entry.archive_contents_fingerprint
                    ),
                    "planner_fingerprint": entry.planner_fingerprint,
                    "runtime_fingerprint": entry.runtime_fingerprint,
                }
                for asset_id, entry in sorted(self.manifest.assets.items())
            },
        }


@dataclass(frozen=True, slots=True)
class SharedPackageIdentity:
    """Shared model, engine, and belief identities for all package points."""

    checkpoint_sha256: str
    model_fingerprint: str
    native_sha256: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str
    belief_prior_path: Path
    belief_prior_sha256: str
    belief_runtime_fingerprint: str


__all__ = [
    "ASSET_IDS",
    "REQUIRED_FILES",
    "PlannerPackageAssetManifestEntry",
    "PlannerPackageAssetsManifest",
    "PlannerProfilePackageAssetsValidation",
    "SharedPackageIdentity",
]
