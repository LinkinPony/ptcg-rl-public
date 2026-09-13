"""Immutable package layout and serving model identities for planner profiling."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PlannerPackageAssetId = Literal["control", "k4", "k8", "k12"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlannerProfilePackageAssetsConfig(BaseModel):
    """Immutable FP16 package inputs and point-specific archive layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    manifest_path: Path
    manifest_sha256_path: Path
    expected_manifest_sha256: str | None = None
    deployment_checkpoint_path: Path
    expected_deployment_checkpoint_sha256: str
    expected_deployment_model_fingerprint: str
    source_checkpoint_sha256: str
    submission_profiles: Mapping[PlannerPackageAssetId, str]
    checkpoint_target: PurePosixPath = PurePosixPath("agent_checkpoint.pt")
    native_library_target: PurePosixPath = PurePosixPath(
        "src/native/cg_probe/libcg_probe.so"
    )
    belief_prior_target: PurePosixPath = PurePosixPath("belief_prior.csv")
    planner_runtime_target: PurePosixPath = PurePosixPath("planner_runtime.json")
    archive_filename: str = "submission.tar.gz"

    @field_validator(
        "expected_manifest_sha256",
        "expected_deployment_checkpoint_sha256",
        "expected_deployment_model_fingerprint",
        "source_checkpoint_sha256",
    )
    @classmethod
    def package_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise ValueError("package asset fingerprints must be lowercase SHA-256")
        return value

    @field_validator("submission_profiles")
    @classmethod
    def exact_submission_profiles(
        cls,
        value: Mapping[PlannerPackageAssetId, str],
    ) -> Mapping[PlannerPackageAssetId, str]:
        if set(value) != {"control", "k4", "k8", "k12"}:
            raise ValueError(
                "package assets require exactly control, k4, k8, and k12 profiles"
            )
        canonical = {key: profile.strip() for key, profile in value.items()}
        if any(
            not profile or "latest" in profile.lower() for profile in canonical.values()
        ):
            raise ValueError("package submission profiles must be immutable")
        if len(set(canonical.values())) != len(canonical):
            raise ValueError("package submission profiles must be unique")
        return canonical

    @field_validator(
        "checkpoint_target",
        "native_library_target",
        "belief_prior_target",
        "planner_runtime_target",
    )
    @classmethod
    def relative_package_target(cls, value: PurePosixPath) -> PurePosixPath:
        if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
            raise ValueError("package asset targets must be safe relative paths")
        return value

    @model_validator(mode="after")
    def exact_layout(self) -> Self:
        if self.manifest_path != self.root / "manifest.json":
            raise ValueError("package manifest must be rooted at manifest.json")
        if self.manifest_sha256_path != self.root / "manifest.sha256":
            raise ValueError("package manifest SHA must be rooted at manifest.sha256")
        if self.deployment_checkpoint_path != (
            self.root / "deployment" / "agent_checkpoint.pt"
        ):
            raise ValueError("package deployment checkpoint path differs from layout")
        expected_targets = {
            "checkpoint": (
                self.checkpoint_target,
                PurePosixPath("agent_checkpoint.pt"),
            ),
            "native": (
                self.native_library_target,
                PurePosixPath("src/native/cg_probe/libcg_probe.so"),
            ),
            "prior": (self.belief_prior_target, PurePosixPath("belief_prior.csv")),
            "runtime": (
                self.planner_runtime_target,
                PurePosixPath("planner_runtime.json"),
            ),
        }
        changed = [
            name
            for name, (actual, expected) in expected_targets.items()
            if actual != expected
        ]
        if changed:
            raise ValueError(f"package asset targets differ from deployment: {changed}")
        if self.archive_filename != "submission.tar.gz":
            raise ValueError("package archive filename must be immutable")
        for path in (
            self.root,
            self.manifest_path,
            self.manifest_sha256_path,
            self.deployment_checkpoint_path,
        ):
            if any("latest" in part.lower() for part in path.parts):
                raise ValueError("package asset paths cannot contain latest aliases")
        return self

    def runtime_path(self, asset_id: PlannerPackageAssetId) -> Path:
        """Return the immutable point-specific planner spec path."""
        return self.root / asset_id / self.planner_runtime_target

    def archive_path(self, asset_id: PlannerPackageAssetId) -> Path:
        """Return the immutable point-specific submission archive path."""
        return self.root / asset_id / self.archive_filename


class PlannerProfileModelIdentity(BaseModel):
    """Checkpoint/model identity expected for one profile point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_path: Path
    checkpoint_sha256: str
    model_fingerprint: str
    source_checkpoint_sha256: str
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)


__all__ = [
    "PlannerPackageAssetId",
    "PlannerProfileModelIdentity",
    "PlannerProfilePackageAssetsConfig",
]
