"""Immutable deployment assets used by the Kaggle submission protocol."""

from ptcg_rl.submission.release_assets.manifest import (
    load_release_bundle,
    load_release_bundle_for_native_execution,
    runtime_config_sha256,
    submission_fingerprint,
)
from ptcg_rl.submission.release_assets.models import (
    ReleaseBundleIdentity,
    ReleaseBundleIdentityLike,
    ReleaseBundleIdentityV2,
    ReleaseBundleIdentityV3,
    ReleaseBundleIdentityV4,
    ReleaseBundleIdentityV5,
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

__all__ = [
    "ReleaseBundleIdentity",
    "ReleaseBundleIdentityLike",
    "ReleaseBundleIdentityV2",
    "ReleaseBundleIdentityV3",
    "ReleaseBundleIdentityV4",
    "ReleaseBundleIdentityV5",
    "ReleaseBundleManifest",
    "ReleaseBundleManifestLike",
    "ReleaseBundleManifestV2",
    "ReleaseBundleManifestV3",
    "ReleaseBundleManifestV4",
    "ReleaseBundleManifestV5",
    "ReleaseDeckCompositionalManifest",
    "ReleaseDeckConditioningManifest",
    "ReleaseDeckDensePrivateManifest",
    "ReleaseDeckLoRAMergeManifest",
    "load_release_bundle",
    "load_release_bundle_for_native_execution",
    "runtime_config_sha256",
    "submission_fingerprint",
]
