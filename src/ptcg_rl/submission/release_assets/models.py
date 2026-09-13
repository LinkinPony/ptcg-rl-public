"""Typed identities for immutable Kaggle deployment bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_HEX_DIGITS = frozenset("0123456789abcdef")


class _ReleaseBundleFields(BaseModel):
    """Fields shared by release protocols without changing v1 identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    checkpoint_tag: str
    checkpoint_path: Path
    checkpoint_sha256: str
    deck_path: Path
    deck_sha256: str
    belief_path: Path | None = None
    belief_sha256: str | None = None
    runtime_archive_path: Path
    runtime_sha256: str
    runtime_config_sha256: str
    bundle_fingerprint: str
    storage_precision: Literal["fp16", "fp32"] = "fp16"
    compute_precision: Literal["fp32"] = "fp32"

    @field_validator("bundle_id", "checkpoint_tag")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Reject blank bundle identifiers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("release bundle identity fields must be non-empty")
        return normalized

    @field_validator(
        "checkpoint_sha256",
        "deck_sha256",
        "belief_sha256",
        "runtime_sha256",
        "runtime_config_sha256",
        "bundle_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str | None) -> str | None:
        """Normalize and validate SHA256-shaped identities."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("release bundle hashes must be 64 hex characters")
        return normalized

    @model_validator(mode="after")
    def paired_belief_fields(self) -> Self:
        """Require belief path and digest to be present or absent together."""
        if (self.belief_path is None) != (self.belief_sha256 is None):
            raise ValueError("belief_path and belief_sha256 must be paired")
        return self


class ReleaseBundleManifest(_ReleaseBundleFields):
    """Portable identity of one legacy-compatible deployment bundle."""

    protocol: Literal["RELEASE-BUNDLE-v1"] = "RELEASE-BUNDLE-v1"


class ReleaseBundleIdentity(ReleaseBundleManifest):
    """Release bundle plus the immutable manifest which declared it."""

    source_manifest_path: Path
    source_manifest_sha256: str

    @field_validator("source_manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        """Validate the source-manifest digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("source manifest SHA256 must be 64 hex characters")
        return normalized


class ReleaseDeckConditioningManifest(BaseModel):
    """Exact deck/private-module binding for a specialized release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int
    canonical_deck_signature: str
    deck_digest: str
    selected_private_profile_module_key: str
    source_registry_sha256: str
    source_checkpoint_sha256: str
    packaged_private_profile_count: int
    pruning_applied: bool = False

    @field_validator("architecture_version", "packaged_private_profile_count")
    @classmethod
    def positive_values(cls, value: int) -> int:
        """Require a concrete architecture and at least one packaged profile."""
        if value <= 0:
            raise ValueError("deck-conditioned release values must be positive")
        return value

    @field_validator("canonical_deck_signature", "selected_private_profile_module_key")
    @classmethod
    def nonempty_deck_identity(cls, value: str) -> str:
        """Reject blank exact-deck semantic identities."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("deck-conditioned release identity cannot be blank")
        return normalized

    @field_validator(
        "deck_digest",
        "source_registry_sha256",
        "source_checkpoint_sha256",
    )
    @classmethod
    def valid_deck_sha256(cls, value: str) -> str:
        """Validate all deck-conditioning SHA256 identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("deck-conditioned release hashes must be SHA256")
        return normalized


class ReleaseDeckLoRAMergeManifest(ReleaseDeckConditioningManifest):
    """Exact fixed-deck identity for a checkpoint with merged LoRA weights."""

    selected_expert_id: str
    lora_merged: Literal[True] = True
    merged_target_count: int
    lora_schema_sha256: str
    deck_context_folded: Literal[True] = True
    card_encoder_alias_pruned: Literal[True] = True
    residual_route_retained: Literal[True] = True

    @field_validator("selected_expert_id", "lora_schema_sha256")
    @classmethod
    def valid_lora_sha256(cls, value: str) -> str:
        """Validate stable expert and merge-schema identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("LoRA release hashes must be SHA256")
        return normalized

    @field_validator("merged_target_count")
    @classmethod
    def positive_target_count(cls, value: int) -> int:
        """Require at least one merged LoRA target."""
        if value <= 0:
            raise ValueError("merged target count must be positive")
        return value

    @model_validator(mode="after")
    def merged_release_is_pruned(self) -> Self:
        """A fixed-deck merge must retain exactly one pruned route."""
        if not self.pruning_applied or self.packaged_private_profile_count != 1:
            raise ValueError("merged LoRA release must contain one pruned route")
        return self


class ReleaseDeckDensePrivateManifest(ReleaseDeckConditioningManifest):
    """Exact fixed-deck identity for one dense private strategy release."""

    architecture_version: Literal[3]
    selected_strategy_id: str
    fixed_strategy: Literal[True] = True
    strategy_schema_sha256: str
    generic_upper_pruned: Literal[True] = True
    generic_policy_decision_pruned: Literal[True] = True
    generic_value_heads_pruned: Literal[True] = True
    lora_runtime_absent: Literal[True] = True
    deck_context_folded: Literal[True] = True
    card_encoder_alias_pruned: Literal[True] = True

    @field_validator("selected_strategy_id", "strategy_schema_sha256")
    @classmethod
    def valid_strategy_sha256(cls, value: str) -> str:
        """Validate immutable strategy and parameter-schema identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("dense-private release hashes must be SHA256")
        return normalized

    @model_validator(mode="after")
    def fixed_release_is_pruned(self) -> Self:
        """A fixed strategy must retain exactly one pruned exact-deck route."""
        if not self.pruning_applied or self.packaged_private_profile_count != 1:
            raise ValueError("fixed dense-private release must contain one route")
        return self


class ReleaseDeckCompositionalManifest(ReleaseDeckConditioningManifest):
    """Exact fixed-deck identity for one materialized DCCR-v4 strategy."""

    architecture_version: Literal[4]
    selected_strategy_id: str
    fixed_strategy: Literal[True] = True
    fixed_compositional: Literal[True] = True
    strategy_schema_sha256: str
    merged_projection_count: int
    fixed_film_site_count: int
    dynamic_router_pruned: Literal[True] = True
    shared_basis_pruned: Literal[True] = True
    non_selected_capsules_pruned: Literal[True] = True
    lora_runtime_absent: Literal[True] = True
    deck_context_folded: Literal[True] = True
    card_encoder_alias_pruned: Literal[True] = True

    @field_validator("selected_strategy_id", "strategy_schema_sha256")
    @classmethod
    def valid_strategy_sha256(cls, value: str) -> str:
        """Validate immutable strategy and parameter-schema identities."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("compositional release hashes must be SHA256")
        return normalized

    @field_validator("merged_projection_count", "fixed_film_site_count")
    @classmethod
    def positive_materialized_count(cls, value: int) -> int:
        """Require every declared materialization inventory to be non-empty."""
        if value <= 0:
            raise ValueError("compositional materialization counts must be positive")
        return value

    @model_validator(mode="after")
    def fixed_release_is_pruned(self) -> Self:
        """A fixed compositional release must retain one exact capsule route."""
        if not self.pruning_applied or self.packaged_private_profile_count != 1:
            raise ValueError("fixed compositional release must contain one route")
        return self


class ReleaseBundleManifestV2(_ReleaseBundleFields):
    """Release bundle carrying an exact deck-conditioned semantic binding."""

    protocol: Literal["RELEASE-BUNDLE-v2"] = "RELEASE-BUNDLE-v2"
    deck_conditioning: ReleaseDeckConditioningManifest


class ReleaseBundleIdentityV2(ReleaseBundleManifestV2):
    """Deck-conditioned release plus its immutable source manifest identity."""

    source_manifest_path: Path
    source_manifest_sha256: str

    @field_validator("source_manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        """Validate the source-manifest digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("source manifest SHA256 must be 64 hex characters")
        return normalized


class ReleaseBundleManifestV3(_ReleaseBundleFields):
    """Release bundle carrying a fixed-deck merged-LoRA semantic binding."""

    protocol: Literal["RELEASE-BUNDLE-v3"] = "RELEASE-BUNDLE-v3"
    deck_conditioning: ReleaseDeckLoRAMergeManifest


class ReleaseBundleIdentityV3(ReleaseBundleManifestV3):
    """Merged-LoRA release plus its immutable source manifest identity."""

    source_manifest_path: Path
    source_manifest_sha256: str

    @field_validator("source_manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        """Validate the source-manifest digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("source manifest SHA256 must be 64 hex characters")
        return normalized


class ReleaseBundleManifestV4(_ReleaseBundleFields):
    """Release bundle carrying one compact fixed dense-private strategy."""

    protocol: Literal["RELEASE-BUNDLE-v4"] = "RELEASE-BUNDLE-v4"
    deck_conditioning: ReleaseDeckDensePrivateManifest


class ReleaseBundleIdentityV4(ReleaseBundleManifestV4):
    """Fixed dense-private release plus its immutable manifest identity."""

    source_manifest_path: Path
    source_manifest_sha256: str

    @field_validator("source_manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        """Validate the source-manifest digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("source manifest SHA256 must be 64 hex characters")
        return normalized


class ReleaseBundleManifestV5(_ReleaseBundleFields):
    """Release bundle carrying one materialized compositional strategy."""

    protocol: Literal["RELEASE-BUNDLE-v5"] = "RELEASE-BUNDLE-v5"
    deck_conditioning: ReleaseDeckCompositionalManifest


class ReleaseBundleIdentityV5(ReleaseBundleManifestV5):
    """Fixed compositional release plus its immutable manifest identity."""

    source_manifest_path: Path
    source_manifest_sha256: str

    @field_validator("source_manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        """Validate the source-manifest digest."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("source manifest SHA256 must be 64 hex characters")
        return normalized


ReleaseBundleManifestLike = (
    ReleaseBundleManifest
    | ReleaseBundleManifestV2
    | ReleaseBundleManifestV3
    | ReleaseBundleManifestV4
    | ReleaseBundleManifestV5
)
ReleaseBundleIdentityLike = (
    ReleaseBundleIdentity
    | ReleaseBundleIdentityV2
    | ReleaseBundleIdentityV3
    | ReleaseBundleIdentityV4
    | ReleaseBundleIdentityV5
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
]
