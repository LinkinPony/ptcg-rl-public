"""Load and verify immutable Kaggle deployment bundle manifests."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256, fingerprint_payload
from ptcg_rl.submission.checkpoint_assets import (
    inspect_runtime_checkpoint_deck_conditioning,
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
)

_REPLACED_RUNTIME_MEMBERS = frozenset(
    {
        "agent_checkpoint.pt",
        "deck.csv",
        "belief_prior.csv",
        "release_decision.json",
        "release_manifest.json",
    }
)


def load_release_bundle(path: Path) -> ReleaseBundleIdentityLike:
    """Load a manifest and verify every referenced deployment asset."""
    return _load_release_bundle(path, inspect_checkpoint_semantics=True)


def load_release_bundle_for_native_execution(
    path: Path,
) -> ReleaseBundleIdentityLike:
    """Verify an immutable release without reinterpreting its checkpoint.

    Historical deployment archives own their model-loading semantics.  This
    path still validates the typed manifest and every referenced artifact
    digest, while deliberately leaving checkpoint topology validation to the
    archive-native runtime that will execute the release.
    """
    return _load_release_bundle(path, inspect_checkpoint_semantics=False)


def _load_release_bundle(
    path: Path,
    *,
    inspect_checkpoint_semantics: bool,
) -> ReleaseBundleIdentityLike:
    """Load one manifest under the requested checkpoint-validation boundary."""
    resolved = _resolved_file(path, label="release bundle manifest")
    payload = _read_object(resolved)
    if payload.get("protocol") == "RELEASE-BUNDLE-v1":
        manifest: ReleaseBundleManifestLike = ReleaseBundleManifest.model_validate(
            payload
        )
    elif payload.get("protocol") == "RELEASE-BUNDLE-v2":
        manifest = ReleaseBundleManifestV2.model_validate(payload)
    elif payload.get("protocol") == "RELEASE-BUNDLE-v3":
        manifest = ReleaseBundleManifestV3.model_validate(payload)
    elif payload.get("protocol") == "RELEASE-BUNDLE-v4":
        manifest = ReleaseBundleManifestV4.model_validate(payload)
    elif payload.get("protocol") == "RELEASE-BUNDLE-v5":
        manifest = ReleaseBundleManifestV5.model_validate(payload)
    elif payload.get("protocol") == "CHECKPOINT-SELECTION-v1-WINNER-BUNDLE":
        manifest = _todo02_bundle_manifest(payload)
    else:
        raise ValueError(f"unsupported release bundle manifest protocol: {resolved}")
    _verify_bundle_assets(
        manifest,
        inspect_checkpoint_semantics=inspect_checkpoint_semantics,
    )
    identity_payload = manifest.model_dump(mode="python")
    identity_payload.update(
        {
            "source_manifest_path": Path(records.display_path(resolved)),
            "source_manifest_sha256": file_sha256(resolved),
        }
    )
    if isinstance(manifest, ReleaseBundleManifestV5):
        return ReleaseBundleIdentityV5.model_validate(identity_payload)
    if isinstance(manifest, ReleaseBundleManifestV4):
        return ReleaseBundleIdentityV4.model_validate(identity_payload)
    if isinstance(manifest, ReleaseBundleManifestV3):
        return ReleaseBundleIdentityV3.model_validate(identity_payload)
    if isinstance(manifest, ReleaseBundleManifestV2):
        return ReleaseBundleIdentityV2.model_validate(identity_payload)
    return ReleaseBundleIdentity.model_validate(identity_payload)


def runtime_config_sha256(bundle_agent: Mapping[str, Any]) -> str:
    """Fingerprint deployment runtime semantics, excluding bundle assets."""
    return fingerprint_payload(
        {
            "kind": bundle_agent.get("kind"),
            "act_time": bundle_agent.get("act_time"),
        }
    )


def submission_fingerprint(bundle: ReleaseBundleIdentityLike) -> str:
    """Fingerprint effective package semantics without labels or tar metadata."""
    runtime_archive = _resolved_file(
        bundle.runtime_archive_path,
        label="release runtime archive",
    )
    payload: dict[str, Any] = {
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "deck_sha256": bundle.deck_sha256,
        "belief_sha256": bundle.belief_sha256,
        "runtime_config_sha256": bundle.runtime_config_sha256,
        "runtime_content_sha256": _runtime_content_sha256(runtime_archive),
    }
    if isinstance(
        bundle,
        (
            ReleaseBundleIdentityV2,
            ReleaseBundleIdentityV3,
            ReleaseBundleIdentityV4,
            ReleaseBundleIdentityV5,
        ),
    ):
        payload["deck_conditioning"] = bundle.deck_conditioning.model_dump(
            mode="json"
        )
    return fingerprint_payload(payload)


def _runtime_content_sha256(path: Path) -> str:
    """Hash runtime member paths and bytes, ignoring archive metadata."""
    hasher = hashlib.sha256()
    with tarfile.open(path, "r:*") as archive:
        members = sorted(archive.getmembers(), key=lambda member: member.name)
        for member in members:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"release runtime archive contains a non-file: {member.name}"
                )
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"release runtime archive contains an unsafe path: {member.name}"
                )
            normalized = member_path.as_posix()
            if normalized in _REPLACED_RUNTIME_MEMBERS:
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"cannot read release runtime member: {member.name}")
            encoded_path = normalized.encode("utf-8")
            hasher.update(len(encoded_path).to_bytes(8, "big"))
            hasher.update(encoded_path)
            while chunk := file_obj.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def _todo02_bundle_manifest(payload: Mapping[str, Any]) -> ReleaseBundleManifest:
    bundle = _mapping(payload.get("bundle"))
    agent = _mapping(bundle.get("agent"))
    checkpoint_asset = _mapping(payload.get("checkpoint_asset"))
    submission_protocol = _mapping(checkpoint_asset.get("submission_protocol"))
    checkpoint_path = checkpoint_asset.get("asset_path") or agent.get(
        "checkpoint_path"
    )
    checkpoint_sha256 = checkpoint_asset.get("asset_sha256")
    if checkpoint_path is None or checkpoint_sha256 is None:
        raise ValueError("winner manifest is missing checkpoint identity")
    deck_path = bundle.get("deck_path")
    if deck_path is None:
        raise ValueError("winner manifest is missing deck_path")
    belief_path = agent.get("belief_summary_path")
    return ReleaseBundleManifest(
        bundle_id=str(payload["winner_bundle_id"]),
        checkpoint_tag=str(payload["checkpoint_tag"]),
        checkpoint_path=Path(str(checkpoint_path)),
        checkpoint_sha256=str(checkpoint_sha256),
        deck_path=Path(str(deck_path)),
        deck_sha256=str(payload["deck_sha256"]),
        belief_path=Path(str(belief_path)) if belief_path is not None else None,
        belief_sha256=(
            str(payload["belief_sha256"])
            if payload.get("belief_sha256") is not None
            else None
        ),
        runtime_archive_path=Path(str(submission_protocol["archive_path"])),
        runtime_sha256=str(submission_protocol["archive_sha256"]),
        runtime_config_sha256=runtime_config_sha256(agent),
        bundle_fingerprint=str(payload["candidate_bundle_fp"]),
        storage_precision=cast(
            Literal["fp16", "fp32"],
            checkpoint_asset.get("storage_precision", "fp16"),
        ),
        compute_precision=cast(
            Literal["fp32"],
            checkpoint_asset.get("compute_precision", "fp32"),
        ),
    )


def _verify_bundle_assets(
    manifest: ReleaseBundleManifestLike,
    *,
    inspect_checkpoint_semantics: bool,
) -> None:
    checkpoint = _resolved_file(manifest.checkpoint_path, label="release checkpoint")
    deck = _resolved_file(manifest.deck_path, label="release deck")
    if file_sha256(checkpoint) != manifest.checkpoint_sha256:
        raise ValueError("release checkpoint SHA256 does not match its manifest")
    if file_sha256(deck) != manifest.deck_sha256:
        raise ValueError("release deck SHA256 does not match its manifest")
    if manifest.belief_path is not None:
        belief = _resolved_file(manifest.belief_path, label="release belief")
        if file_sha256(belief) != manifest.belief_sha256:
            raise ValueError("release belief SHA256 does not match its manifest")
    runtime_archive = _resolved_file(
        manifest.runtime_archive_path,
        label="release runtime archive",
    )
    if file_sha256(runtime_archive) != manifest.runtime_sha256:
        raise ValueError("release runtime SHA256 does not match its manifest")
    if inspect_checkpoint_semantics and isinstance(
        manifest,
        (
            ReleaseBundleManifestV2,
            ReleaseBundleManifestV3,
            ReleaseBundleManifestV4,
            ReleaseBundleManifestV5,
        ),
    ):
        checkpoint_identity = inspect_runtime_checkpoint_deck_conditioning(
            checkpoint,
            deck_path=deck,
            strict_runtime_load=True,
        )
        if (
            checkpoint_identity
            != manifest.deck_conditioning.model_dump(mode="python")
        ):
            raise ValueError(
                "release deck-conditioning manifest does not match checkpoint"
            )


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return raw


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "load_release_bundle",
    "load_release_bundle_for_native_execution",
    "runtime_config_sha256",
    "submission_fingerprint",
]
