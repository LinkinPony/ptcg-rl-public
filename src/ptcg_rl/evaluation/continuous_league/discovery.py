"""Content-identity discovery for decks, checkpoint pairs, and releases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.continuous_league.ledger import LeagueLedger
from ptcg_rl.evaluation.continuous_league.models import ContinuousLeagueConfig
from ptcg_rl.model.simple_stateless.network import (
    materialize_simple_stateless_checkpoint_model,
)
from ptcg_rl.rl.stateless_checkpoint import load_stateless_policy_checkpoint
from ptcg_rl.submission.release_assets.manifest import (
    load_release_bundle_for_native_execution,
)
from ptcg_rl.submission.release_assets.models import (
    ReleaseBundleIdentityV2,
    ReleaseBundleIdentityV3,
    ReleaseBundleIdentityV4,
    ReleaseBundleIdentityV5,
)


@dataclass(frozen=True, slots=True)
class DiscoverySummary:
    """One bounded discovery pass outcome."""

    decks_added: int = 0
    checkpoints_added: int = 0
    releases_added: int = 0
    skipped: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointPairAsset:
    """Strictly verified immutable checkpoint-pair identity."""

    manifest_path: Path
    manifest_sha256: str
    policy_path: Path
    policy_sha256: str
    active_deck_digests: tuple[str, ...]
    version: int


class AssetDiscovery:
    """Discover only new checkpoint/release paths after a durable first cursor."""

    def __init__(
        self,
        config: ContinuousLeagueConfig,
        ledger: LeagueLedger,
        *,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.repo_root = repo_root.resolve()

    def initialize(self) -> DiscoverySummary:
        """Index existing decks, but cursor existing pairs/releases without backfill."""
        decks_added = self.scan_decks()
        skipped = 0
        for asset_type, paths in (
            ("checkpoint", self._checkpoint_paths()),
            ("release", self._release_paths()),
        ):
            if self.ledger.discovery_initialized(asset_type):
                continue
            for path in paths:
                self.ledger.mark_discovery_seen(
                    asset_type,
                    path,
                    fingerprint=_file_sha256(path),
                )
                skipped += 1
            self.ledger.set_metadata(f"discovery_initialized:{asset_type}", "1")
        return DiscoverySummary(decks_added=decks_added, skipped=skipped)

    def scan(self) -> DiscoverySummary:
        """Admit all valid new assets; invalid assets remain durably inspected."""
        decks_added = self.scan_decks()
        checkpoints_added = 0
        releases_added = 0
        skipped = 0
        if not self.ledger.discovery_initialized("checkpoint"):
            return self.initialize()
        for path in self._checkpoint_paths():
            if self.ledger.discovery_seen("checkpoint", path):
                continue
            fingerprint = _file_sha256(path)
            try:
                self.add_checkpoint(path, automatic=True)
                checkpoints_added += 1
            except (OSError, RuntimeError, TypeError, ValueError):
                skipped += 1
            finally:
                self.ledger.mark_discovery_seen(
                    "checkpoint", path, fingerprint=fingerprint
                )
        if not self.ledger.discovery_initialized("release"):
            for path in self._release_paths():
                self.ledger.mark_discovery_seen(
                    "release", path, fingerprint=_file_sha256(path)
                )
            self.ledger.set_metadata("discovery_initialized:release", "1")
        else:
            for path in self._release_paths():
                if self.ledger.discovery_seen("release", path):
                    continue
                fingerprint = _file_sha256(path)
                try:
                    self.add_release(path, alias=path.parents[1].name)
                    releases_added += 1
                except (OSError, RuntimeError, TypeError, ValueError):
                    skipped += 1
                finally:
                    self.ledger.mark_discovery_seen(
                        "release", path, fingerprint=fingerprint
                    )
        return DiscoverySummary(
            decks_added=decks_added,
            checkpoints_added=checkpoints_added,
            releases_added=releases_added,
            skipped=skipped,
        )

    def scan_decks(self) -> int:
        """Index every readable 60-card CSV and reconcile compatible pairs."""
        added = 0
        for path in self._deck_paths():
            if self.ledger.discovery_seen("deck", path):
                continue
            fingerprint = _file_sha256(path)
            try:
                added += int(self.add_deck(path))
            except (OSError, TypeError, ValueError):
                pass
            finally:
                self.ledger.mark_discovery_seen("deck", path, fingerprint=fingerprint)
        return added

    def add_deck(self, path: Path, *, label: str | None = None) -> bool:
        """Manually or automatically register one exact 60-card CSV."""
        resolved = self._required_repo_file(path)
        deck = canonicalize_deck(records.read_deck(resolved))
        return self.ledger.register_deck(
            deck_digest=deck.deck_digest,
            signature=deck.signature,
            label=label or _deck_label(resolved),
            path=resolved,
            file_sha256=_file_sha256(resolved),
        )

    def add_checkpoint(
        self,
        path: Path,
        *,
        automatic: bool,
        label: str | None = None,
    ) -> CheckpointPairAsset:
        """Strictly verify and register one exact checkpoint pair."""
        asset = inspect_checkpoint_pair(self._required_repo_file(path))
        self.ledger.register_controller(
            controller_id=f"checkpoint:{asset.policy_sha256}",
            kind="checkpoint",
            label=label or f"checkpoint v{asset.version}",
            asset_path=asset.policy_path,
            asset_sha256=asset.policy_sha256,
            pair_manifest_path=asset.manifest_path,
            candidate_kind="automatic" if automatic else "manual",
            compatible_deck_digests=asset.active_deck_digests,
            requires_cuda=True,
        )
        return asset

    def add_release(self, path: Path, *, alias: str) -> str:
        """Map a release to its source checkpoint or admit its actual checkpoint."""
        resolved = self._required_repo_file(path)
        release = load_release_bundle_for_native_execution(resolved)
        deck_path = records.repo_path(release.deck_path)
        self.add_deck(deck_path, label=f"release {release.bundle_id}")
        deck = canonicalize_deck(records.read_deck(deck_path))
        source_sha = release.checkpoint_sha256
        if isinstance(
            release,
            (
                ReleaseBundleIdentityV2,
                ReleaseBundleIdentityV3,
                ReleaseBundleIdentityV4,
                ReleaseBundleIdentityV5,
            ),
        ):
            source_sha = release.deck_conditioning.source_checkpoint_sha256
        source_id = f"checkpoint:{source_sha}"
        if not self.ledger.controller_exists(source_id):
            actual_id = f"checkpoint:{release.checkpoint_sha256}"
            self.ledger.register_controller(
                controller_id=actual_id,
                kind="checkpoint",
                label=f"release {release.bundle_id}",
                asset_path=records.repo_path(release.checkpoint_path),
                asset_sha256=release.checkpoint_sha256,
                pair_manifest_path=None,
                candidate_kind="manual",
                compatible_deck_digests=(deck.deck_digest,),
                requires_cuda=True,
            )
            source_id = actual_id
        self.ledger.add_submission_alias(
            alias=alias,
            controller_id=source_id,
            release_manifest_path=resolved,
            release_manifest_sha256=_file_sha256(resolved),
        )
        return source_id

    def add_anchors(self) -> int:
        """Register explicitly configured, code-fingerprinted script anchors."""
        added = 0
        for anchor in self.config.anchors:
            deck_path = self._required_repo_file(anchor.deck_path)
            self.add_deck(deck_path, label=anchor.label)
            deck = canonicalize_deck(records.read_deck(deck_path))
            implementation = tuple(
                self._required_repo_file(path) for path in anchor.implementation_paths
            )
            code_sha = _paths_fingerprint(
                implementation,
                repo_root=self.repo_root,
            )
            self._migrate_legacy_anchor_fingerprint(
                anchor.controller_id,
                implementation_paths=implementation,
                stable_sha256=code_sha,
            )
            added += int(
                self.ledger.register_controller(
                    controller_id=anchor.controller_id,
                    kind="script",
                    label=anchor.label,
                    asset_path=implementation[0],
                    asset_sha256=code_sha,
                    pair_manifest_path=None,
                    candidate_kind="anchor",
                    compatible_deck_digests=(deck.deck_digest,),
                    requires_cuda=anchor.requires_cuda,
                )
            )
        return added

    def _migrate_legacy_anchor_fingerprint(
        self,
        controller_id: str,
        *,
        implementation_paths: Sequence[Path],
        stable_sha256: str,
    ) -> None:
        """Repair the original absolute-path hash after proving equal contents."""
        binding = self.ledger.controller_asset_binding(controller_id)
        if binding is None or binding[1] == stable_sha256:
            return
        old_first_path, old_sha256 = binding
        if old_first_path is None or not implementation_paths:
            return
        relative_paths = tuple(
            path.resolve().relative_to(self.repo_root) for path in implementation_paths
        )
        old_root = old_first_path.resolve()
        for _part in relative_paths[0].parts:
            old_root = old_root.parent
        old_paths = tuple(old_root / relative for relative in relative_paths)
        if not all(path.is_file() for path in old_paths):
            return
        if _legacy_paths_fingerprint(old_paths) != old_sha256:
            return
        if _paths_fingerprint(old_paths, repo_root=old_root) != stable_sha256:
            return
        self.ledger.migrate_legacy_script_fingerprint(
            controller_id=controller_id,
            expected_legacy_sha256=old_sha256,
            stable_sha256=stable_sha256,
            asset_path=implementation_paths[0],
        )

    def _checkpoint_paths(self) -> tuple[Path, ...]:
        return self._glob_roots(self.config.checkpoint_roots, "checkpoint_pair_v*.json")

    def _release_paths(self) -> tuple[Path, ...]:
        return self._glob_roots(self.config.release_roots, "bundle_manifest.json")

    def _deck_paths(self) -> tuple[Path, ...]:
        return self._glob_roots(self.config.deck_roots, "*.csv")

    def _glob_roots(self, roots: Sequence[Path], pattern: str) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for root in roots:
            resolved = root if root.is_absolute() else self.repo_root / root
            if resolved.is_dir():
                paths.update(path.resolve() for path in resolved.rglob(pattern))
        return tuple(sorted(paths))

    def _required_repo_file(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved


def inspect_checkpoint_pair(path: Path) -> CheckpointPairAsset:
    """Verify pair bytes, policy loadability, bindings, and exact-deck identity."""
    raw = path.read_bytes()
    payload = _mapping(json.loads(raw), "checkpoint pair")
    if payload.get("format") != "exact_policy_learner_pair_v1":
        raise ValueError("unsupported checkpoint pair format")
    policy = _mapping(payload.get("policy"), "checkpoint policy")
    learner = _mapping(payload.get("training_state"), "checkpoint learner state")
    policy_path = Path(str(policy["path"])).resolve()
    learner_path = Path(str(learner["path"])).resolve()
    policy_sha = _verify_file_record(policy_path, policy, label="policy")
    _verify_file_record(learner_path, learner, label="learner state")
    if str(learner.get("policy_sha256")) != policy_sha:
        raise ValueError("learner state is not bound to checkpoint policy")
    loaded = load_stateless_policy_checkpoint(policy_path)
    model = materialize_simple_stateless_checkpoint_model(
        loaded.model_config_value,
        loaded.model_state,
    )
    del model
    metadata = _mapping(payload.get("metadata"), "checkpoint metadata")
    active = _find_active_deck_digests(metadata)
    payload_active = tuple(sorted(loaded.identity.active_exact_deck_digests))
    if payload_active != active:
        raise ValueError("checkpoint policy and pair active deck identities differ")
    model_fingerprint = policy.get("model_fingerprint")
    if (
        model_fingerprint is not None
        and str(model_fingerprint) != loaded.artifact.policy_model_fingerprint
    ):
        raise ValueError("checkpoint pair model fingerprint differs from policy")
    return CheckpointPairAsset(
        manifest_path=path.resolve(),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        policy_path=policy_path,
        policy_sha256=policy_sha,
        active_deck_digests=active,
        version=int(payload["version"]),
    )


def _find_active_deck_digests(
    payload: Mapping[str, Any], *, required: bool = True
) -> tuple[str, ...]:
    queue: list[Mapping[str, Any]] = [payload]
    while queue:
        current = queue.pop(0)
        value = current.get("active_exact_deck_digests")
        if isinstance(value, (list, tuple)):
            digests = tuple(sorted({str(item).strip().lower() for item in value}))
            if digests and all(len(item) == 64 for item in digests):
                return digests
        queue.extend(item for item in current.values() if isinstance(item, Mapping))
    if required:
        raise ValueError("checkpoint pair has no active exact-deck roster")
    return ()


def _verify_file_record(path: Path, record: Mapping[str, Any], *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} size differs from pair manifest")
    expected = str(record["sha256"])
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs from pair manifest")
    return actual


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _deck_label(path: Path) -> str:
    return path.parent.name if path.name == "deck.csv" else path.stem


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _paths_fingerprint(paths: Sequence[Path], *, repo_root: Path) -> str:
    """Hash stable repository-relative names and immutable file contents."""
    root = repo_root.resolve()
    digest = hashlib.sha256()
    relative_paths = sorted(path.resolve().relative_to(root) for path in paths)
    for relative in relative_paths:
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_file_sha256(root / relative)))
    return digest.hexdigest()


def _legacy_paths_fingerprint(paths: Sequence[Path]) -> str:
    """Reproduce the initial path-bound hash solely for verified migration."""
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


__all__ = [
    "AssetDiscovery",
    "CheckpointPairAsset",
    "DiscoverySummary",
    "inspect_checkpoint_pair",
]
