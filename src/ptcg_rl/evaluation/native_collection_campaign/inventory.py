"""Discover auditable historical model states and their exact deck routes."""

from __future__ import annotations

import glob
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    PolicyEvaluationBinding,
    policy_evaluation_binding_fingerprint,
)
from ptcg_rl.evaluation.native_collection_campaign.artifact_io import (
    file_sha256,
    read_json,
    write_json_atomic,
)
from ptcg_rl.evaluation.native_collection_campaign.inventory_provenance import (
    ResolvedRouteEvidence,
    checkpoint_deck_records,
    discover_catalogs,
    discover_decks,
    discover_resolved_route_evidence,
    extract_authoritative_deck_hash,
)
from ptcg_rl.evaluation.native_collection_campaign.models import (
    INVENTORY_FORMAT,
    HistoricalCheckpointInventory,
    HistoricalCheckpointRecord,
    HistoricalDeckRecord,
    InventoryExclusion,
    artifact_fingerprint,
)
from ptcg_rl.rl.stateless_checkpoint import StatelessPolicyIdentity

_POLICY_NAME = re.compile(r"policy_v(?P<version>[0-9]+)\.pt")


@dataclass(frozen=True)
class _PolicyCandidate:
    checkpoint_id: str
    checkpoint_path: Path
    checkpoint_size_bytes: int
    checkpoint_version: int
    model_fingerprint: str
    identity: StatelessPolicyIdentity
    run_root: Path
    source_identity_path: Path
    source_identity_sha256: str
    checkpoint_source_commit: str
    provenance_fingerprint: str | None
    resolved_config_path: Path
    resolved_config_sha256: str
    public_catalog_manifest_path: Path
    public_catalog_manifest_sha256: str
    decks: tuple[HistoricalDeckRecord, ...]


class InventoryBuildConfig(BaseModel):
    """Hydra-friendly discovery inputs; globs are resolved below the repo root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_manifest_globs: tuple[str, ...]
    policy_checkpoint_globs: tuple[str, ...] = ()
    resolved_config_globs: tuple[str, ...] = ()
    deck_registry_globs: tuple[str, ...]
    public_catalog_manifest_globs: tuple[str, ...]
    output_path: Path
    evaluation_binding_dir: Path | None = None
    checkpoint_hash_workers: int = Field(default=4, ge=1, le=32)
    fail_on_exclusion: bool = False

    @field_validator(
        "pair_manifest_globs",
        "deck_registry_globs",
        "public_catalog_manifest_globs",
    )
    @classmethod
    def nonempty_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("inventory discovery globs must be nonempty")
        return normalized

    @field_validator(
        "policy_checkpoint_globs",
        "resolved_config_globs",
    )
    @classmethod
    def normalized_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("inventory discovery globs cannot contain empty values")
        return normalized

    @model_validator(mode="after")
    def policy_binding_target(self) -> InventoryBuildConfig:
        if self.policy_checkpoint_globs and self.evaluation_binding_dir is None:
            raise ValueError(
                "policy checkpoint discovery requires evaluation_binding_dir"
            )
        return self


def build_historical_inventory(
    config: InventoryBuildConfig,
    *,
    root: Path | None = None,
) -> HistoricalCheckpointInventory:
    """Build and atomically publish a deduplicated checkpoint inventory."""
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    deck_records = discover_decks(
        _glob_paths(repo_root, config.deck_registry_globs),
        root=repo_root,
    )
    catalogs = discover_catalogs(
        _glob_paths(repo_root, config.public_catalog_manifest_globs),
        root=repo_root,
    )
    route_evidence = discover_resolved_route_evidence(
        _glob_paths(repo_root, config.resolved_config_globs),
        root=repo_root,
    )
    discovered: list[HistoricalCheckpointRecord] = []
    exclusions: list[InventoryExclusion] = []
    for pair_path in _glob_paths(repo_root, config.pair_manifest_globs):
        try:
            discovered.append(
                _checkpoint_from_pair(
                    pair_path,
                    decks=deck_records,
                    catalogs=catalogs,
                    route_evidence=route_evidence,
                    root=repo_root,
                )
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            exclusions.append(
                InventoryExclusion(
                    pair_manifest_path=_display_path(pair_path, root=repo_root),
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    policy_candidates: list[_PolicyCandidate] = []
    for checkpoint_path in _glob_paths(repo_root, config.policy_checkpoint_globs):
        try:
            policy_candidates.append(
                _inspect_policy_checkpoint(
                    checkpoint_path,
                    decks=deck_records,
                    catalogs=catalogs,
                    route_evidence=route_evidence,
                    root=repo_root,
                )
            )
        except Exception as error:
            exclusions.append(
                InventoryExclusion(
                    policy_checkpoint_path=_display_path(
                        checkpoint_path,
                        root=repo_root,
                    ),
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    grouped_policies: defaultdict[str, list[_PolicyCandidate]] = defaultdict(list)
    for candidate in policy_candidates:
        grouped_policies[candidate.checkpoint_id].append(candidate)
    pair_contexts = {item.checkpoint_id for item in discovered}
    selected_policies = {
        checkpoint_id: max(
            candidates,
            key=lambda item: (item.checkpoint_version, str(item.checkpoint_path)),
        )
        for checkpoint_id, candidates in grouped_policies.items()
        if checkpoint_id not in pair_contexts
    }
    policy_hashes: dict[str, str] = {}
    if selected_policies:
        with ThreadPoolExecutor(
            max_workers=min(config.checkpoint_hash_workers, len(selected_policies))
        ) as executor:
            hashes = executor.map(
                file_sha256,
                (item.checkpoint_path for item in selected_policies.values()),
            )
            policy_hashes = dict(zip(selected_policies, hashes, strict=True))
        if config.evaluation_binding_dir is None:
            raise AssertionError("validated policy binding directory is missing")
        binding_dir = _resolve_path(config.evaluation_binding_dir, root=repo_root)
        for checkpoint_id, candidate in selected_policies.items():
            discovered.append(
                _checkpoint_from_policy(
                    candidate,
                    checkpoint_sha256=policy_hashes[checkpoint_id],
                    binding_dir=binding_dir,
                    root=repo_root,
                )
            )
    discovered_paths = {
        checkpoint_id: tuple(
            sorted(
                {
                    _display_path(item.checkpoint_path, root=repo_root)
                    for item in candidates
                },
                key=str,
            )
        )
        for checkpoint_id, candidates in grouped_policies.items()
    }
    exclusions.sort(key=_exclusion_sort_key)
    checkpoint_records = _deduplicate_checkpoints(
        discovered,
        discovered_paths=discovered_paths,
    )
    exclusion_records = tuple(exclusions)
    payload = {
        "format": INVENTORY_FORMAT,
        "checkpoints": [item.model_dump(mode="json") for item in checkpoint_records],
        "exclusions": [item.model_dump(mode="json") for item in exclusion_records],
    }
    inventory = HistoricalCheckpointInventory(
        inventory_fingerprint=artifact_fingerprint(
            "native-collection-checkpoint-inventory/v1",
            payload,
        ),
        checkpoints=checkpoint_records,
        exclusions=exclusion_records,
    )
    if config.fail_on_exclusion and exclusions:
        raise ValueError(f"historical inventory excluded {len(exclusions)} artifacts")
    output_path = _resolve_path(config.output_path, root=repo_root)
    if output_path.exists():
        existing = load_historical_inventory(output_path, root=repo_root)
        if existing != inventory:
            raise ValueError(
                "historical inventory path already binds a different snapshot"
            )
    else:
        write_json_atomic(output_path, inventory)
    return inventory


def load_historical_inventory(
    path: Path,
    *,
    root: Path | None = None,
) -> HistoricalCheckpointInventory:
    """Load an inventory and verify its canonical payload fingerprint."""
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    raw = read_json(_resolve_path(path, root=repo_root))
    payload = dict(raw)
    payload.pop("inventory_fingerprint", None)
    observed = artifact_fingerprint(
        "native-collection-checkpoint-inventory/v1",
        payload,
    )
    if observed != raw.get("inventory_fingerprint"):
        raise ValueError("historical inventory fingerprint differs from its payload")
    return HistoricalCheckpointInventory.model_validate(raw)


def _checkpoint_from_pair(
    pair_path: Path,
    *,
    decks: Mapping[str, Sequence[HistoricalDeckRecord]],
    catalogs: Mapping[str, tuple[Path, str]],
    route_evidence: Mapping[tuple[str, str], ResolvedRouteEvidence],
    root: Path,
) -> HistoricalCheckpointRecord:
    payload = read_json(pair_path)
    policy = _mapping(payload.get("policy"), name="policy")
    metadata = _mapping(payload.get("metadata"), name="metadata")
    identity = StatelessPolicyIdentity.model_validate(
        metadata.get("simple_stateless_identity")
    )
    model_fingerprint = _required_text(policy, "model_fingerprint")
    checkpoint_sha256 = _required_text(policy, "sha256")
    checkpoint_size = int(policy.get("size_bytes", 0))
    checkpoint_path = _resolve_checkpoint_path(
        _required_text(policy, "path"),
        pair_path=pair_path,
        root=root,
    )
    if checkpoint_size <= 0 or checkpoint_path.stat().st_size != checkpoint_size:
        raise ValueError("checkpoint bytes differ from pair manifest size")
    active = identity.active_exact_deck_digests
    if identity.sequence_contract_fingerprint is None:
        raise ValueError("native collection campaign requires a sequence checkpoint")
    missing_decks = tuple(digest for digest in active if digest not in decks)
    if missing_decks:
        raise ValueError(
            "authoritative deck provenance is unavailable for: "
            + ", ".join(missing_decks)
        )
    catalog_fingerprint = _required_text(
        identity.model_dump(mode="json"),
        "public_deck_catalog_fingerprint",
    )
    try:
        catalog_path, catalog_sha = catalogs[catalog_fingerprint]
    except KeyError as error:
        raise ValueError("checkpoint public catalog manifest is unavailable") from error
    run_root = pair_path.parent.parent
    source_identity = read_json(run_root / "control" / "training_source_identity.json")
    source_commit = _required_text(source_identity, "source_git_commit")
    provenance = source_identity.get("training_source_fingerprint")
    source_identity_path = run_root / "control" / "training_source_identity.json"
    resolved_config_path = run_root / "resolved_config.json"
    checkpoint_decks = checkpoint_deck_records(
        active,
        run_root=run_root,
        identity=identity.model_dump(mode="json"),
        candidates=decks,
        route_evidence=route_evidence,
        root=root,
    )
    return HistoricalCheckpointRecord(
        checkpoint_id=_checkpoint_context_id(model_fingerprint, identity),
        label=f"{run_root.name}:v{int(payload.get('version', 0))}",
        model_fingerprint=model_fingerprint,
        checkpoint_path=_display_path(checkpoint_path, root=root),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_version=int(payload.get("version", 0)),
        checkpoint_source_commit=source_commit,
        model_config_fingerprint=identity.model_config_fingerprint,
        source_identity_path=_display_path(source_identity_path, root=root),
        source_identity_sha256=file_sha256(source_identity_path),
        resolved_config_path=_display_path(resolved_config_path, root=root),
        resolved_config_sha256=file_sha256(resolved_config_path),
        pair_manifest_path=_display_path(pair_path, root=root),
        pair_manifest_sha256=file_sha256(pair_path),
        public_catalog_manifest_path=catalog_path,
        public_catalog_manifest_sha256=catalog_sha,
        public_catalog_fingerprint=catalog_fingerprint,
        exact_registry_fingerprint=_required_text(
            identity.model_dump(mode="json"),
            "exact_registry_fingerprint",
        ),
        input_contract_fingerprint=_required_text(
            identity.model_dump(mode="json"),
            "input_contract_fingerprint",
        ),
        provenance_fingerprint=(str(provenance) if provenance is not None else None),
        decks=checkpoint_decks,
        alias_pair_manifests=(_display_path(pair_path, root=root),),
        discovered_checkpoint_paths=(_display_path(checkpoint_path, root=root),),
    )


def _inspect_policy_checkpoint(
    checkpoint_path: Path,
    *,
    decks: Mapping[str, Sequence[HistoricalDeckRecord]],
    catalogs: Mapping[str, tuple[Path, str]],
    route_evidence: Mapping[tuple[str, str], ResolvedRouteEvidence],
    root: Path,
) -> _PolicyCandidate:
    """Read trusted checkpoint metadata without materializing tensor storage."""
    name_match = _POLICY_NAME.fullmatch(checkpoint_path.name)
    if name_match is None:
        raise ValueError("policy checkpoint filename has no immutable version")
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    try:
        payload = _mapping(loaded, name="policy checkpoint")
        if payload.get("format") != "simple_stateless_policy_v1":
            raise ValueError("unsupported standalone policy checkpoint format")
        version = int(payload.get("version", -1))
        if version != int(name_match.group("version")):
            raise ValueError("policy checkpoint version differs from its filename")
        model_fingerprint = _required_sha(payload, "model_fingerprint")
        identity = StatelessPolicyIdentity.model_validate(payload.get("identity"))
        if identity.sequence_contract_fingerprint is None:
            raise ValueError(
                "native collection campaign requires a sequence checkpoint"
            )
        if not isinstance(payload.get("model_config"), Mapping):
            raise ValueError("policy checkpoint has no model configuration")
        active = identity.active_exact_deck_digests
        missing_decks = tuple(digest for digest in active if digest not in decks)
        if missing_decks:
            raise ValueError(
                "authoritative deck provenance is unavailable for: "
                + ", ".join(missing_decks)
            )
        try:
            catalog_path, catalog_sha = catalogs[
                identity.public_deck_catalog_fingerprint
            ]
        except KeyError as error:
            raise ValueError(
                "checkpoint public catalog manifest is unavailable"
            ) from error
        run_root = checkpoint_path.parent.parent
        source_identity_path = run_root / "control" / "training_source_identity.json"
        source_identity = read_json(source_identity_path)
        source_commit = _required_text(source_identity, "source_git_commit")
        provenance = source_identity.get("training_source_fingerprint")
        resolved_config_path = run_root / "resolved_config.json"
        checkpoint_decks = checkpoint_deck_records(
            active,
            run_root=run_root,
            identity=identity.model_dump(mode="json"),
            candidates=decks,
            route_evidence=route_evidence,
            root=root,
        )
        return _PolicyCandidate(
            checkpoint_id=_checkpoint_context_id(model_fingerprint, identity),
            checkpoint_path=checkpoint_path,
            checkpoint_size_bytes=checkpoint_path.stat().st_size,
            checkpoint_version=version,
            model_fingerprint=model_fingerprint,
            identity=identity,
            run_root=run_root,
            source_identity_path=source_identity_path,
            source_identity_sha256=file_sha256(source_identity_path),
            checkpoint_source_commit=source_commit,
            provenance_fingerprint=(
                str(provenance) if provenance is not None else None
            ),
            resolved_config_path=resolved_config_path,
            resolved_config_sha256=file_sha256(resolved_config_path),
            public_catalog_manifest_path=catalog_path,
            public_catalog_manifest_sha256=catalog_sha,
            decks=checkpoint_decks,
        )
    finally:
        del loaded


def _checkpoint_from_policy(
    candidate: _PolicyCandidate,
    *,
    checkpoint_sha256: str,
    binding_dir: Path,
    root: Path,
) -> HistoricalCheckpointRecord:
    binding_payload: dict[str, Any] = {
        "format": "policy_evaluation_binding_v1",
        "checkpoint_path": str(_display_path(candidate.checkpoint_path, root=root)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": candidate.checkpoint_size_bytes,
        "checkpoint_version": candidate.checkpoint_version,
        "model_fingerprint": candidate.model_fingerprint,
        "policy_identity": candidate.identity.model_dump(mode="json"),
        "source_identity_path": str(
            _display_path(candidate.source_identity_path, root=root)
        ),
        "source_identity_sha256": candidate.source_identity_sha256,
        "checkpoint_source_commit": candidate.checkpoint_source_commit,
        "provenance_fingerprint": candidate.provenance_fingerprint,
        "resolved_config_path": str(
            _display_path(candidate.resolved_config_path, root=root)
        ),
        "resolved_config_sha256": candidate.resolved_config_sha256,
        "public_catalog_manifest_path": str(candidate.public_catalog_manifest_path),
        "public_catalog_manifest_sha256": (candidate.public_catalog_manifest_sha256),
        "resume_capability": "policy_only_not_resumable",
    }
    binding = PolicyEvaluationBinding(
        **binding_payload,
        binding_fingerprint=policy_evaluation_binding_fingerprint(binding_payload),
    )
    binding_path = binding_dir / f"{candidate.checkpoint_id}.json"
    if binding_path.exists():
        existing = PolicyEvaluationBinding.model_validate(read_json(binding_path))
        if existing != binding:
            raise ValueError(
                "policy evaluation binding path already binds another artifact"
            )
    else:
        write_json_atomic(binding_path, binding)
    return HistoricalCheckpointRecord(
        checkpoint_id=candidate.checkpoint_id,
        label=(f"{candidate.run_root.name}:v{candidate.checkpoint_version}"),
        model_fingerprint=candidate.model_fingerprint,
        checkpoint_path=_display_path(candidate.checkpoint_path, root=root),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=candidate.checkpoint_size_bytes,
        checkpoint_version=candidate.checkpoint_version,
        checkpoint_source_commit=candidate.checkpoint_source_commit,
        model_config_fingerprint=candidate.identity.model_config_fingerprint,
        source_identity_path=_display_path(
            candidate.source_identity_path,
            root=root,
        ),
        source_identity_sha256=candidate.source_identity_sha256,
        resolved_config_path=_display_path(
            candidate.resolved_config_path,
            root=root,
        ),
        resolved_config_sha256=candidate.resolved_config_sha256,
        evaluation_binding_path=_display_path(binding_path, root=root),
        evaluation_binding_sha256=file_sha256(binding_path),
        public_catalog_manifest_path=candidate.public_catalog_manifest_path,
        public_catalog_manifest_sha256=(candidate.public_catalog_manifest_sha256),
        public_catalog_fingerprint=(candidate.identity.public_deck_catalog_fingerprint),
        exact_registry_fingerprint=candidate.identity.exact_registry_fingerprint,
        input_contract_fingerprint=candidate.identity.input_contract_fingerprint,
        provenance_fingerprint=candidate.provenance_fingerprint,
        decks=candidate.decks,
        discovered_checkpoint_paths=(
            _display_path(candidate.checkpoint_path, root=root),
        ),
    )


def _checkpoint_context_id(
    model_fingerprint: str,
    identity: StatelessPolicyIdentity,
) -> str:
    """Separate one model state only when evaluator-visible contracts differ."""
    payload = {
        "model_fingerprint": model_fingerprint,
        "model_config_fingerprint": identity.model_config_fingerprint,
        "exact_registry_fingerprint": identity.exact_registry_fingerprint,
        "active_exact_deck_digests": identity.active_exact_deck_digests,
        "action_schema_fingerprint": identity.action_schema_fingerprint,
        "public_context_fingerprint": identity.public_context_fingerprint,
        "card_catalog_fingerprint": identity.card_catalog_fingerprint,
        "belief_target_semantics_fingerprint": (
            identity.belief_target_semantics_fingerprint
        ),
        "public_deck_catalog_fingerprint": (identity.public_deck_catalog_fingerprint),
        "input_contract_fingerprint": identity.input_contract_fingerprint,
        "fragment_static_contract_fingerprint": (
            identity.fragment_static_contract_fingerprint
        ),
        "sequence_contract_fingerprint": identity.sequence_contract_fingerprint,
    }
    return artifact_fingerprint(
        "native-collection-checkpoint-context/v1",
        payload,
    )


def _deduplicate_checkpoints(
    records_: Iterable[HistoricalCheckpointRecord],
    *,
    discovered_paths: Mapping[str, tuple[Path, ...]],
) -> tuple[HistoricalCheckpointRecord, ...]:
    grouped: defaultdict[str, list[HistoricalCheckpointRecord]] = defaultdict(list)
    for item in records_:
        grouped[item.checkpoint_id].append(item)
    output: list[HistoricalCheckpointRecord] = []
    for checkpoint_id, aliases in sorted(grouped.items()):
        contracts = {
            (
                item.model_fingerprint,
                item.model_config_fingerprint,
                item.exact_registry_fingerprint,
                item.input_contract_fingerprint,
                item.public_catalog_fingerprint,
                tuple(deck.deck_digest for deck in item.decks),
            )
            for item in aliases
        }
        if len(contracts) != 1:
            raise ValueError(
                f"checkpoint context {checkpoint_id} has conflicting contracts"
            )
        selected = max(
            aliases,
            key=lambda item: (
                item.pair_manifest_path is not None,
                item.checkpoint_version,
                str(item.pair_manifest_path or item.evaluation_binding_path),
            ),
        )
        output.append(
            selected.model_copy(
                update={
                    "alias_pair_manifests": tuple(
                        sorted(
                            {
                                path
                                for item in aliases
                                for path in item.alias_pair_manifests
                            },
                            key=str,
                        )
                    ),
                    "discovered_checkpoint_paths": tuple(
                        sorted(
                            {
                                *(
                                    path
                                    for item in aliases
                                    for path in item.discovered_checkpoint_paths
                                ),
                                *discovered_paths.get(checkpoint_id, ()),
                            },
                            key=str,
                        )
                    ),
                }
            )
        )
    return tuple(output)


def _glob_paths(root: Path, patterns: Sequence[str]) -> tuple[Path, ...]:
    paths = {
        Path(path).resolve()
        for pattern in patterns
        for path in glob.glob(str(root / pattern), recursive=True)
        if Path(path).is_file()
    }
    return tuple(sorted(paths))


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact {name} must be an object")
    return value


def _required_text(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact is missing {key}")
    return value.strip()


def _required_sha(values: Mapping[str, Any], key: str) -> str:
    value = _required_text(values, key).lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"artifact {key} must be a full lowercase SHA-256")
    return value


def _exclusion_sort_key(item: InventoryExclusion) -> str:
    return str(item.pair_manifest_path or item.policy_checkpoint_path)


def _resolve_checkpoint_path(raw: str, *, pair_path: Path, root: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = (
        path,
        pair_path.parent / path.name,
        root / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint bound by pair does not exist: {raw}")


def _resolve_path(path: Path, *, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _display_path(path: Path, *, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved


__all__ = [
    "InventoryBuildConfig",
    "build_historical_inventory",
    "extract_authoritative_deck_hash",
    "load_historical_inventory",
]
