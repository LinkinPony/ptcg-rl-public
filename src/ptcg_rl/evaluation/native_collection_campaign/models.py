"""Immutable contracts for large training-path native evaluation campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    NativeCheckpointGauntletConfig,
)

LEGACY_INVENTORY_FORMAT: Final = "native_collection_checkpoint_inventory_v1"
INVENTORY_FORMAT: Final = "native_collection_checkpoint_inventory_v2"
PLAN_FORMAT: Final = "native_collection_campaign_plan_v1"
SELECTION_FORMAT: Final = "native_collection_campaign_selection_v1"


def _sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("identity must be a full lowercase SHA-256")
    return normalized


def _commit(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source commit must be a full lowercase Git commit")
    return normalized


def _deck_hash(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 12 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("deck_hash must be the historical 12-character ID")
    return normalized


class HistoricalDeckRecord(BaseModel):
    """One exact deck with its authoritative compact display identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    deck_hash: str
    label: str
    path: Path
    deck_signature: str
    provenance_paths: tuple[Path, ...] = ()

    @field_validator("deck_digest")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("deck_hash")
    @classmethod
    def valid_deck_hash(cls, value: str) -> str:
        return _deck_hash(value)

    @field_validator("label", "deck_signature")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("deck identity text must be nonempty")
        return normalized


class HistoricalCheckpointRecord(BaseModel):
    """One deduplicated model state and every exact route it can serve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: str
    label: str
    model_fingerprint: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int = Field(gt=0)
    checkpoint_version: int = Field(ge=0)
    checkpoint_source_commit: str
    model_config_fingerprint: str | None = None
    source_identity_path: Path | None = None
    source_identity_sha256: str | None = None
    resolved_config_path: Path | None = None
    resolved_config_sha256: str | None = None
    pair_manifest_path: Path | None = None
    pair_manifest_sha256: str | None = None
    evaluation_binding_path: Path | None = None
    evaluation_binding_sha256: str | None = None
    public_catalog_manifest_path: Path
    public_catalog_manifest_sha256: str
    public_catalog_fingerprint: str
    exact_registry_fingerprint: str
    input_contract_fingerprint: str
    provenance_fingerprint: str | None = None
    decks: tuple[HistoricalDeckRecord, ...]
    alias_pair_manifests: tuple[Path, ...] = ()
    discovered_checkpoint_paths: tuple[Path, ...] = ()

    @field_validator(
        "checkpoint_id",
        "model_fingerprint",
        "checkpoint_sha256",
        "public_catalog_manifest_sha256",
        "public_catalog_fingerprint",
        "exact_registry_fingerprint",
        "input_contract_fingerprint",
    )
    @classmethod
    def valid_sha(cls, value: str) -> str:
        return _sha256(value)

    @field_validator(
        "model_config_fingerprint",
        "source_identity_sha256",
        "resolved_config_sha256",
        "pair_manifest_sha256",
        "evaluation_binding_sha256",
    )
    @classmethod
    def valid_optional_artifact_sha(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value)

    @field_validator("provenance_fingerprint")
    @classmethod
    def valid_optional_sha(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value)

    @field_validator("checkpoint_source_commit")
    @classmethod
    def valid_source_commit(cls, value: str) -> str:
        return _commit(value)

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("checkpoint label must be nonempty")
        return normalized

    @model_validator(mode="after")
    def unique_routes(self) -> HistoricalCheckpointRecord:
        common_pairs = (
            (self.source_identity_path, self.source_identity_sha256),
            (self.resolved_config_path, self.resolved_config_sha256),
        )
        if any((path is None) != (sha256 is None) for path, sha256 in common_pairs):
            raise ValueError("checkpoint provenance path/SHA fields are incomplete")
        pair_complete = (
            self.pair_manifest_path is not None
            and self.pair_manifest_sha256 is not None
        )
        if (self.pair_manifest_path is None) != (self.pair_manifest_sha256 is None):
            raise ValueError("checkpoint pair path/SHA fields are incomplete")
        binding_complete = (
            self.evaluation_binding_path is not None
            and self.evaluation_binding_sha256 is not None
        )
        if (self.evaluation_binding_path is None) != (
            self.evaluation_binding_sha256 is None
        ):
            raise ValueError("evaluation binding path/SHA fields are incomplete")
        if pair_complete == binding_complete:
            raise ValueError(
                "checkpoint requires exactly one pair or evaluation binding"
            )
        if not self.decks:
            raise ValueError("checkpoint requires at least one exact route")
        digests = tuple(deck.deck_digest for deck in self.decks)
        if len(digests) != len(set(digests)):
            raise ValueError("checkpoint exact routes must be unique")
        if self.discovered_checkpoint_paths and len(
            self.discovered_checkpoint_paths
        ) != len(set(self.discovered_checkpoint_paths)):
            raise ValueError("discovered checkpoint paths must be unique")
        return self


class InventoryExclusion(BaseModel):
    """One discovered artifact that could not become an auditable participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_manifest_path: Path | None = None
    policy_checkpoint_path: Path | None = None
    reason: str

    @field_validator("reason")
    @classmethod
    def nonempty_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("inventory exclusion reason must be nonempty")
        return normalized

    @model_validator(mode="after")
    def exactly_one_artifact(self) -> InventoryExclusion:
        if (self.pair_manifest_path is None) == (self.policy_checkpoint_path is None):
            raise ValueError("inventory exclusion requires exactly one artifact path")
        return self


class HistoricalCheckpointInventory(BaseModel):
    """Auditable, deduplicated historical checkpoint and route inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "native_collection_checkpoint_inventory_v1",
        "native_collection_checkpoint_inventory_v2",
    ] = INVENTORY_FORMAT
    inventory_fingerprint: str
    checkpoints: tuple[HistoricalCheckpointRecord, ...]
    exclusions: tuple[InventoryExclusion, ...] = ()

    @field_validator("inventory_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        return _sha256(value)

    @model_validator(mode="after")
    def unique_checkpoints(self) -> HistoricalCheckpointInventory:
        ids = tuple(item.checkpoint_id for item in self.checkpoints)
        if len(ids) != len(set(ids)):
            raise ValueError("inventory checkpoint IDs must be unique")
        return self


class BundleSelector(BaseModel):
    """Declarative filter over exact checkpoint-route bundles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    include_all: bool = False
    checkpoint_ids: tuple[str, ...] = ()
    deck_digests: tuple[str, ...] = ()
    deck_hashes: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()

    @field_validator("checkpoint_ids", "bundle_ids")
    @classmethod
    def nonempty_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_sha256(item) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("selector IDs must be unique full SHA-256 values")
        return normalized

    @field_validator("deck_digests")
    @classmethod
    def valid_deck_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_sha256(item) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("selector deck digests must be unique")
        return normalized

    @field_validator("deck_hashes")
    @classmethod
    def valid_deck_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip().lower() for item in value)
        if any(
            len(item) != 12
            or any(character not in "0123456789abcdef" for character in item)
            for item in normalized
        ) or len(normalized) != len(set(normalized)):
            raise ValueError("selector deck_hash values must be unique compact IDs")
        return normalized

    @model_validator(mode="after")
    def nonempty_selector(self) -> BundleSelector:
        filters = (
            self.checkpoint_ids,
            self.deck_digests,
            self.deck_hashes,
            self.bundle_ids,
        )
        if self.include_all and any(filters):
            raise ValueError("include_all cannot be combined with selector filters")
        if not self.include_all and not any(filters):
            raise ValueError("bundle selector must include at least one filter")
        return self


class PlannedBundle(BaseModel):
    """One deployable checkpoint × exact-deck route in a frozen campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    checkpoint_id: str
    model_fingerprint: str
    deck_digest: str
    deck_hash: str
    deck_label: str
    deck_signature: str
    policy_temperature: float

    @field_validator("bundle_id", "checkpoint_id", "model_fingerprint", "deck_digest")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("deck_hash")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _deck_hash(value)

    @field_validator("policy_temperature")
    @classmethod
    def valid_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("bundle temperature must be finite and non-negative")
        return value

    @field_validator("deck_label", "deck_signature")
    @classmethod
    def nonempty_deck_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("planned deck identity text must be nonempty")
        return normalized


class CampaignTask(BaseModel):
    """One sequential model-pair task assigned to exactly one device shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    device_index: int = Field(ge=0)
    device_binding: str
    games_per_cell_per_seat: int = Field(gt=0)
    candidate_bundle_ids: tuple[str, ...]
    opponent_bundle_ids: tuple[str, ...]
    gauntlet: NativeCheckpointGauntletConfig

    @field_validator("task_id")
    @classmethod
    def valid_task_id(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("candidate_bundle_ids", "opponent_bundle_ids")
    @classmethod
    def valid_bundle_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_sha256(item) for item in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("campaign task bundle IDs must be nonempty and unique")
        return normalized

    @field_validator("device_binding")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device binding must be nonempty")
        return normalized

    @model_validator(mode="after")
    def strict_greedy_work_unit(self) -> CampaignTask:
        expected_games = (
            len(self.candidate_bundle_ids)
            * len(self.opponent_bundle_ids)
            * 2
            * self.games_per_cell_per_seat
        )
        if self.gauntlet.total_games != expected_games:
            raise ValueError("campaign task game count differs from its exact cells")
        if (
            self.gauntlet.backend != "native_collection"
            or not self.gauntlet.evaluation_action_only
            or self.gauntlet.candidate.policy_temperature != 0.0
            or self.gauntlet.baseline.policy_temperature != 0.0
        ):
            raise ValueError("campaign task must use strict greedy action-only")
        return self


class NativeCollectionCampaignPlan(BaseModel):
    """Complete immutable work graph for one independently scored stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["native_collection_campaign_plan_v1"] = PLAN_FORMAT
    plan_fingerprint: str
    stage_id: str
    inventory_path: Path
    inventory_fingerprint: str
    output_dir: Path
    device_bindings: tuple[str, ...]
    candidates: tuple[PlannedBundle, ...]
    opponents: tuple[PlannedBundle, ...]
    tasks: tuple[CampaignTask, ...]

    @field_validator("plan_fingerprint", "inventory_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("stage_id")
    @classmethod
    def nonempty_stage(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("campaign stage ID must be nonempty")
        return normalized

    @model_validator(mode="after")
    def coherent_work_graph(self) -> NativeCollectionCampaignPlan:
        if not self.device_bindings or not self.candidates or not self.opponents:
            raise ValueError("campaign requires devices, candidates, and opponents")
        if len(self.device_bindings) != len(set(self.device_bindings)):
            raise ValueError("campaign device bindings must be unique")
        task_ids = tuple(task.task_id for task in self.tasks)
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("campaign task IDs must be nonempty and unique")
        candidate_ids = {bundle.bundle_id for bundle in self.candidates}
        opponent_ids = {bundle.bundle_id for bundle in self.opponents}
        if len(candidate_ids) != len(self.candidates) or len(opponent_ids) != len(
            self.opponents
        ):
            raise ValueError("campaign bundle identities must be unique by role")
        if any(
            not set(task.candidate_bundle_ids) <= candidate_ids
            or not set(task.opponent_bundle_ids) <= opponent_ids
            for task in self.tasks
        ):
            raise ValueError("campaign task references an unknown bundle")
        if any(
            task.device_index >= len(self.device_bindings)
            or task.device_binding != self.device_bindings[task.device_index]
            for task in self.tasks
        ):
            raise ValueError("campaign task device binding differs from its shard")
        output_dirs = tuple(task.gauntlet.output_dir for task in self.tasks)
        if len(output_dirs) != len(set(output_dirs)):
            raise ValueError("campaign task output directories must be unique")
        return self


class CampaignSelection(BaseModel):
    """Immutable promotion decision consumed by a later campaign stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["native_collection_campaign_selection_v1"] = SELECTION_FORMAT
    source_plan_fingerprint: str
    standings_sha256: str
    selected_bundle_ids: tuple[str, ...]
    score_column: str

    @field_validator("source_plan_fingerprint", "standings_sha256")
    @classmethod
    def valid_sha(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("selected_bundle_ids")
    @classmethod
    def valid_selected_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_sha256(item) for item in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("promotion bundle IDs must be nonempty and unique")
        return normalized

    @field_validator("score_column")
    @classmethod
    def nonempty_score_column(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("promotion score column must be nonempty")
        return normalized


def artifact_fingerprint(domain: str, payload: object) -> str:
    """Hash one canonical JSON payload under an explicit identity domain."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(f"ptcg-rl/{domain}\0".encode() + encoded).hexdigest()


__all__ = [
    "INVENTORY_FORMAT",
    "LEGACY_INVENTORY_FORMAT",
    "PLAN_FORMAT",
    "SELECTION_FORMAT",
    "BundleSelector",
    "CampaignSelection",
    "CampaignTask",
    "HistoricalCheckpointInventory",
    "HistoricalCheckpointRecord",
    "HistoricalDeckRecord",
    "InventoryExclusion",
    "NativeCollectionCampaignPlan",
    "PlannedBundle",
    "artifact_fingerprint",
]
