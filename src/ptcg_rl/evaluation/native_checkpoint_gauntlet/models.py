"""Validated contracts for a bounded native cross-checkpoint gauntlet."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.evaluation.native_deck_elo.models import DeckAsset
from ptcg_rl.rl.stateless_checkpoint import StatelessPolicyIdentity

FORMAT = "native_checkpoint_gauntlet_v1"
RESULTS_FORMAT = "native_checkpoint_gauntlet_games_v1"
POLICY_EVALUATION_BINDING_FORMAT: Final = "policy_evaluation_binding_v1"


def _validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("artifact identity must be a full lowercase SHA-256")
    return normalized


def policy_evaluation_binding_fingerprint(
    payload: Mapping[str, Any],
) -> str:
    """Hash the canonical payload of a policy-only evaluation capability."""
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/policy-evaluation-binding/v1\0" + encoded
    ).hexdigest()


class PolicyEvaluationBinding(BaseModel):
    """Immutable evaluation-only provenance for a standalone policy weight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["policy_evaluation_binding_v1"] = POLICY_EVALUATION_BINDING_FORMAT
    binding_fingerprint: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int = Field(gt=0)
    checkpoint_version: int = Field(ge=0)
    model_fingerprint: str
    policy_identity: StatelessPolicyIdentity
    source_identity_path: Path
    source_identity_sha256: str
    checkpoint_source_commit: str
    provenance_fingerprint: str | None = None
    resolved_config_path: Path
    resolved_config_sha256: str
    public_catalog_manifest_path: Path
    public_catalog_manifest_sha256: str
    resume_capability: Literal["policy_only_not_resumable"] = (
        "policy_only_not_resumable"
    )

    @field_validator(
        "binding_fingerprint",
        "checkpoint_sha256",
        "model_fingerprint",
        "source_identity_sha256",
        "resolved_config_sha256",
        "public_catalog_manifest_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("provenance_fingerprint")
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value)

    @field_validator("checkpoint_source_commit")
    @classmethod
    def valid_source_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("checkpoint source identity must be a full Git commit")
        return normalized

    @model_validator(mode="after")
    def coherent_fingerprint(self) -> PolicyEvaluationBinding:
        payload = self.model_dump(mode="json", exclude={"binding_fingerprint"})
        if policy_evaluation_binding_fingerprint(payload) != (self.binding_fingerprint):
            raise ValueError("policy evaluation binding fingerprint differs")
        return self


class CheckpointRosterDeckConfig(BaseModel):
    """One exact deck routed by a checkpoint participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    path: Path
    deck_hash: str | None = None

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("deck label must be non-empty")
        return cleaned

    @field_validator("deck_hash")
    @classmethod
    def valid_optional_deck_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 12 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("deck_hash must be the historical 12-character ID")
        return normalized


class CheckpointRosterParticipantConfig(BaseModel):
    """One frozen checkpoint, its catalog, and its checkpoint-native roster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    checkpoint_path: Path
    expected_checkpoint_sha256: str
    checkpoint_source_commit: str
    source_identity_path: Path | None = None
    expected_source_identity_sha256: str | None = None
    resolved_config_path: Path | None = None
    expected_resolved_config_sha256: str | None = None
    pair_manifest_path: Path | None = None
    expected_pair_manifest_sha256: str | None = None
    policy_evaluation_binding_path: Path | None = None
    expected_policy_evaluation_binding_sha256: str | None = None
    public_catalog_manifest_path: Path
    expected_public_catalog_manifest_sha256: str
    provenance_fingerprint: str | None = None
    policy_temperature: float = 1.0
    roster_scope: Literal["full", "subset"] = "full"
    decks: tuple[CheckpointRosterDeckConfig, ...]

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("checkpoint label must be non-empty")
        return cleaned

    @field_validator(
        "expected_checkpoint_sha256",
        "expected_public_catalog_manifest_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator(
        "expected_source_identity_sha256",
        "expected_resolved_config_sha256",
        "expected_pair_manifest_sha256",
        "expected_policy_evaluation_binding_sha256",
    )
    @classmethod
    def valid_optional_artifact_sha256(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value)

    @field_validator("provenance_fingerprint")
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value)

    @field_validator("policy_temperature")
    @classmethod
    def valid_policy_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy_temperature must be finite and non-negative")
        return value

    @field_validator("checkpoint_source_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("checkpoint source identity must be a full Git commit")
        return normalized

    @model_validator(mode="after")
    def unique_nonempty_roster(self) -> CheckpointRosterParticipantConfig:
        common_pairs = (
            (self.source_identity_path, self.expected_source_identity_sha256),
            (self.resolved_config_path, self.expected_resolved_config_sha256),
        )
        if any((path is None) != (sha256 is None) for path, sha256 in common_pairs):
            raise ValueError("participant provenance path/SHA fields are incomplete")
        pair_complete = (
            self.pair_manifest_path is not None
            and self.expected_pair_manifest_sha256 is not None
        )
        if (self.pair_manifest_path is None) != (
            self.expected_pair_manifest_sha256 is None
        ):
            raise ValueError("checkpoint pair path/SHA fields are incomplete")
        binding_complete = (
            self.policy_evaluation_binding_path is not None
            and self.expected_policy_evaluation_binding_sha256 is not None
        )
        if (self.policy_evaluation_binding_path is None) != (
            self.expected_policy_evaluation_binding_sha256 is None
        ):
            raise ValueError("policy evaluation binding path/SHA fields are incomplete")
        if pair_complete == binding_complete:
            raise ValueError(
                "participant requires exactly one checkpoint pair or policy "
                "evaluation binding"
            )
        if not self.decks:
            raise ValueError("checkpoint participant requires an exact roster")
        labels = [deck.label for deck in self.decks]
        if len(labels) != len(set(labels)):
            raise ValueError("checkpoint roster labels must be unique")
        return self


class NativeCheckpointGauntletConfig(BaseModel):
    """Immutable inputs and operational settings for one cross-roster campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: CheckpointRosterParticipantConfig
    baseline: CheckpointRosterParticipantConfig
    runner_source_commit: str
    native_library_path: Path = Path("src/native/cg_train/libcg_train.so")
    expected_native_library_sha256: str
    output_dir: Path
    backend: Literal["native_collection", "native_match"] = "native_collection"
    total_games: int = 8_000
    concurrency: int = 48
    policy_batch_max_rows: int = 16
    policy_batch_wait_ms: float = 2.0
    policy_batch_coalesce_temperatures: bool = True
    native_lane_worker_count: int = 1
    native_worker_replicas: int = 1
    native_worker_torch_threads: int = 1
    native_inductor_compile_threads: int | None = Field(default=None, gt=0)
    native_option_capacity: int = 2_048
    native_arena_capacity: int = 512
    native_engine_shards: int = 8
    native_engine_fact_workers: int = 8
    native_policy_cohort_slots: int = 512
    native_policy_group_bank_limit: int = 3
    native_policy_cohort_wait_ms: float = 0.0
    native_frozen_batch_min_rows: int = 48
    native_frozen_batch_max_wait_waves: int = 4
    collection_part_games: int = 1_024
    maximum_engine_steps: int = 10_000
    flush_shard_games: int = 32
    seed: int = 20_260_810
    match_seed_namespace: str | None = None
    device: str = "cuda"
    checkpoint_resident_precision: Literal["source", "bfloat16"] = "source"
    checkpoint_rollout_inductor: bool = False
    evaluation_action_only: bool = False
    candidate_intervention: Literal[
        "baseline", "reset_temporal", "mask_engine_facts", "both"
    ] = "baseline"
    compression: str = "zstd"
    act_time: ActTimeConfig = Field(default_factory=ActTimeConfig)

    @field_validator("expected_native_library_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("runner_source_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("runner source identity must be a full Git commit")
        return normalized

    @field_validator("match_seed_namespace")
    @classmethod
    def valid_match_seed_namespace(cls, value: str | None) -> str | None:
        """Require an explicit nonempty identity for common-random-number runs."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("match_seed_namespace must be nonempty")
        return normalized

    @field_validator(
        "total_games",
        "concurrency",
        "policy_batch_max_rows",
        "native_lane_worker_count",
        "native_worker_replicas",
        "native_worker_torch_threads",
        "native_option_capacity",
        "native_arena_capacity",
        "native_engine_shards",
        "native_engine_fact_workers",
        "native_policy_cohort_slots",
        "native_policy_group_bank_limit",
        "native_frozen_batch_min_rows",
        "native_frozen_batch_max_wait_waves",
        "collection_part_games",
        "maximum_engine_steps",
        "flush_shard_games",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("native checkpoint gauntlet counts must be positive")
        return value

    @field_validator("policy_batch_wait_ms", "native_policy_cohort_wait_ms")
    @classmethod
    def nonnegative_float(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("batch wait must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def valid_cross_roster_schedule(self) -> NativeCheckpointGauntletConfig:
        if self.candidate_intervention != "baseline" and not (
            self.backend == "native_collection" and self.evaluation_action_only
        ):
            raise ValueError("input interventions require native greedy evaluation")
        if self.checkpoint_rollout_inductor and (
            self.checkpoint_resident_precision != "bfloat16"
        ):
            raise ValueError(
                "checkpoint rollout Inductor requires bfloat16 resident weights"
            )
        if self.checkpoint_resident_precision == "bfloat16" and not (
            self.device.lower().startswith("cuda")
        ):
            raise ValueError("bfloat16 resident checkpoint inference requires CUDA")
        if self.backend == "native_collection" and (
            self.checkpoint_resident_precision != "source"
            or self.checkpoint_rollout_inductor
        ):
            raise ValueError(
                "checkpoint resident precision controls apply only to native_match"
            )
        if self.candidate.label == self.baseline.label:
            raise ValueError("candidate and baseline labels must differ")
        cell_count = len(self.candidate.decks) * len(self.baseline.decks)
        if self.total_games < cell_count * 2 or self.total_games % 2:
            raise ValueError(
                "total_games must cover every cross-roster cell with mirrored seats"
            )
        if self.policy_batch_max_rows <= 1:
            raise ValueError("cross-checkpoint evaluation requires CUDA batching")
        if not 4 <= self.native_engine_shards <= 8:
            raise ValueError("native collection requires 4-8 engine shards")
        if not 1 <= self.native_policy_group_bank_limit <= 4:
            raise ValueError("native policy bank limit must be within [1, 4]")
        if self.native_policy_cohort_slots > self.native_arena_capacity:
            raise ValueError("native policy cohort exceeds live arena capacity")
        if self.collection_part_games < self.native_arena_capacity:
            raise ValueError("collection parts must fill the native arena")
        if self.act_time.planner is not None:
            raise ValueError("native checkpoint gauntlet has no planner lane service")
        if self.evaluation_action_only:
            if self.backend != "native_collection":
                raise ValueError(
                    "evaluation_action_only applies only to native_collection"
                )
            if (
                self.candidate.policy_temperature != 0.0
                or self.baseline.policy_temperature != 0.0
            ):
                raise ValueError("evaluation_action_only requires both policies at T=0")
        elif self.backend == "native_collection" and (
            self.candidate.policy_temperature != 1.0
            or self.baseline.policy_temperature != 1.0
        ):
            raise ValueError(
                "native_collection accepts T=1 training semantics or explicit "
                "T=0 evaluation_action_only semantics"
            )
        return self


@dataclass(frozen=True)
class ScheduledCrossCheckpointGame:
    """One deterministic game in a mirrored cross-checkpoint schedule."""

    game_index: int
    match_id: str
    candidate_deck: DeckAsset
    baseline_deck: DeckAsset
    candidate_seat: Literal[0, 1]


__all__ = [
    "FORMAT",
    "POLICY_EVALUATION_BINDING_FORMAT",
    "RESULTS_FORMAT",
    "CheckpointRosterDeckConfig",
    "CheckpointRosterParticipantConfig",
    "NativeCheckpointGauntletConfig",
    "PolicyEvaluationBinding",
    "ScheduledCrossCheckpointGame",
    "policy_evaluation_binding_fingerprint",
]
