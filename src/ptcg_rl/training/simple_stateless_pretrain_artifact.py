"""Immutable supervised-policy artifacts and RL handoff verification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from torch import Tensor

from ptcg_rl.model.simple_stateless import (
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
    materialize_simple_stateless_checkpoint_model,
    uses_exact_v2_topology,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.training.simple_stateless_pretrain_data import ReplaySplit, file_sha256
from ptcg_rl.training.simple_stateless_pretrain_weighting import (
    ReplayPretrainingOutcomeWeightsConfig,
)

LEGACY_SUPERVISED_POLICY_ARTIFACT_SCHEMA: Final[
    Literal["simple-stateless-supervised-policy-v1"]
] = "simple-stateless-supervised-policy-v1"
SUPERVISED_POLICY_ARTIFACT_SCHEMA: Final[
    Literal["simple-stateless-supervised-policy-v2"]
] = "simple-stateless-supervised-policy-v2"
TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA: Final[
    Literal["simple-stateless-supervised-policy-v3"]
] = "simple-stateless-supervised-policy-v3"
WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA: Final[
    Literal["simple-stateless-supervised-policy-v4"]
] = "simple-stateless-supervised-policy-v4"
POLICY_ACTION_TYPE_BUCKETS: Final[tuple[str, ...]] = (
    "PLAY",
    "ABILITY",
    "ATTACH",
    "ATTACK",
    "RETREAT",
    "END",
    "OTHER",
)
BASELINE_POLICY_FORWARD_KL_SEMANTICS: Final[
    Literal["initialization-baseline-visited-prefix-forward-kl-v1"]
] = "initialization-baseline-visited-prefix-forward-kl-v1"
OUTCOME_WEIGHTING_SEMANTICS: Final[
    Literal["acting-side-outcome-weighted-episode-mean-v1"]
] = "acting-side-outcome-weighted-episode-mean-v1"
OUTCOME_BUCKETS: Final[tuple[str, ...]] = ("win", "draw", "loss")


class PrivateParameterBankTrainingRecord(BaseModel):
    """Observed movement within one route-keyed physical parameter bank."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bank_name: str = Field(min_length=1)
    tensor_count: int = Field(ge=1)
    parameter_elements: int = Field(ge=1)
    trainable_tensor_count: int = Field(ge=0)
    trainable_parameter_elements: int = Field(ge=0)
    changed_tensor_count: int = Field(ge=0)
    parameter_delta_l2: float = Field(ge=0.0)
    initial_fingerprint: str
    final_fingerprint: str

    @field_validator("initial_fingerprint", "final_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require an immutable bank-state fingerprint."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_counts(self) -> Self:
        """Keep bank movement and trainable ownership internally consistent."""
        if self.trainable_tensor_count > self.tensor_count:
            raise ValueError("trainable tensor count exceeds bank tensor count")
        if self.trainable_parameter_elements > self.parameter_elements:
            raise ValueError("trainable parameter count exceeds bank parameter count")
        if self.changed_tensor_count > self.tensor_count:
            raise ValueError("changed tensor count exceeds bank tensor count")
        if self.changed_tensor_count == 0 and (
            self.parameter_delta_l2 != 0.0
            or self.initial_fingerprint != self.final_fingerprint
        ):
            raise ValueError("unchanged bank reports parameter movement")
        if self.changed_tensor_count > 0 and (
            self.initial_fingerprint == self.final_fingerprint
        ):
            raise ValueError("changed bank is missing movement evidence")
        return self


class TrainableParameterScopeAudit(BaseModel):
    """Validated optimizer whitelist and global frozen-state invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["full_model", "exact_actor_private_v2"]
    target_deck_digest: str | None = None
    target_expert_id: str | None = None
    trainable_parameter_names: tuple[str, ...]
    trainable_tensor_count: int = Field(ge=1)
    trainable_parameter_elements: int = Field(ge=1)
    frozen_tensor_count: int = Field(ge=0)
    frozen_initial_fingerprint: str
    frozen_final_fingerprint: str
    frozen_bitwise_identical: bool

    @field_validator("target_deck_digest", "target_expert_id")
    @classmethod
    def valid_optional_identity(cls, value: str | None) -> str | None:
        """Validate exact route identities when private mode owns the update."""
        return None if value is None else _validate_sha256(value)

    @field_validator("frozen_initial_fingerprint", "frozen_final_fingerprint")
    @classmethod
    def valid_frozen_fingerprint(cls, value: str) -> str:
        """Require complete frozen-state identities."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_scope(self) -> Self:
        """Fail closed on ambiguous ownership or a changed frozen tensor."""
        if (
            len(self.trainable_parameter_names) != self.trainable_tensor_count
            or tuple(sorted(set(self.trainable_parameter_names)))
            != self.trainable_parameter_names
        ):
            raise ValueError("trainable parameter names must be unique and sorted")
        if self.mode == "full_model":
            if self.target_deck_digest is not None or self.target_expert_id is not None:
                raise ValueError("full-model audit cannot name an exact target")
        elif self.target_deck_digest is None or self.target_expert_id is None:
            raise ValueError("exact-private audit requires a complete target identity")
        if (
            not self.frozen_bitwise_identical
            or self.frozen_initial_fingerprint != self.frozen_final_fingerprint
        ):
            raise ValueError("frozen model state changed during supervised training")
        return self


class SupervisedPolicyOutcomeWeightingRecord(BaseModel):
    """Split-specific effective mass for acting-side outcome weighting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantics: Literal["acting-side-outcome-weighted-episode-mean-v1"] = (
        OUTCOME_WEIGHTING_SEMANTICS
    )
    split: ReplaySplit
    multipliers: ReplayPretrainingOutcomeWeightsConfig
    episodes: int = Field(ge=1)
    source_example_weight_sum: float = Field(gt=0.0)
    episode_counts: dict[str, int]
    raw_episode_weight_sums: dict[str, float]
    normalized_episode_weight_sums: dict[str, float]
    effective_per_episode_multipliers: dict[str, float]

    @field_validator(
        "source_example_weight_sum",
    )
    @classmethod
    def finite_positive_sum(cls, value: float) -> float:
        """Require finite source normalization evidence."""
        if not math.isfinite(value):
            raise ValueError("outcome weighting source weight sum must be finite")
        return value

    @field_validator("episode_counts")
    @classmethod
    def complete_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require every terminal outcome, including an empty bucket."""
        if set(value) != set(OUTCOME_BUCKETS) or any(
            isinstance(count, bool) or count < 0 for count in value.values()
        ):
            raise ValueError("outcome weighting episode counts are invalid")
        return value

    @field_validator(
        "raw_episode_weight_sums",
        "normalized_episode_weight_sums",
        "effective_per_episode_multipliers",
    )
    @classmethod
    def complete_finite_weights(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Require finite weights for every supported terminal outcome."""
        if set(value) != set(OUTCOME_BUCKETS) or any(
            not math.isfinite(weight) or weight < 0.0 for weight in value.values()
        ):
            raise ValueError("outcome weighting effective weights are invalid")
        return value

    @model_validator(mode="after")
    def coherent_effective_weights(self) -> Self:
        """Bind reported masses to the declared normalized objective."""
        if sum(self.episode_counts.values()) != self.episodes:
            raise ValueError("outcome weighting episode count differs from total")
        if not math.isclose(
            self.source_example_weight_sum,
            float(self.episodes),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError("source example weights are not episode normalized")
        declared = self.multipliers.model_dump(mode="python")
        expected_raw = {
            outcome: self.episode_counts[outcome] * float(declared[outcome])
            for outcome in OUTCOME_BUCKETS
        }
        if any(
            not math.isclose(
                self.raw_episode_weight_sums[outcome],
                expected_raw[outcome],
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            for outcome in OUTCOME_BUCKETS
        ):
            raise ValueError("raw outcome weights differ from declaration")
        raw_total = sum(expected_raw.values())
        if raw_total <= 0.0:
            raise ValueError("outcome weighting has no effective episode mass")
        expected_normalized = {
            outcome: expected_raw[outcome] / raw_total for outcome in OUTCOME_BUCKETS
        }
        if any(
            not math.isclose(
                self.normalized_episode_weight_sums[outcome],
                expected_normalized[outcome],
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            for outcome in OUTCOME_BUCKETS
        ):
            raise ValueError("normalized outcome weights are incoherent")
        mean_multiplier = raw_total / self.episodes
        expected_effective = {
            outcome: float(declared[outcome]) / mean_multiplier
            for outcome in OUTCOME_BUCKETS
        }
        if any(
            not math.isclose(
                self.effective_per_episode_multipliers[outcome],
                expected_effective[outcome],
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            for outcome in OUTCOME_BUCKETS
        ):
            raise ValueError("per-episode outcome multipliers are incoherent")
        return self


def supervised_outcome_weighting_record(
    *,
    split: ReplaySplit,
    multipliers: ReplayPretrainingOutcomeWeightsConfig,
    episode_outcomes: Mapping[Any, float],
    source_example_weight_sum: float,
) -> SupervisedPolicyOutcomeWeightingRecord:
    """Build the normalized effective mass for one immutable split."""
    counts = dict.fromkeys(OUTCOME_BUCKETS, 0)
    for outcome in episode_outcomes.values():
        counts[_outcome_bucket(outcome)] += 1
    episodes = len(episode_outcomes)
    if episodes <= 0:
        raise ValueError("outcome weighting requires terminal episodes")
    declared = multipliers.model_dump(mode="python")
    raw = {
        outcome: counts[outcome] * float(declared[outcome])
        for outcome in OUTCOME_BUCKETS
    }
    raw_total = sum(raw.values())
    if raw_total <= 0.0 or not math.isfinite(raw_total):
        raise ValueError("outcome weighting has invalid total mass")
    mean_multiplier = raw_total / episodes
    return SupervisedPolicyOutcomeWeightingRecord(
        split=split,
        multipliers=multipliers,
        episodes=episodes,
        source_example_weight_sum=source_example_weight_sum,
        episode_counts=counts,
        raw_episode_weight_sums=raw,
        normalized_episode_weight_sums={
            outcome: raw[outcome] / raw_total for outcome in OUTCOME_BUCKETS
        },
        effective_per_episode_multipliers={
            outcome: float(declared[outcome]) / mean_multiplier
            for outcome in OUTCOME_BUCKETS
        },
    )


def _outcome_bucket(outcome: float) -> str:
    """Map an acting-side terminal value without accepting unknown outcomes."""
    if outcome == 1.0:
        return "win"
    if outcome == 0.0:
        return "draw"
    if outcome == -1.0:
        return "loss"
    raise ValueError(f"unsupported pretraining outcome: {outcome}")


class SupervisedPolicyEvaluationMetrics(BaseModel):
    """Streaming held-out policy metrics with episode-equal aggregation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: ReplaySplit
    examples: int = Field(ge=1)
    episodes: int = Field(ge=1)
    decode_tokens: int = Field(ge=1)
    episode_normalized_policy_nll: float = Field(ge=0.0)
    outcome_weighted_episode_policy_nll: float | None = Field(
        default=None,
        ge=0.0,
    )
    decision_policy_nll: float = Field(ge=0.0)
    token_policy_nll: float = Field(ge=0.0)
    exact_action_sequence_accuracy: float = Field(ge=0.0, le=1.0)
    episode_normalized_baseline_policy_forward_kl: float | None = Field(
        default=None,
        ge=0.0,
    )
    decision_baseline_policy_forward_kl: float | None = Field(
        default=None,
        ge=0.0,
    )
    outcome_weighting: SupervisedPolicyOutcomeWeightingRecord | None = None
    action_types: dict[str, SupervisedPolicyActionTypeMetrics]

    @field_validator(
        "episode_normalized_policy_nll",
        "decision_policy_nll",
        "token_policy_nll",
        "exact_action_sequence_accuracy",
    )
    @classmethod
    def finite_metric(cls, value: float) -> float:
        """Reject non-finite selection evidence."""
        if not math.isfinite(value):
            raise ValueError("supervised evaluation metric must be finite")
        return value

    @field_validator(
        "outcome_weighted_episode_policy_nll",
        "episode_normalized_baseline_policy_forward_kl",
        "decision_baseline_policy_forward_kl",
    )
    @classmethod
    def finite_optional_metric(cls, value: float | None) -> float | None:
        """Reject non-finite optional baseline-anchor evidence."""
        if value is not None and not math.isfinite(value):
            raise ValueError("supervised baseline KL metric must be finite")
        return value

    @model_validator(mode="after")
    def coherent_baseline_metrics(self) -> Self:
        """Publish complete baseline and outcome-aware evidence."""
        metrics = (
            self.episode_normalized_baseline_policy_forward_kl,
            self.decision_baseline_policy_forward_kl,
        )
        if any(value is None for value in metrics) and any(
            value is not None for value in metrics
        ):
            raise ValueError("supervised baseline KL metrics are incomplete")
        outcome_metrics = (
            self.outcome_weighted_episode_policy_nll,
            self.outcome_weighting,
        )
        if any(value is None for value in outcome_metrics) and any(
            value is not None for value in outcome_metrics
        ):
            raise ValueError("supervised outcome-weighted metrics are incomplete")
        if (
            self.outcome_weighting is not None
            and self.outcome_weighting.split != self.split
        ):
            raise ValueError("outcome weighting split differs from evaluation")
        return self

    @field_validator("action_types")
    @classmethod
    def complete_action_types(
        cls,
        value: dict[str, SupervisedPolicyActionTypeMetrics],
    ) -> dict[str, SupervisedPolicyActionTypeMetrics]:
        """Always report the six critical macro types plus other actions."""
        if set(value) != set(POLICY_ACTION_TYPE_BUCKETS):
            raise ValueError("supervised action-type metrics are incomplete")
        return value


class SupervisedPolicyActionTypeMetrics(BaseModel):
    """Held-out sequence metrics for one teacher first-action type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int = Field(ge=0)
    policy_nll: float | None = Field(default=None, ge=0.0)
    exact_action_sequence_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def coherent_rows(self) -> Self:
        """Keep empty and populated type buckets unambiguous."""
        metrics = (
            self.policy_nll,
            self.exact_action_sequence_accuracy,
        )
        if self.rows == 0 and any(value is not None for value in metrics):
            raise ValueError("empty action-type bucket has metrics")
        if self.rows > 0 and any(value is None for value in metrics):
            raise ValueError("populated action-type bucket is missing metrics")
        if any(value is not None and not math.isfinite(value) for value in metrics):
            raise ValueError("action-type metric must be finite")
        return self


class SupervisedPolicySelectionRecord(BaseModel):
    """Best-checkpoint selection bound to one immutable split assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_assignment_fingerprint: str
    selected_optimizer_step: int = Field(ge=0)
    selected_epoch: int = Field(ge=0)
    final_optimizer_steps: int = Field(ge=1)
    selected_model_state_fingerprint: str | None = None
    selected_checkpoint_sha256: str | None = None
    selected_checkpoint_size_bytes: int | None = Field(default=None, ge=1)
    validation: SupervisedPolicyEvaluationMetrics

    @field_validator(
        "split_assignment_fingerprint",
        "selected_model_state_fingerprint",
        "selected_checkpoint_sha256",
    )
    @classmethod
    def valid_optional_fingerprint(cls, value: str | None) -> str | None:
        """Require complete hashes when selection binding fields are present."""
        return None if value is None else _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_selection(self) -> Self:
        """Ensure validation alone selected an already observed checkpoint."""
        if self.validation.split != "validation":
            raise ValueError("best checkpoint must be selected on validation")
        if self.selected_optimizer_step > self.final_optimizer_steps:
            raise ValueError("selected checkpoint is after final optimizer state")
        binding = (
            self.selected_model_state_fingerprint,
            self.selected_checkpoint_sha256,
            self.selected_checkpoint_size_bytes,
        )
        if any(value is None for value in binding) and any(
            value is not None for value in binding
        ):
            raise ValueError("selected checkpoint binding is incomplete")
        return self


class SupervisedPolicyTrainMonitorSelectionRecord(BaseModel):
    """Best checkpoint selected on a deterministic in-training monitor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: Literal["train_monitor"] = "train_monitor"
    unvalidated: Literal[True] = True
    monitor_fingerprint: str
    selected_optimizer_step: int = Field(ge=0)
    selected_epoch: int = Field(ge=0)
    final_optimizer_steps: int = Field(ge=1)
    selected_model_state_fingerprint: str
    selected_checkpoint_sha256: str
    selected_checkpoint_size_bytes: int = Field(ge=1)
    initial: SupervisedPolicyEvaluationMetrics
    best: SupervisedPolicyEvaluationMetrics
    final: SupervisedPolicyEvaluationMetrics
    stop_reason: Literal[
        "patience_exhausted",
        "maximum_epochs",
        "maximum_steps",
    ]

    @field_validator(
        "monitor_fingerprint",
        "selected_model_state_fingerprint",
        "selected_checkpoint_sha256",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable monitor and checkpoint identities."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_selection(self) -> Self:
        """Bind the selected state to the lowest observed training NLL."""
        if self.selected_optimizer_step > self.final_optimizer_steps:
            raise ValueError("selected checkpoint is after final optimizer state")
        if any(
            metrics.split != "train"
            for metrics in (self.initial, self.best, self.final)
        ):
            raise ValueError("train-monitor selection contains a held-out split")
        if self.best.episode_normalized_policy_nll > min(
            self.initial.episode_normalized_policy_nll,
            self.final.episode_normalized_policy_nll,
        ):
            raise ValueError("selected train monitor is not the best recorded state")
        return self


class SupervisedPolicyFinalEpochSelectionRecord(BaseModel):
    """Explicit unvalidated selection of the completed train-only state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: Literal["final_epoch"] = "final_epoch"
    unvalidated: Literal[True] = True
    reason: Literal["explicit_train_only"] = "explicit_train_only"
    selected_optimizer_step: int = Field(ge=1)
    selected_epoch: int = Field(ge=1)
    final_optimizer_steps: int = Field(ge=1)
    selected_model_state_fingerprint: str

    @field_validator("selected_model_state_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Bind the unvalidated choice to one complete model state."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_selection(self) -> Self:
        """A train-only artifact may select only its final optimizer state."""
        if self.selected_optimizer_step != self.final_optimizer_steps:
            raise ValueError("train-only selection is not the final optimizer state")
        return self


class PrivateResidualTrainingRecord(BaseModel):
    """Observed parameter movement for one active exact strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    expert_id: str
    matched_examples: int = Field(ge=0)
    policy_parameter_delta_l2: float = Field(ge=0.0)
    value_parameter_delta_l2: float = Field(ge=0.0)
    parameter_banks: tuple[PrivateParameterBankTrainingRecord, ...] = ()

    @model_validator(mode="after")
    def unique_parameter_banks(self) -> Self:
        """Keep every physical route bank represented at most once."""
        names = tuple(record.bank_name for record in self.parameter_banks)
        if len(names) != len(set(names)):
            raise ValueError("private parameter bank records must be unique")
        return self


class SupervisedPolicyTopologyTransitionRecord(BaseModel):
    """Compact immutable evidence for a cross-topology policy initialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "generalist-sequence-v2-to-family-private-v3-audit-v1"
    ] = "generalist-sequence-v2-to-family-private-v3-audit-v1"
    declaration_fingerprint: str
    source_state_fingerprint: str
    target_state_fingerprint: str
    state_mapping_fingerprint: str
    initialized_state_fingerprint: str
    source_tensors: int = Field(gt=0)
    inherited_target_tensors: int = Field(gt=0)
    cloned_upper_target_tensors: int = Field(gt=0)
    initialized_exact_target_tensors: int = Field(ge=0)
    initialized_appended_target_tensors: int = Field(gt=0)
    initialized_tensors: int = Field(gt=0)

    @field_validator(
        "declaration_fingerprint",
        "source_state_fingerprint",
        "target_state_fingerprint",
        "state_mapping_fingerprint",
        "initialized_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable transition and state identities."""
        return _validate_sha256(value)


class SupervisedPolicyInitializationRecord(BaseModel):
    """Immutable provenance for the policy state before supervised updates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["random", "rl_pair"]
    source_pair_manifest_path: str | None = None
    source_pair_manifest_sha256: str | None = None
    source_pair_version: int | None = Field(default=None, ge=0)
    source_policy_path: str | None = None
    source_policy_sha256: str | None = None
    source_policy_model_fingerprint: str | None = None
    random_seed: int | None = Field(default=None, ge=0)
    initial_model_state_fingerprint: str | None = None
    topology_transition: SupervisedPolicyTopologyTransitionRecord | None = None

    @field_validator(
        "source_pair_manifest_sha256",
        "source_policy_sha256",
        "source_policy_model_fingerprint",
        "initial_model_state_fingerprint",
    )
    @classmethod
    def optional_sha256(cls, value: str | None) -> str | None:
        """Validate fingerprints when the initialization has a source."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("initialization identity must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_source(self) -> Self:
        """Keep random and pair-based provenance unambiguous."""
        source_values = (
            self.source_pair_manifest_path,
            self.source_pair_manifest_sha256,
            self.source_pair_version,
            self.source_policy_path,
            self.source_policy_sha256,
            self.source_policy_model_fingerprint,
        )
        if self.mode == "random" and any(value is not None for value in source_values):
            raise ValueError("random initialization cannot carry source artifacts")
        if self.mode == "random" and self.topology_transition is not None:
            raise ValueError("random initialization cannot carry a topology transition")
        if self.mode == "rl_pair" and any(value is None for value in source_values):
            raise ValueError("RL-pair initialization provenance is incomplete")
        if self.mode == "rl_pair" and self.random_seed is not None:
            raise ValueError("RL-pair initialization cannot carry a random seed")
        if self.mode == "rl_pair":
            if self.topology_transition is None:
                if self.initial_model_state_fingerprint is not None:
                    raise ValueError(
                        "same-topology RL initialization cannot carry a migrated state"
                    )
            elif (
                self.initial_model_state_fingerprint
                != self.topology_transition.target_state_fingerprint
                or self.source_policy_model_fingerprint
                != self.topology_transition.source_state_fingerprint
            ):
                raise ValueError(
                    "RL topology transition differs from initialization provenance"
                )
        random_values = (self.random_seed, self.initial_model_state_fingerprint)
        if self.mode == "random" and any(
            value is None for value in random_values
        ) and any(value is not None for value in random_values):
            raise ValueError("random initialization provenance is incomplete")
        return self


class SupervisedPolicyBaselineAnchorRecord(BaseModel):
    """Immutable source and objective semantics for baseline policy anchoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantics: Literal["initialization-baseline-visited-prefix-forward-kl-v1"] = (
        BASELINE_POLICY_FORWARD_KL_SEMANTICS
    )
    weight: float = Field(gt=0.0)
    source_pair_manifest_sha256: str
    source_policy_sha256: str
    source_policy_model_fingerprint: str

    @field_validator(
        "source_pair_manifest_sha256",
        "source_policy_sha256",
        "source_policy_model_fingerprint",
    )
    @classmethod
    def valid_source_identity(cls, value: str) -> str:
        """Require the exact RL initialization identities."""
        return _validate_sha256(value)

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        """Reject a non-finite anchor coefficient."""
        if not math.isfinite(value):
            raise ValueError("baseline policy KL weight must be finite")
        return value


class SupervisedPolicyArtifactManifest(BaseModel):
    """Complete identity required for a weights-only RL initialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "simple-stateless-supervised-policy-v1",
        "simple-stateless-supervised-policy-v2",
        "simple-stateless-supervised-policy-v3",
        "simple-stateless-supervised-policy-v4",
    ] = SUPERVISED_POLICY_ARTIFACT_SCHEMA
    policy_filename: str
    policy_size_bytes: int = Field(ge=1)
    policy_sha256: str
    model_state_fingerprint: str
    model_config_fingerprint: str
    exact_registry_fingerprint: str
    public_catalog_fingerprint: str
    input_contract_fingerprint: str
    event_contract_fingerprint: str | None = None
    sequence_contract_fingerprint: str | None = None
    dataset_manifest_sha256: str
    dataset_fingerprint: str
    optimizer_steps: int = Field(ge=1)
    completed_epochs: int = Field(ge=0)
    initialization: SupervisedPolicyInitializationRecord
    baseline_policy_anchor: SupervisedPolicyBaselineAnchorRecord | None = None
    outcome_weighting: SupervisedPolicyOutcomeWeightingRecord | None = None
    residual_training: tuple[PrivateResidualTrainingRecord, ...]
    trainable_scope: TrainableParameterScopeAudit | None = None
    selection: (
        SupervisedPolicySelectionRecord
        | SupervisedPolicyTrainMonitorSelectionRecord
        | SupervisedPolicyFinalEpochSelectionRecord
        | None
    ) = None

    @field_validator("policy_filename")
    @classmethod
    def local_policy_filename(cls, value: str) -> str:
        """Keep the payload relocatable beside its manifest."""
        if Path(value).name != value or not value.endswith(".pt"):
            raise ValueError("supervised policy filename is invalid")
        return value

    @field_validator(
        "policy_sha256",
        "model_state_fingerprint",
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "public_catalog_fingerprint",
        "input_contract_fingerprint",
        "dataset_manifest_sha256",
        "dataset_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str | None) -> str | None:
        """Require full artifact fingerprints."""
        return None if value is None else _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_schema(self) -> Self:
        """Require complete ownership evidence for newly published artifacts."""
        anchor = self.baseline_policy_anchor
        if anchor is not None:
            initialization = self.initialization
            if (
                initialization.mode != "rl_pair"
                or anchor.source_pair_manifest_sha256
                != initialization.source_pair_manifest_sha256
                or anchor.source_policy_sha256 != initialization.source_policy_sha256
                or anchor.source_policy_model_fingerprint
                != initialization.source_policy_model_fingerprint
            ):
                raise ValueError(
                    "baseline policy anchor differs from RL initialization"
                )
        if isinstance(
            self.selection,
            (
                SupervisedPolicySelectionRecord,
                SupervisedPolicyTrainMonitorSelectionRecord,
            ),
        ):
            validation = (
                self.selection.validation
                if isinstance(self.selection, SupervisedPolicySelectionRecord)
                else self.selection.best
            )
            has_anchor_metrics = (
                validation.episode_normalized_baseline_policy_forward_kl is not None
            )
            if has_anchor_metrics != (anchor is not None):
                raise ValueError(
                    "validation baseline KL differs from artifact objective"
                )
            if isinstance(self.selection, SupervisedPolicySelectionRecord):
                validation_weighting = validation.outcome_weighting
                if self.outcome_weighting is None:
                    if validation_weighting is not None:
                        raise ValueError(
                            "validation outcome weighting differs from "
                            "artifact objective"
                        )
                elif validation_weighting is None:
                    if self.format not in {
                        TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
                        WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
                    }:
                        raise ValueError(
                            "validation outcome weighting differs from "
                            "artifact objective"
                        )
                elif (
                    self.outcome_weighting.split != "train"
                    or validation_weighting.split != "validation"
                    or validation_weighting.multipliers
                    != self.outcome_weighting.multipliers
                ):
                    raise ValueError(
                        "validation outcome weighting differs from artifact objective"
                    )
        if self.format in {
            SUPERVISED_POLICY_ARTIFACT_SCHEMA,
            TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
            WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
        }:
            if self.trainable_scope is None:
                raise ValueError("v2 supervised artifact is missing scope audit")
            if (
                self.trainable_scope.mode == "exact_actor_private_v2"
                and self.initialization.mode != "rl_pair"
            ):
                raise ValueError(
                    "exact actor-private artifact is not bound to an RL pair"
                )
            if self.trainable_scope.mode == "exact_actor_private_v2" and not isinstance(
                self.selection,
                (
                    SupervisedPolicySelectionRecord,
                    SupervisedPolicyFinalEpochSelectionRecord,
                ),
            ):
                raise ValueError(
                    "exact actor-private artifact has no declared selection"
                )
            if isinstance(
                self.selection,
                SupervisedPolicyFinalEpochSelectionRecord,
            ) and not (
                (
                    self.format == SUPERVISED_POLICY_ARTIFACT_SCHEMA
                    and self.trainable_scope.mode == "exact_actor_private_v2"
                )
                or (
                    self.format == WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA
                    and self.trainable_scope.mode == "full_model"
                )
            ):
                raise ValueError(
                    "train-only final selection requires a v2 exact-private or "
                    "v4 temporal full-model artifact"
                )
            if not self.residual_training or any(
                not record.parameter_banks for record in self.residual_training
            ):
                raise ValueError("v2 supervised artifact is missing bank audits")
            if self.selection is not None and (
                self.selection.final_optimizer_steps != self.optimizer_steps
                or self.selection.selected_epoch > self.completed_epochs
            ):
                raise ValueError("supervised selection differs from training record")
            if (
                self.selection is not None
                and self.selection.selected_model_state_fingerprint is not None
                and self.selection.selected_model_state_fingerprint
                != self.model_state_fingerprint
            ):
                raise ValueError("selected model differs from published policy")
            if (
                isinstance(
                    self.selection,
                    SupervisedPolicyFinalEpochSelectionRecord,
                )
                and self.selection.selected_epoch != self.completed_epochs
            ):
                raise ValueError("train-only selection is not the final epoch")
            if (
                self.trainable_scope.mode == "full_model"
                and self.selection is not None
                and self.selection.selected_model_state_fingerprint is None
            ):
                raise ValueError("full-model selection is missing checkpoint binding")
            if self.format == TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA:
                if (
                    self.initialization.mode != "random"
                    or self.initialization.random_seed is None
                    or self.initialization.initial_model_state_fingerprint is None
                ):
                    raise ValueError(
                        "v3 supervised artifact requires random initialization"
                    )
                if not isinstance(
                    self.selection,
                    (
                        SupervisedPolicySelectionRecord,
                        SupervisedPolicyTrainMonitorSelectionRecord,
                    ),
                ):
                    raise ValueError(
                        "v3 supervised artifact requires checkpoint selection"
                    )
                if (
                    self.event_contract_fingerprint is None
                    or self.sequence_contract_fingerprint is None
                ):
                    raise ValueError(
                        "v3 supervised artifact is missing temporal contracts"
                    )
            elif self.format == WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA:
                if self.initialization.mode != "rl_pair" or not isinstance(
                    self.selection,
                    (
                        SupervisedPolicySelectionRecord,
                        SupervisedPolicyFinalEpochSelectionRecord,
                    ),
                ):
                    raise ValueError(
                        "v4 supervised artifact requires pair initialization "
                        "and validation or explicit final-epoch selection"
                    )
                if (
                    self.event_contract_fingerprint is None
                    or self.sequence_contract_fingerprint is None
                ):
                    raise ValueError(
                        "v4 supervised artifact is missing temporal contracts"
                    )
            elif (
                self.event_contract_fingerprint is not None
                or self.sequence_contract_fingerprint is not None
                or isinstance(
                    self.selection,
                    SupervisedPolicyTrainMonitorSelectionRecord,
                )
            ):
                raise ValueError("v2 supervised artifact carries v3 metadata")
        elif (
            self.trainable_scope is not None
            or self.selection is not None
            or self.event_contract_fingerprint is not None
            or self.sequence_contract_fingerprint is not None
        ):
            raise ValueError("legacy supervised artifact carries v2 audit metadata")
        return self


def load_supervised_policy_artifact(
    manifest_path: Path,
    *,
    expected_model_config: SimpleStatelessModelConfig,
    expected_exact_registry_fingerprint: str,
    expected_public_catalog_fingerprint: str,
    expected_input_contract_fingerprint: str,
    expected_event_contract_fingerprint: str | None = None,
    expected_sequence_contract_fingerprint: str | None = None,
) -> tuple[
    SupervisedPolicyArtifactManifest,
    Mapping[str, Tensor],
]:
    """Load policy tensors only after all RL input/topology identities match."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = SupervisedPolicyArtifactManifest.model_validate(json.load(handle))
    expected = (
        model_config_fingerprint(expected_model_config),
        expected_exact_registry_fingerprint,
        expected_public_catalog_fingerprint,
        expected_input_contract_fingerprint,
    )
    actual = (
        manifest.model_config_fingerprint,
        manifest.exact_registry_fingerprint,
        manifest.public_catalog_fingerprint,
        manifest.input_contract_fingerprint,
    )
    if actual != expected:
        raise ValueError("supervised policy artifact is incompatible with RL")
    if manifest.format in {
        TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
        WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    } and (
        manifest.event_contract_fingerprint != expected_event_contract_fingerprint
        or manifest.sequence_contract_fingerprint
        != expected_sequence_contract_fingerprint
    ):
        raise ValueError("supervised policy temporal contract is incompatible with RL")
    policy_path = manifest_path.parent / manifest.policy_filename
    if (
        not policy_path.is_file()
        or policy_path.stat().st_size != manifest.policy_size_bytes
        or file_sha256(policy_path) != manifest.policy_sha256
    ):
        raise ValueError("supervised policy payload failed fingerprint checks")
    payload = torch.load(policy_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("supervised policy payload must be a mapping")
    if payload.get("format") != manifest.format:
        raise ValueError("supervised policy payload schema differs from manifest")
    raw_state = payload.get("model_state")
    raw_config = payload.get("model_config")
    if not isinstance(raw_state, Mapping) or not isinstance(raw_config, Mapping):
        raise TypeError("supervised policy payload is incomplete")
    payload_config = SimpleStatelessModelConfig.model_validate(dict(raw_config))
    if payload_config != expected_model_config:
        raise ValueError("supervised policy payload topology changed")
    state = {
        str(key): value
        for key, value in raw_state.items()
        if isinstance(key, str) and isinstance(value, Tensor)
    }
    if len(state) != len(raw_state):
        raise TypeError("supervised policy state contains invalid entries")
    if canonical_model_state_fingerprint(state) != manifest.model_state_fingerprint:
        raise ValueError("supervised policy model-state fingerprint changed")
    probe = materialize_simple_stateless_checkpoint_model(
        expected_model_config,
        state,
    )
    _validate_scope_audit_against_state(manifest, probe, state)
    return (manifest, state)


def _validate_scope_audit_against_state(
    manifest: SupervisedPolicyArtifactManifest,
    model: SimpleStatelessPolicyValueNet,
    state: Mapping[str, Tensor],
) -> None:
    """Recompute every final-state fact asserted by the V2 artifact audit."""
    scope = manifest.trainable_scope
    if scope is None:
        return
    named_parameters = dict(model.named_parameters())
    if scope.mode == "full_model":
        expected_trainable = tuple(sorted(named_parameters))
        target_route = None
    else:
        if not uses_exact_v2_topology(model.config):
            raise ValueError("exact actor-private artifact requires V2 exact routes")
        matches = tuple(
            route
            for route in model.config.exact_routes
            if (
                route.deck_digest == scope.target_deck_digest
                and route.expert_id == scope.target_expert_id
            )
        )
        if len(matches) != 1:
            raise ValueError("supervised scope target differs from model registry")
        target_route = matches[0]
        expected_trainable = _exact_actor_parameter_names(
            model,
            module_key=target_route.module_key,
        )
    if scope.trainable_parameter_names != expected_trainable:
        raise ValueError("supervised trainable whitelist differs from model topology")
    parameter_elements = sum(
        named_parameters[name].numel() for name in expected_trainable
    )
    if parameter_elements != scope.trainable_parameter_elements:
        raise ValueError("supervised trainable parameter count changed")
    frozen_names = tuple(
        name for name in sorted(state) if name not in frozenset(expected_trainable)
    )
    if len(frozen_names) != scope.frozen_tensor_count:
        raise ValueError("supervised frozen tensor count changed")
    frozen_fingerprint = _state_subset_fingerprint(state, frozen_names)
    if frozen_fingerprint != scope.frozen_final_fingerprint:
        raise ValueError("supervised frozen-state fingerprint changed")
    expected_routes = tuple(
        (route.deck_digest, route.expert_id) for route in model.config.exact_routes
    )
    actual_routes = tuple(
        (record.deck_digest, record.expert_id) for record in manifest.residual_training
    )
    if actual_routes != expected_routes:
        raise ValueError("supervised private-bank audit differs from model routes")
    bank_inventory = _private_bank_state_names(model)
    trainable_names = frozenset(expected_trainable)
    for route, record in zip(
        model.config.exact_routes,
        manifest.residual_training,
        strict=True,
    ):
        expected_banks = bank_inventory[route.module_key]
        if tuple(bank.bank_name for bank in record.parameter_banks) != tuple(
            expected_banks
        ):
            raise ValueError("supervised private-bank inventory changed")
        for bank_record in record.parameter_banks:
            names = expected_banks[bank_record.bank_name]
            final_bank = {name: state[name] for name in names}
            selected = tuple(name for name in names if name in trainable_names)
            if (
                bank_record.tensor_count != len(names)
                or bank_record.parameter_elements
                != sum(state[name].numel() for name in names)
                or bank_record.trainable_tensor_count != len(selected)
                or bank_record.trainable_parameter_elements
                != sum(state[name].numel() for name in selected)
                or bank_record.final_fingerprint
                != canonical_model_state_fingerprint(final_bank)
            ):
                raise ValueError("supervised private-bank audit changed")
    if target_route is not None:
        target_record = next(
            record
            for record in manifest.residual_training
            if record.deck_digest == target_route.deck_digest
        )
        if (
            target_record.matched_examples <= 0
            or target_record.policy_parameter_delta_l2 <= 0.0
            or target_record.value_parameter_delta_l2 != 0.0
        ):
            raise ValueError("exact actor-private artifact has invalid target movement")


def _exact_actor_parameter_names(
    model: SimpleStatelessPolicyValueNet,
    *,
    module_key: str,
) -> tuple[str, ...]:
    """Reconstruct the only authorized V2 actor-private whitelist."""
    banks = _private_bank_state_names(model)[module_key]
    names = tuple(
        sorted(
            name
            for bank_names in banks.values()
            for name in bank_names
            if _is_exact_actor_trainable_name(name)
        )
    )
    if not names or any(name not in dict(model.named_parameters()) for name in names):
        raise ValueError("supervised actor-private whitelist is incomplete")
    return names


def _private_bank_state_names(
    model: SimpleStatelessPolicyValueNet,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Enumerate route-private banks independently while loading an artifact."""
    module_keys = tuple(route.module_key for route in model.config.exact_routes)
    inventory: dict[str, dict[str, tuple[str, ...]]] = {
        module_key: {} for module_key in module_keys
    }
    state_names = frozenset(model.state_dict())
    for bank_name, bank in model.route_private_banks():
        if tuple(sorted(bank)) != tuple(sorted(module_keys)):
            raise ValueError("supervised private bank differs from model registry")
        for module_key in module_keys:
            names = tuple(
                sorted(
                    f"{bank_name}.{module_key}.{local_name}"
                    for local_name in bank[module_key].state_dict()
                )
            )
            if not names or any(name not in state_names for name in names):
                raise ValueError("supervised private bank has incomplete model state")
            inventory[module_key][bank_name] = names
    return inventory


def _is_exact_actor_trainable_name(name: str) -> bool:
    """Recognize prompt, staged residual, option, and query actor parameters."""
    if name.startswith(("heads.policy_residuals.", "heads.option_residuals.")):
        return True
    if name.startswith("backbone.v2_adapters.prompts."):
        return name.endswith(".policy_and_scratch")
    return ".exact_capsules." in name and ".policy_residual." in name


def _state_subset_fingerprint(
    state: Mapping[str, Tensor],
    names: tuple[str, ...],
) -> str:
    """Hash a named state subset with the pretrainer's empty-set identity."""
    selected = {name: state[name] for name in names}
    if selected:
        return canonical_model_state_fingerprint(selected)
    return hashlib.sha256(
        b"ptcg-rl/simple-stateless-pretraining-empty-state/v1\x00"
    ).hexdigest()


def supervised_policy_payload(
    model: SimpleStatelessPolicyValueNet,
    *,
    artifact_format: Literal[
        "simple-stateless-supervised-policy-v2",
        "simple-stateless-supervised-policy-v3",
        "simple-stateless-supervised-policy-v4",
    ] = SUPERVISED_POLICY_ARTIFACT_SCHEMA,
) -> dict[str, Any]:
    """Return a CPU policy-only checkpoint payload."""
    return {
        "format": artifact_format,
        "model_config": model.config.model_dump(mode="json"),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def _validate_sha256(value: str) -> str:
    """Normalize one SHA-256 identity shared by artifact audit records."""
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("supervised artifact identity must be SHA-256")
    return normalized


__all__ = [
    "BASELINE_POLICY_FORWARD_KL_SEMANTICS",
    "LEGACY_SUPERVISED_POLICY_ARTIFACT_SCHEMA",
    "OUTCOME_BUCKETS",
    "OUTCOME_WEIGHTING_SEMANTICS",
    "POLICY_ACTION_TYPE_BUCKETS",
    "PrivateParameterBankTrainingRecord",
    "PrivateResidualTrainingRecord",
    "SUPERVISED_POLICY_ARTIFACT_SCHEMA",
    "TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA",
    "WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA",
    "SupervisedPolicyArtifactManifest",
    "SupervisedPolicyActionTypeMetrics",
    "SupervisedPolicyBaselineAnchorRecord",
    "SupervisedPolicyEvaluationMetrics",
    "SupervisedPolicyFinalEpochSelectionRecord",
    "SupervisedPolicyInitializationRecord",
    "SupervisedPolicyOutcomeWeightingRecord",
    "SupervisedPolicySelectionRecord",
    "SupervisedPolicyTrainMonitorSelectionRecord",
    "SupervisedPolicyTopologyTransitionRecord",
    "TrainableParameterScopeAudit",
    "load_supervised_policy_artifact",
    "supervised_outcome_weighting_record",
    "supervised_policy_payload",
]
