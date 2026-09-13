"""Validated Hydra configuration for simple-stateless replay pretraining."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.model.simple_stateless.config import uses_generalist_sequence
from ptcg_rl.rl.stateless_family_private_transition import (
    StatelessFamilyPrivateTransitionDeclaration,
)
from ptcg_rl.rl.stateless_training_config import SimpleStatelessTrainingConfig
from ptcg_rl.training.simple_stateless_pretrain_weighting import (
    ReplayPretrainingOutcomeWeightsConfig,
)


class ReplayPretrainingDataConfig(BaseModel):
    """Source replay corpus and compact extraction settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_root: Path
    source_manifest_path: Path
    replay_manifest_path: Path
    source_selection: Literal[
        "team_allowlist", "episode_team_bindings", "all_sides"
    ] = "team_allowlist"
    top_teams_path: Path | None = None
    episode_teams_path: Path | None = None
    dataset_dir: Path | None = None
    shard_rows: int = Field(default=4096, ge=1)
    shard_uncompressed_bytes: int = Field(default=256 << 20, ge=1 << 20)
    extraction_workers: int = Field(default=16, ge=1, le=64)
    extraction_batch_replays: int = Field(default=1, ge=1, le=64)
    pending_replays: int = Field(default=32, ge=1, le=4096)
    maximum_inflight_source_bytes: int = Field(default=1 << 30, ge=1 << 20)
    maximum_ready_result_bytes: int = Field(default=4 << 30, ge=1 << 20)
    compression_workers: int = Field(default=2, ge=1, le=2)
    prefix_bytes: int = Field(default=1 << 20, ge=4096)
    json_chunk_bytes: int = Field(default=1 << 20, ge=4096)
    maximum_replays: int | None = Field(default=None, ge=1)
    require_done_status: bool = True
    drop_forced_actions: bool = True
    temporal_sequence: bool = False
    verify_replay_sizes: bool = True
    replay_attempts: int = Field(default=2, ge=1, le=5)
    maximum_replay_errors: int = Field(default=256, ge=0)
    maximum_replay_error_fraction: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
    )
    replay_error_fraction_minimum_cursor: int = Field(default=1000, ge=1)
    expected_target_deck_digest: str | None = None
    resume: bool = True

    @model_validator(mode="after")
    def temporal_prompts_are_nonforced(self) -> Self:
        """Forced callbacks remain history but can never become BC targets."""
        if self.temporal_sequence and not self.drop_forced_actions:
            raise ValueError("temporal pretraining must drop forced-action targets")
        return self

    @model_validator(mode="after")
    def coherent_source_selection(self) -> Self:
        """Require a cohort file only when selecting an explicit team allowlist."""
        if self.source_selection in {"team_allowlist", "episode_team_bindings"}:
            if self.top_teams_path is None:
                raise ValueError("explicit source selection requires top_teams_path")
        elif self.top_teams_path is not None:
            raise ValueError("all-sides selection cannot name top_teams_path")
        if (
            self.source_selection == "episode_team_bindings"
        ) != (self.episode_teams_path is not None):
            raise ValueError(
                "episode-team selection and episode_teams_path must be set together"
            )
        return self

    @field_validator("expected_target_deck_digest")
    @classmethod
    def valid_expected_target_deck_digest(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize an optional single-deck corpus identity."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("expected target deck digest must be a lowercase SHA-256")
        return normalized


class ReplayPretrainingInitializationConfig(BaseModel):
    """Policy-only initialization for a new supervised optimizer lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["random", "rl_pair"] = "random"
    pair_manifest_path: Path | None = None
    family_private_transition: (
        StatelessFamilyPrivateTransitionDeclaration | None
    ) = None

    @model_validator(mode="after")
    def coherent_source(self) -> Self:
        """Require exactly the source artifact selected by the mode."""
        if self.mode == "random" and self.pair_manifest_path is not None:
            raise ValueError("random initialization cannot name an RL pair")
        if self.mode == "rl_pair" and self.pair_manifest_path is None:
            raise ValueError("RL-pair initialization requires a pair manifest")
        if self.mode == "random" and self.family_private_transition is not None:
            raise ValueError(
                "random initialization cannot declare a family-private transition"
            )
        return self


class ReplayPretrainingTrainableScopeConfig(BaseModel):
    """Fail-closed parameter ownership for one supervised optimizer lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["full_model", "exact_actor_private_v2"] = "full_model"
    target_deck_digest: str | None = None

    @field_validator("target_deck_digest")
    @classmethod
    def valid_optional_deck_digest(cls, value: str | None) -> str | None:
        """Normalize a configured exact-deck identity."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("target deck digest must be a lowercase SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_target(self) -> Self:
        """Bind exact-private mode to one immutable deck identity."""
        if self.mode == "full_model" and self.target_deck_digest is not None:
            raise ValueError("full-model pretraining cannot name a target deck")
        if self.mode == "exact_actor_private_v2" and self.target_deck_digest is None:
            raise ValueError("exact actor-private pretraining requires a target deck")
        return self


class ReplayPretrainingEvaluationConfig(BaseModel):
    """Training selection and held-out evaluation for a supervised artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["validation", "test"] = "validation"
    artifact_manifest_path: Path | None = None
    require_validation: bool = False
    train_only: bool = False

    @model_validator(mode="after")
    def coherent_training_selection(self) -> Self:
        """Keep held-out selection distinct from an explicit final-epoch run."""
        if self.require_validation and self.train_only:
            raise ValueError(
                "train-only pretraining cannot require validation examples"
            )
        return self


class ReplayPretrainingOptimizationConfig(BaseModel):
    """One-pass supervised optimization settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    minimum_learning_rate_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    warmup_steps: int = Field(default=500, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    adam_epsilon: float = Field(default=1.0e-8, gt=0.0)
    maximum_gradient_norm: float = Field(default=1.0, gt=0.0)
    policy_loss_weight: float = Field(default=1.0, ge=0.0)
    root_value_loss_weight: float = Field(default=0.25, ge=0.0)
    prefix_value_loss_weight: float = Field(default=0.10, ge=0.0)
    belief_loss_weight: float = Field(default=0.05, ge=0.0)
    baseline_policy_forward_kl_weight: float = Field(default=0.0, ge=0.0)
    episode_normalized_weighting: bool = True
    outcome_weights: ReplayPretrainingOutcomeWeightsConfig = Field(
        default_factory=ReplayPretrainingOutcomeWeightsConfig
    )
    shuffle_seed: int = Field(default=20260723, ge=0)
    log_interval_steps: int = Field(default=10, ge=1)
    checkpoint_interval_steps: int = Field(default=500, ge=1)
    validation_interval_epochs: int = Field(default=1, ge=1)
    maximum_steps: int | None = Field(default=None, ge=1)
    resume: bool = True
    minimum_epochs: int = Field(default=20, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    monitor_relative_improvement: float = Field(default=0.002, gt=0.0, lt=1.0)
    monitor_max_targets: int = Field(default=65_536, ge=1)
    target_chunk_decisions: int = Field(default=16, ge=1)
    maximum_batch_context_blocks: int = Field(default=1_600, ge=1)
    maximum_batch_context_state_tokens: int = Field(default=160_000, ge=1)
    maximum_batch_target_options: int = Field(default=4_096, ge=1)

    @field_validator(
        "learning_rate",
        "minimum_learning_rate_ratio",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "maximum_gradient_norm",
        "policy_loss_weight",
        "root_value_loss_weight",
        "prefix_value_loss_weight",
        "belief_loss_weight",
        "baseline_policy_forward_kl_weight",
        "monitor_relative_improvement",
    )
    @classmethod
    def finite_float(cls, value: float) -> float:
        """Reject non-finite optimizer and objective values."""
        if not math.isfinite(value):
            raise ValueError("pretraining numeric settings must be finite")
        return value

    @model_validator(mode="after")
    def non_empty_objective(self) -> Self:
        """Require at least one supervised signal."""
        if (
            self.policy_loss_weight
            + self.root_value_loss_weight
            + self.prefix_value_loss_weight
            + self.belief_loss_weight
            <= 0.0
        ):
            raise ValueError("pretraining objective cannot be empty")
        if (
            self.baseline_policy_forward_kl_weight > 0.0
            and self.policy_loss_weight <= 0.0
        ):
            raise ValueError("baseline policy KL requires behavior-cloning loss")
        if (
            not self.episode_normalized_weighting
            and self.outcome_weights != ReplayPretrainingOutcomeWeightsConfig()
        ):
            raise ValueError("outcome-aware weighting requires episode normalization")
        return self


class SimpleStatelessPretrainingConfig(SimpleStatelessTrainingConfig):
    """Current RL topology plus a replay-pretraining execution profile."""

    trainer: Literal["simple_stateless_pretraining"] = "simple_stateless_pretraining"  # type: ignore[assignment]
    stage: Literal[
        "pipeline",
        "extract",
        "train",
        "finalize",
        "validate",
        "evaluate",
    ] = "pipeline"
    data: ReplayPretrainingDataConfig
    initialization: ReplayPretrainingInitializationConfig = Field(
        default_factory=ReplayPretrainingInitializationConfig
    )
    trainable_scope: ReplayPretrainingTrainableScopeConfig = Field(
        default_factory=ReplayPretrainingTrainableScopeConfig
    )
    evaluation: ReplayPretrainingEvaluationConfig = Field(
        default_factory=ReplayPretrainingEvaluationConfig
    )
    optimization: ReplayPretrainingOptimizationConfig = Field(
        default_factory=ReplayPretrainingOptimizationConfig
    )

    @model_validator(mode="after")
    def fresh_supervised_lineage(self) -> Self:
        """Pretraining never consumes an RL exact-resume pair."""
        if (
            self.resume.mode != "fresh"
            or self.resume.pair_manifest_path is not None
            or self.resume.supervised_artifact_manifest_path is not None
        ):
            raise ValueError("replay pretraining requires a fresh RL topology")
        return self

    @model_validator(mode="after")
    def exact_actor_private_is_policy_only(self) -> Self:
        """Keep target-only actor cloning out of critic and belief graphs."""
        if (
            self.stage == "evaluate"
            or self.trainable_scope.mode != "exact_actor_private_v2"
        ):
            return self
        if self.initialization.mode != "rl_pair":
            raise ValueError(
                "exact actor-private pretraining requires an immutable RL pair"
            )
        if self.optimization.policy_loss_weight <= 0.0:
            raise ValueError("exact actor-private pretraining requires policy loss")
        if (
            self.optimization.root_value_loss_weight != 0.0
            or self.optimization.prefix_value_loss_weight != 0.0
            or self.optimization.belief_loss_weight != 0.0
        ):
            raise ValueError(
                "exact actor-private pretraining requires a policy-only objective"
            )
        return self

    @model_validator(mode="after")
    def train_only_scope_is_explicit(self) -> Self:
        """Allow final-state selection only for exact-private or temporal BC."""
        if not self.evaluation.train_only:
            return self
        valid_scope = self.trainable_scope.mode == "exact_actor_private_v2" or (
            self.data.temporal_sequence and self.trainable_scope.mode == "full_model"
        )
        if not valid_scope:
            raise ValueError(
                "train-only pretraining requires exact-private scope or "
                "temporal full-model scope"
            )
        return self

    @model_validator(mode="after")
    def baseline_anchor_uses_rl_pair(self) -> Self:
        """Bind behavior anchoring to the immutable initialization policy."""
        if (
            self.optimization.baseline_policy_forward_kl_weight > 0.0
            and self.initialization.mode != "rl_pair"
        ):
            raise ValueError("baseline policy KL requires RL-pair initialization")
        return self

    @model_validator(mode="after")
    def family_private_transition_is_full_model_v3(self) -> Self:
        """Confine the one-to-many RL migration to its declared target topology."""
        if self.initialization.family_private_transition is None:
            return self
        if (
            self.initialization.mode != "rl_pair"
            or self.model.architecture != "generalist_sequence_v3"
            or self.trainable_scope.mode != "full_model"
            or self.optimization.baseline_policy_forward_kl_weight != 0.0
        ):
            raise ValueError(
                "family-private transition requires unanchored, pair-initialized "
                "full-model v3"
            )
        return self

    @model_validator(mode="after")
    def coherent_evaluation_stage(self) -> Self:
        """Require an artifact only for explicit held-out evaluation."""
        has_artifact = self.evaluation.artifact_manifest_path is not None
        if self.stage == "evaluate" and not has_artifact:
            raise ValueError("evaluation stage requires a supervised artifact")
        if self.stage != "evaluate" and has_artifact:
            raise ValueError(
                "supervised evaluation artifact is only valid in evaluate stage"
            )
        if self.stage == "evaluate" and self.evaluation.train_only:
            raise ValueError("train-only selection is unavailable in evaluate stage")
        return self

    @model_validator(mode="after")
    def temporal_pretraining_contract(self) -> Self:
        """Keep causal BC on production sequence geometry and split semantics."""
        if not self.data.temporal_sequence:
            return self
        if not uses_generalist_sequence(self.model):
            raise ValueError("temporal pretraining requires generalist sequence")
        if self.trainable_scope.mode != "full_model":
            raise ValueError("temporal pretraining requires full-model scope")
        if (
            self.initialization.family_private_transition is not None
            and self.model.architecture != "generalist_sequence_v3"
        ):
            raise ValueError(
                "family-private transition requires generalist sequence v3"
            )
        if self.optimization.baseline_policy_forward_kl_weight != 0.0:
            raise ValueError("temporal pretraining cannot anchor another policy")
        if self.optimization.epochs < self.optimization.minimum_epochs:
            raise ValueError("temporal pretraining epochs are below its minimum")
        sequence = self.model.sequence
        if (
            sequence is None
            or self.optimization.target_chunk_decisions > sequence.learner_target_blocks
        ):
            raise ValueError("temporal target chunks exceed production geometry")
        if self.model.architecture in {
            "generalist_sequence_v2",
            "generalist_sequence_v3",
        } and not (
            self.evaluation.require_validation or self.evaluation.train_only
        ):
            raise ValueError(
                "temporal pretraining requires held-out validation or an "
                "explicit final full-data run"
            )
        return self


__all__ = [
    "ReplayPretrainingDataConfig",
    "ReplayPretrainingEvaluationConfig",
    "ReplayPretrainingInitializationConfig",
    "ReplayPretrainingOptimizationConfig",
    "ReplayPretrainingOutcomeWeightsConfig",
    "ReplayPretrainingTrainableScopeConfig",
    "SimpleStatelessPretrainingConfig",
]
