"""Prepare final-precision assets for the preregistered architecture comparison."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.bundle_gauntlet import (
    BundleAgentConfig,
    BundleGauntletConfig,
    BundleScheduleConfig,
    EvaluationBundleConfig,
)
from ptcg_rl.evaluation.deck_strength import DeckStrengthConfig
from ptcg_rl.evaluation.meta_distribution import MetaDistributionConfig
from ptcg_rl.evaluation.posterior import PosteriorEvaluationConfig
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic
from ptcg_rl.model import AgentNetworkConfig
from ptcg_rl.submission import RuntimeCheckpointExportConfig, export_runtime_checkpoint


class ComparisonDeckConfig(BaseModel):
    """One exact deployment-facing deck in the fixed comparison support."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    archetype: str
    path: Path
    sha256: str
    weight: float

    @field_validator("label", "archetype")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        """Reject blank bundle metadata."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("comparison deck identities must be non-empty")
        return cleaned

    @field_validator("weight")
    @classmethod
    def positive_weight(cls, value: float) -> float:
        """Require a finite positive primary-deck weight."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("comparison deck weights must be finite and positive")
        return value

    @field_validator("sha256")
    @classmethod
    def full_sha256(cls, value: str) -> str:
        """Require an immutable exact deck-file identity."""
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("comparison deck SHA256 must be a lowercase full hash")
        return value


class DeckConditioningComparisonConfig(BaseModel):
    """Immutable inputs and budgets for one baseline/candidate comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["DECK-CONDITIONING-DIRECT-COMPARISON-v1"]
    source_checkpoint: Path
    source_checkpoint_sha256: str
    expected_registry_sha256: str
    expected_final_version: int
    arm_a_checkpoint: Path
    arm_b_checkpoint: Path
    arm_a_runtime_checkpoint: Path
    arm_b_runtime_checkpoint: Path
    opponent_checkpoint: Path
    opponent_checkpoint_sha256: str
    static_features_path: Path
    static_features_sha256: str
    belief_summary_path: Path
    belief_summary_sha256: str
    side_observations_path: Path
    side_observations_sha256: str
    bundle_output_dir: Path
    score_output_dir: Path
    launch_dir: Path
    decks: tuple[ComparisonDeckConfig, ...]
    games_per_matchup: int
    evaluation_seed: int
    posterior_seed: int
    posterior_samples: int
    reference_date: date
    decision_rule: Literal["candidate_if_weighted_deploy_mean_delta_strictly_positive"]

    @field_validator(
        "source_checkpoint_sha256",
        "expected_registry_sha256",
        "opponent_checkpoint_sha256",
        "static_features_sha256",
        "belief_summary_sha256",
        "side_observations_sha256",
    )
    @classmethod
    def full_sha256(cls, value: str) -> str:
        """Require full lowercase SHA256 identities."""
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("comparison SHA256 values must be lowercase full hashes")
        return value

    @field_validator(
        "expected_final_version",
        "games_per_matchup",
        "posterior_samples",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Reject empty training or evaluation budgets."""
        if value <= 0:
            raise ValueError("comparison integer budgets must be positive")
        return value

    @model_validator(mode="after")
    def valid_deck_support(self) -> DeckConditioningComparisonConfig:
        """Require unique exact supports and normalized primary weights."""
        if not self.decks:
            raise ValueError("comparison requires at least one exact deck")
        labels = [deck.label for deck in self.decks]
        if len(set(labels)) != len(labels):
            raise ValueError("comparison deck labels must be unique")
        paths = [records.repo_path(deck.path).resolve() for deck in self.decks]
        if len(set(paths)) != len(paths):
            raise ValueError("comparison deck paths must be unique")
        if not math.isclose(
            sum(deck.weight for deck in self.decks),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("comparison deck weights must sum to one")
        if self.games_per_matchup % 2:
            raise ValueError("mirrored games_per_matchup must be even")
        return self


def prepare_deck_conditioning_comparison(
    config: DeckConditioningComparisonConfig,
) -> dict[str, Any]:
    """Validate final arms, export FP16 checkpoints, and publish runner JSON."""
    _require_sha256(
        config.source_checkpoint,
        config.source_checkpoint_sha256,
        label="source checkpoint",
    )
    _require_sha256(
        config.opponent_checkpoint,
        config.opponent_checkpoint_sha256,
        label="opponent checkpoint",
    )
    _require_sha256(
        config.static_features_path,
        config.static_features_sha256,
        label="static feature table",
    )
    _require_sha256(
        config.belief_summary_path,
        config.belief_summary_sha256,
        label="belief summary",
    )
    _require_sha256(
        config.side_observations_path,
        config.side_observations_sha256,
        label="side observations",
    )
    deck_paths = tuple(
        _require_sha256(deck.path, deck.sha256, label=f"comparison deck {deck.label}")
        for deck in config.decks
    )
    arm_a = _validate_final_checkpoint(
        config.arm_a_checkpoint,
        version=config.expected_final_version,
        expected_conditioning=False,
        expected_registry=None,
    )
    arm_b = _validate_final_checkpoint(
        config.arm_b_checkpoint,
        version=config.expected_final_version,
        expected_conditioning=True,
        expected_registry=config.expected_registry_sha256,
    )
    launch_dir = records.repo_path(config.launch_dir).resolve()
    bundle_path = launch_dir / "bundle_config.json"
    score_path = launch_dir / "score_config.json"
    summary_path = launch_dir / "prepared_assets.json"
    published_paths = (
        records.repo_path(config.arm_a_runtime_checkpoint),
        records.repo_path(config.arm_b_runtime_checkpoint),
        bundle_path,
        score_path,
        summary_path,
    )
    existing = [path for path in published_paths if path.exists()]
    if existing:
        raise FileExistsError(f"immutable comparison outputs already exist: {existing}")
    launch_dir.mkdir(parents=True, exist_ok=True)
    first_deck = deck_paths[0]
    arm_a_export = export_runtime_checkpoint(
        RuntimeCheckpointExportConfig(
            source_checkpoint=records.repo_path(config.arm_a_checkpoint),
            output_checkpoint=records.repo_path(config.arm_a_runtime_checkpoint),
            precision="fp16",
            deck_path=first_deck,
        )
    )
    arm_b_export = export_runtime_checkpoint(
        RuntimeCheckpointExportConfig(
            source_checkpoint=records.repo_path(config.arm_b_checkpoint),
            output_checkpoint=records.repo_path(config.arm_b_runtime_checkpoint),
            precision="fp16",
            deck_path=first_deck,
        )
    )
    for runtime_path in (
        config.arm_a_runtime_checkpoint,
        config.arm_b_runtime_checkpoint,
    ):
        for deck_path in deck_paths:
            CheckpointPolicy(
                runtime_path, device="cpu", own_deck=records.read_deck(deck_path)
            ).prewarm()
    bundle_config = build_deck_conditioning_bundle_config(config)
    score_config = build_deck_conditioning_score_config(config)
    _write_model_json(bundle_path, bundle_config)
    _write_model_json(score_path, score_config)
    summary = {
        "protocol": config.protocol,
        "source_checkpoint_sha256": config.source_checkpoint_sha256,
        "expected_registry_sha256": config.expected_registry_sha256,
        "expected_final_version": config.expected_final_version,
        "arm_a": {
            **arm_a,
            "runtime_checkpoint": records.display_path(
                records.repo_path(config.arm_a_runtime_checkpoint)
            ),
            "runtime_sha256": file_sha256(
                records.repo_path(config.arm_a_runtime_checkpoint)
            ),
            "export": dict(arm_a_export),
        },
        "arm_b": {
            **arm_b,
            "runtime_checkpoint": records.display_path(
                records.repo_path(config.arm_b_runtime_checkpoint)
            ),
            "runtime_sha256": file_sha256(
                records.repo_path(config.arm_b_runtime_checkpoint)
            ),
            "export": dict(arm_b_export),
        },
        "bundle_config": records.display_path(bundle_path),
        "score_config": records.display_path(score_path),
        "primary_deck_weights": {deck.label: deck.weight for deck in config.decks},
        "decision_rule": config.decision_rule,
    }
    write_identity_atomic(summary_path, summary)
    return summary


def build_deck_conditioning_bundle_config(
    config: DeckConditioningComparisonConfig,
) -> BundleGauntletConfig:
    """Build a host-balanced candidate order and fixed exact support."""
    belief_path = config.belief_summary_path
    arm_agents = {
        "arm_a_baseline": BundleAgentConfig(
            kind="runtime",
            checkpoint_path=config.arm_a_runtime_checkpoint,
            device="cuda",
            belief_summary_path=belief_path,
        ),
        "arm_b_conditioned": BundleAgentConfig(
            kind="runtime",
            checkpoint_path=config.arm_b_runtime_checkpoint,
            device="cuda",
            belief_summary_path=belief_path,
        ),
    }
    candidates = tuple(
        candidate
        for deck_index, deck in enumerate(config.decks)
        for candidate in (
            EvaluationBundleConfig(
                bundle_id=f"{pilot}_{deck.label}",
                pilot_id=pilot,
                archetype=deck.archetype,
                deck_path=deck.path,
                variant_weight=1.0,
                agent=arm_agents[pilot],
            )
            for pilot in (
                ("arm_a_baseline", "arm_b_conditioned")
                if deck_index % 2 == 0
                else ("arm_b_conditioned", "arm_a_baseline")
            )
        )
    )
    opponent_agent = BundleAgentConfig(
        kind="runtime",
        checkpoint_path=config.opponent_checkpoint,
        device="cuda",
        belief_summary_path=belief_path,
    )
    opponents = tuple(
        EvaluationBundleConfig(
            bundle_id=f"opponent_v24500_{deck.label}",
            pilot_id="opponent_v24500",
            archetype=deck.archetype,
            deck_path=deck.path,
            variant_weight=1.0,
            agent=opponent_agent,
        )
        for deck in config.decks
    )
    return BundleGauntletConfig(
        protocol="deck-conditioning-direct-comparison-v1",
        candidates=candidates,
        opponents=opponents,
        output_dir=config.bundle_output_dir,
        schedule=BundleScheduleConfig(
            games_per_matchup=config.games_per_matchup,
            mirror_sides=True,
        ),
        num_workers=8,
        max_steps_per_game=1_000,
        result_shard_size=16,
        seed=config.evaluation_seed,
        fail_on_error=False,
    )


def build_deck_conditioning_score_config(
    config: DeckConditioningComparisonConfig,
) -> DeckStrengthConfig:
    """Build the fixed recent-meta posterior configuration."""
    return DeckStrengthConfig(
        games_paths=(config.bundle_output_dir / "games.parquet",),
        side_observations_path=config.side_observations_path,
        output_dir=config.score_output_dir,
        meta=MetaDistributionConfig(
            recency_half_life_days=7.0,
            reference_date=config.reference_date,
            equalize_team_days=True,
        ),
        posterior=PosteriorEvaluationConfig(
            sample_count=config.posterior_samples,
            sample_batch_size=2_048,
            seed=config.posterior_seed,
            meta_concentration=200.0,
            top_k=3,
        ),
        pilot_weights={
            "arm_a_baseline": 0.5,
            "arm_b_conditioned": 0.5,
            "opponent_v24500": 1.0,
        },
        require_balanced_seats=True,
        require_explicit_multi_pilot_weights=True,
    )


def _validate_final_checkpoint(
    path: Path,
    *,
    version: int,
    expected_conditioning: bool,
    expected_registry: str | None,
) -> dict[str, Any]:
    resolved = _required_file(path, "final arm checkpoint")
    if resolved.name != f"policy_v{version}.pt":
        raise ValueError(
            f"final checkpoint name does not bind version {version}: {resolved}"
        )
    checkpoint = torch.load(resolved, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_config"), dict
    ):
        raise ValueError(f"final checkpoint lacks portable model_config: {resolved}")
    model_config = AgentNetworkConfig.model_validate(checkpoint["model_config"])
    conditioning = model_config.deck_conditioning
    enabled = conditioning is not None and conditioning.enabled
    if enabled != expected_conditioning:
        raise ValueError(f"final checkpoint conditioning mismatch: {resolved}")
    registry = (
        conditioning.resolved_registry_sha256 if conditioning is not None else None
    )
    if registry != expected_registry:
        raise ValueError(f"final checkpoint registry mismatch: {resolved}")
    return {
        "checkpoint": records.display_path(resolved),
        "checkpoint_sha256": file_sha256(resolved),
        "deck_conditioning_enabled": enabled,
        "registry_sha256": registry,
    }


def _require_sha256(path: Path, expected: str, *, label: str) -> Path:
    resolved = _required_file(path, label)
    actual = file_sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return resolved


def _required_file(path: Path, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _write_model_json(path: Path, model: BaseModel) -> None:
    write_identity_atomic(path, json.loads(model.model_dump_json()))


__all__ = [
    "ComparisonDeckConfig",
    "DeckConditioningComparisonConfig",
    "build_deck_conditioning_bundle_config",
    "build_deck_conditioning_score_config",
    "prepare_deck_conditioning_comparison",
]
