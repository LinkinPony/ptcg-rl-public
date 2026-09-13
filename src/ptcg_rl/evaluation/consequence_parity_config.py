"""Validated Hydra contract for bounded decision-transition parity audits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.evaluation.consequence_audit_sampling import COVERAGE_LABELS


class DecisionTransitionParityConfig(BaseModel):
    """Hydra-backed bounded consequence parity audit configuration."""

    model_config = ConfigDict(extra="forbid")

    steps_globs: tuple[str, ...]
    library_path: Path
    output_dir: Path
    reservoir_capacity_per_label: int = 16
    reservoir_seed: str = "stage-a"
    scan_batch_size: int = 65_536
    exhaustive_action_cap: int = 24
    max_candidates_per_root: int = 24
    worlds_per_root: int = 1
    max_cells: int = 24
    max_engine_steps: int = 216
    max_forced_steps: int = 8
    max_observation_bytes: int = 64 * 1024 * 1024
    effect_atol: float = 1.0e-6
    manual_coin: bool = True
    output_format: Literal["parquet", "npz"] = "parquet"
    output_shard_rows: int = 512
    max_output_rows: int = 2304
    compression: str = "zstd"
    fallback_card_id: int = 1
    fallback_basic_pokemon_id: int = 463
    hydra: Mapping[str, Any] | None = None

    @field_validator(
        "reservoir_capacity_per_label",
        "scan_batch_size",
        "exhaustive_action_cap",
        "max_candidates_per_root",
        "worlds_per_root",
        "max_cells",
        "max_engine_steps",
        "max_observation_bytes",
        "output_shard_rows",
        "max_output_rows",
        "fallback_card_id",
        "fallback_basic_pokemon_id",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Reject non-positive resource bounds and card IDs."""
        if value <= 0:
            raise ValueError("audit integer bounds must be positive")
        return value

    @field_validator("max_forced_steps")
    @classmethod
    def nonnegative_forced_steps(cls, value: int) -> int:
        """Allow a root-only transition audit but reject negative caps."""
        if value < 0:
            raise ValueError("max_forced_steps must be non-negative")
        return value

    @field_validator("effect_atol")
    @classmethod
    def valid_tolerance(cls, value: float) -> float:
        """Require a finite non-negative exact-feature tolerance."""
        if not 0.0 <= value < float("inf"):
            raise ValueError("effect_atol must be finite and non-negative")
        return value

    @field_validator("steps_globs")
    @classmethod
    def nonempty_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require at least one non-empty input pattern."""
        if not value or any(not pattern for pattern in value):
            raise ValueError("steps_globs must be non-empty")
        return value

    @model_validator(mode="after")
    def coherent_bounds(self) -> Self:
        """Prove the sample cannot overrun native or output bounds."""
        if self.exhaustive_action_cap > self.max_candidates_per_root:
            raise ValueError(
                "exhaustive_action_cap cannot exceed max_candidates_per_root"
            )
        cells_per_root = self.max_candidates_per_root * self.worlds_per_root
        if self.max_cells < cells_per_root:
            raise ValueError("max_cells cannot cover the candidate-by-world grid")
        required_steps = cells_per_root * (self.max_forced_steps + 1)
        if self.max_engine_steps < required_steps:
            raise ValueError("max_engine_steps cannot cover the bounded candidate grid")
        maximum_rows = (
            len(COVERAGE_LABELS)
            * self.reservoir_capacity_per_label
            * self.max_candidates_per_root
            * self.worlds_per_root
        )
        if self.max_output_rows < maximum_rows:
            raise ValueError(
                "max_output_rows cannot cover the deterministic reservoir bound"
            )
        if self.output_shard_rows > self.max_output_rows:
            raise ValueError("output_shard_rows cannot exceed max_output_rows")
        return self


__all__ = ["DecisionTransitionParityConfig"]
