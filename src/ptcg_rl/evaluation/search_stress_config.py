"""Strict S2 frozen-trace and ActTime stress configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.config import MacroSearchConfig
from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_steps.records import DEFAULT_CHUNK_SIZE

StressController = Literal["disabled", "shadow", "override"]


class SearchTraceCellConfig(BaseModel):
    """One controller × seat × trace length × virtual slowdown cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str
    controller: StressController
    seat: Literal[0, 1]
    global_steps: int
    slowdown_factor: float = 1.0

    @field_validator("cell_id")
    @classmethod
    def immutable_cell_id(cls, value: str) -> str:
        """Require a stable non-moving cell identifier."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("stress cell_id must be non-empty and immutable")
        return normalized

    @field_validator("global_steps")
    @classmethod
    def positive_steps(cls, value: int) -> int:
        """Require a non-empty global replay prefix."""
        if value <= 0:
            raise ValueError("stress global_steps must be positive")
        return value

    @field_validator("slowdown_factor")
    @classmethod
    def positive_slowdown(cls, value: float) -> float:
        """Require a finite positive virtual-clock scale."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("slowdown_factor must be finite and positive")
        return value


class SearchStressReferences(BaseModel):
    """Pre-registered correctness and ActTime diagnostic references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_overage_seconds: float = 600.0
    startup_charge_seconds: float = 20.0
    max_startup_p99_seconds: float = 25.0
    max_search_bank_seconds: float = 180.0
    hard_search_cutoff_seconds: float = 180.0
    min_reserve_1x_826_seconds: float = 120.0
    min_reserve_1_25x_826_seconds: float = 60.0
    max_deadline_overshoot_seconds: float = 0.25
    min_telemetry_coverage: float = 0.999

    @field_validator(
        "total_overage_seconds",
        "startup_charge_seconds",
        "max_startup_p99_seconds",
        "max_search_bank_seconds",
        "hard_search_cutoff_seconds",
        "min_reserve_1x_826_seconds",
        "min_reserve_1_25x_826_seconds",
        "max_deadline_overshoot_seconds",
    )
    @classmethod
    def finite_non_negative(cls, value: float) -> float:
        """Reject negative or non-finite timing references."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("stress timing references must be finite and non-negative")
        return value

    @field_validator("min_telemetry_coverage")
    @classmethod
    def valid_coverage(cls, value: float) -> float:
        """Restrict telemetry coverage to a probability."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("min_telemetry_coverage must be in [0, 1]")
        return value


def _default_trace_cells() -> tuple[SearchTraceCellConfig, ...]:
    cells: list[SearchTraceCellConfig] = []
    for seat in (0, 1):
        cells.extend(
            [
                SearchTraceCellConfig(
                    cell_id=f"disabled_s{seat}_g300_x1",
                    controller="disabled",
                    seat=seat,
                    global_steps=300,
                ),
                SearchTraceCellConfig(
                    cell_id=f"shadow_s{seat}_g300_x1",
                    controller="shadow",
                    seat=seat,
                    global_steps=300,
                ),
            ]
        )
        for global_steps in (300, 500, 826):
            cells.append(
                SearchTraceCellConfig(
                    cell_id=f"override_s{seat}_g{global_steps}_x1",
                    controller="override",
                    seat=seat,
                    global_steps=global_steps,
                )
            )
        for factor, label in ((1.25, "1_25"), (1.5, "1_5"), (2.0, "2")):
            cells.append(
                SearchTraceCellConfig(
                    cell_id=f"override_s{seat}_g826_x{label}",
                    controller="override",
                    seat=seat,
                    global_steps=826,
                    slowdown_factor=factor,
                )
            )
    return tuple(cells)


class SearchTraceStressConfig(BaseModel):
    """Hydra configuration for S2 synthetic long-trace runtime stress."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = "v12395_paired_k4_w3_s2_trace_20260711"
    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = (
        "outputs/kaggle_submission_replays/54498922_comfey_v12395/*.json"
    )
    team_name: str = "Marshall Maximizer"
    deck_path: Path = Path(
        "docs/experiments/rl_dynamic_deck_pool_20260708/decks/"
        "29_comfey_yveltal_shaymin_4f8e151b4dd0.csv"
    )
    checkpoint_path: Path = Path(
        "outputs/inference_time_search/p0/assets/policy_v12395.pt"
    )
    belief: OpponentBeliefFeatureConfig = Field(
        default_factory=OpponentBeliefFeatureConfig
    )
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="archetype")
    )
    macro: MacroSearchConfig = Field(
        default_factory=lambda: MacroSearchConfig(mode="override")
    )
    device: str = "cpu"
    precision: Literal["float32"] = "float32"
    cells: tuple[SearchTraceCellConfig, ...] = Field(
        default_factory=_default_trace_cells
    )
    overage_boundaries: tuple[float, ...] = (180.0, 120.0, 60.0, 30.0)
    seed: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    output_dir: Path = Path(
        "outputs/evaluation/inference_time_search/"
        "v12395_paired_k4_w3_s2_trace_20260711"
    )
    compression: str = "zstd"
    overwrite: bool = False
    torch_num_threads: int = 1
    cpu_affinity: tuple[int, ...] = (0, 1, 2, 3)
    references: SearchStressReferences = Field(default_factory=SearchStressReferences)
    runtime_source_paths: tuple[Path, ...] = (
        Path("src/ptcg_rl/agent/runtime.py"),
        Path("src/ptcg_rl/agent/probe.py"),
        Path("src/ptcg_rl/agent/search/budget.py"),
        Path("src/ptcg_rl/agent/search/candidates.py"),
        Path("src/ptcg_rl/agent/search/config.py"),
        Path("src/ptcg_rl/agent/search/context.py"),
        Path("src/ptcg_rl/agent/search/macro.py"),
        Path("src/ptcg_rl/agent/search/policy_inputs.py"),
        Path("src/ptcg_rl/agent/search/reranker.py"),
        Path("src/ptcg_rl/agent/search/scoring.py"),
    )
    engine_asset_paths: tuple[Path, ...] = (
        Path("data/sample_submission/cg/api.py"),
        Path("data/sample_submission/cg/libcg.so"),
        Path("src/ptcg_rl/engine/effect_types.py"),
        Path("src/ptcg_rl/engine/effects.py"),
        Path("src/ptcg_rl/engine/forward_model.py"),
        Path("src/ptcg_rl/engine/session.py"),
    )

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        """Reject empty or moving stress campaign labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("experiment_id must be non-empty and cannot contain latest")
        return normalized

    @field_validator("overage_boundaries")
    @classmethod
    def valid_boundaries(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Require unique non-negative finite overage boundary probes."""
        if not value or len(set(value)) != len(value):
            raise ValueError("overage_boundaries must be non-empty and unique")
        if any(not math.isfinite(item) or item < 0.0 for item in value):
            raise ValueError("overage boundaries must be finite and non-negative")
        return value

    @field_validator("torch_num_threads")
    @classmethod
    def positive_threads(cls, value: int) -> int:
        """Require a positive fixed Torch thread count."""
        if value <= 0:
            raise ValueError("torch_num_threads must be positive")
        return value

    @field_validator("cpu_affinity")
    @classmethod
    def valid_affinity(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require unique non-negative reference CPU indices."""
        if not value or len(set(value)) != len(value) or any(cpu < 0 for cpu in value):
            raise ValueError("cpu_affinity must contain unique non-negative CPUs")
        return value

    @model_validator(mode="after")
    def complete_stress_matrix(self) -> SearchTraceStressConfig:
        """Require unique cells and every pre-registered primary stress stratum."""
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("stress cell_id values must be unique")
        signatures = {
            (
                cell.controller,
                cell.seat,
                cell.global_steps,
                cell.slowdown_factor,
            )
            for cell in self.cells
        }
        required: set[tuple[StressController, int, int, float]] = set()
        for seat in (0, 1):
            required.update(
                {
                    ("disabled", seat, 300, 1.0),
                    ("shadow", seat, 300, 1.0),
                    ("override", seat, 300, 1.0),
                    ("override", seat, 500, 1.0),
                    ("override", seat, 826, 1.0),
                    ("override", seat, 826, 1.25),
                    ("override", seat, 826, 1.5),
                    ("override", seat, 826, 2.0),
                }
            )
        missing = sorted(required - signatures, key=str)
        if missing:
            raise ValueError(f"stress matrix is missing required cells: {missing}")
        if self.macro.mode != "override":
            raise ValueError("stress base macro config must use override mode")
        belief_path = self.belief.deck_signature_summary_path
        sampler_path = self.sampler.prior_deck_signature_summary_path
        if (
            belief_path is not None
            and sampler_path is not None
            and belief_path != sampler_path
        ):
            raise ValueError("belief and sampler prior assets must match")
        return self
