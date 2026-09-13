"""Typed configuration for deck-conditioned policy verification."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class DeckConditioningAuditArchitecture(BaseModel):
    """Conditioning dimensions fixed before running the migration audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder_hidden_dim: int = 256
    adapter_layer_indices: tuple[int, ...] = (9, 10, 11)
    adapter_bottleneck_dim: int = 64
    policy_bottleneck_dim: int = 64
    value_bottleneck_dim: int = 64
    adapter_dropout: float = 0.0


class DeckConditioningAuditConfig(BaseModel):
    """Hydra-facing config for one immutable verification campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_checkpoint: Path
    source_checkpoint_sha256: str
    resolved_registry_sha256: str
    private_deck_paths: tuple[Path, ...]
    generic_deck_path: Path
    selected_release_deck_path: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 20260715
    migration_atol: float = 1.0e-6
    migration_rtol: float = 1.0e-6
    mixed_eager_atol: float = 3.0e-5
    mixed_eager_rtol: float = 1.0e-5
    benchmark_warmup_iterations: int = 5
    benchmark_iterations: int = 30
    runtime_iterations: int = 20
    require_cuda: bool = True
    architecture: DeckConditioningAuditArchitecture = (
        DeckConditioningAuditArchitecture()
    )

    @field_validator(
        "benchmark_warmup_iterations",
        "benchmark_iterations",
        "runtime_iterations",
    )
    @classmethod
    def positive_iterations(cls, value: int) -> int:
        """Reject empty timing campaigns."""
        if value <= 0:
            raise ValueError("audit iteration counts must be positive")
        return value

    @field_validator("source_checkpoint_sha256", "resolved_registry_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require lowercase full SHA256 identities in the audit profile."""
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("audit SHA256 values must be 64 lowercase hex characters")
        return value

    @field_validator(
        "migration_atol",
        "migration_rtol",
        "mixed_eager_atol",
        "mixed_eager_rtol",
    )
    @classmethod
    def non_negative_tolerance(cls, value: float) -> float:
        """Require preregistered non-negative numerical tolerances."""
        if value < 0.0:
            raise ValueError("audit tolerances must be non-negative")
        return value

    @model_validator(mode="after")
    def nonempty_private_registry(self) -> DeckConditioningAuditConfig:
        """Require one or more exact private profiles."""
        if not self.private_deck_paths:
            raise ValueError("private_deck_paths must be non-empty")
        return self


__all__ = [
    "DeckConditioningAuditArchitecture",
    "DeckConditioningAuditConfig",
]
