"""Validated packaged ActTime replay contract for handoff adapter A/B."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.config import MacroSearchConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HandoffReplayAsset(BaseModel):
    """One immutable ordered episode and its expected seat callback counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str
    active_callbacks_by_seat: tuple[int, int]

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a canonical content fingerprint."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("replay fingerprint must be lowercase SHA-256")
        return value

    @field_validator("active_callbacks_by_seat")
    @classmethod
    def positive_callback_counts(cls, values: tuple[int, int]) -> tuple[int, int]:
        """Require an exact non-empty callback workload for both seats."""
        if any(value <= 0 for value in values):
            raise ValueError("replay callback counts must be positive")
        return values


class HandoffReplayArm(BaseModel):
    """One fixed handoff scorer semantic arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str
    handoff_score_mode: Literal["engine_only", "root_value_adapter"]

    @field_validator("arm_id")
    @classmethod
    def nonempty_arm_id(cls, value: str) -> str:
        """Keep artifact keys compact and unambiguous."""
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("arm_id must be non-empty without edge whitespace")
        return value


class HandoffAdapterActTimeReplayConfig(BaseModel):
    """Hydra-backed immutable package and replay comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    package_archive_path: Path
    expected_package_archive_sha256: str
    expected_archive_contents_fingerprint: str
    expected_checkpoint_sha256: str
    expected_deck_sha256: str
    replay_assets: tuple[HandoffReplayAsset, ...]
    seats: tuple[Literal[0, 1], ...]
    arms: tuple[HandoffReplayArm, ...]
    repetitions: int = Field(gt=0)
    initial_overage_seconds: float = Field(gt=0.0)
    seed: int
    macro: MacroSearchConfig
    child_timeout_seconds: float = Field(gt=0.0)
    output_dir: Path
    compression: str = "zstd"

    @field_validator(
        "expected_package_archive_sha256",
        "expected_archive_contents_fingerprint",
        "expected_checkpoint_sha256",
        "expected_deck_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require canonical immutable identities."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("artifact fingerprint must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def coherent_comparison(self) -> Self:
        """Require one direct A/B with identical non-handoff semantics."""
        if not self.replay_assets:
            raise ValueError("packaged replay requires at least one episode")
        if len(set(self.seats)) != len(self.seats) or not self.seats:
            raise ValueError("packaged replay seats must be non-empty and unique")
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        modes = tuple(arm.handoff_score_mode for arm in self.arms)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("packaged replay arm IDs must be unique")
        if set(modes) != {"engine_only", "root_value_adapter"} or len(modes) != 2:
            raise ValueError("packaged replay requires exactly the direct scorer A/B")
        if self.macro.mode != "conditioned":
            raise ValueError("handoff adapter replay requires conditioned serving")
        if self.macro.rerank.score_mode == "engine_only":
            raise ValueError("handoff adapter replay requires a value-aware scorer")
        return self


__all__ = [
    "HandoffAdapterActTimeReplayConfig",
    "HandoffReplayArm",
    "HandoffReplayAsset",
]
