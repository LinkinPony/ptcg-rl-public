"""Validated contracts for the persistent evaluation league."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.agent.search.budget import ActTimeLedgerConfig

MU0 = 12.5
SIGMA0 = (25.0 / 3.0) / math.sqrt(2.0)
BETA = (25.0 / 6.0) / math.sqrt(2.0)
TAU = 0.0
DRAW_PROBABILITY = 0.001

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}")

ControllerKind = Literal["checkpoint", "script"]
CandidateKind = Literal["automatic", "manual", "anchor"]
CandidateState = Literal["candidate", "incumbent", "rejected", "anchor"]
MatchState = Literal["queued", "leased", "completed", "cancelled"]
GameOutcome = Literal[
    "side_a_win",
    "side_b_win",
    "draw",
    "unresolved",
]
TerminalReason = Literal[
    "normal",
    "side_a_timeout",
    "side_b_timeout",
    "side_a_act_error",
    "side_b_act_error",
    "side_a_illegal_action",
    "side_b_illegal_action",
    "infrastructure_error",
    "both_sides_error",
    "max_steps",
]


class RatingConfig(BaseModel):
    """Immutable classic TrueSkill season parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mu0: float = MU0
    sigma0: float = SIGMA0
    beta: float = BETA
    tau: float = TAU
    draw_probability: float = DRAW_PROBABILITY
    conservative_sigma: float = 3.0

    @model_validator(mode="after")
    def fixed_season_parameters(self) -> Self:
        expected = (MU0, SIGMA0, BETA, TAU, DRAW_PROBABILITY, 3.0)
        actual = (
            self.mu0,
            self.sigma0,
            self.beta,
            self.tau,
            self.draw_probability,
            self.conservative_sigma,
        )
        if actual != expected:
            raise ValueError("continuous league TrueSkill parameters are fixed")
        return self


class PromotionConfig(BaseModel):
    """Sequential automatic-candidate gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_games: int = 32
    maximum_games: int = 256
    early_accept_probability: float = 0.80
    early_reject_probability: float = 0.05
    final_accept_probability: float = 0.50
    top_fraction: float = 0.20
    monte_carlo_samples: int = 20_000
    monte_carlo_seed: int = 20260804

    @model_validator(mode="after")
    def valid_gate(self) -> Self:
        if self.minimum_games <= 0 or self.maximum_games < self.minimum_games:
            raise ValueError("promotion game limits are inconsistent")
        probabilities = (
            self.early_accept_probability,
            self.early_reject_probability,
            self.final_accept_probability,
            self.top_fraction,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("promotion probabilities must be in (0, 1)")
        if self.early_reject_probability >= self.early_accept_probability:
            raise ValueError("early rejection must be below early acceptance")
        if self.monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be positive")
        return self


class SchedulingConfig(BaseModel):
    """Bounded queue and challenge-ladder policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queue_target: int = 64
    lease_seconds: float = 300.0
    stale_pair_games: int = 64
    anchor_frequency: int = 8

    @field_validator("queue_target", "stale_pair_games", "anchor_frequency")
    @classmethod
    def positive_counts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("schedule counts must be positive")
        return value

    @field_validator("lease_seconds")
    @classmethod
    def positive_lease(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("lease_seconds must be finite and positive")
        return value


class ResourceConfig(BaseModel):
    """Opportunity-based worker admission thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quiet_probes: int = 2
    probe_seconds: float = 5.0
    max_load_per_cpu: float = 0.90
    minimum_available_memory_bytes: int = 4 * 1024**3
    maximum_gpu_utilization_percent: int = 60
    minimum_gpu_free_memory_bytes: int = 8 * 1024**3

    @model_validator(mode="after")
    def valid_resources(self) -> Self:
        if self.quiet_probes <= 0 or self.probe_seconds <= 0.0:
            raise ValueError("resource quiet probe policy must be positive")
        if not 0.0 < self.max_load_per_cpu <= 1.0:
            raise ValueError("max_load_per_cpu must be in (0, 1]")
        if self.minimum_available_memory_bytes < 0:
            raise ValueError("minimum memory cannot be negative")
        if not 0 <= self.maximum_gpu_utilization_percent <= 100:
            raise ValueError("GPU utilization threshold must be in [0, 100]")
        if self.minimum_gpu_free_memory_bytes < 0:
            raise ValueError("minimum GPU memory cannot be negative")
        return self


class AnchorConfig(BaseModel):
    """One explicitly pinned public/script controller and exact deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    controller_id: str
    label: str
    deck_path: Path
    implementation_paths: tuple[Path, ...]
    requires_cuda: bool = False

    @field_validator("controller_id")
    @classmethod
    def safe_controller_id(cls, value: str) -> str:
        return validate_safe_id(value)

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("anchor label must be non-empty")
        return cleaned

    @field_validator("implementation_paths")
    @classmethod
    def nonempty_implementation(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value:
            raise ValueError("anchor implementation_paths cannot be empty")
        return value


class NativeMatchConfig(BaseModel):
    """Frozen thin adapter over the project-owned native training lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["continuous_league_native_match_v1"] = (
        "continuous_league_native_match_v1"
    )
    library_path: Path = Path("src/native/cg_train/libcg_train.so")
    expected_library_sha256: str | None = None
    public_catalog_manifest_path: Path | None = None
    expected_public_catalog_manifest_sha256: str | None = None
    checkpoint_device: str = "cuda"
    checkpoint_resident_precision: Literal["source", "bfloat16"] = "source"
    checkpoint_rollout_inductor: bool = False
    lane_worker_count: int = 1
    option_capacity: int = 2048
    maximum_engine_steps: int = 10_000
    checkpoint_cache_entries: int = 2
    policy_batch_max_rows: int = 1
    policy_batch_wait_ms: float = 0.0
    policy_batch_coalesce_temperatures: bool = True
    act_time: ActTimeConfig = Field(default_factory=ActTimeConfig)
    act_time_ledger: ActTimeLedgerConfig = Field(default_factory=ActTimeLedgerConfig)

    @field_validator(
        "expected_library_sha256",
        "expected_public_catalog_manifest_sha256",
    )
    @classmethod
    def valid_optional_asset_sha(cls, value: str | None) -> str | None:
        return None if value is None else validate_sha256(value)

    @field_validator(
        "lane_worker_count",
        "option_capacity",
        "maximum_engine_steps",
        "checkpoint_cache_entries",
        "policy_batch_max_rows",
    )
    @classmethod
    def positive_native_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("native match counts must be positive")
        return value

    @field_validator("checkpoint_cache_entries")
    @classmethod
    def cache_holds_both_sides(cls, value: int) -> int:
        if value < 2:
            raise ValueError("checkpoint cache must hold both match sides")
        return value

    @field_validator("checkpoint_device")
    @classmethod
    def nonempty_checkpoint_device(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("checkpoint_device must be non-empty")
        return normalized

    @field_validator("policy_batch_wait_ms")
    @classmethod
    def nonnegative_batch_wait(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy batch wait must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def supported_runtime(self) -> Self:
        if self.checkpoint_rollout_inductor and (
            self.checkpoint_resident_precision != "bfloat16"
        ):
            raise ValueError(
                "checkpoint rollout Inductor requires bfloat16 resident weights"
            )
        if self.checkpoint_resident_precision == "bfloat16" and not (
            self.checkpoint_device.lower().startswith("cuda")
        ):
            raise ValueError("bfloat16 resident checkpoint inference requires CUDA")
        if self.act_time.planner is not None:
            raise ValueError(
                "continuous league native adapter does not yet expose the "
                "packaged planner lane service"
            )
        catalog_values = (
            self.public_catalog_manifest_path,
            self.expected_public_catalog_manifest_sha256,
        )
        if (catalog_values[0] is None) != (catalog_values[1] is None):
            raise ValueError("public catalog path and SHA-256 must be set together")
        return self


class ContinuousLeagueConfig(BaseModel):
    """Hydra-facing persistent league configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    database_path: Path = Path("outputs/evaluation/continuous_league/league.sqlite3")
    telemetry_dir: Path = Path("outputs/evaluation/continuous_league/telemetry")
    checkpoint_roots: tuple[Path, ...] = (Path("outputs/training/rl"),)
    deck_roots: tuple[Path, ...] = (
        Path("configs/decks"),
        Path("docs/experiments"),
        Path("data/public_opponents"),
        Path("outputs/decks"),
        Path("outputs/evaluation_launch"),
        Path("outputs/submission"),
        Path("src/ptcg_rl/evaluation/assets"),
    )
    release_roots: tuple[Path, ...] = (Path("outputs/submission"),)
    discovery_seconds: float = 30.0
    scheduling_seconds: float = 2.0
    telemetry_shard_rows: int = 256
    coordinator_host: str = "127.0.0.1"
    coordinator_port: int = 8788
    native_match_command: tuple[str, ...] = ()
    native_match: NativeMatchConfig = Field(default_factory=NativeMatchConfig)
    runtime_fingerprint: str
    belief_fingerprint: str
    rating: RatingConfig = Field(default_factory=RatingConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    anchors: tuple[AnchorConfig, ...] = ()

    @field_validator("runtime_fingerprint", "belief_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("discovery_seconds", "scheduling_seconds")
    @classmethod
    def positive_interval(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("league intervals must be finite and positive")
        return value

    @field_validator("telemetry_shard_rows")
    @classmethod
    def positive_shard_rows(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("telemetry_shard_rows must be positive")
        return value

    @field_validator("coordinator_port")
    @classmethod
    def valid_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("coordinator_port must be in [1, 65535]")
        return value


class ComponentRating(BaseModel):
    """One controller or exact-deck component rating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str
    component_kind: Literal["controller", "deck"]
    mu: float
    sigma: float
    games: int = 0

    @field_validator("component_id")
    @classmethod
    def safe_component_id(cls, value: str) -> str:
        return validate_safe_id(value)

    @model_validator(mode="after")
    def finite_rating(self) -> Self:
        if not math.isfinite(self.mu) or not math.isfinite(self.sigma):
            raise ValueError("rating values must be finite")
        if self.sigma <= 0.0 or self.games < 0:
            raise ValueError("rating sigma must be positive and games non-negative")
        return self


class BundleIdentity(BaseModel):
    """One playable exact controller/deck combination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    controller_id: str
    deck_digest: str
    active: bool = True

    @field_validator("bundle_id", "controller_id")
    @classmethod
    def safe_ids(cls, value: str) -> str:
        return validate_safe_id(value)

    @field_validator("deck_digest")
    @classmethod
    def valid_deck_digest(cls, value: str) -> str:
        return validate_sha256(value)


class ResourceSnapshot(BaseModel):
    """Worker resource evidence attached to heartbeats and lease requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_affinity_count: int = Field(gt=0)
    load_1m: float = Field(ge=0.0)
    memory_available_bytes: int = Field(ge=0)
    cuda_available: bool
    gpu_index: int | None = None
    gpu_utilization_percent: int | None = Field(default=None, ge=0, le=100)
    gpu_memory_free_bytes: int | None = Field(default=None, ge=0)
    quiet: bool


class WorkerHeartbeat(BaseModel):
    """Periodic liveness and resource report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    hostname: str
    source_commit: str
    runtime_fingerprint: str
    belief_fingerprint: str
    resources: ResourceSnapshot
    current_match_id: str | None = None
    games_completed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)

    @field_validator("worker_id", "hostname")
    @classmethod
    def safe_worker_identity(cls, value: str) -> str:
        return validate_safe_id(value)

    @field_validator("runtime_fingerprint", "belief_fingerprint")
    @classmethod
    def valid_worker_fingerprint(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("source_commit")
    @classmethod
    def valid_source_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("source_commit must be a full hexadecimal commit")
        return normalized


class LeaseRequest(BaseModel):
    """Atomic request for one small match lease."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heartbeat: WorkerHeartbeat


class MatchLease(BaseModel):
    """Fully resolved immutable game request returned to a worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_id: str
    side_a: BundleIdentity
    side_b: BundleIdentity
    side_a_controller_kind: ControllerKind
    side_b_controller_kind: ControllerKind
    side_a_controller_path: Path | None
    side_b_controller_path: Path | None
    side_a_deck_path: Path
    side_b_deck_path: Path
    requires_cuda: bool
    runtime_fingerprint: str
    belief_fingerprint: str
    lease_expires_at: str


class MatchResult(BaseModel):
    """Minimal idempotent game result plus compact telemetry payload."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    match_id: str
    worker_id: str
    outcome: GameOutcome
    terminal_reason: TerminalReason
    started_at: str
    finished_at: str
    steps: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    telemetry_msgpack: bytes = b""

    @field_validator("match_id", "worker_id")
    @classmethod
    def safe_match_identity(cls, value: str) -> str:
        return validate_safe_id(value)

    @model_validator(mode="after")
    def coherent_outcome(self) -> Self:
        unresolved = self.terminal_reason in {
            "infrastructure_error",
            "both_sides_error",
            "max_steps",
        }
        if unresolved != (self.outcome == "unresolved"):
            raise ValueError("terminal reason and outcome resolution disagree")
        attributed = {
            "side_a_timeout": "side_b_win",
            "side_a_act_error": "side_b_win",
            "side_a_illegal_action": "side_b_win",
            "side_b_timeout": "side_a_win",
            "side_b_act_error": "side_a_win",
            "side_b_illegal_action": "side_a_win",
        }
        expected = attributed.get(self.terminal_reason)
        if expected is not None and self.outcome != expected:
            raise ValueError("attributed participant fault must be scored as a loss")
        return self


class ManualCheckpointRequest(BaseModel):
    """Manual checkpoint-pair admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_manifest_path: Path
    label: str | None = None


class ManualDeckRequest(BaseModel):
    """Manual exact 60-card CSV admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_path: Path
    label: str | None = None


class ManualReleaseRequest(BaseModel):
    """Manual immutable release admission or alias attachment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    release_manifest_path: Path
    alias: str

    @field_validator("alias")
    @classmethod
    def nonempty_alias(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("release alias must be non-empty")
        return cleaned


class ForceChallengeRequest(BaseModel):
    """Manual request for one exact bundle matchup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    side_a_bundle_id: str
    side_b_bundle_id: str

    @field_validator("side_a_bundle_id", "side_b_bundle_id")
    @classmethod
    def safe_bundle_id(cls, value: str) -> str:
        return validate_safe_id(value)

    @model_validator(mode="after")
    def distinct_sides(self) -> Self:
        if self.side_a_bundle_id == self.side_b_bundle_id:
            raise ValueError("challenge sides must be distinct")
        return self


class ActivationRequest(BaseModel):
    """Enable or disable a controller and its combinations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active: bool
    reason: str

    @field_validator("reason")
    @classmethod
    def nonempty_reason(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("activation reason must be non-empty")
        return cleaned


def validate_sha256(value: str) -> str:
    """Normalize one full SHA-256 identity."""
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError("identity must be 64 lowercase hexadecimal characters")
    return normalized


def validate_safe_id(value: str) -> str:
    """Validate one compact database and URL identity."""
    normalized = value.strip()
    if _SAFE_ID.fullmatch(normalized) is None:
        raise ValueError("identifier contains unsupported characters")
    return normalized


__all__ = [
    "ActivationRequest",
    "AnchorConfig",
    "BETA",
    "BundleIdentity",
    "CandidateKind",
    "CandidateState",
    "ComponentRating",
    "ContinuousLeagueConfig",
    "DRAW_PROBABILITY",
    "ForceChallengeRequest",
    "GameOutcome",
    "LeaseRequest",
    "MU0",
    "ManualCheckpointRequest",
    "ManualDeckRequest",
    "ManualReleaseRequest",
    "MatchLease",
    "MatchResult",
    "NativeMatchConfig",
    "PromotionConfig",
    "RatingConfig",
    "ResourceConfig",
    "ResourceSnapshot",
    "SIGMA0",
    "SchedulingConfig",
    "TAU",
    "TerminalReason",
    "WorkerHeartbeat",
    "validate_safe_id",
    "validate_sha256",
]
