"""Opponent and deck curriculum sampling for RL rollout."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.opponents import OpponentSpec, opponent_registry
from ptcg_rl.rl.curriculum_assignment import (
    DeterministicWeightedCycle,
    WeightedCycleItem,
    automatic_cycle_size,
)
from ptcg_rl.rl.curriculum_schedule import (
    FixedFrozenPolicyConfig,
    StaticOpponentScheduleConfig,
)
from ptcg_rl.rl.curriculum_specialist import (
    CANDIDATE_LANE_ORDER,
    CandidateLaneConfig,
    MatchupPriorityConfig,
    MatchupStatistic,
    blended_matchup_ema,
    lane_probabilities,
    matchup_key,
    parse_matchup_key,
    validate_matchup_statistics,
)

OpponentKind = Literal["self_play", "frozen", "scripted"]
FrozenSamplingLane = Literal["pfsp", "hard"]
_OpponentSamplingKind = Literal["self_play", "frozen", "hard", "scripted"]
_T = TypeVar("_T")

_RANK1_DECK_PATH = Path(
    "docs/experiments/marnie_grimmsnarl_rank1_6f4cfbebac40/deck.csv"
)
_CHECKPOINT_HASH_CHUNK_BYTES = 1024 * 1024
_LANE_REORDER_WINDOW_MULTIPLIER = 4


@dataclass(frozen=True)
class CheckpointFingerprint:
    """Content fingerprint used to publish one frozen checkpoint safely."""

    size_bytes: int
    sha256: str


class CurriculumMixConfig(BaseModel):
    """Sampling weights for the opponent curriculum categories."""

    model_config = ConfigDict(extra="forbid")

    self_play: float = 0.5
    frozen: float = 0.35
    hard: float = 0.0
    scripted: float = 0.15

    @field_validator("self_play", "frozen", "hard", "scripted")
    @classmethod
    def valid_non_negative_weight(cls, value: float) -> float:
        """Reject negative sampling weights."""
        if value < 0.0:
            raise ValueError("curriculum mix weights must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_nonzero_total(self) -> CurriculumMixConfig:
        """Require at least one non-zero category weight."""
        if self.self_play + self.frozen + self.hard + self.scripted <= 0.0:
            raise ValueError("at least one curriculum mix weight must be positive")
        return self


class CandidatePriorityConfig(BaseModel):
    """PLR-lite weighting for candidate deck sampling."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    power: float = 2.0
    epsilon: float = 0.1
    floor_ratio: float = 0.5
    cap_ratio: float = 3.0

    @field_validator("power")
    @classmethod
    def valid_power(cls, value: float) -> float:
        """Reject invalid PFSP-style powers."""
        if value <= 0.0:
            raise ValueError("candidate priority power must be positive")
        return value

    @field_validator("epsilon", "floor_ratio")
    @classmethod
    def valid_non_negative(cls, value: float) -> float:
        """Reject negative candidate priority settings."""
        if value < 0.0:
            raise ValueError("candidate priority settings must be non-negative")
        return value

    @field_validator("cap_ratio")
    @classmethod
    def valid_cap_ratio(cls, value: float) -> float:
        """Reject invalid candidate priority caps."""
        if value <= 0.0:
            raise ValueError("candidate priority cap_ratio must be positive")
        return value

    @model_validator(mode="after")
    def valid_bounds(self) -> CandidatePriorityConfig:
        """Reject impossible floor/cap bounds."""
        if self.floor_ratio > 1.0:
            raise ValueError("candidate priority floor_ratio must be <= 1")
        if self.cap_ratio < 1.0:
            raise ValueError("candidate priority cap_ratio must be >= 1")
        return self


class OpponentFocusConfig(BaseModel):
    """Fixed mass for a preregistered group of opponent deck labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    probability: float = 0.0
    label_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("probability")
    @classmethod
    def valid_probability(cls, value: float) -> float:
        """Require the fixed focus mass to be a probability."""
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("opponent focus probability must be in [0, 1]")
        return value

    @field_validator("label_weights")
    @classmethod
    def valid_label_weights(cls, value: dict[str, float]) -> dict[str, float]:
        """Normalize labels and reject non-positive fixed weights."""
        cleaned = {label.strip(): float(weight) for label, weight in value.items()}
        if any(not label for label in cleaned):
            raise ValueError("opponent focus labels must be non-empty")
        if any(
            not math.isfinite(weight) or weight <= 0.0 for weight in cleaned.values()
        ):
            raise ValueError("opponent focus label weights must be positive")
        return cleaned

    @model_validator(mode="after")
    def valid_enabled_focus(self) -> OpponentFocusConfig:
        """Require labels and positive mass when fixed focus is enabled."""
        if self.enabled and (self.probability <= 0.0 or not self.label_weights):
            raise ValueError(
                "enabled opponent focus requires positive probability and labels"
            )
        return self


class FrozenPoolMember(BaseModel):
    """One frozen checkpoint opponent in the PFSP-lite pool."""

    model_config = ConfigDict(extra="forbid")

    opponent_id: str
    checkpoint_path: Path
    winrate_ema: float = 0.5
    games: int = 0
    added_order: int = 0
    pinned: bool = False
    recurrent: bool = False
    retired: bool = False
    deck_winrate_ema: dict[str, float] = Field(default_factory=dict)

    @field_validator("opponent_id")
    @classmethod
    def valid_opponent_id(cls, value: str) -> str:
        """Reject empty opponent identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("opponent_id must be non-empty")
        return cleaned

    @field_validator("winrate_ema")
    @classmethod
    def valid_winrate_ema(cls, value: float) -> float:
        """Reject EMA values outside [0, 1]."""
        if value < 0.0 or value > 1.0:
            raise ValueError("winrate_ema must be in [0, 1]")
        return value

    @field_validator("games", "added_order")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative counters."""
        if value < 0:
            raise ValueError("frozen pool counters must be non-negative")
        return value

    @field_validator("deck_winrate_ema")
    @classmethod
    def valid_deck_winrate_ema(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject invalid per-deck EMA values."""
        cleaned: dict[str, float] = {}
        for raw_label, raw_ema in value.items():
            label = raw_label.strip()
            if not label:
                raise ValueError("deck_winrate_ema labels must be non-empty")
            ema = float(raw_ema)
            if ema < 0.0 or ema > 1.0:
                raise ValueError("deck_winrate_ema values must be in [0, 1]")
            cleaned[label] = ema
        return cleaned


class AssignmentLaneState(BaseModel):
    """One durably reserved route-local assignment block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_index: int
    cursor: int = 0
    reserved_until: int = 0

    @field_validator("template_index", "cursor", "reserved_until")
    @classmethod
    def valid_non_negative_position(cls, value: int) -> int:
        """Reject negative template and in-block positions."""
        if value < 0:
            raise ValueError("assignment lane positions must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_reservation(self) -> AssignmentLaneState:
        """Require the durable boundary to cover every issued position."""
        if self.reserved_until < self.cursor:
            raise ValueError("assignment lane reservation cannot trail its cursor")
        return self


class FrozenPoolState(BaseModel):
    """Persistent frozen-pool state."""

    model_config = ConfigDict(extra="forbid")

    members: tuple[FrozenPoolMember, ...] = Field(default_factory=tuple)
    next_added_order: int = 0
    candidate_deck_winrate_ema: dict[str, float] = Field(default_factory=dict)
    matchup_statistics: dict[str, MatchupStatistic] = Field(default_factory=dict)
    assignment_cursor: int = 0
    assignment_reserved_until: int = 0
    assignment_schedule_fingerprint: str = ""
    assignment_next_template_index: int = 0
    assignment_pending_template_indices: tuple[int, ...] = ()
    assignment_lanes: dict[str, AssignmentLaneState] = Field(default_factory=dict)
    assignment_layout_fingerprint: str = ""

    @field_validator("members")
    @classmethod
    def valid_unique_member_ids(
        cls,
        value: tuple[FrozenPoolMember, ...],
    ) -> tuple[FrozenPoolMember, ...]:
        """Require one unambiguous checkpoint route per opponent identity."""
        opponent_ids = tuple(member.opponent_id for member in value)
        if len(opponent_ids) != len(set(opponent_ids)):
            raise ValueError("frozen pool opponent_id values must be unique")
        return value

    @field_validator("next_added_order")
    @classmethod
    def valid_next_added_order(cls, value: int) -> int:
        """Reject negative next-order counters."""
        if value < 0:
            raise ValueError("next_added_order must be non-negative")
        return value

    @field_validator(
        "assignment_cursor",
        "assignment_reserved_until",
        "assignment_next_template_index",
    )
    @classmethod
    def valid_assignment_position(cls, value: int) -> int:
        """Reject negative assignment-stream positions."""
        if value < 0:
            raise ValueError("assignment stream positions must be non-negative")
        return value

    @field_validator(
        "assignment_schedule_fingerprint",
        "assignment_layout_fingerprint",
    )
    @classmethod
    def valid_assignment_fingerprint(cls, value: str) -> str:
        """Validate an optional persisted assignment schedule fingerprint."""
        cleaned = value.strip().lower()
        if cleaned and (
            len(cleaned) != 64
            or any(character not in "0123456789abcdef" for character in cleaned)
        ):
            raise ValueError("assignment schedule fingerprint must be SHA-256")
        return cleaned

    @field_validator("assignment_lanes")
    @classmethod
    def valid_assignment_lanes(
        cls,
        value: dict[str, AssignmentLaneState],
    ) -> dict[str, AssignmentLaneState]:
        """Normalize lane keys to canonical non-negative integer strings."""
        normalized: dict[str, AssignmentLaneState] = {}
        for raw_key, lane in value.items():
            try:
                lane_index = int(raw_key)
            except ValueError as error:
                raise ValueError("assignment lane keys must be integers") from error
            if lane_index < 0 or str(lane_index) != raw_key:
                raise ValueError(
                    "assignment lane keys must be canonical non-negative integers"
                )
            normalized[raw_key] = lane
        return normalized

    @field_validator("assignment_pending_template_indices")
    @classmethod
    def valid_pending_template_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject negative or duplicated pending statistical templates."""
        if any(index < 0 for index in value):
            raise ValueError("pending assignment template indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("pending assignment template indices must be unique")
        return value

    @model_validator(mode="after")
    def valid_assignment_reservation(self) -> FrozenPoolState:
        """Require the durable reservation boundary to cover the cursor."""
        if self.assignment_reserved_until < self.assignment_cursor:
            raise ValueError("assignment reservation cannot trail its cursor")
        return self

    @field_validator("candidate_deck_winrate_ema")
    @classmethod
    def valid_candidate_deck_winrate_ema(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Reject invalid candidate-deck EMA values."""
        cleaned: dict[str, float] = {}
        for raw_label, raw_ema in value.items():
            label = raw_label.strip()
            if not label:
                raise ValueError("candidate_deck_winrate_ema labels must be non-empty")
            ema = float(raw_ema)
            if ema < 0.0 or ema > 1.0:
                raise ValueError("candidate_deck_winrate_ema values must be in [0, 1]")
            cleaned[label] = ema
        return cleaned

    @field_validator("matchup_statistics")
    @classmethod
    def valid_matchup_statistics(
        cls,
        value: dict[str, MatchupStatistic],
    ) -> dict[str, MatchupStatistic]:
        """Reject malformed persisted matchup triple keys."""
        return validate_matchup_statistics(value)


class FrozenPoolAddition(BaseModel):
    """Journal event requesting one frozen-pool member addition."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    opponent_id: str
    checkpoint_path: Path
    checkpoint_size_bytes: int | None = None
    checkpoint_sha256: str | None = None
    winrate_ema: float = 0.5
    pinned: bool = False
    created_at_utc: str = Field(default_factory=lambda: _utc_now())

    @field_validator("schema_version")
    @classmethod
    def valid_schema_version(cls, value: int) -> int:
        """Reject unsupported addition event versions."""
        if value not in (1, 2):
            raise ValueError("unsupported frozen pool addition schema_version")
        return value

    @field_validator("checkpoint_size_bytes")
    @classmethod
    def valid_checkpoint_size(cls, value: int | None) -> int | None:
        """Reject invalid optional checkpoint sizes."""
        if value is not None and value < 0:
            raise ValueError("checkpoint_size_bytes must be non-negative")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def valid_checkpoint_sha256(cls, value: str | None) -> str | None:
        """Normalize and validate optional checkpoint SHA256 values."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(
            char not in "0123456789abcdef" for char in cleaned
        ):
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest")
        return cleaned

    @field_validator("opponent_id")
    @classmethod
    def valid_opponent_id(cls, value: str) -> str:
        """Reject empty opponent identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("opponent_id must be non-empty")
        return cleaned

    @field_validator("winrate_ema")
    @classmethod
    def valid_winrate_ema(cls, value: float) -> float:
        """Reject EMA values outside [0, 1]."""
        if value < 0.0 or value > 1.0:
            raise ValueError("winrate_ema must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def valid_fingerprint(self) -> FrozenPoolAddition:
        """Require complete fingerprints for newly emitted events."""
        has_size = self.checkpoint_size_bytes is not None
        has_sha = self.checkpoint_sha256 is not None
        if has_size != has_sha:
            raise ValueError("checkpoint size and SHA256 must be provided together")
        if self.schema_version >= 2 and not has_size:
            raise ValueError("schema_version 2 requires a checkpoint fingerprint")
        return self


class WeightedDeckEntry(BaseModel):
    """One weighted deck source for curriculum sampling."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    weight: float
    label: str = ""

    @field_validator("weight")
    @classmethod
    def valid_weight(cls, value: float) -> float:
        """Reject non-positive deck sampling weights."""
        if value <= 0.0:
            raise ValueError("deck pool weights must be positive")
        return value

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        """Normalize optional deck labels."""
        return value.strip()


class FixedFrozenBundleConfig(BaseModel):
    """One immutable frozen policy paired with one exact opponent deck.

    The configured weight is a fixed multiplier on the member's live PFSP
    score. It does not replace PFSP or reserve a fixed probability mass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_id: str
    checkpoint_path: Path
    opponent_deck_path: Path
    opponent_deck_label: str
    weight: float = 1.0
    checkpoint_size_bytes: int | None = None
    checkpoint_sha256: str | None = None

    @field_validator("opponent_id", "opponent_deck_label")
    @classmethod
    def valid_non_empty_text(cls, value: str) -> str:
        """Normalize and reject empty bundle identifiers and labels."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("fixed frozen bundle identifiers must be non-empty")
        return cleaned

    @field_validator("weight")
    @classmethod
    def valid_weight(cls, value: float) -> float:
        """Require a finite positive fixed sampling weight."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fixed frozen bundle weight must be positive")
        return value

    @field_validator("checkpoint_size_bytes")
    @classmethod
    def valid_checkpoint_size(cls, value: int | None) -> int | None:
        """Reject invalid optional checkpoint sizes."""
        if value is not None and value < 0:
            raise ValueError("checkpoint_size_bytes must be non-negative")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def valid_checkpoint_sha256(cls, value: str | None) -> str | None:
        """Normalize and validate an optional checkpoint digest."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(
            char not in "0123456789abcdef" for char in cleaned
        ):
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest")
        return cleaned

    @model_validator(mode="after")
    def valid_fingerprint(self) -> FixedFrozenBundleConfig:
        """Require the optional checkpoint fingerprint as a complete pair."""
        has_size = self.checkpoint_size_bytes is not None
        has_sha = self.checkpoint_sha256 is not None
        if has_size != has_sha:
            raise ValueError("checkpoint size and SHA256 must be provided together")
        return self


class AssignmentScheduleConfig(BaseModel):
    """Persistent per-game assignment-stream settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["random", "stratified"] = "random"
    cycle_size: int = 0
    reservation_size: int = 128
    seed: int = 0
    execution_lanes: int = 0
    lane_block_size: int = 0

    @field_validator("cycle_size")
    @classmethod
    def valid_cycle_size(cls, value: int) -> int:
        """Allow zero for automatic sizing and reject negative sizes."""
        if value < 0:
            raise ValueError("assignment cycle_size must be non-negative")
        return value

    @field_validator("reservation_size")
    @classmethod
    def valid_reservation_size(cls, value: int) -> int:
        """Require a positive durable cursor reservation."""
        if value <= 0:
            raise ValueError("assignment reservation_size must be positive")
        return value

    @field_validator("execution_lanes", "lane_block_size")
    @classmethod
    def valid_lane_geometry(cls, value: int) -> int:
        """Allow zero for disabled lane scheduling and reject negatives."""
        if value < 0:
            raise ValueError("assignment lane geometry must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_lane_mode(self) -> AssignmentScheduleConfig:
        """Require lane count and block size as one stratified setting."""
        enabled = self.execution_lanes > 0 or self.lane_block_size > 0
        if enabled and (
            self.mode != "stratified"
            or self.execution_lanes <= 0
            or self.lane_block_size <= 0
        ):
            raise ValueError(
                "assignment execution_lanes and lane_block_size must both be "
                "positive in stratified mode"
            )
        return self


class CurriculumConfig(BaseModel):
    """Hydra-backed config for rollout opponent curriculum sampling."""

    model_config = ConfigDict(extra="forbid")

    mix: CurriculumMixConfig = Field(default_factory=CurriculumMixConfig)
    candidate_deck_path: Path = _RANK1_DECK_PATH
    candidate_deck_pool: tuple[WeightedDeckEntry, ...] = ()
    additional_candidate_deck_pool: tuple[WeightedDeckEntry, ...] = ()
    meta_deck_paths: tuple[Path, ...] = (
        Path("data/sample_submission/deck.csv"),
        _RANK1_DECK_PATH,
    )
    opponent_deck_pool: tuple[WeightedDeckEntry, ...] = ()
    additional_opponent_deck_pool: tuple[WeightedDeckEntry, ...] = ()
    opponent_focus: OpponentFocusConfig = Field(default_factory=OpponentFocusConfig)
    include_public_fixture_decks: bool = True
    scripted_opponents: tuple[str, ...] = ("heuristic", "mixed75")
    mirror_selfplay_probability: float = 0.5
    train_offdeck_selfplay: bool = False
    assignment_block_size: int = 1
    assignment_schedule: AssignmentScheduleConfig = Field(
        default_factory=AssignmentScheduleConfig
    )
    frozen_capacity: int = 6
    fixed_frozen_bundles: tuple[FixedFrozenBundleConfig, ...] = ()
    fixed_frozen_policies: tuple[FixedFrozenPolicyConfig, ...] = ()
    static_opponent_schedule: StaticOpponentScheduleConfig = Field(
        default_factory=StaticOpponentScheduleConfig
    )
    seed_anchor_in_frozen_pool: bool = True
    pfsp_power: float = 2.0
    pfsp_epsilon: float = 0.1
    winrate_ema_alpha: float = 0.004
    frozen_state_path: Path | None = None
    candidate_priority: CandidatePriorityConfig = Field(
        default_factory=CandidatePriorityConfig
    )
    candidate_lanes: CandidateLaneConfig = Field(default_factory=CandidateLaneConfig)
    matchup_priority: MatchupPriorityConfig = Field(
        default_factory=MatchupPriorityConfig
    )
    anchor_opponent_id: str = "anchor"
    anchor_max_total_probability: float | None = None

    @field_validator("mirror_selfplay_probability", "winrate_ema_alpha")
    @classmethod
    def valid_probability(cls, value: float) -> float:
        """Reject probabilities outside [0, 1]."""
        if value < 0.0 or value > 1.0:
            raise ValueError("curriculum probabilities must be in [0, 1]")
        return value

    @field_validator("assignment_block_size")
    @classmethod
    def valid_assignment_block_size(cls, value: int) -> int:
        """Require each route-local assignment block to contain a game."""
        if value <= 0:
            raise ValueError("assignment_block_size must be positive")
        return value

    @field_validator("frozen_capacity")
    @classmethod
    def valid_frozen_capacity(cls, value: int) -> int:
        """Reject invalid frozen pool capacities."""
        if value < 0:
            raise ValueError("frozen_capacity must be non-negative")
        return value

    @field_validator("pfsp_power")
    @classmethod
    def valid_pfsp_power(cls, value: float) -> float:
        """Reject invalid PFSP powers."""
        if value <= 0.0:
            raise ValueError("pfsp_power must be positive")
        return value

    @field_validator("pfsp_epsilon")
    @classmethod
    def valid_pfsp_epsilon(cls, value: float) -> float:
        """Reject invalid PFSP epsilon values."""
        if value < 0.0:
            raise ValueError("pfsp_epsilon must be non-negative")
        return value

    @field_validator("anchor_max_total_probability")
    @classmethod
    def valid_optional_anchor_cap(cls, value: float | None) -> float | None:
        """Restrict the optional aggregate anchor fraction to [0, 1]."""
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("anchor_max_total_probability must be in [0, 1]")
        return value

    @field_validator("anchor_opponent_id")
    @classmethod
    def valid_anchor_opponent_id(cls, value: str) -> str:
        """Reject an empty anchor identity."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("anchor_opponent_id must be non-empty")
        return cleaned

    @field_validator("scripted_opponents")
    @classmethod
    def valid_scripted_opponents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty scripted opponent names."""
        cleaned = tuple(name.strip() for name in value)
        if any(not name for name in cleaned):
            raise ValueError("scripted opponent names must be non-empty")
        return cleaned

    @field_validator(
        "candidate_deck_pool",
        "additional_candidate_deck_pool",
        "opponent_deck_pool",
        "additional_opponent_deck_pool",
    )
    @classmethod
    def valid_deck_pool_paths(
        cls,
        value: tuple[WeightedDeckEntry, ...],
    ) -> tuple[WeightedDeckEntry, ...]:
        """Reject repeated deck paths in one weighted pool."""
        seen: set[str] = set()
        for entry in value:
            key = str(entry.path)
            if key in seen:
                raise ValueError(f"deck pool paths must be unique: {entry.path}")
            seen.add(key)
        return value

    @model_validator(mode="after")
    def valid_combined_deck_pools(self) -> CurriculumConfig:
        """Reject path or label collisions after additive pool composition."""
        for pool_name, entries in (
            (
                "candidate",
                self.candidate_deck_pool + self.additional_candidate_deck_pool,
            ),
            (
                "opponent",
                self.opponent_deck_pool + self.additional_opponent_deck_pool,
            ),
        ):
            paths = [str(entry.path) for entry in entries]
            if len(paths) != len(set(paths)):
                raise ValueError(f"{pool_name} deck pool paths must be unique")
            labels = [entry.label for entry in entries if entry.label]
            if len(labels) != len(set(labels)):
                raise ValueError(f"{pool_name} deck pool labels must be unique")
        fixed_ids = [bundle.opponent_id for bundle in self.fixed_frozen_bundles]
        fixed_ids.extend(policy.opponent_id for policy in self.fixed_frozen_policies)
        if len(fixed_ids) != len(set(fixed_ids)):
            raise ValueError("fixed frozen opponent_id values must be unique")
        if self.assignment_schedule.mode == "stratified":
            schedule = self.assignment_schedule
            if schedule.execution_lanes > 0:
                if schedule.lane_block_size % self.assignment_block_size != 0:
                    raise ValueError(
                        "stratified lane_block_size must be divisible by "
                        "assignment_block_size"
                    )
                if schedule.lane_block_size % schedule.reservation_size != 0:
                    raise ValueError(
                        "stratified lane_block_size must be divisible by "
                        "reservation_size"
                    )
            elif schedule.reservation_size % self.assignment_block_size != 0:
                raise ValueError(
                    "stratified assignment reservation_size must be divisible by "
                    "assignment_block_size"
                )
        if self.assignment_schedule.mode == "stratified" and (
            not self.static_opponent_schedule.enabled
            or self.candidate_priority.enabled
            or self.matchup_priority.enabled
        ):
            raise ValueError(
                "stratified assignment scheduling requires a static opponent "
                "schedule with online candidate/matchup priorities disabled"
            )
        if self.mix.hard > 0.0 and self.static_opponent_schedule.enabled:
            raise ValueError(
                "hard frozen sampling requires live PFSP; static opponent "
                "schedules bypass online difficulty weights"
            )
        return self

    def candidate_deck_labels(self) -> tuple[str, ...]:
        """Return candidate labels without loading deck contents."""
        entries = self.candidate_deck_pool + self.additional_candidate_deck_pool
        if entries:
            return tuple(_deck_label(entry.path, entry.label) for entry in entries)
        return (_deck_label(self.candidate_deck_path, ""),)


@dataclass(frozen=True)
class _WeightedDeck:
    """Loaded deck plus normalized sampling metadata."""

    cards: tuple[int, ...]
    weight: float
    label: str
    source_path: Path


@dataclass(frozen=True)
class GameAssignment:
    """One per-game curriculum assignment."""

    opponent_kind: OpponentKind
    opponent_id: str
    candidate_deck: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    candidate_deck_label: str = ""
    opponent_deck_label: str = ""
    candidate_lane: str = "broad"
    frozen_sampling_lane: FrozenSamplingLane | None = None
    train_opponent_seat: bool = False


@dataclass(frozen=True)
class CurriculumOutcome:
    """Finished-game result used to update online curriculum statistics."""

    opponent_kind: OpponentKind
    opponent_id: str
    candidate_reward: float
    candidate_deck_label: str = ""
    opponent_deck_label: str = ""
    candidate_policy_version: int | None = None


class CurriculumSampler:
    """Sample opponent/deck assignments and maintain PFSP-lite frozen state."""

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        *,
        state: FrozenPoolState | None = None,
        rng: random.Random | None = None,
        registry: Mapping[str, OpponentSpec] | None = None,
    ) -> None:
        """Initialize decks, scripted specs, frozen state, and RNG."""
        self.config = config or CurriculumConfig()
        self._rng = rng or random.Random()
        self._registry = dict(registry or opponent_registry())
        self._candidate_decks = self._load_candidate_decks()
        self._opponent_decks = self._load_opponent_decks()
        self._fixed_frozen_decks = self._load_fixed_frozen_decks()
        self._validate_fixed_frozen_policy_checkpoints()
        if not self._candidate_decks:
            raise ValueError("curriculum needs at least one candidate deck")
        if not self._opponent_decks:
            raise ValueError("curriculum needs at least one meta deck")
        self._validate_candidate_lanes()
        self._validate_opponent_focus()
        self._validate_static_opponent_schedule()
        self._scripted_specs = tuple(
            self._scripted_spec(name) for name in self.config.scripted_opponents
        )
        self._scripted_decks = {
            spec.name: _WeightedDeck(
                cards=_read_deck(spec.deck_path),
                weight=1.0,
                label=spec.name,
                source_path=spec.deck_path,
            )
            for spec in self._scripted_specs
            if spec.deck_path is not None
        }
        self._frozen_probe_sampling_fraction = 0.0
        self._frozen_probe_member_weights: dict[str, float] = {}
        self._frozen_probe_deck_weights: dict[str, dict[str, float]] = {}
        self._state = state or self._load_state_or_default()
        self._seed_fixed_frozen_members()
        self._enforce_capacity()
        self._recover_assignment_reservations()
        self._assignment_cycle: DeterministicWeightedCycle[GameAssignment] | None = None
        self._blocked_assignment: GameAssignment | None = None
        self._blocked_assignment_remaining = 0

    @property
    def frozen_members(self) -> tuple[FrozenPoolMember, ...]:
        """Return current frozen pool members."""
        return self._state.members

    @property
    def scripted_specs(self) -> tuple[OpponentSpec, ...]:
        """Return configured scripted opponents (search-based ones rejected)."""
        return self._scripted_specs

    @property
    def opponent_deck_labels(self) -> tuple[str, ...]:
        """Return the exact opponent-deck labels available to frozen pilots."""
        return tuple(deck.label for deck in self._opponent_decks)

    def opponent_deck_base_probabilities(self) -> dict[str, float]:
        """Return normalized deployment-independent opponent-deck weights."""
        weights = self._opponent_deck_base_weights()
        total = sum(weights)
        if total <= 0.0:
            return {}
        return {
            deck.label: weight / total
            for deck, weight in zip(self._opponent_decks, weights, strict=True)
        }

    @property
    def candidate_deck_distribution(self) -> tuple[dict[str, object], ...]:
        """Return configured candidate deck sampling metadata."""
        total = sum(deck.weight for deck in self._candidate_decks)
        if total <= 0.0:
            return ()
        target_probabilities = self.candidate_deck_sampling_probabilities()
        return tuple(
            {
                "label": deck.label,
                "path": deck_records.display_path(
                    deck_records.repo_path(deck.source_path)
                ),
                "weight": deck.weight,
                "probability": deck.weight / total,
                "target_probability": target_probabilities.get(deck.label, 0.0),
                "winrate_ema": self._candidate_deck_ema(deck.label),
                "lane": self._candidate_lane(deck.label),
            }
            for deck in self._candidate_decks
        )

    @property
    def candidate_lane_distribution(self) -> tuple[dict[str, object], ...]:
        """Return normalized configured mass and deck counts for every lane."""
        probabilities = self.candidate_deck_sampling_probabilities()
        rows: list[dict[str, object]] = []
        for lane in CANDIDATE_LANE_ORDER:
            labels = tuple(
                deck.label
                for deck in self._candidate_decks
                if self._candidate_lane(deck.label) == lane
            )
            rows.append(
                {
                    "lane": lane,
                    "labels": labels,
                    "decks": len(labels),
                    "target_probability": sum(
                        probabilities.get(label, 0.0) for label in labels
                    ),
                }
            )
        return tuple(rows)

    @property
    def matchup_statistics(self) -> tuple[dict[str, object], ...]:
        """Return observable games/EMA rows for exact curriculum triples."""
        rows: list[dict[str, object]] = []
        for key, statistic in self._state.matchup_statistics.items():
            candidate, opponent, pilot = parse_matchup_key(key)
            rows.append(
                {
                    "candidate_deck": candidate,
                    "opponent_deck": opponent,
                    "opponent_pilot": pilot,
                    "games": statistic.games,
                    "winrate_ema": statistic.winrate_ema,
                }
            )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    str(row["candidate_deck"]),
                    str(row["opponent_deck"]),
                    str(row["opponent_pilot"]),
                ),
            )
        )

    def assign(self, *, lane_index: int | None = None) -> GameAssignment:
        """Sample one per-game opponent/deck assignment."""
        if self.config.assignment_schedule.mode == "stratified":
            return self._assign_stratified(lane_index=lane_index)
        if lane_index is not None:
            raise ValueError("assignment lanes require stratified scheduling")
        if self._blocked_assignment_remaining > 0:
            if self._blocked_assignment is None:
                raise RuntimeError("curriculum assignment block lost its template")
            self._blocked_assignment_remaining -= 1
            return self._blocked_assignment

        assignment = self._sample_assignment()
        self._blocked_assignment = assignment
        self._blocked_assignment_remaining = self.config.assignment_block_size - 1
        return assignment

    def finalize_assignment_state(self) -> None:
        """Commit the exact cursor before a graceful worker shutdown."""
        if self.config.assignment_schedule.mode != "stratified":
            return
        updates: dict[str, object] = {
            "assignment_reserved_until": self._state.assignment_cursor
        }
        if self.config.assignment_schedule.execution_lanes > 0:
            updates["assignment_lanes"] = {
                key: lane.model_copy(update={"reserved_until": lane.cursor})
                for key, lane in self._state.assignment_lanes.items()
            }
        self._state = self._state.model_copy(update=updates)

    def _assign_stratified(self, *, lane_index: int | None) -> GameAssignment:
        """Serve one persisted, route-local assignment without RNG coupling.

        The weighted cycle controls the statistical order of matchup templates.
        A short assignment block repeats each template only long enough for the
        concurrently running actors to share an efficient inference route.  The
        durable cursor remains game-based so restart reservations have the same
        meaning regardless of the hardware block size.
        """
        cycle = self._stratified_assignment_cycle()
        persisted_fingerprint = self._state.assignment_schedule_fingerprint
        if persisted_fingerprint and persisted_fingerprint != cycle.fingerprint:
            raise RuntimeError(
                "persisted assignment schedule differs from the active static roster"
            )
        if not persisted_fingerprint:
            self._state = self._state.model_copy(
                update={"assignment_schedule_fingerprint": cycle.fingerprint}
            )
        if self.config.assignment_schedule.execution_lanes > 0:
            if lane_index is None:
                raise ValueError("lane_index is required by the assignment schedule")
            return self._assign_stratified_lane(cycle, lane_index=lane_index)
        if lane_index is not None:
            raise ValueError("lane_index requires assignment execution lanes")
        cursor = self._state.assignment_cursor
        if cursor >= self._state.assignment_reserved_until:
            reserved_until = cursor + self.config.assignment_schedule.reservation_size
            self._state = self._state.model_copy(
                update={"assignment_reserved_until": reserved_until}
            )
            if self.config.frozen_state_path is not None:
                self.save_state()
        template_index = cursor // self.config.assignment_block_size
        assignment = cycle.value_at(template_index)
        self._state = self._state.model_copy(update={"assignment_cursor": cursor + 1})
        return assignment

    def _assign_stratified_lane(
        self,
        cycle: DeterministicWeightedCycle[GameAssignment],
        *,
        lane_index: int,
    ) -> GameAssignment:
        """Serve one assignment from a durable, route-sticky execution lane."""
        schedule = self.config.assignment_schedule
        if lane_index < 0 or lane_index >= schedule.execution_lanes:
            raise ValueError(
                f"assignment lane_index {lane_index} is outside "
                f"[0, {schedule.execution_lanes})"
            )
        layout_fingerprint = self._assignment_layout_fingerprint(cycle)
        persisted_layout = self._state.assignment_layout_fingerprint
        if persisted_layout and persisted_layout != layout_fingerprint:
            raise RuntimeError(
                "persisted assignment lane layout differs from the active geometry"
            )
        if not persisted_layout:
            # A global assignment stream may be migrated at a partially consumed
            # hardware tile. Start after that tile, then allocate consecutive
            # templates dynamically as lanes need them. This preserves the global
            # exact-quota order without binding particular matchups to faster lanes.
            next_template = self._state.assignment_next_template_index
            if next_template == 0 and self._state.assignment_cursor > 0:
                next_template = math.ceil(
                    self._state.assignment_cursor / self.config.assignment_block_size
                )
            self._state = self._state.model_copy(
                update={
                    "assignment_layout_fingerprint": layout_fingerprint,
                    "assignment_next_template_index": next_template,
                }
            )

        lane_key = str(lane_index)
        lanes = dict(self._state.assignment_lanes)
        lane = lanes.get(lane_key)
        must_persist_reservation = False
        if lane is None or lane.cursor >= schedule.lane_block_size:
            previous_template_index = None if lane is None else lane.template_index
            lane = AssignmentLaneState(
                template_index=self._take_next_lane_template(
                    cycle,
                    previous_template_index=previous_template_index,
                )
            )
            lanes[lane_key] = lane
            self._state = self._state.model_copy(
                update={
                    "assignment_lanes": lanes,
                }
            )
            must_persist_reservation = True

        if lane.cursor >= lane.reserved_until:
            lane = lane.model_copy(
                update={
                    "reserved_until": min(
                        schedule.lane_block_size,
                        lane.cursor + schedule.reservation_size,
                    )
                }
            )
            lanes = dict(self._state.assignment_lanes)
            lanes[lane_key] = lane
            self._state = self._state.model_copy(update={"assignment_lanes": lanes})
            must_persist_reservation = True
        if must_persist_reservation and self.config.frozen_state_path is not None:
            # Publish the reservation before the assignment can leave this process.
            self.save_state()

        assignment = cycle.value_at(lane.template_index)
        lane = lane.model_copy(update={"cursor": lane.cursor + 1})
        lanes = dict(self._state.assignment_lanes)
        lanes[lane_key] = lane
        cursor = self._state.assignment_cursor + 1
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": cursor,
                "assignment_reserved_until": cursor,
                "assignment_lanes": lanes,
            }
        )
        return assignment

    def _take_next_lane_template(
        self,
        cycle: DeterministicWeightedCycle[GameAssignment],
        *,
        previous_template_index: int | None,
    ) -> int:
        """Take one bounded-reorder template that reuses an outgoing route."""
        schedule = self.config.assignment_schedule
        window_size = max(
            1,
            schedule.execution_lanes * _LANE_REORDER_WINDOW_MULTIPLIER,
        )
        pending = list(self._state.assignment_pending_template_indices)
        next_index = self._state.assignment_next_template_index
        while len(pending) < window_size:
            pending.append(next_index)
            next_index += 1

        oldest = min(pending)
        force_oldest = next_index - oldest >= window_size * 2
        if previous_template_index is None or force_oldest:
            selected = oldest
        else:
            previous = cycle.value_at(previous_template_index)
            selected = max(
                pending,
                key=lambda index: (
                    _assignment_route_overlap(previous, cycle.value_at(index)),
                    -index,
                ),
            )
        pending.remove(selected)
        self._state = self._state.model_copy(
            update={
                "assignment_next_template_index": next_index,
                "assignment_pending_template_indices": tuple(pending),
            }
        )
        return selected

    def _assignment_layout_fingerprint(
        self,
        cycle: DeterministicWeightedCycle[GameAssignment],
    ) -> str:
        """Hash the execution geometry separately from statistical quotas."""
        schedule = self.config.assignment_schedule
        digest = hashlib.sha256()
        digest.update(b"ptcg-rl/curriculum-execution-lanes/v2\0")
        for value in (
            cycle.fingerprint,
            str(self.config.assignment_block_size),
            str(schedule.execution_lanes),
            str(schedule.lane_block_size),
            str(schedule.reservation_size),
            str(_LANE_REORDER_WINDOW_MULTIPLIER),
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _recover_assignment_reservations(self) -> None:
        """Skip possibly issued positions after an unclean sampler shutdown."""
        schedule = self.config.assignment_schedule
        if schedule.mode != "stratified":
            return
        if schedule.execution_lanes <= 0 or not self._state.assignment_lanes:
            if self._state.assignment_cursor < self._state.assignment_reserved_until:
                self._state = self._state.model_copy(
                    update={
                        "assignment_cursor": self._state.assignment_reserved_until,
                    }
                )
            return
        skipped = 0
        recovered: dict[str, AssignmentLaneState] = {}
        for key, lane in self._state.assignment_lanes.items():
            skipped += lane.reserved_until - lane.cursor
            recovered[key] = lane.model_copy(update={"cursor": lane.reserved_until})
        cursor = self._state.assignment_cursor + skipped
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": cursor,
                "assignment_reserved_until": cursor,
                "assignment_lanes": recovered,
            }
        )

    def _stratified_assignment_cycle(
        self,
    ) -> DeterministicWeightedCycle[GameAssignment]:
        """Build and cache the immutable weighted assignment population."""
        if self._assignment_cycle is not None:
            return self._assignment_cycle
        items = self._weighted_assignment_population()
        configured_size = self.config.assignment_schedule.cycle_size
        cycle_size = configured_size or automatic_cycle_size(items)
        self._assignment_cycle = DeterministicWeightedCycle(
            items,
            cycle_size=cycle_size,
            seed=self.config.assignment_schedule.seed,
        )
        return self._assignment_cycle

    def _weighted_assignment_population(
        self,
    ) -> tuple[WeightedCycleItem[GameAssignment], ...]:
        """Enumerate the exact static hierarchy and its target joint masses."""
        weighted: dict[str, tuple[GameAssignment, float]] = {}

        def add(assignment: GameAssignment, weight: float) -> None:
            if weight <= 0.0:
                return
            key = _assignment_identity(assignment)
            previous = weighted.get(key)
            if previous is None:
                weighted[key] = (assignment, weight)
                return
            if previous[0] != assignment:
                raise RuntimeError("assignment identity collision")
            weighted[key] = (assignment, previous[1] + weight)

        kinds, kind_weights = self._kind_choices_and_weights()
        kind_probabilities = _normalize_positive_weights(kind_weights)
        candidate_probabilities = _normalize_positive_weights(
            self._candidate_deck_weights()
        )
        for kind, kind_probability in zip(
            kinds,
            kind_probabilities,
            strict=True,
        ):
            for candidate, candidate_probability in zip(
                self._candidate_decks,
                candidate_probabilities,
                strict=True,
            ):
                base_probability = kind_probability * candidate_probability
                if kind == "self_play":
                    mirror_probability = self.config.mirror_selfplay_probability
                    add(
                        self._assignment_for_decks(
                            kind="self_play",
                            opponent_id="self_play",
                            candidate=candidate,
                            opponent=candidate,
                            train_opponent_seat=True,
                        ),
                        base_probability * mirror_probability,
                    )
                    opponent_probabilities = _normalize_positive_weights(
                        self._scheduled_opponent_deck_weights(candidate.label)
                    )
                    for opponent, opponent_probability in zip(
                        self._opponent_decks,
                        opponent_probabilities,
                        strict=True,
                    ):
                        add(
                            self._assignment_for_decks(
                                kind="self_play",
                                opponent_id="self_play",
                                candidate=candidate,
                                opponent=opponent,
                                train_opponent_seat=(
                                    self.config.train_offdeck_selfplay
                                ),
                            ),
                            base_probability
                            * (1.0 - mirror_probability)
                            * opponent_probability,
                        )
                    continue
                if kind == "frozen":
                    member_probabilities = _normalize_positive_weights(
                        self._frozen_member_weights(
                            candidate_deck_label=candidate.label
                        )
                    )
                    for member, member_probability in zip(
                        self._state.members,
                        member_probabilities,
                        strict=True,
                    ):
                        fixed_deck = self._fixed_frozen_decks.get(member.opponent_id)
                        if fixed_deck is not None:
                            add(
                                self._assignment_for_decks(
                                    kind="frozen",
                                    opponent_id=member.opponent_id,
                                    candidate=candidate,
                                    opponent=fixed_deck,
                                    train_opponent_seat=False,
                                ),
                                base_probability * member_probability,
                            )
                            continue
                        opponent_probabilities = _normalize_positive_weights(
                            self._frozen_opponent_deck_weights(
                                member,
                                candidate_deck_label=candidate.label,
                            )
                        )
                        for opponent, opponent_probability in zip(
                            self._opponent_decks,
                            opponent_probabilities,
                            strict=True,
                        ):
                            add(
                                self._assignment_for_decks(
                                    kind="frozen",
                                    opponent_id=member.opponent_id,
                                    candidate=candidate,
                                    opponent=opponent,
                                    train_opponent_seat=False,
                                ),
                                base_probability
                                * member_probability
                                * opponent_probability,
                            )
                    continue
                scripted_probability = 1.0 / len(self._scripted_specs)
                for spec in self._scripted_specs:
                    fixed_deck = self._scripted_decks.get(spec.name)
                    if fixed_deck is not None:
                        add(
                            self._assignment_for_decks(
                                kind="scripted",
                                opponent_id=spec.name,
                                candidate=candidate,
                                opponent=fixed_deck,
                                train_opponent_seat=False,
                            ),
                            base_probability * scripted_probability,
                        )
                        continue
                    opponent_probabilities = _normalize_positive_weights(
                        self._scheduled_opponent_deck_weights(candidate.label)
                    )
                    for opponent, opponent_probability in zip(
                        self._opponent_decks,
                        opponent_probabilities,
                        strict=True,
                    ):
                        add(
                            self._assignment_for_decks(
                                kind="scripted",
                                opponent_id=spec.name,
                                candidate=candidate,
                                opponent=opponent,
                                train_opponent_seat=False,
                            ),
                            base_probability
                            * scripted_probability
                            * opponent_probability,
                        )
        return tuple(
            WeightedCycleItem(key=key, value=assignment, weight=weight)
            for key, (assignment, weight) in sorted(weighted.items())
        )

    def _assignment_for_decks(
        self,
        *,
        kind: OpponentKind,
        opponent_id: str,
        candidate: _WeightedDeck,
        opponent: _WeightedDeck,
        train_opponent_seat: bool,
    ) -> GameAssignment:
        """Build one exact joint-cell assignment without sampling."""
        return GameAssignment(
            opponent_kind=kind,
            opponent_id=opponent_id,
            candidate_deck=candidate.cards,
            opponent_deck=opponent.cards,
            candidate_deck_label=candidate.label,
            opponent_deck_label=opponent.label,
            candidate_lane=self._candidate_lane(candidate.label),
            frozen_sampling_lane="pfsp" if kind == "frozen" else None,
            train_opponent_seat=train_opponent_seat,
        )

    def _sample_assignment(self) -> GameAssignment:
        """Sample a new assignment template for one route-local block."""
        kind = self._sample_kind()
        if kind == "self_play":
            return self._self_play_assignment()
        if kind == "frozen":
            return self._frozen_assignment(sampling_lane="pfsp")
        if kind == "hard":
            return self._frozen_assignment(sampling_lane="hard")
        return self._scripted_assignment()

    def observe(self, outcome: CurriculumOutcome) -> None:
        """Update online curriculum EMAs from one completed game."""
        score = _score_from_reward(outcome.candidate_reward)
        self._observe_candidate_deck(outcome.candidate_deck_label, score=score)
        self._observe_matchup(outcome, score=score)
        if outcome.opponent_kind != "frozen":
            return
        members: list[FrozenPoolMember] = []
        for member in self._state.members:
            if member.opponent_id != outcome.opponent_id:
                members.append(member)
                continue
            alpha = self.config.winrate_ema_alpha
            updated = (1.0 - alpha) * member.winrate_ema + alpha * score
            deck_winrate_ema = dict(member.deck_winrate_ema)
            opponent_deck_label = outcome.opponent_deck_label.strip()
            if opponent_deck_label:
                previous_deck_ema = deck_winrate_ema.get(
                    opponent_deck_label,
                    member.winrate_ema,
                )
                deck_winrate_ema[opponent_deck_label] = (
                    1.0 - alpha
                ) * previous_deck_ema + alpha * score
            members.append(
                member.model_copy(
                    update={
                        "winrate_ema": updated,
                        "games": member.games + 1,
                        "deck_winrate_ema": deck_winrate_ema,
                    }
                )
            )
        self._state = self._state.model_copy(update={"members": tuple(members)})

    def add_frozen_member(
        self,
        *,
        opponent_id: str,
        checkpoint_path: Path,
        winrate_ema: float = 0.5,
        pinned: bool = False,
        recurrent: bool = False,
    ) -> None:
        """Add one immutable frozen checkpoint opponent."""
        self._state = add_frozen_pool_member(
            self._state,
            opponent_id=opponent_id,
            checkpoint_path=checkpoint_path,
            winrate_ema=winrate_ema,
            pinned=pinned,
            recurrent=recurrent,
            capacity=self.config.frozen_capacity,
        )
        self._assignment_cycle = None
        self._blocked_assignment = None
        self._blocked_assignment_remaining = 0

    def remove_frozen_member(self, opponent_id: str) -> None:
        """Remove one non-pinned frozen member by immutable opponent identity."""
        member = self._frozen_member(opponent_id)
        if member is None:
            raise ValueError(f"unknown frozen opponent: {opponent_id}")
        if member.pinned:
            raise ValueError(f"cannot remove pinned frozen opponent: {opponent_id}")
        self._state = self._state.model_copy(
            update={
                "members": tuple(
                    existing
                    for existing in self._state.members
                    if existing.opponent_id != opponent_id
                )
            }
        )
        self._assignment_cycle = None
        self._blocked_assignment = None
        self._blocked_assignment_remaining = 0

    def retire_frozen_member(self, opponent_id: str) -> None:
        """Stop new assignments while retaining one in-flight opponent runtime."""
        member = self._frozen_member(opponent_id)
        if member is None:
            raise ValueError(f"unknown frozen opponent: {opponent_id}")
        if member.pinned:
            raise ValueError(f"cannot retire pinned frozen opponent: {opponent_id}")
        if member.retired:
            return
        self._state = self._state.model_copy(
            update={
                "members": tuple(
                    existing.model_copy(update={"retired": True})
                    if existing.opponent_id == opponent_id
                    else existing
                    for existing in self._state.members
                )
            }
        )
        self._assignment_cycle = None
        self._blocked_assignment = None
        self._blocked_assignment_remaining = 0

    def set_frozen_probe_schedule(
        self,
        *,
        member_weights: Mapping[str, float],
        deck_weights: Mapping[str, Mapping[str, float]],
        fraction: float,
    ) -> None:
        """Temporarily blend a balanced promotion probe into frozen sampling."""
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("frozen probe fraction must be in [0, 1]")
        member_ids = {member.opponent_id for member in self._state.members}
        unknown_members = set(member_weights) - member_ids
        unknown_members.update(set(deck_weights) - member_ids)
        if unknown_members:
            raise ValueError(
                f"frozen probe schedule references unknown members: {unknown_members}"
            )
        labels = set(self.opponent_deck_labels)
        unknown_labels = {
            label
            for weights in deck_weights.values()
            for label in weights
            if label not in labels
        }
        if unknown_labels:
            raise ValueError(
                f"frozen probe schedule references unknown decks: {unknown_labels}"
            )
        if any(weight < 0.0 for weight in member_weights.values()) or any(
            weight < 0.0
            for weights in deck_weights.values()
            for weight in weights.values()
        ):
            raise ValueError("frozen probe weights must be non-negative")
        self._frozen_probe_sampling_fraction = fraction
        self._frozen_probe_member_weights = dict(member_weights)
        self._frozen_probe_deck_weights = {
            member_id: dict(weights) for member_id, weights in deck_weights.items()
        }
        self._assignment_cycle = None

    def clear_frozen_probe_schedule(self) -> None:
        """Restore ordinary PFSP member and deck sampling."""
        self._frozen_probe_sampling_fraction = 0.0
        self._frozen_probe_member_weights = {}
        self._frozen_probe_deck_weights = {}
        self._assignment_cycle = None

    def frozen_sampling_probabilities(self) -> dict[str, float]:
        """Return normalized PFSP-lite probabilities by frozen opponent id."""
        return self._frozen_sampling_probabilities(sampling_lane="pfsp")

    def hard_frozen_sampling_probabilities(self) -> dict[str, float]:
        """Return hard-lane probabilities by frozen opponent id."""
        return self._frozen_sampling_probabilities(sampling_lane="hard")

    def _frozen_sampling_probabilities(
        self,
        *,
        sampling_lane: FrozenSamplingLane,
    ) -> dict[str, float]:
        """Return one normalized frozen sampling lane."""
        members = self._state.members
        if not members:
            return {}
        weights = self._frozen_member_weights(
            candidate_deck_label="",
            sampling_lane=sampling_lane,
        )
        total = sum(weights)
        return {
            member.opponent_id: weight / total
            for member, weight in zip(members, weights, strict=True)
        }

    def frozen_deck_sampling_probabilities(
        self,
        opponent_id: str,
        *,
        candidate_deck_label: str = "",
        sampling_lane: FrozenSamplingLane = "pfsp",
    ) -> dict[str, float]:
        """Return normalized opponent-deck probabilities for one frozen member."""
        member = self._frozen_member(opponent_id)
        if member is None:
            return {}
        fixed_deck = self._fixed_frozen_decks.get(opponent_id)
        if fixed_deck is not None:
            return {fixed_deck.label: 1.0}
        weights = self._frozen_opponent_deck_weights(
            member,
            candidate_deck_label=candidate_deck_label,
            sampling_lane=sampling_lane,
        )
        total = sum(weights)
        if total <= 0.0:
            return {}
        return {
            deck.label: weight / total
            for deck, weight in zip(self._opponent_decks, weights, strict=True)
        }

    def candidate_deck_sampling_probabilities(self) -> dict[str, float]:
        """Return normalized candidate-deck sampling probabilities."""
        weights = self._candidate_deck_weights()
        total = sum(weights)
        if total <= 0.0:
            return {}
        return {
            deck.label: weight / total
            for deck, weight in zip(self._candidate_decks, weights, strict=True)
        }

    def opponent_deck_sampling_probabilities(
        self,
        *,
        candidate_deck_label: str = "",
    ) -> dict[str, float]:
        """Return fixed opponent-deck probabilities for one candidate label."""
        weights = self._scheduled_opponent_deck_weights(candidate_deck_label)
        total = sum(weights)
        if total <= 0.0:
            return {}
        return {
            deck.label: weight / total
            for deck, weight in zip(self._opponent_decks, weights, strict=True)
        }

    def save_state(self, path: Path | None = None) -> None:
        """Persist frozen-pool state as a small JSON file."""
        output_path = path or self.config.frozen_state_path
        if output_path is None:
            raise ValueError("save_state requires a path")
        write_frozen_pool_state(output_path, self._state)

    def state(self) -> FrozenPoolState:
        """Return a copy of the current persistent state."""
        return self._state.model_copy(deep=True)

    def _sample_kind(self) -> _OpponentSamplingKind:
        choices, weights = self._kind_choices_and_weights()
        return _sample_weighted(self._rng, choices, weights)

    def _kind_choices_and_weights(
        self,
    ) -> tuple[list[_OpponentSamplingKind], list[float]]:
        """Return the available category hierarchy after the anchor cap."""
        choices: list[_OpponentSamplingKind] = []
        weights: list[float] = []
        if self.config.mix.self_play > 0.0:
            choices.append("self_play")
            weights.append(self.config.mix.self_play)
        if self.config.mix.frozen > 0.0 and self._state.members:
            choices.append("frozen")
            weights.append(self.config.mix.frozen)
        if self.config.mix.hard > 0.0 and self._state.members:
            choices.append("hard")
            weights.append(self.config.mix.hard)
        if self.config.mix.scripted > 0.0 and self._scripted_specs:
            choices.append("scripted")
            weights.append(self.config.mix.scripted)
        if not choices:
            raise ValueError("curriculum has no available opponent categories")
        self._apply_anchor_kind_cap(choices, weights)
        return (choices, weights)

    def _self_play_assignment(self) -> GameAssignment:
        candidate = self._sample_candidate_deck()
        mirror = self._rng.random() < self.config.mirror_selfplay_probability
        opponent = (
            candidate
            if mirror
            else self._sample_opponent_deck(
                candidate_deck_label=candidate.label,
                opponent_pilot="self_play",
            )
        )
        return GameAssignment(
            opponent_kind="self_play",
            opponent_id="self_play",
            candidate_deck=candidate.cards,
            opponent_deck=opponent.cards,
            candidate_deck_label=candidate.label,
            opponent_deck_label=opponent.label,
            candidate_lane=self._candidate_lane(candidate.label),
            train_opponent_seat=mirror or self.config.train_offdeck_selfplay,
        )

    def _frozen_assignment(
        self,
        *,
        sampling_lane: FrozenSamplingLane = "pfsp",
    ) -> GameAssignment:
        candidate = self._sample_candidate_deck()
        member = _sample_weighted(
            self._rng,
            self._state.members,
            self._frozen_member_weights(
                candidate_deck_label=candidate.label,
                sampling_lane=sampling_lane,
            ),
        )
        opponent = self._fixed_frozen_decks.get(member.opponent_id)
        if opponent is None:
            opponent = self._sample_frozen_opponent_deck(
                member,
                candidate_deck_label=candidate.label,
                sampling_lane=sampling_lane,
            )
        return GameAssignment(
            opponent_kind="frozen",
            opponent_id=member.opponent_id,
            candidate_deck=candidate.cards,
            opponent_deck=opponent.cards,
            candidate_deck_label=candidate.label,
            opponent_deck_label=opponent.label,
            candidate_lane=self._candidate_lane(candidate.label),
            frozen_sampling_lane=sampling_lane,
            train_opponent_seat=False,
        )

    def _scripted_assignment(self) -> GameAssignment:
        candidate = self._sample_candidate_deck()
        spec = self._rng.choice(self._scripted_specs)
        # Deck-specific scripted opponents (e.g. tier-3 public bots) must play
        # their own bundled deck; generic baselines sample from the pool.
        opponent = self._scripted_decks.get(spec.name) or self._sample_opponent_deck(
            candidate_deck_label=candidate.label,
            opponent_pilot=spec.name,
        )
        return GameAssignment(
            opponent_kind="scripted",
            opponent_id=spec.name,
            candidate_deck=candidate.cards,
            opponent_deck=opponent.cards,
            candidate_deck_label=candidate.label,
            opponent_deck_label=opponent.label,
            candidate_lane=self._candidate_lane(candidate.label),
            train_opponent_seat=False,
        )

    def _sample_candidate_deck(self) -> _WeightedDeck:
        return _sample_weighted(
            self._rng,
            self._candidate_decks,
            self._candidate_deck_weights(),
        )

    def _sample_opponent_deck(
        self,
        *,
        candidate_deck_label: str = "",
        opponent_pilot: str = "",
    ) -> _WeightedDeck:
        weights = list(self._scheduled_opponent_deck_weights(candidate_deck_label))
        if (
            not self.config.static_opponent_schedule.enabled
            and self.config.matchup_priority.enabled
            and candidate_deck_label
            and opponent_pilot
        ):
            weights = [
                weights[index]
                * self._matchup_priority_weight(
                    candidate_deck_label,
                    deck.label,
                    opponent_pilot,
                    fallback=self._candidate_deck_ema(candidate_deck_label),
                )
                for index, deck in enumerate(self._opponent_decks)
            ]
            priority = self.config.matchup_priority
            weights = list(
                _bounded_probabilities(
                    weights,
                    floor_ratio=priority.floor_ratio,
                    cap_ratio=priority.cap_ratio,
                )
            )
        return _sample_weighted(
            self._rng,
            self._opponent_decks,
            weights,
        )

    def _sample_frozen_opponent_deck(
        self,
        member: FrozenPoolMember,
        *,
        candidate_deck_label: str = "",
        sampling_lane: FrozenSamplingLane = "pfsp",
    ) -> _WeightedDeck:
        return _sample_weighted(
            self._rng,
            self._opponent_decks,
            self._frozen_opponent_deck_weights(
                member,
                candidate_deck_label=candidate_deck_label,
                sampling_lane=sampling_lane,
            ),
        )

    def _candidate_deck_weights(self) -> tuple[float, ...]:
        raw_weights = tuple(
            deck.weight
            * (
                self._candidate_deck_priority(deck.label)
                if self.config.candidate_priority.enabled
                else 1.0
            )
            for deck in self._candidate_decks
        )
        if self.config.candidate_lanes.enabled:
            bounded = list(raw_weights)
            for lane in CANDIDATE_LANE_ORDER:
                indices = [
                    index
                    for index, deck in enumerate(self._candidate_decks)
                    if self._candidate_lane(deck.label) == lane
                ]
                if not indices:
                    continue
                local = [raw_weights[index] for index in indices]
                if self.config.candidate_priority.enabled:
                    priority = self.config.candidate_priority
                    local = list(
                        _bounded_probabilities(
                            local,
                            floor_ratio=priority.floor_ratio,
                            cap_ratio=priority.cap_ratio,
                        )
                    )
                for index, weight in zip(indices, local, strict=True):
                    bounded[index] = weight
            return lane_probabilities(
                [deck.label for deck in self._candidate_decks],
                bounded,
                self.config.candidate_lanes,
            )
        if not self.config.candidate_priority.enabled:
            return raw_weights
        priority = self.config.candidate_priority
        return _bounded_probabilities(
            raw_weights,
            floor_ratio=priority.floor_ratio,
            cap_ratio=priority.cap_ratio,
        )

    def _candidate_deck_priority(self, label: str) -> float:
        priority = self.config.candidate_priority
        ema = self._candidate_deck_ema(label)
        return float((1.0 - ema) ** priority.power + priority.epsilon)

    def _candidate_deck_ema(self, label: str) -> float:
        return float(self._state.candidate_deck_winrate_ema.get(label, 0.5))

    def _observe_candidate_deck(self, raw_label: str, *, score: float) -> None:
        label = raw_label.strip() or "candidate"
        alpha = self.config.winrate_ema_alpha
        previous = self._state.candidate_deck_winrate_ema.get(label, 0.5)
        updated = (1.0 - alpha) * previous + alpha * score
        deck_winrate_ema = dict(self._state.candidate_deck_winrate_ema)
        deck_winrate_ema[label] = updated
        self._state = self._state.model_copy(
            update={"candidate_deck_winrate_ema": deck_winrate_ema}
        )

    def _observe_matchup(self, outcome: CurriculumOutcome, *, score: float) -> None:
        candidate = outcome.candidate_deck_label.strip() or "candidate"
        opponent = outcome.opponent_deck_label.strip() or "unknown"
        pilot = outcome.opponent_id.strip() or outcome.opponent_kind
        key = matchup_key(candidate, opponent, pilot)
        statistics = self._state.matchup_statistics
        previous = statistics.get(key)
        fallback = self._candidate_deck_ema(candidate)
        previous_ema = previous.winrate_ema if previous is not None else fallback
        alpha = self.config.winrate_ema_alpha
        statistics[key] = MatchupStatistic(
            games=(previous.games if previous is not None else 0) + 1,
            winrate_ema=(1.0 - alpha) * previous_ema + alpha * score,
        )

    def _frozen_opponent_deck_weights(
        self,
        member: FrozenPoolMember,
        *,
        candidate_deck_label: str = "",
        sampling_lane: FrozenSamplingLane = "pfsp",
    ) -> tuple[float, ...]:
        base_weights = self._scheduled_opponent_deck_weights(candidate_deck_label)
        if self.config.static_opponent_schedule.enabled:
            return self._apply_frozen_probe_deck_weights(
                member.opponent_id,
                base_weights,
            )
        weights = tuple(
            base_weight
            * self._frozen_deck_weight(
                member,
                deck.label,
                sampling_lane=sampling_lane,
            )
            for base_weight, deck in zip(
                base_weights,
                self._opponent_decks,
                strict=True,
            )
        )

        if sampling_lane == "hard":
            if sum(weights) > 0.0:
                return self._apply_frozen_probe_deck_weights(
                    member.opponent_id,
                    weights,
                )
            fallback_weights = tuple(
                base_weight
                * self._frozen_deck_weight(
                    member,
                    deck.label,
                    sampling_lane="pfsp",
                )
                for base_weight, deck in zip(
                    base_weights,
                    self._opponent_decks,
                    strict=True,
                )
            )
            return self._apply_frozen_probe_deck_weights(
                member.opponent_id,
                fallback_weights,
            )

        if not self.config.matchup_priority.enabled or not candidate_deck_label:
            return self._apply_frozen_probe_deck_weights(
                member.opponent_id,
                weights,
            )
        weights = tuple(
            base_weights[index]
            * self._matchup_priority_weight(
                candidate_deck_label,
                deck.label,
                member.opponent_id,
                fallback=member.deck_winrate_ema.get(
                    deck.label,
                    member.winrate_ema,
                ),
            )
            for index, deck in enumerate(self._opponent_decks)
        )
        priority = self.config.matchup_priority
        return self._apply_frozen_probe_deck_weights(
            member.opponent_id,
            _bounded_probabilities(
                weights,
                floor_ratio=priority.floor_ratio,
                cap_ratio=priority.cap_ratio,
            ),
        )

    def _frozen_deck_weight(
        self,
        member: FrozenPoolMember,
        label: str,
        *,
        sampling_lane: FrozenSamplingLane,
    ) -> float:
        ema = member.deck_winrate_ema.get(label, member.winrate_ema)
        epsilon = self.config.pfsp_epsilon if sampling_lane == "pfsp" else 0.0
        return float((1.0 - ema) ** self.config.pfsp_power + epsilon)

    def _frozen_weight(
        self,
        member: FrozenPoolMember,
        *,
        sampling_lane: FrozenSamplingLane,
    ) -> float:
        epsilon = self.config.pfsp_epsilon if sampling_lane == "pfsp" else 0.0
        return float((1.0 - member.winrate_ema) ** self.config.pfsp_power + epsilon)

    def _frozen_member_weights(
        self,
        *,
        candidate_deck_label: str,
        sampling_lane: FrozenSamplingLane = "pfsp",
    ) -> tuple[float, ...]:
        if self.config.static_opponent_schedule.enabled:
            return self._apply_frozen_probe_member_weights(
                tuple(
                    self._fixed_frozen_weight_multiplier(member.opponent_id)
                    for member in self._state.members
                )
            )
        if (
            sampling_lane == "hard"
            or not self.config.matchup_priority.enabled
            or not candidate_deck_label
        ):
            lane_weights = tuple(
                self._frozen_weight(member, sampling_lane=sampling_lane)
                * self._fixed_frozen_weight_multiplier(member.opponent_id)
                for member in self._state.members
            )
            if sum(lane_weights) > 0.0 or sampling_lane == "pfsp":
                return self._apply_frozen_probe_member_weights(lane_weights)
            return self._frozen_member_weights(
                candidate_deck_label=candidate_deck_label,
                sampling_lane="pfsp",
            )
        base_weights = self._opponent_deck_base_weights()
        weights: list[float] = []
        for member in self._state.members:
            fixed_deck = self._fixed_frozen_decks.get(member.opponent_id)
            if fixed_deck is not None:
                member_weight = self._matchup_priority_weight(
                    candidate_deck_label,
                    fixed_deck.label,
                    member.opponent_id,
                    fallback=member.deck_winrate_ema.get(
                        fixed_deck.label,
                        member.winrate_ema,
                    ),
                )
            else:
                deck_weights = [
                    base_weights[index]
                    * self._matchup_priority_weight(
                        candidate_deck_label,
                        deck.label,
                        member.opponent_id,
                        fallback=member.deck_winrate_ema.get(
                            deck.label,
                            member.winrate_ema,
                        ),
                    )
                    for index, deck in enumerate(self._opponent_decks)
                ]
                member_weight = sum(deck_weights) / max(1, len(deck_weights))
            weights.append(
                member_weight * self._fixed_frozen_weight_multiplier(member.opponent_id)
            )
        priority = self.config.matchup_priority
        return self._apply_frozen_probe_member_weights(
            _bounded_probabilities(
                weights,
                floor_ratio=priority.floor_ratio,
                cap_ratio=priority.cap_ratio,
            )
        )

    def _apply_frozen_probe_member_weights(
        self,
        weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        weights = tuple(
            0.0 if member.retired else weight
            for member, weight in zip(self._state.members, weights, strict=True)
        )
        probe = tuple(
            (
                0.0
                if member.retired
                else self._frozen_probe_member_weights.get(member.opponent_id, 0.0)
            )
            for member in self._state.members
        )
        return _blend_sampling_weights(
            weights,
            probe,
            fraction=self._frozen_probe_sampling_fraction,
        )

    def _apply_frozen_probe_deck_weights(
        self,
        opponent_id: str,
        weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        probe_by_label = self._frozen_probe_deck_weights.get(opponent_id, {})
        probe = tuple(
            probe_by_label.get(deck.label, 0.0) for deck in self._opponent_decks
        )
        return _blend_sampling_weights(
            weights,
            probe,
            fraction=self._frozen_probe_sampling_fraction,
        )

    def _fixed_frozen_weight_multiplier(self, opponent_id: str) -> float:
        """Return one configured frozen pilot's fixed sampling multiplier."""
        for bundle in self.config.fixed_frozen_bundles:
            if bundle.opponent_id == opponent_id:
                return bundle.weight
        for policy in self.config.fixed_frozen_policies:
            if policy.opponent_id == opponent_id:
                return policy.weight
        return 1.0

    def _matchup_priority_weight(
        self,
        candidate_deck: str,
        opponent_deck: str,
        opponent_pilot: str,
        *,
        fallback: float,
    ) -> float:
        priority = self.config.matchup_priority
        statistic = self._state.matchup_statistics.get(
            matchup_key(candidate_deck, opponent_deck, opponent_pilot)
        )
        ema = blended_matchup_ema(
            statistic,
            fallback=fallback,
            minimum_games=priority.minimum_games,
        )
        return float((1.0 - ema) ** priority.power + priority.epsilon)

    def _candidate_lane(self, label: str) -> str:
        if not self.config.candidate_lanes.enabled:
            return "broad"
        return self.config.candidate_lanes.lane_for(label)

    def _opponent_deck_base_weights(self) -> tuple[float, ...]:
        """Return base weights after applying the fixed opponent focus mass."""
        raw = tuple(deck.weight for deck in self._opponent_decks)
        focus = self.config.opponent_focus
        if not focus.enabled:
            return raw
        focus_labels = set(focus.label_weights)
        focus_total = sum(focus.label_weights.values())
        broad_total = sum(
            deck.weight
            for deck in self._opponent_decks
            if deck.label not in focus_labels
        )
        weights: list[float] = []
        for deck in self._opponent_decks:
            if deck.label in focus_labels:
                local_weight = focus.label_weights[deck.label] / focus_total
                weights.append(focus.probability * local_weight)
                continue
            if broad_total <= 0.0:
                weights.append(0.0)
                continue
            weights.append((1.0 - focus.probability) * deck.weight / broad_total)
        return tuple(weights)

    def _scheduled_opponent_deck_weights(
        self,
        candidate_deck_label: str,
    ) -> tuple[float, ...]:
        """Apply the fixed candidate-specific matrix to opponent base weights."""
        base_weights = self._opponent_deck_base_weights()
        schedule = self.config.static_opponent_schedule
        if not schedule.enabled:
            return base_weights
        return tuple(
            base_weight * schedule.matchup_multiplier(candidate_deck_label, deck.label)
            for base_weight, deck in zip(
                base_weights,
                self._opponent_decks,
                strict=True,
            )
        )

    def _validate_candidate_lanes(self) -> None:
        lanes = self.config.candidate_lanes
        if not lanes.enabled:
            return
        labels = {deck.label for deck in self._candidate_decks}
        unknown = (set(lanes.target_labels) | set(lanes.near_labels)) - labels
        if unknown:
            raise ValueError(
                f"candidate lane labels are absent from pool: {sorted(unknown)}"
            )
        counts = Counter(lanes.lane_for(label) for label in labels)
        for lane in CANDIDATE_LANE_ORDER:
            if lanes.lane_mass(lane) > 0.0 and counts[lane] <= 0:
                raise ValueError(
                    f"candidate lane {lane!r} has probability but no decks"
                )

    def _validate_opponent_focus(self) -> None:
        """Require every fixed-focus label and any requested broad remainder."""
        focus = self.config.opponent_focus
        if not focus.enabled:
            return
        labels = {deck.label for deck in self._opponent_decks}
        unknown = set(focus.label_weights) - labels
        if unknown:
            raise ValueError(
                f"opponent focus labels are absent from pool: {sorted(unknown)}"
            )
        if focus.probability < 1.0 and labels <= set(focus.label_weights):
            raise ValueError("opponent focus leaves broad mass but no broad decks")

    def _validate_static_opponent_schedule(self) -> None:
        """Reject static matrix rows that do not belong to the loaded roster."""
        schedule = self.config.static_opponent_schedule
        if not schedule.enabled:
            return
        candidate_labels = {deck.label for deck in self._candidate_decks}
        opponent_labels = {deck.label for deck in self._opponent_decks}
        unknown_candidates = set(schedule.matchup_multipliers) - candidate_labels
        if unknown_candidates:
            raise ValueError(
                "static schedule candidate labels are absent from pool: "
                f"{sorted(unknown_candidates)}"
            )
        unknown_opponents = {
            opponent
            for row in schedule.matchup_multipliers.values()
            for opponent in row
            if opponent not in opponent_labels
        }
        if unknown_opponents:
            raise ValueError(
                "static schedule opponent labels are absent from pool: "
                f"{sorted(unknown_opponents)}"
            )

    def _apply_anchor_kind_cap(
        self,
        choices: Sequence[_OpponentSamplingKind],
        weights: list[float],
    ) -> None:
        cap = self.config.anchor_max_total_probability
        frozen_indices = [
            index for index, kind in enumerate(choices) if kind in ("frozen", "hard")
        ]
        if cap is None or not frozen_indices or not self._state.members:
            return
        frozen_weight = 0.0
        anchor_weight = 0.0
        for index in frozen_indices:
            sampling_lane: FrozenSamplingLane = (
                "hard" if choices[index] == "hard" else "pfsp"
            )
            member_weights = self._frozen_member_weights(
                candidate_deck_label="",
                sampling_lane=sampling_lane,
            )
            member_total = sum(member_weights)
            if member_total <= 0.0:
                continue
            kind_weight = weights[index]
            frozen_weight += kind_weight
            anchor_given_kind = (
                sum(
                    weight
                    for member, weight in zip(
                        self._state.members,
                        member_weights,
                        strict=True,
                    )
                    if member.opponent_id == self.config.anchor_opponent_id
                )
                / member_total
            )
            anchor_weight += kind_weight * anchor_given_kind
        if frozen_weight <= 0.0 or anchor_weight <= cap * frozen_weight:
            return
        other_kind_weight = sum(
            weight
            for index, weight in enumerate(weights)
            if index not in frozen_indices
        )
        if other_kind_weight <= 0.0:
            raise ValueError(
                "anchor total-probability cap requires a non-frozen curriculum kind"
            )
        scale = min(
            1.0,
            cap * other_kind_weight / (anchor_weight - cap * frozen_weight),
        )
        for index in frozen_indices:
            weights[index] *= scale

    def _frozen_member(self, opponent_id: str) -> FrozenPoolMember | None:
        for member in self._state.members:
            if member.opponent_id == opponent_id:
                return member
        return None

    def _enforce_capacity(self) -> None:
        self._state = enforce_frozen_pool_capacity(
            self._state,
            capacity=self.config.frozen_capacity,
        )

    def _seed_fixed_frozen_members(self) -> None:
        """Ensure configured bundles and deck-independent pilots are pinned."""
        seeds = tuple(self.config.fixed_frozen_bundles) + tuple(
            self.config.fixed_frozen_policies
        )
        for seed in seeds:
            self._state = add_frozen_pool_member(
                self._state,
                opponent_id=seed.opponent_id,
                checkpoint_path=seed.checkpoint_path,
                winrate_ema=0.5,
                pinned=True,
                capacity=self.config.frozen_capacity,
            )

    def _load_state_or_default(self) -> FrozenPoolState:
        path = self.config.frozen_state_path
        if path is None:
            return FrozenPoolState()
        return read_frozen_pool_state(path)

    def _meta_deck_paths(self) -> tuple[Path, ...]:
        paths = list(self.config.meta_deck_paths)
        if self.config.include_public_fixture_decks:
            paths.extend(
                spec.deck_path
                for spec in self._registry.values()
                if spec.deck_path is not None
            )
        return tuple(_unique_paths(paths))

    def _load_candidate_decks(self) -> tuple[_WeightedDeck, ...]:
        entries = (
            self.config.candidate_deck_pool + self.config.additional_candidate_deck_pool
        )
        if entries:
            return tuple(_weighted_deck_from_entry(entry) for entry in entries)
        return (
            _WeightedDeck(
                cards=_read_deck(self.config.candidate_deck_path),
                weight=1.0,
                label=_deck_label(self.config.candidate_deck_path, ""),
                source_path=self.config.candidate_deck_path,
            ),
        )

    def _load_opponent_decks(self) -> tuple[_WeightedDeck, ...]:
        entries = (
            self.config.opponent_deck_pool + self.config.additional_opponent_deck_pool
        )
        if entries:
            return tuple(_weighted_deck_from_entry(entry) for entry in entries)
        return tuple(
            _WeightedDeck(
                cards=_read_deck(path),
                weight=1.0,
                label=_deck_label(path, ""),
                source_path=path,
            )
            for path in self._meta_deck_paths()
        )

    def _load_fixed_frozen_decks(self) -> dict[str, _WeightedDeck]:
        """Load exact decks and verify optional immutable checkpoint identities."""
        loaded: dict[str, _WeightedDeck] = {}
        for bundle in self.config.fixed_frozen_bundles:
            if bundle.checkpoint_size_bytes is not None:
                fingerprint = fingerprint_checkpoint(bundle.checkpoint_path)
                if fingerprint.size_bytes != bundle.checkpoint_size_bytes:
                    raise ValueError(
                        "fixed frozen checkpoint size mismatch: "
                        f"{bundle.checkpoint_path} "
                        f"expected={bundle.checkpoint_size_bytes} "
                        f"actual={fingerprint.size_bytes}"
                    )
                if fingerprint.sha256 != bundle.checkpoint_sha256:
                    raise ValueError(
                        "fixed frozen checkpoint SHA256 mismatch: "
                        f"{bundle.checkpoint_path} "
                        f"expected={bundle.checkpoint_sha256} "
                        f"actual={fingerprint.sha256}"
                    )
            loaded[bundle.opponent_id] = _WeightedDeck(
                cards=_read_deck(bundle.opponent_deck_path),
                weight=bundle.weight,
                label=bundle.opponent_deck_label,
                source_path=bundle.opponent_deck_path,
            )
        return loaded

    def _validate_fixed_frozen_policy_checkpoints(self) -> None:
        """Verify optional immutable identities for deck-independent pilots."""
        for policy in self.config.fixed_frozen_policies:
            if policy.checkpoint_size_bytes is None:
                continue
            fingerprint = fingerprint_checkpoint(policy.checkpoint_path)
            if fingerprint.size_bytes != policy.checkpoint_size_bytes:
                raise ValueError(
                    "fixed frozen checkpoint size mismatch: "
                    f"{policy.checkpoint_path} "
                    f"expected={policy.checkpoint_size_bytes} "
                    f"actual={fingerprint.size_bytes}"
                )
            if fingerprint.sha256 != policy.checkpoint_sha256:
                raise ValueError(
                    "fixed frozen checkpoint SHA256 mismatch: "
                    f"{policy.checkpoint_path} "
                    f"expected={policy.checkpoint_sha256} "
                    f"actual={fingerprint.sha256}"
                )

    def _scripted_spec(self, name: str) -> OpponentSpec:
        spec = self._registry[name]
        if spec.requires_search:
            # Search-based opponents drive the process-global engine Search
            # API, which conflicts with the vectorized rollout pool and the
            # rollout probe backends running in the same actor process.
            raise ValueError(
                f"scripted curriculum opponent requires engine search: {name}"
            )
        return spec


def _read_deck(path: Path) -> tuple[int, ...]:
    return tuple(deck_records.read_deck(deck_records.repo_path(path)))


def _weighted_deck_from_entry(entry: WeightedDeckEntry) -> _WeightedDeck:
    return _WeightedDeck(
        cards=_read_deck(entry.path),
        weight=entry.weight,
        label=_deck_label(entry.path, entry.label),
        source_path=entry.path,
    )


def _deck_label(path: Path, label: str) -> str:
    if label:
        return label
    return path.stem


def read_frozen_pool_state(path: Path) -> FrozenPoolState:
    """Read a frozen-pool state file, returning an empty state when absent."""
    if not path.exists():
        return FrozenPoolState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"frozen pool state must be a JSON object: {path}")
    return FrozenPoolState.model_validate(raw)


def write_frozen_pool_state(path: Path, state: FrozenPoolState) -> None:
    """Atomically write a frozen-pool state JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def emit_frozen_pool_addition(
    journal_dir: Path,
    *,
    opponent_id: str,
    checkpoint_path: Path,
    checkpoint_size_bytes: int | None = None,
    checkpoint_sha256: str | None = None,
    winrate_ema: float = 0.5,
    pinned: bool = False,
) -> Path:
    """Atomically queue one frozen-pool addition event."""
    if (checkpoint_size_bytes is None) != (checkpoint_sha256 is None):
        raise ValueError("checkpoint size and SHA256 must be provided together")
    if checkpoint_size_bytes is None or checkpoint_sha256 is None:
        fingerprint = fingerprint_checkpoint(checkpoint_path)
        checkpoint_size_bytes = fingerprint.size_bytes
        checkpoint_sha256 = fingerprint.sha256
    addition = FrozenPoolAddition(
        opponent_id=opponent_id,
        checkpoint_path=checkpoint_path,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_sha256=checkpoint_sha256,
        winrate_ema=winrate_ema,
        pinned=pinned,
    )
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = frozen_pool_addition_path(journal_dir, addition.opponent_id)
    tmp_path = journal_dir / f".{path.name}.{time.time_ns()}.tmp"
    tmp_path.write_text(
        json.dumps(addition.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def frozen_pool_addition_path(journal_dir: Path, opponent_id: str) -> Path:
    """Return the canonical journal path for one frozen opponent ID."""
    return journal_dir / f"{_safe_event_stem(opponent_id)}.json"


def read_frozen_pool_addition(path: Path) -> FrozenPoolAddition:
    """Read one frozen-pool addition event."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"frozen pool addition must be a JSON object: {path}")
    return FrozenPoolAddition.model_validate(raw)


def consume_frozen_pool_additions(
    sampler: CurriculumSampler,
    journal_dir: Path,
    *,
    state_path: Path | None = None,
) -> tuple[FrozenPoolAddition, ...]:
    """Persist queued additions before deleting their journal events."""
    if not journal_dir.exists():
        return ()
    output_path = state_path or sampler.config.frozen_state_path
    if output_path is None:
        raise ValueError("consuming frozen additions requires a state path")
    consumed: list[FrozenPoolAddition] = []
    consumed_paths: list[Path] = []
    pending = tuple(
        (path, read_frozen_pool_addition(path)) for path in journal_dir.glob("*.json")
    )
    for path, addition in sorted(
        pending,
        key=lambda item: (item[1].created_at_utc, item[0].name),
    ):
        verify_frozen_pool_addition(addition)
        sampler.add_frozen_member(
            opponent_id=addition.opponent_id,
            checkpoint_path=addition.checkpoint_path,
            winrate_ema=addition.winrate_ema,
            pinned=addition.pinned,
        )
        consumed.append(addition)
        consumed_paths.append(path)
    if not consumed:
        return ()
    sampler.save_state(output_path)
    for path in consumed_paths:
        path.unlink(missing_ok=True)
    return tuple(consumed)


def fingerprint_checkpoint(path: Path) -> CheckpointFingerprint:
    """Stream a checkpoint once and return its size and SHA256."""
    resolved = deck_records.repo_path(path)
    digest = hashlib.sha256()
    size_bytes = 0
    with resolved.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(_CHECKPOINT_HASH_CHUNK_BYTES):
            size_bytes += len(chunk)
            digest.update(chunk)
    return CheckpointFingerprint(size_bytes=size_bytes, sha256=digest.hexdigest())


def verify_frozen_pool_addition(addition: FrozenPoolAddition) -> None:
    """Reject an addition whose checkpoint is absent or has changed."""
    fingerprint = fingerprint_checkpoint(addition.checkpoint_path)
    if (
        addition.checkpoint_size_bytes is not None
        and fingerprint.size_bytes != addition.checkpoint_size_bytes
    ):
        raise ValueError(
            "frozen checkpoint size mismatch: "
            f"{addition.checkpoint_path} expected={addition.checkpoint_size_bytes} "
            f"actual={fingerprint.size_bytes}"
        )
    if (
        addition.checkpoint_sha256 is not None
        and fingerprint.sha256 != addition.checkpoint_sha256
    ):
        raise ValueError(
            "frozen checkpoint SHA256 mismatch: "
            f"{addition.checkpoint_path} expected={addition.checkpoint_sha256} "
            f"actual={fingerprint.sha256}"
        )


def add_frozen_pool_member(
    state: FrozenPoolState,
    *,
    opponent_id: str,
    checkpoint_path: Path,
    winrate_ema: float = 0.5,
    pinned: bool = False,
    recurrent: bool = False,
    capacity: int | None = None,
) -> FrozenPoolState:
    """Add one immutable frozen checkpoint member to persistent state."""
    existing = next(
        (member for member in state.members if member.opponent_id == opponent_id),
        None,
    )
    if existing is not None:
        same_checkpoint = (
            deck_records.repo_path(existing.checkpoint_path).resolve()
            == deck_records.repo_path(checkpoint_path).resolve()
        )
        if not same_checkpoint or existing.recurrent != recurrent:
            raise ValueError(
                "frozen opponent identity is immutable; use a new opponent_id "
                f"for checkpoint or topology changes: {opponent_id}"
            )
        if existing.pinned or not pinned:
            return state
        replacement = existing.model_copy(
            update={
                "pinned": True,
            }
        )
        updated = state.model_copy(
            update={
                "members": tuple(
                    replacement if member.opponent_id == opponent_id else member
                    for member in state.members
                )
            }
        )
        if capacity is None:
            return updated
        return enforce_frozen_pool_capacity(updated, capacity=capacity)
    member = FrozenPoolMember(
        opponent_id=opponent_id,
        checkpoint_path=checkpoint_path,
        winrate_ema=winrate_ema,
        added_order=state.next_added_order,
        pinned=pinned,
        recurrent=recurrent,
    )
    members = [
        existing for existing in state.members if existing.opponent_id != opponent_id
    ]
    updated = state.model_copy(
        update={
            "members": (*members, member),
            "next_added_order": state.next_added_order + 1,
        }
    )
    if capacity is None:
        return updated
    return enforce_frozen_pool_capacity(updated, capacity=capacity)


def enforce_frozen_pool_capacity(
    state: FrozenPoolState,
    *,
    capacity: int,
) -> FrozenPoolState:
    """Evict high-EMA non-pinned members until the pool is within capacity."""
    if capacity < 0:
        raise ValueError("frozen pool capacity must be non-negative")
    members = list(state.members)
    while _non_pinned_count(members) > capacity:
        evict = max(
            (member for member in members if not member.pinned),
            key=lambda member: (member.winrate_ema, -member.added_order),
        )
        members.remove(evict)
    return state.model_copy(update={"members": tuple(members)})


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return tuple(output)


def _sample_weighted(
    rng: random.Random,
    items: Sequence[_T],
    weights: Sequence[float],
) -> _T:
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("at least one sampling weight must be positive")
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights, strict=True):
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1]


def _normalize_positive_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite non-negative hierarchy weights to unit mass."""
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("assignment hierarchy weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("assignment hierarchy needs positive total mass")
    return tuple(weight / total for weight in weights)


def _blend_sampling_weights(
    base_weights: Sequence[float],
    probe_weights: Sequence[float],
    *,
    fraction: float,
) -> tuple[float, ...]:
    """Blend two sampling distributions without changing category-level mass."""
    if len(base_weights) != len(probe_weights):
        raise ValueError("base and probe distributions must have the same length")
    if fraction <= 0.0 or sum(probe_weights) <= 0.0:
        return tuple(base_weights)
    base = _normalize_positive_weights(base_weights)
    probe = _normalize_positive_weights(probe_weights)
    return tuple(
        (1.0 - fraction) * base_weight + fraction * probe_weight
        for base_weight, probe_weight in zip(base, probe, strict=True)
    )


def _assignment_route_overlap(left: GameAssignment, right: GameAssignment) -> int:
    """Count model/deck routes reusable across one lane transition."""

    def route_keys(assignment: GameAssignment) -> frozenset[tuple[str, object]]:
        keys: set[tuple[str, object]] = {("candidate", assignment.candidate_deck)}
        if assignment.opponent_kind == "self_play":
            keys.add(("candidate", assignment.opponent_deck))
        elif assignment.opponent_kind == "frozen":
            keys.add((assignment.opponent_id, assignment.opponent_deck))
        return frozenset(keys)

    return len(route_keys(left).intersection(route_keys(right)))


def _assignment_identity(assignment: GameAssignment) -> str:
    """Return a stable collision-resistant identity for one exact joint cell."""
    payload = (
        assignment.opponent_kind,
        assignment.opponent_id,
        assignment.candidate_deck_label,
        assignment.opponent_deck_label,
        assignment.candidate_lane,
        assignment.frozen_sampling_lane,
        assignment.train_opponent_seat,
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _bounded_probabilities(
    weights: Sequence[float],
    *,
    floor_ratio: float,
    cap_ratio: float,
) -> tuple[float, ...]:
    if not weights:
        return ()
    if any(weight < 0.0 for weight in weights):
        raise ValueError("sampling weights must be non-negative")
    count = len(weights)
    floor = floor_ratio / float(count)
    cap = cap_ratio / float(count)
    remaining = set(range(count))
    fixed: dict[int, float] = {}
    remaining_mass = 1.0
    while remaining:
        remaining_weight = sum(weights[index] for index in remaining)
        if remaining_weight <= 0.0:
            proposal = {
                index: remaining_mass / float(len(remaining)) for index in remaining
            }
        else:
            proposal = {
                index: remaining_mass * weights[index] / remaining_weight
                for index in remaining
            }
        to_fix = {
            index: cap for index, probability in proposal.items() if probability > cap
        }
        if not to_fix:
            to_fix = {
                index: floor
                for index, probability in proposal.items()
                if probability < floor
            }
        if not to_fix:
            fixed.update(proposal)
            break
        for index, probability in to_fix.items():
            fixed[index] = probability
            remaining.remove(index)
            remaining_mass -= probability
    return tuple(float(fixed[index]) for index in range(count))


def _score_from_reward(candidate_reward: float) -> float:
    if candidate_reward > 0.0:
        return 1.0
    if candidate_reward < 0.0:
        return 0.0
    return 0.5


def _non_pinned_count(members: Sequence[FrozenPoolMember]) -> int:
    return sum(1 for member in members if not member.pinned)


def _safe_event_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
