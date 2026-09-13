"""Specialist lane and matchup-PFSP configuration for RL curriculum."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CandidateLane = Literal["target", "near", "broad"]
CANDIDATE_LANE_ORDER: tuple[CandidateLane, ...] = ("target", "near", "broad")


class CandidateLaneConfig(BaseModel):
    """Two-level target/near/broad candidate-deck allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    target_labels: tuple[str, ...] = ()
    near_labels: tuple[str, ...] = ()
    target_probability: float = 0.70
    near_probability: float = 0.15
    broad_probability: float = 0.15

    @field_validator("target_probability", "near_probability", "broad_probability")
    @classmethod
    def non_negative_probability(cls, value: float) -> float:
        """Reject invalid lane masses."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("candidate lane probabilities must be finite and non-negative")
        return value

    @field_validator("target_labels", "near_labels")
    @classmethod
    def clean_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize and reject duplicate or empty lane labels."""
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("candidate lane labels must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("candidate lane labels must be unique")
        return cleaned

    @model_validator(mode="after")
    def valid_lanes(self) -> CandidateLaneConfig:
        """Require disjoint labels and a positive enabled lane distribution."""
        overlap = set(self.target_labels) & set(self.near_labels)
        if overlap:
            raise ValueError(f"candidate lane labels overlap: {sorted(overlap)}")
        if self.enabled:
            if not self.target_labels:
                raise ValueError("enabled candidate lanes require target_labels")
            if self.target_probability <= 0.0:
                raise ValueError("enabled candidate lanes require target probability")
            if self.probability_total <= 0.0:
                raise ValueError("enabled candidate lane probability total must be positive")
        return self

    @property
    def probability_total(self) -> float:
        """Return the configured lane-mass denominator."""
        return self.target_probability + self.near_probability + self.broad_probability

    def lane_for(self, label: str) -> CandidateLane:
        """Return the configured lane for one candidate label."""
        if label in self.target_labels:
            return "target"
        if label in self.near_labels:
            return "near"
        return "broad"

    def lane_mass(self, lane: CandidateLane) -> float:
        """Return the unnormalized configured probability for one lane."""
        if lane == "target":
            return self.target_probability
        if lane == "near":
            return self.near_probability
        return self.broad_probability


class MatchupPriorityConfig(BaseModel):
    """Low-sample-safe PFSP settings for exact deployment matchups."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    minimum_games: int = 32
    power: float = 2.0
    epsilon: float = 0.1
    floor_ratio: float = 0.5
    cap_ratio: float = 3.0

    @field_validator("minimum_games")
    @classmethod
    def positive_minimum_games(cls, value: int) -> int:
        """Require a positive fallback sample count."""
        if value <= 0:
            raise ValueError("matchup minimum_games must be positive")
        return value

    @field_validator("power")
    @classmethod
    def positive_power(cls, value: float) -> float:
        """Require a finite positive PFSP power."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("matchup priority power must be finite and positive")
        return value

    @field_validator("epsilon", "floor_ratio", "cap_ratio")
    @classmethod
    def non_negative_finite(cls, value: float) -> float:
        """Reject non-finite or negative PFSP bounds."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("matchup priority settings must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def valid_bounds(self) -> MatchupPriorityConfig:
        """Reject floor/cap settings that cannot bound a base distribution."""
        if self.floor_ratio > 1.0:
            raise ValueError("matchup floor_ratio must be <= 1")
        if self.cap_ratio < 1.0:
            raise ValueError("matchup cap_ratio must be >= 1")
        return self


class MatchupStatistic(BaseModel):
    """Games and candidate score EMA for one exact curriculum cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int = Field(default=0, ge=0)
    winrate_ema: float = Field(default=0.5, ge=0.0, le=1.0)


def matchup_key(
    candidate_deck: str,
    opponent_deck: str,
    opponent_pilot: str,
) -> str:
    """Encode one triple without delimiter ambiguity."""
    values = (candidate_deck.strip(), opponent_deck.strip(), opponent_pilot.strip())
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def parse_matchup_key(key: str) -> tuple[str, str, str]:
    """Decode and validate one persisted matchup key."""
    values = json.loads(key)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str)
        or len(values) != 3
        or any(not isinstance(value, str) for value in values)
    ):
        raise ValueError(f"invalid matchup statistic key: {key!r}")
    return (str(values[0]), str(values[1]), str(values[2]))


def validate_matchup_statistics(
    values: Mapping[str, MatchupStatistic],
) -> dict[str, MatchupStatistic]:
    """Normalize persisted mapping keys and reject malformed triples."""
    normalized: dict[str, MatchupStatistic] = {}
    for key, statistic in values.items():
        candidate, opponent, pilot = parse_matchup_key(str(key))
        canonical = matchup_key(candidate, opponent, pilot)
        if canonical in normalized:
            raise ValueError(f"duplicate matchup statistic: {canonical}")
        normalized[canonical] = statistic
    return normalized


def blended_matchup_ema(
    statistic: MatchupStatistic | None,
    *,
    fallback: float,
    minimum_games: int,
) -> float:
    """Blend sparse cells toward the aggregate EMA until sufficiently sampled."""
    if statistic is None or statistic.games <= 0:
        return float(fallback)
    confidence = min(1.0, statistic.games / float(minimum_games))
    return float(
        (1.0 - confidence) * fallback + confidence * statistic.winrate_ema
    )


def lane_probabilities(
    labels: Sequence[str],
    within_lane_weights: Sequence[float],
    config: CandidateLaneConfig,
) -> tuple[float, ...]:
    """Allocate fixed lane masses, then normalize PLR only within each lane."""
    if len(labels) != len(within_lane_weights):
        raise ValueError("candidate labels and weights must align")
    if not labels:
        return ()
    if not config.enabled:
        return _normalize(within_lane_weights)
    indices_by_lane: dict[CandidateLane, list[int]] = {
        "target": [],
        "near": [],
        "broad": [],
    }
    for index, label in enumerate(labels):
        indices_by_lane[config.lane_for(label)].append(index)
    available_lanes: tuple[CandidateLane, ...] = tuple(
        lane
        for lane in CANDIDATE_LANE_ORDER
        if indices_by_lane[lane] and config.lane_mass(lane) > 0.0
    )
    lane_total = sum(config.lane_mass(lane) for lane in available_lanes)
    if lane_total <= 0.0:
        raise ValueError("available candidate lanes have zero probability")
    probabilities = [0.0] * len(labels)
    for lane in available_lanes:
        indices = indices_by_lane[lane]
        local = _normalize([within_lane_weights[index] for index in indices])
        lane_probability = config.lane_mass(lane) / lane_total
        for index, probability in zip(indices, local, strict=True):
            probabilities[index] = lane_probability * probability
    return tuple(probabilities)


def _normalize(weights: Sequence[float]) -> tuple[float, ...]:
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("sampling weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("sampling weights must have a positive total")
    return tuple(float(weight) / total for weight in weights)
