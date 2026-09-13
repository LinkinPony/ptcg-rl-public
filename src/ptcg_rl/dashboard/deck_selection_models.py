"""Typed API models for checkpoint-bound deck selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeckSelectionQuality(BaseModel):
    """Selection-readiness checks over one complete round robin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    expected_decks: int = Field(ge=0)
    observed_decks: int = Field(ge=0)
    expected_matchups: int = Field(ge=0)
    observed_matchups: int = Field(ge=0)
    expected_games_per_matchup: int = Field(ge=0)
    expected_games: int = Field(ge=0)
    observed_games: int = Field(ge=0)
    seat_balanced: bool
    agent_error_games: int = Field(ge=0)
    truncated_games: int = Field(ge=0)
    warnings: tuple[str, ...]


class DeckSelectionCell(BaseModel):
    """One directed candidate-versus-opponent matrix cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    candidate_hash: str
    opponent_id: str
    opponent_hash: str
    candidate_label: str
    opponent_label: str
    games: int = Field(ge=0)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    truncated: int = Field(ge=0)
    agent_error_games: int = Field(ge=0)
    seat_0_games: int = Field(ge=0)
    seat_1_games: int = Field(ge=0)
    score_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    posterior_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_low: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_high: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_state: Literal["ready", "incomplete", "missing"]


class DeckSelectionStanding(BaseModel):
    """One deck ranked by equal weight over every observed opponent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    deck_id: str
    deck_hash: str
    deck_label: str
    games: int = Field(ge=0)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    truncated: int = Field(ge=0)
    opponent_count: int = Field(ge=0)
    equal_score_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    posterior_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_low: float | None = Field(default=None, ge=0.0, le=1.0)
    credible_high: float | None = Field(default=None, ge=0.0, le=1.0)
    worst_opponent_id: str | None = None
    worst_opponent_label: str | None = None
    worst_matchup_score: float | None = Field(default=None, ge=0.0, le=1.0)
    best_opponent_id: str | None = None
    best_opponent_label: str | None = None
    best_matchup_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_state: Literal["ready", "incomplete", "missing"]


class DeckSelectionPayload(BaseModel):
    """Decision-oriented view of one same-checkpoint runtime ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task_state: str
    semantics: Literal["same_checkpoint_equal_opponent_deck_selection_v1"]
    checkpoint_id: str
    checkpoint_fingerprint: str | None = None
    spec_fingerprint: str
    result_fingerprint: str | None = None
    quality: DeckSelectionQuality
    standings: tuple[DeckSelectionStanding, ...]
    cells: tuple[DeckSelectionCell, ...]
