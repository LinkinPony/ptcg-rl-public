"""Non-blocking rolling deck and seat deficit assignment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONFIG_DOMAIN = b"ptcg-rl/stateless-deck-balance-config/v1\x00"
_CONFIG_V2_DOMAIN = b"ptcg-rl/stateless-deck-balance-config/v2\x00"
_CONFIG_V3_DOMAIN = b"ptcg-rl/stateless-deck-balance-config/v3\x00"
_ASSIGNMENT_DOMAIN = b"ptcg-rl/stateless-deck-balance-assignment/v1\x00"
_ASSIGNMENT_V2_DOMAIN = b"ptcg-rl/stateless-deck-balance-assignment/v2\x00"
_ASSIGNMENT_V3_DOMAIN = b"ptcg-rl/stateless-deck-balance-assignment/v3\x00"
ChoiceT = TypeVar("ChoiceT")


class StatelessDeckBalanceConfig(BaseModel):
    """Mechanical rolling sampler settings supplied by the Hydra profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    active_deck_digests: tuple[str, ...]
    rolling_window_decisions: int = Field(gt=0)
    assignment_seed: int = Field(ge=0)
    inflight_decision_credit: float = Field(default=1.0, ge=0.0)
    deficit_exponent: float = Field(default=1.0, gt=0.0)

    @field_validator("active_deck_digests")
    @classmethod
    def valid_active_decks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a non-empty immutable exact-deck roster."""
        normalized = tuple(_fingerprint(item, "active deck") for item in value)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("active deck digests must be non-empty and unique")
        return normalized

    @field_validator("inflight_decision_credit", "deficit_exponent")
    @classmethod
    def finite_float(cls, value: float) -> float:
        """Reject non-finite sampler settings."""
        if not math.isfinite(value):
            raise ValueError("deck balance settings must be finite")
        return value

    @property
    def fingerprint(self) -> str:
        """Return the exact resume identity of the sampler settings."""
        return _canonical_fingerprint(
            _CONFIG_DOMAIN,
            self.model_dump(mode="json"),
        )


class DeckTargetShare(BaseModel):
    """One canonical target probability in a weighted deck roster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    target_share: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("deck_digest")
    @classmethod
    def valid_deck(cls, value: str) -> str:
        """Require an immutable exact deck identity."""
        return _fingerprint(value, "target deck")


class StatelessWeightedDeckBalanceConfig(BaseModel):
    """Version-two scheduler settings with explicit deck target shares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    target_deck_shares: tuple[DeckTargetShare, ...]
    rolling_window_decisions: int = Field(gt=0)
    assignment_seed: int = Field(ge=0)
    inflight_decision_credit: float = Field(default=1.0, gt=0.0)
    deficit_exponent: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def valid_target_simplex(self) -> Self:
        """Require a sorted, unique, strictly positive probability simplex."""
        digests = tuple(item.deck_digest for item in self.target_deck_shares)
        if not digests or digests != tuple(sorted(set(digests))):
            raise ValueError("weighted deck targets must be sorted and unique")
        if not math.isclose(
            sum(item.target_share for item in self.target_deck_shares),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("weighted deck target shares must sum to one")
        return self

    @field_validator("inflight_decision_credit", "deficit_exponent")
    @classmethod
    def finite_float(cls, value: float) -> float:
        """Reject non-finite sampler settings."""
        if not math.isfinite(value):
            raise ValueError("deck balance settings must be finite")
        return value

    @property
    def active_deck_digests(self) -> tuple[str, ...]:
        """Return the exact roster in canonical order."""
        return tuple(item.deck_digest for item in self.target_deck_shares)

    @property
    def target_share_mapping(self) -> dict[str, float]:
        """Return a fresh digest-to-share mapping."""
        return {item.deck_digest: item.target_share for item in self.target_deck_shares}

    @property
    def fingerprint(self) -> str:
        """Return the exact resume identity of the weighted sampler."""
        return _canonical_fingerprint(
            _CONFIG_V2_DOMAIN,
            self.model_dump(mode="json"),
        )


class StatelessDynamicDeckBalanceConfig(BaseModel):
    """Difficulty-adaptive scheduler with bounded uniform coverage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    active_deck_digests: tuple[str, ...]
    rolling_window_decisions: int = Field(gt=0)
    assignment_seed: int = Field(ge=0)
    inflight_decision_credit: float = Field(default=1.0, gt=0.0)
    deficit_exponent: float = Field(default=1.0, gt=0.0)
    performance_window_games: int = Field(gt=0)
    evidence_prior_games: float = Field(ge=0.0, allow_inf_nan=False)
    uniform_mix: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    difficulty_temperature: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_share_ratio: float = Field(ge=1.0, allow_inf_nan=False)

    @field_validator("active_deck_digests")
    @classmethod
    def valid_active_decks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require one canonical ordered exact-deck roster."""
        normalized = tuple(_fingerprint(item, "active deck") for item in value)
        if not normalized or normalized != tuple(sorted(set(normalized))):
            raise ValueError("dynamic active deck digests must be sorted and unique")
        return normalized

    @model_validator(mode="after")
    def coherent_share_cap(self) -> Self:
        """Require a cap that admits at least one probability simplex."""
        if self.maximum_share_ratio > float(len(self.active_deck_digests)):
            raise ValueError("dynamic maximum share ratio exceeds the deck count")
        return self

    @field_validator("inflight_decision_credit", "deficit_exponent")
    @classmethod
    def finite_float(cls, value: float) -> float:
        """Reject non-finite sampler settings."""
        if not math.isfinite(value):
            raise ValueError("deck balance settings must be finite")
        return value

    @property
    def fingerprint(self) -> str:
        """Return the exact resume identity of the adaptive sampler."""
        return _canonical_fingerprint(
            _CONFIG_V3_DOMAIN,
            self.model_dump(mode="json"),
        )


StatelessDeckBalanceConfigValue = (
    StatelessDeckBalanceConfig
    | StatelessWeightedDeckBalanceConfig
    | StatelessDynamicDeckBalanceConfig
)


class DeckSeatAssignment(BaseModel):
    """One immutable candidate deck/seat assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    assignment_cursor: int = Field(ge=0)
    deck_digest: str
    seat: Literal[0, 1]

    @field_validator("assignment_id", "deck_digest")
    @classmethod
    def valid_fingerprints(cls, value: str) -> str:
        """Require content-addressed assignment fields."""
        return _fingerprint(value, "deck balance assignment")


class RollingDecisionEvent(BaseModel):
    """One completed decision contribution retained in the rolling window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    seat: Literal[0, 1]
    decisions: int = Field(gt=0)
    end_decision_cursor: int = Field(gt=0)

    @field_validator("deck_digest")
    @classmethod
    def valid_deck(cls, value: str) -> str:
        """Require an exact deck identity."""
        return _fingerprint(value, "rolling deck")


class DeckTerminalScoreEvent(BaseModel):
    """One engine-terminal candidate result retained for dynamic allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    candidate_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("deck_digest")
    @classmethod
    def valid_deck(cls, value: str) -> str:
        """Require an exact deck identity."""
        return _fingerprint(value, "terminal-score deck")

    @field_validator("candidate_score")
    @classmethod
    def valid_score(cls, value: float) -> float:
        """Admit only engine terminal win, draw, and loss scores."""
        if value not in (0.0, 0.5, 1.0):
            raise ValueError("terminal score must be zero, one half, or one")
        return value


class DeckSeatHistory(BaseModel):
    """Lifetime starvation counters for one deck and seat cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    seat: Literal[0, 1]
    lifetime_finished_decisions: int = Field(default=0, ge=0)
    last_finish_cursor: int = Field(default=0, ge=0)
    longest_absence_decisions: int = Field(default=0, ge=0)

    @field_validator("deck_digest")
    @classmethod
    def valid_deck(cls, value: str) -> str:
        """Require an exact deck identity."""
        return _fingerprint(value, "history deck")


class StatelessDeckBalanceState(BaseModel):
    """Exact-resumable rolling sampler cursor and counters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2, 3] = 1
    config_fingerprint: str
    assignment_cursor: int = Field(default=0, ge=0)
    decision_cursor: int = Field(default=0, ge=0)
    rolling_events: tuple[RollingDecisionEvent, ...] = ()
    terminal_score_events: tuple[DeckTerminalScoreEvent, ...] = ()
    inflight_assignments: tuple[DeckSeatAssignment, ...] = ()
    history: tuple[DeckSeatHistory, ...]

    @field_validator("config_fingerprint")
    @classmethod
    def valid_config(cls, value: str) -> str:
        """Require a full sampler config identity."""
        return _fingerprint(value, "deck balance config")

    @model_validator(mode="after")
    def coherent_state(self) -> Self:
        """Reject duplicate leases and impossible monotonic cursors."""
        assignment_ids = tuple(
            assignment.assignment_id for assignment in self.inflight_assignments
        )
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("deck balance state has duplicate assignments")
        keys = tuple((item.deck_digest, item.seat) for item in self.history)
        if len(set(keys)) != len(keys):
            raise ValueError("deck balance state has duplicate history cells")
        if any(
            event.end_decision_cursor > self.decision_cursor
            for event in self.rolling_events
        ):
            raise ValueError("rolling event is ahead of the decision cursor")
        if any(
            left.end_decision_cursor > right.end_decision_cursor
            for left, right in zip(
                self.rolling_events,
                self.rolling_events[1:],
                strict=False,
            )
        ):
            raise ValueError("rolling events are not chronological")
        return self


class DeckSeatBalanceCellReport(BaseModel):
    """Observable rolling and lifetime coverage for one deck/seat cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    seat: Literal[0, 1]
    rolling_finished_decisions: int = Field(ge=0)
    rolling_share: float = Field(ge=0.0, le=1.0)
    target_share: float = Field(gt=0.0, le=1.0)
    inflight_assignments: int = Field(ge=0)
    lifetime_finished_decisions: int = Field(ge=0)
    current_absence_decisions: int = Field(ge=0)
    longest_absence_decisions: int = Field(ge=0)


class StatelessDeckBalanceReport(BaseModel):
    """Status payload for starvation and seat-skew monitoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_cursor: int = Field(ge=0)
    decision_cursor: int = Field(ge=0)
    rolling_finished_decisions: int = Field(ge=0)
    cells: tuple[DeckSeatBalanceCellReport, ...]


class StatelessDeckBalanceTransitionAudit(BaseModel):
    """Evidence for one settled dynamic-to-weighted scheduler migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-deck-balance-transition-v1"] = (
        "stateless-deck-balance-transition-v1"
    )
    source_schema_version: Literal[2, 3]
    target_schema_version: Literal[2] = 2
    source_config_fingerprint: str
    target_config_fingerprint: str
    assignment_cursor: int = Field(ge=0)
    decision_cursor: int = Field(ge=0)
    preserved_rolling_events: int = Field(ge=0)
    preserved_history_cells: int = Field(gt=0)
    discarded_terminal_score_events: int = Field(ge=0)

    @field_validator("source_config_fingerprint", "target_config_fingerprint")
    @classmethod
    def valid_config_fingerprint(cls, value: str) -> str:
        """Require full scheduler identities in transition evidence."""
        return _fingerprint(value, "deck balance config")


@dataclass(frozen=True)
class PreparedDeckBalanceOutcomes:
    """Fully validated deck-balance cohort awaiting an in-memory publish."""

    base_state: StatelessDeckBalanceState
    final_state: StatelessDeckBalanceState


class StatelessDeckBalanceSampler:
    """Choose underrepresented deck/seat cells without learner barriers."""

    def __init__(
        self,
        config: StatelessDeckBalanceConfigValue,
        *,
        state: StatelessDeckBalanceState | None = None,
    ) -> None:
        """Create a fresh sampler or restore its complete deterministic state."""
        self.config = config
        self._state = state or StatelessDeckBalanceState(
            schema_version=config.schema_version,
            config_fingerprint=config.fingerprint,
            history=tuple(
                DeckSeatHistory(
                    deck_digest=deck,
                    seat=seat,
                )
                for deck in config.active_deck_digests
                for seat in (0, 1)
            ),
        )
        self._validate_state_against_config()

    @property
    def state(self) -> StatelessDeckBalanceState:
        """Return the immutable exact-resume state."""
        return self._state

    @property
    def target_share_mapping(self) -> dict[str, float]:
        """Return the targets used for the next assignment cohort."""
        return _deck_target_shares(self.config, state=self._state)

    def assign(self) -> DeckSeatAssignment:
        """Return immediately with a deficit-weighted deck and seat lease."""
        rolling = _rolling_counts(self._state.rolling_events)
        inflight: Counter[tuple[str, int]] = Counter(
            (assignment.deck_digest, assignment.seat)
            for assignment in self._state.inflight_assignments
        )
        cursor = self._state.assignment_cursor
        assignment = self._assignment(
            rolling=rolling,
            inflight=inflight,
            cursor=cursor,
            config_fingerprint=self.config.fingerprint,
        )
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": cursor + 1,
                "inflight_assignments": (
                    self._state.inflight_assignments + (assignment,)
                ),
            }
        )
        return assignment

    def assign_many(self, count: int) -> tuple[DeckSeatAssignment, ...]:
        """Lease one ordered cohort with a single immutable state publish."""
        if count < 0:
            raise ValueError("assignment count must be non-negative")
        if count == 0:
            return ()
        rolling = _rolling_counts(self._state.rolling_events)
        inflight: Counter[tuple[str, int]] = Counter(
            (assignment.deck_digest, assignment.seat)
            for assignment in self._state.inflight_assignments
        )
        first_cursor = self._state.assignment_cursor
        config_fingerprint = self.config.fingerprint
        assignments: list[DeckSeatAssignment] = []
        for offset in range(count):
            assignment = self._assignment(
                rolling=rolling,
                inflight=inflight,
                cursor=first_cursor + offset,
                config_fingerprint=config_fingerprint,
            )
            assignments.append(assignment)
            inflight[(assignment.deck_digest, assignment.seat)] += 1
        cohort = tuple(assignments)
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": first_cursor + count,
                "inflight_assignments": (self._state.inflight_assignments + cohort),
            }
        )
        return cohort

    def preview_cells(
        self,
        count: int,
        *,
        target_shares: Mapping[str, float] | None = None,
    ) -> tuple[tuple[str, Literal[0, 1]], ...]:
        """Preview a cohort as compact cells without leasing or hashing it."""
        if count < 0:
            raise ValueError("assignment count must be non-negative")
        rolling = _rolling_counts(self._state.rolling_events)
        inflight: Counter[tuple[str, int]] = Counter(
            (assignment.deck_digest, assignment.seat)
            for assignment in self._state.inflight_assignments
        )
        first_cursor = self._state.assignment_cursor
        cells: list[tuple[str, Literal[0, 1]]] = []
        weighted_cells: tuple[tuple[str, Literal[0, 1]], ...] | None = None
        weighted_targets: Mapping[tuple[str, int], float] | None = None
        if isinstance(
            self.config,
            (StatelessWeightedDeckBalanceConfig, StatelessDynamicDeckBalanceConfig),
        ):
            deck_targets = (
                self.target_share_mapping
                if target_shares is None
                else dict(target_shares)
            )
            if set(deck_targets) != set(self.config.active_deck_digests):
                raise ValueError("preview target shares differ from active decks")
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in deck_targets.values()
            ) or not math.isclose(sum(deck_targets.values()), 1.0, abs_tol=1e-9):
                raise ValueError("preview target shares must form a positive simplex")
            resolved_preview_cells: tuple[tuple[str, Literal[0, 1]], ...] = tuple(
                (deck_digest, seat)
                for deck_digest in self.config.active_deck_digests
                for seat in (0, 1)
            )
            weighted_cells = resolved_preview_cells
            weighted_targets = {
                (deck_digest, seat): deck_targets[deck_digest] / 2.0
                for deck_digest, seat in resolved_preview_cells
            }
        elif target_shares is not None:
            raise ValueError("explicit preview targets require a weighted scheduler")
        for offset in range(count):
            cell = (
                self._weighted_assignment_cell(
                    rolling=rolling,
                    inflight=inflight,
                    cursor=first_cursor + offset,
                    cells=weighted_cells,
                    target_shares=weighted_targets,
                )
                if weighted_cells is not None and weighted_targets is not None
                else self._assignment_cell(
                    rolling=rolling,
                    inflight=inflight,
                    cursor=first_cursor + offset,
                )
            )
            cells.append(cell)
            inflight[cell] += 1
        return tuple(cells)

    def materialize_cells(
        self,
        cells: Sequence[tuple[str, Literal[0, 1]]],
        *,
        start_cursor: int,
    ) -> tuple[DeckSeatAssignment, ...]:
        """Build explicit leases without mutating the sampler state."""
        if start_cursor < 0:
            raise ValueError("deck-balance start cursor must be non-negative")
        active = set(self.config.active_deck_digests)
        normalized = tuple(cells)
        if any(deck not in active or seat not in (0, 1) for deck, seat in normalized):
            raise ValueError("materialized deck-balance cell is not active")
        config_fingerprint = self.config.fingerprint
        return tuple(
            DeckSeatAssignment(
                assignment_id=_canonical_fingerprint(
                    _assignment_domain(self.config),
                    {
                        "config_fingerprint": config_fingerprint,
                        "assignment_cursor": start_cursor + offset,
                        "deck_digest": deck,
                        "seat": seat,
                    },
                ),
                assignment_cursor=start_cursor + offset,
                deck_digest=deck,
                seat=seat,
            )
            for offset, (deck, seat) in enumerate(normalized)
        )

    def adopt_assignments(
        self,
        assignments: Sequence[DeckSeatAssignment],
        *,
        reservation_count: int | None = None,
    ) -> None:
        """Attach canonical leases while consuming their reservation span."""
        cohort = tuple(assignments)
        reserved = len(cohort) if reservation_count is None else reservation_count
        if reserved < len(cohort) or reserved < 0:
            raise ValueError("deck-balance reservation count is invalid")
        if not cohort and reserved == 0:
            return
        first_cursor = self._state.assignment_cursor
        reserved_until = first_cursor + reserved
        cohort_cursors = tuple(item.assignment_cursor for item in cohort)
        if tuple(sorted(set(cohort_cursors))) != cohort_cursors or any(
            cursor < first_cursor or cursor >= reserved_until
            for cursor in cohort_cursors
        ):
            raise ValueError("deck-balance adoption leaves its reservation span")
        expected = tuple(
            self.materialize_cells(
                ((item.deck_digest, item.seat),),
                start_cursor=item.assignment_cursor,
            )[0]
            for item in cohort
        )
        if cohort != expected:
            raise ValueError("deck-balance adoption cohort is not canonical")
        active_ids = {item.assignment_id for item in self._state.inflight_assignments}
        cohort_ids = tuple(item.assignment_id for item in cohort)
        if len(set(cohort_ids)) != len(cohort_ids) or active_ids.intersection(
            cohort_ids
        ):
            raise ValueError("deck-balance adoption contains duplicate leases")
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": reserved_until,
                "inflight_assignments": self._state.inflight_assignments + cohort,
            }
        )

    def assign_mixed(
        self,
        requested_cells: Sequence[tuple[str, Literal[0, 1]] | None],
    ) -> tuple[DeckSeatAssignment, ...]:
        """Lease a cohort with optional controller-prescribed deck/seat cells."""
        if not requested_cells:
            return ()
        active = set(self.config.active_deck_digests)
        for cell in requested_cells:
            if cell is not None and cell[0] not in active:
                raise ValueError("requested deck-balance cell is not active")
        rolling = _rolling_counts(self._state.rolling_events)
        inflight: Counter[tuple[str, int]] = Counter(
            (assignment.deck_digest, assignment.seat)
            for assignment in self._state.inflight_assignments
        )
        first_cursor = self._state.assignment_cursor
        config_fingerprint = self.config.fingerprint
        assignments: list[DeckSeatAssignment] = []
        for offset, requested in enumerate(requested_cells):
            cursor = first_cursor + offset
            if requested is None:
                assignment = self._assignment(
                    rolling=rolling,
                    inflight=inflight,
                    cursor=cursor,
                    config_fingerprint=config_fingerprint,
                )
            else:
                deck, seat = requested
                assignment = DeckSeatAssignment(
                    assignment_id=_canonical_fingerprint(
                        _assignment_domain(self.config),
                        {
                            "config_fingerprint": config_fingerprint,
                            "assignment_cursor": cursor,
                            "deck_digest": deck,
                            "seat": seat,
                        },
                    ),
                    assignment_cursor=cursor,
                    deck_digest=deck,
                    seat=seat,
                )
            assignments.append(assignment)
            inflight[(assignment.deck_digest, assignment.seat)] += 1
        cohort = tuple(assignments)
        self._state = self._state.model_copy(
            update={
                "assignment_cursor": first_cursor + len(cohort),
                "inflight_assignments": self._state.inflight_assignments + cohort,
            }
        )
        return cohort

    def _assignment(
        self,
        *,
        rolling: Counter[tuple[str, int]],
        inflight: Counter[tuple[str, int]],
        cursor: int,
        config_fingerprint: str,
    ) -> DeckSeatAssignment:
        """Build the deterministic next lease from explicit coverage counts."""
        deck, seat = self._assignment_cell(
            rolling=rolling,
            inflight=inflight,
            cursor=cursor,
        )
        return DeckSeatAssignment(
            assignment_id=_canonical_fingerprint(
                _assignment_domain(self.config),
                {
                    "config_fingerprint": config_fingerprint,
                    "assignment_cursor": cursor,
                    "deck_digest": deck,
                    "seat": seat,
                },
            ),
            assignment_cursor=cursor,
            deck_digest=deck,
            seat=seat,
        )

    def _assignment_cell(
        self,
        *,
        rolling: Counter[tuple[str, int]],
        inflight: Counter[tuple[str, int]],
        cursor: int,
    ) -> tuple[str, Literal[0, 1]]:
        """Choose only the next compact cell, omitting object and hash work."""
        if isinstance(
            self.config,
            (StatelessWeightedDeckBalanceConfig, StatelessDynamicDeckBalanceConfig),
        ):
            return self._weighted_assignment_cell(
                rolling=rolling,
                inflight=inflight,
                cursor=cursor,
            )
        deck_values = []
        for deck in self.config.active_deck_digests:
            completed = sum(rolling[(deck, seat)] for seat in (0, 1))
            leases = sum(inflight[(deck, seat)] for seat in (0, 1))
            deck_values.append(
                float(completed) + self.config.inflight_decision_credit * float(leases)
            )
        deck = _deficit_choice(
            self.config.active_deck_digests,
            deck_values,
            exponent=self.config.deficit_exponent,
            random_unit=_counter_random(
                self.config.assignment_seed,
                cursor,
                "deck",
            ),
        )
        seat_values = tuple(
            float(rolling[(deck, seat)])
            + self.config.inflight_decision_credit * float(inflight[(deck, seat)])
            for seat in (0, 1)
        )
        seat_choices: tuple[Literal[0], Literal[1]] = (0, 1)
        seat = cast(
            Literal[0, 1],
            _deficit_choice(
                seat_choices,
                seat_values,
                exponent=self.config.deficit_exponent,
                random_unit=_counter_random(
                    self.config.assignment_seed,
                    cursor,
                    "seat",
                ),
            ),
        )
        return deck, seat

    def _weighted_assignment(
        self,
        *,
        rolling: Counter[tuple[str, int]],
        inflight: Counter[tuple[str, int]],
        cursor: int,
        config_fingerprint: str,
    ) -> DeckSeatAssignment:
        """Choose one cell from positive projected decision-share deficits."""
        deck, seat = self._weighted_assignment_cell(
            rolling=rolling,
            inflight=inflight,
            cursor=cursor,
        )
        return DeckSeatAssignment(
            assignment_id=_canonical_fingerprint(
                _assignment_domain(self.config),
                {
                    "config_fingerprint": config_fingerprint,
                    "assignment_cursor": cursor,
                    "deck_digest": deck,
                    "seat": seat,
                },
            ),
            assignment_cursor=cursor,
            deck_digest=deck,
            seat=seat,
        )

    def _weighted_assignment_cell(
        self,
        *,
        rolling: Counter[tuple[str, int]],
        inflight: Counter[tuple[str, int]],
        cursor: int,
        cells: tuple[tuple[str, Literal[0, 1]], ...] | None = None,
        target_shares: Mapping[tuple[str, int], float] | None = None,
    ) -> tuple[str, Literal[0, 1]]:
        """Choose a weighted cell without constructing its durable lease."""
        if not isinstance(
            self.config,
            (StatelessWeightedDeckBalanceConfig, StatelessDynamicDeckBalanceConfig),
        ):
            raise TypeError("weighted assignment requires a weighted config")
        resolved_cells: tuple[tuple[str, Literal[0, 1]], ...]
        resolved_targets: Mapping[tuple[str, int], float]
        if cells is None or target_shares is None:
            deck_targets = self.target_share_mapping
            resolved_cells = tuple(
                (deck_digest, seat)
                for deck_digest in self.config.active_deck_digests
                for seat in (0, 1)
            )
            resolved_targets = {
                (deck_digest, seat): deck_targets[deck_digest] / 2.0
                for deck_digest, seat in resolved_cells
            }
        else:
            resolved_cells = cells
            resolved_targets = target_shares
        credit = self.config.inflight_decision_credit
        projected_total = float(sum(rolling.values())) + credit * float(
            sum(inflight.values()) + 1
        )
        deficits = tuple(
            resolved_targets[cell] * projected_total
            - (float(rolling[cell]) + credit * float(inflight[cell]))
            for cell in resolved_cells
        )
        positive = tuple(
            max(value, 0.0) ** self.config.deficit_exponent for value in deficits
        )
        if sum(positive) <= 0.0:
            highest = max(deficits)
            positive = tuple(1.0 if value == highest else 0.0 for value in deficits)
        chosen = _weighted_choice(
            resolved_cells,
            positive,
            random_unit=_counter_random(
                self.config.assignment_seed,
                cursor,
                "weighted-cell",
            ),
        )
        deck, seat = chosen
        return deck, seat

    def finish(
        self,
        assignment_id: str,
        *,
        finished_decisions: int,
        candidate_score: float | None = None,
    ) -> None:
        """Commit valid completed decisions to rolling and lifetime coverage."""
        if finished_decisions <= 0:
            raise ValueError("finished decision count must be positive")
        score = _candidate_score(candidate_score)
        assignment, remaining = self._take_assignment(assignment_id)
        end_cursor = self._state.decision_cursor + finished_decisions
        event = RollingDecisionEvent(
            deck_digest=assignment.deck_digest,
            seat=assignment.seat,
            decisions=finished_decisions,
            end_decision_cursor=end_cursor,
        )
        events = _trim_events(
            self._state.rolling_events + (event,),
            maximum_decisions=self.config.rolling_window_decisions,
        )
        history = []
        for item in self._state.history:
            if (item.deck_digest, item.seat) != (
                assignment.deck_digest,
                assignment.seat,
            ):
                history.append(item)
                continue
            absence = end_cursor - item.last_finish_cursor
            history.append(
                item.model_copy(
                    update={
                        "lifetime_finished_decisions": (
                            item.lifetime_finished_decisions + finished_decisions
                        ),
                        "last_finish_cursor": end_cursor,
                        "longest_absence_decisions": max(
                            item.longest_absence_decisions,
                            absence,
                        ),
                    }
                )
            )
        self._state = self._state.model_copy(
            update={
                "decision_cursor": end_cursor,
                "rolling_events": events,
                "terminal_score_events": _append_terminal_scores(
                    self.config,
                    self._state.terminal_score_events,
                    assignment.deck_digest,
                    score,
                ),
                "inflight_assignments": remaining,
                "history": tuple(history),
            }
        )

    def cancel(self, assignment_id: str) -> None:
        """Release an unfinished lease without altering decision coverage."""
        _assignment, remaining = self._take_assignment(assignment_id)
        self._state = self._state.model_copy(update={"inflight_assignments": remaining})

    def prepare_outcomes(
        self,
        outcomes: Sequence[
            tuple[str, int | None] | tuple[str, int | None, float | None]
        ],
    ) -> PreparedDeckBalanceOutcomes:
        """Validate a complete cohort and derive its immutable final state.

        A positive decision count finishes a lease; ``None`` cancels it.  The
        live sampler is not changed when any later outcome is invalid.
        """
        base_state = self._state
        if len(outcomes) == 0:
            return PreparedDeckBalanceOutcomes(
                base_state=base_state,
                final_state=base_state,
            )
        active_assignments = {
            assignment.assignment_id: assignment
            for assignment in base_state.inflight_assignments
        }
        retained_events = deque(
            (event, event.decisions) for event in base_state.rolling_events
        )
        retained_decisions = sum(event.decisions for event in base_state.rolling_events)
        decision_cursor = base_state.decision_cursor
        history_indexes = {
            (item.deck_digest, item.seat): index
            for index, item in enumerate(base_state.history)
        }
        lifetime_decisions = [
            item.lifetime_finished_decisions for item in base_state.history
        ]
        last_finish_cursors = [item.last_finish_cursor for item in base_state.history]
        longest_absences = [
            item.longest_absence_decisions for item in base_state.history
        ]
        terminal_scores = deque(base_state.terminal_score_events)
        maximum_decisions = self.config.rolling_window_decisions
        for raw_outcome in outcomes:
            if len(raw_outcome) == 2:
                assignment_id, finished_decisions = raw_outcome
                candidate_score = None
            else:
                assignment_id, finished_decisions, candidate_score = raw_outcome
            if finished_decisions is not None and finished_decisions <= 0:
                raise ValueError("finished decision count must be positive")
            score = _candidate_score(candidate_score)
            normalized = _fingerprint(assignment_id, "assignment")
            assignment = active_assignments.get(normalized)
            if assignment is None:
                raise KeyError("deck balance assignment is not active")
            if finished_decisions is None:
                if score is not None and isinstance(
                    self.config, StatelessDynamicDeckBalanceConfig
                ):
                    terminal_scores.append(
                        DeckTerminalScoreEvent(
                            deck_digest=assignment.deck_digest,
                            candidate_score=score,
                        )
                    )
                    while len(terminal_scores) > self.config.performance_window_games:
                        terminal_scores.popleft()
                del active_assignments[normalized]
                continue

            end_cursor = decision_cursor + finished_decisions
            event = RollingDecisionEvent(
                deck_digest=assignment.deck_digest,
                seat=assignment.seat,
                decisions=finished_decisions,
                end_decision_cursor=end_cursor,
            )
            retained_events.append((event, event.decisions))
            retained_decisions += event.decisions
            while retained_decisions > maximum_decisions:
                first, first_decisions = retained_events[0]
                excess = retained_decisions - maximum_decisions
                if first_decisions <= excess:
                    retained_events.popleft()
                    retained_decisions -= first_decisions
                    continue
                retained_events[0] = (first, first_decisions - excess)
                retained_decisions -= excess

            history_index = history_indexes[(assignment.deck_digest, assignment.seat)]
            absence = end_cursor - last_finish_cursors[history_index]
            lifetime_decisions[history_index] += finished_decisions
            last_finish_cursors[history_index] = end_cursor
            longest_absences[history_index] = max(
                longest_absences[history_index],
                absence,
            )
            if score is not None and isinstance(
                self.config, StatelessDynamicDeckBalanceConfig
            ):
                terminal_scores.append(
                    DeckTerminalScoreEvent(
                        deck_digest=assignment.deck_digest,
                        candidate_score=score,
                    )
                )
                while len(terminal_scores) > self.config.performance_window_games:
                    terminal_scores.popleft()
            decision_cursor = end_cursor
            del active_assignments[normalized]

        rolling_events = tuple(
            event
            if decisions == event.decisions
            else event.model_copy(update={"decisions": decisions})
            for event, decisions in retained_events
        )
        history = tuple(
            item
            if (
                lifetime == item.lifetime_finished_decisions
                and last_finish == item.last_finish_cursor
                and longest_absence == item.longest_absence_decisions
            )
            else item.model_copy(
                update={
                    "lifetime_finished_decisions": lifetime,
                    "last_finish_cursor": last_finish,
                    "longest_absence_decisions": longest_absence,
                }
            )
            for item, lifetime, last_finish, longest_absence in zip(
                base_state.history,
                lifetime_decisions,
                last_finish_cursors,
                longest_absences,
                strict=True,
            )
        )
        final_state = base_state.model_copy(
            update={
                "decision_cursor": decision_cursor,
                "rolling_events": rolling_events,
                "terminal_score_events": tuple(terminal_scores),
                "inflight_assignments": tuple(
                    assignment
                    for assignment in base_state.inflight_assignments
                    if assignment.assignment_id in active_assignments
                ),
                "history": history,
            }
        )
        return PreparedDeckBalanceOutcomes(
            base_state=base_state,
            final_state=final_state,
        )

    def validate_prepared_outcomes(
        self,
        prepared: PreparedDeckBalanceOutcomes,
    ) -> None:
        """Reject a prepared transition after any intervening mutation."""
        if self._state is not prepared.base_state:
            raise RuntimeError("deck balance changed after outcomes were prepared")

    def publish_prepared_outcomes(
        self,
        prepared: PreparedDeckBalanceOutcomes,
    ) -> None:
        """Publish one already validated immutable cohort state."""
        self.validate_prepared_outcomes(prepared)
        self._state = prepared.final_state

    def report(self) -> StatelessDeckBalanceReport:
        """Return rolling shares, seat coverage, and starvation counters."""
        rolling = _rolling_counts(self._state.rolling_events)
        inflight = Counter(
            (assignment.deck_digest, assignment.seat)
            for assignment in self._state.inflight_assignments
        )
        total = sum(rolling.values())
        deck_targets = self.target_share_mapping
        cells = tuple(
            DeckSeatBalanceCellReport(
                deck_digest=item.deck_digest,
                seat=item.seat,
                rolling_finished_decisions=rolling[(item.deck_digest, item.seat)],
                rolling_share=(
                    0.0
                    if total == 0
                    else rolling[(item.deck_digest, item.seat)] / float(total)
                ),
                target_share=deck_targets[item.deck_digest] / 2.0,
                inflight_assignments=inflight[(item.deck_digest, item.seat)],
                lifetime_finished_decisions=item.lifetime_finished_decisions,
                current_absence_decisions=(
                    self._state.decision_cursor - item.last_finish_cursor
                ),
                longest_absence_decisions=max(
                    item.longest_absence_decisions,
                    self._state.decision_cursor - item.last_finish_cursor,
                ),
            )
            for item in self._state.history
        )
        return StatelessDeckBalanceReport(
            assignment_cursor=self._state.assignment_cursor,
            decision_cursor=self._state.decision_cursor,
            rolling_finished_decisions=total,
            cells=cells,
        )

    def _take_assignment(
        self,
        assignment_id: str,
    ) -> tuple[DeckSeatAssignment, tuple[DeckSeatAssignment, ...]]:
        normalized = _fingerprint(assignment_id, "assignment")
        matches = tuple(
            assignment
            for assignment in self._state.inflight_assignments
            if assignment.assignment_id == normalized
        )
        if len(matches) != 1:
            raise KeyError("deck balance assignment is not active")
        remaining = tuple(
            assignment
            for assignment in self._state.inflight_assignments
            if assignment.assignment_id != normalized
        )
        return (matches[0], remaining)

    def _validate_state_against_config(self) -> None:
        if self._state.schema_version != self.config.schema_version:
            raise ValueError("deck balance state schema differs")
        if self._state.config_fingerprint != self.config.fingerprint:
            raise ValueError("deck balance resume config differs")
        expected = {
            (deck, seat) for deck in self.config.active_deck_digests for seat in (0, 1)
        }
        actual = {(item.deck_digest, item.seat) for item in self._state.history}
        if actual != expected:
            raise ValueError("deck balance resume roster differs")
        if (
            any(
                assignment.deck_digest not in self.config.active_deck_digests
                for assignment in self._state.inflight_assignments
            )
            or any(
                event.deck_digest not in self.config.active_deck_digests
                for event in self._state.rolling_events
            )
            or any(
                event.deck_digest not in self.config.active_deck_digests
                for event in self._state.terminal_score_events
            )
        ):
            raise ValueError("deck balance state references an inactive deck")
        if (
            sum(event.decisions for event in self._state.rolling_events)
            > self.config.rolling_window_decisions
        ):
            raise ValueError("deck balance rolling window exceeds its limit")
        if any(
            item.last_finish_cursor > self._state.decision_cursor
            for item in self._state.history
        ):
            raise ValueError("deck balance history is ahead of its decision cursor")
        if isinstance(self.config, StatelessDynamicDeckBalanceConfig):
            if len(self._state.terminal_score_events) > (
                self.config.performance_window_games
            ):
                raise ValueError("dynamic deck score window exceeds its limit")
        elif self._state.terminal_score_events:
            raise ValueError("static deck-balance state carried dynamic score events")
        for assignment in self._state.inflight_assignments:
            expected_id = _canonical_fingerprint(
                _assignment_domain(self.config),
                {
                    "config_fingerprint": self.config.fingerprint,
                    "assignment_cursor": assignment.assignment_cursor,
                    "deck_digest": assignment.deck_digest,
                    "seat": assignment.seat,
                },
            )
            if (
                assignment.assignment_cursor >= self._state.assignment_cursor
                or assignment.assignment_id != expected_id
            ):
                raise ValueError("deck balance in-flight assignment is corrupt")


def migrate_settled_dynamic_deck_balance_to_weighted(
    source: StatelessDeckBalanceState,
    *,
    target_config: StatelessWeightedDeckBalanceConfig,
    expected_source_config_fingerprint: str,
) -> tuple[StatelessDeckBalanceState, StatelessDeckBalanceTransitionAudit]:
    """Preserve generic history while retiring dynamic-only score evidence."""
    source_fingerprint = _fingerprint(
        expected_source_config_fingerprint,
        "source deck balance config",
    )
    if source.schema_version != 3:
        raise ValueError("deck-balance transition source is not dynamic schema three")
    if source.config_fingerprint != source_fingerprint:
        raise ValueError("deck-balance transition source fingerprint mismatch")
    if source.inflight_assignments:
        raise ValueError("deck-balance transition has in-flight assignments")
    migrated = source.model_copy(
        update={
            "schema_version": target_config.schema_version,
            "config_fingerprint": target_config.fingerprint,
            "terminal_score_events": (),
        }
    )
    validated = StatelessDeckBalanceSampler(target_config, state=migrated).state
    audit = StatelessDeckBalanceTransitionAudit(
        source_schema_version=source.schema_version,
        source_config_fingerprint=source_fingerprint,
        target_config_fingerprint=target_config.fingerprint,
        assignment_cursor=source.assignment_cursor,
        decision_cursor=source.decision_cursor,
        preserved_rolling_events=len(source.rolling_events),
        preserved_history_cells=len(source.history),
        discarded_terminal_score_events=len(source.terminal_score_events),
    )
    return validated, audit


def migrate_settled_deck_balance_to_weighted(
    source: StatelessDeckBalanceState,
    *,
    target_config: StatelessWeightedDeckBalanceConfig,
    expected_source_config_fingerprint: str,
) -> tuple[StatelessDeckBalanceState, StatelessDeckBalanceTransitionAudit]:
    """Rebind a settled dynamic or weighted scheduler to fixed target shares."""
    if source.schema_version == 3:
        return migrate_settled_dynamic_deck_balance_to_weighted(
            source,
            target_config=target_config,
            expected_source_config_fingerprint=(
                expected_source_config_fingerprint
            ),
        )
    source_fingerprint = _fingerprint(
        expected_source_config_fingerprint,
        "source deck balance config",
    )
    if source.schema_version != 2:
        raise ValueError(
            "deck-balance transition source is not weighted schema two or "
            "dynamic schema three"
        )
    if source.config_fingerprint != source_fingerprint:
        raise ValueError("deck-balance transition source fingerprint mismatch")
    if source.inflight_assignments:
        raise ValueError("deck-balance transition has in-flight assignments")
    migrated = source.model_copy(
        update={
            "config_fingerprint": target_config.fingerprint,
            "terminal_score_events": (),
        }
    )
    validated = StatelessDeckBalanceSampler(target_config, state=migrated).state
    audit = StatelessDeckBalanceTransitionAudit(
        source_schema_version=source.schema_version,
        source_config_fingerprint=source_fingerprint,
        target_config_fingerprint=target_config.fingerprint,
        assignment_cursor=source.assignment_cursor,
        decision_cursor=source.decision_cursor,
        preserved_rolling_events=len(source.rolling_events),
        preserved_history_cells=len(source.history),
        discarded_terminal_score_events=len(source.terminal_score_events),
    )
    return validated, audit


def _rolling_counts(
    events: tuple[RollingDecisionEvent, ...],
) -> Counter[tuple[str, int]]:
    return Counter(
        {
            key: sum(
                event.decisions
                for event in events
                if (event.deck_digest, event.seat) == key
            )
            for key in {(event.deck_digest, event.seat) for event in events}
        }
    )


def _trim_events(
    events: tuple[RollingDecisionEvent, ...],
    *,
    maximum_decisions: int,
) -> tuple[RollingDecisionEvent, ...]:
    retained = list(events)
    excess = sum(event.decisions for event in retained) - maximum_decisions
    while retained and excess > 0:
        first = retained[0]
        if first.decisions <= excess:
            excess -= first.decisions
            retained.pop(0)
            continue
        retained[0] = first.model_copy(update={"decisions": first.decisions - excess})
        excess = 0
    return tuple(retained)


def _deficit_choice(
    choices: tuple[ChoiceT, ...],
    values: list[float] | tuple[float, ...],
    *,
    exponent: float,
    random_unit: float,
) -> ChoiceT:
    high = max(values)
    weights = tuple((high - value + 1.0) ** exponent for value in values)
    threshold = random_unit * sum(weights)
    cumulative = 0.0
    for choice, weight in zip(choices, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return choice
    return choices[-1]


def _weighted_choice(
    choices: tuple[ChoiceT, ...],
    weights: tuple[float, ...],
    *,
    random_unit: float,
) -> ChoiceT:
    """Sample one deterministic counter-RNG choice from non-negative weights."""
    if len(choices) != len(weights) or not choices:
        raise ValueError("weighted choice inputs are misaligned")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("weighted choice weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weighted choice requires positive mass")
    threshold = random_unit * total
    cumulative = 0.0
    for choice, weight in zip(choices, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return choice
    return choices[-1]


def _deck_target_shares(
    config: StatelessDeckBalanceConfigValue,
    *,
    state: StatelessDeckBalanceState | None = None,
) -> dict[str, float]:
    """Return uniform, explicit, or evidence-adaptive deck targets."""
    if isinstance(config, StatelessWeightedDeckBalanceConfig):
        return config.target_share_mapping
    if isinstance(config, StatelessDynamicDeckBalanceConfig):
        if state is None:
            raise ValueError("dynamic deck targets require scheduler state")
        scores: defaultdict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for event in state.terminal_score_events:
            scores[event.deck_digest] += event.candidate_score
            counts[event.deck_digest] += 1
        posterior_scores = {
            deck: (0.5 * config.evidence_prior_games + float(scores[deck]))
            / (config.evidence_prior_games + float(counts[deck]))
            if config.evidence_prior_games + float(counts[deck]) > 0.0
            else 0.5
            for deck in config.active_deck_digests
        }
        logits = tuple(
            (0.5 - posterior_scores[deck]) / config.difficulty_temperature
            for deck in config.active_deck_digests
        )
        highest = max(logits)
        priorities = tuple(math.exp(value - highest) for value in logits)
        priority_total = sum(priorities)
        deck_count = len(config.active_deck_digests)
        uniform_share = 1.0 / float(deck_count)
        raw = tuple(
            config.uniform_mix * uniform_share
            + (1.0 - config.uniform_mix) * priority / priority_total
            for priority in priorities
        )
        bounded = _cap_probability_simplex(
            raw,
            maximum=config.maximum_share_ratio * uniform_share,
        )
        return dict(zip(config.active_deck_digests, bounded, strict=True))
    share = 1.0 / float(len(config.active_deck_digests))
    return dict.fromkeys(config.active_deck_digests, share)


def _assignment_domain(config: StatelessDeckBalanceConfigValue) -> bytes:
    """Keep historical assignment identities byte-compatible."""
    if isinstance(config, StatelessDynamicDeckBalanceConfig):
        return _ASSIGNMENT_V3_DOMAIN
    if isinstance(config, StatelessWeightedDeckBalanceConfig):
        return _ASSIGNMENT_V2_DOMAIN
    return _ASSIGNMENT_DOMAIN


def _append_terminal_scores(
    config: StatelessDeckBalanceConfigValue,
    events: tuple[DeckTerminalScoreEvent, ...],
    deck_digest: str,
    candidate_score: float | None,
) -> tuple[DeckTerminalScoreEvent, ...]:
    """Append one bounded engine-terminal result for an adaptive config."""
    if candidate_score is None or not isinstance(
        config,
        StatelessDynamicDeckBalanceConfig,
    ):
        return events
    updated = events + (
        DeckTerminalScoreEvent(
            deck_digest=deck_digest,
            candidate_score=candidate_score,
        ),
    )
    return updated[-config.performance_window_games :]


def _candidate_score(value: float | None) -> float | None:
    """Normalize the only engine-terminal scores admitted by the scheduler."""
    if value is None:
        return None
    if not math.isfinite(value) or value not in (0.0, 0.5, 1.0):
        raise ValueError("candidate score must be an engine terminal result")
    return float(value)


def _cap_probability_simplex(
    values: tuple[float, ...],
    *,
    maximum: float,
) -> tuple[float, ...]:
    """Project positive simplex values onto a common upper bound."""
    if not values or maximum <= 0.0 or maximum * len(values) < 1.0 - 1e-12:
        raise ValueError("probability cap cannot represent a simplex")
    result = list(values)
    free = set(range(len(result)))
    fixed_total = 0.0
    while free:
        free_total = sum(result[index] for index in free)
        remaining = 1.0 - fixed_total
        if free_total <= 0.0:
            raise ValueError("probability simplex has no free mass")
        scale = remaining / free_total
        overflowing = {
            index for index in free if result[index] * scale > maximum + 1e-15
        }
        if not overflowing:
            for index in free:
                result[index] *= scale
            break
        for index in overflowing:
            result[index] = maximum
        fixed_total += maximum * float(len(overflowing))
        free -= overflowing
    correction = 1.0 - sum(result)
    if abs(correction) > 1e-12:
        raise RuntimeError("bounded probability projection lost simplex mass")
    if correction:
        adjustable = next(
            (index for index, value in enumerate(result) if value < maximum),
            0,
        )
        result[adjustable] += correction
    if any(value <= 0.0 or value > maximum + 1e-12 for value in result):
        raise RuntimeError("bounded probability projection violated its limits")
    return tuple(result)


def _counter_random(seed: int, cursor: int, namespace: str) -> float:
    payload = f"{seed}:{cursor}:{namespace}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def _fingerprint(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return normalized


def _canonical_fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


__all__ = [
    "DeckTargetShare",
    "DeckSeatAssignment",
    "DeckSeatBalanceCellReport",
    "DeckSeatHistory",
    "DeckTerminalScoreEvent",
    "PreparedDeckBalanceOutcomes",
    "RollingDecisionEvent",
    "StatelessDeckBalanceConfig",
    "StatelessDeckBalanceConfigValue",
    "StatelessDeckBalanceReport",
    "StatelessDeckBalanceSampler",
    "StatelessDeckBalanceState",
    "StatelessDeckBalanceTransitionAudit",
    "StatelessDynamicDeckBalanceConfig",
    "StatelessWeightedDeckBalanceConfig",
    "migrate_settled_deck_balance_to_weighted",
    "migrate_settled_dynamic_deck_balance_to_weighted",
]
