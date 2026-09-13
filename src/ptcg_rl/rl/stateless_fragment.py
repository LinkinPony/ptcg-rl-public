"""Fixed-horizon public trajectory fragments for stateless PPO."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.context.public_events import PublicEventDelta
from ptcg_rl.decks.identity import DECK_SIZE, canonicalize_deck
from ptcg_rl.model.sequence.action import (
    AcceptedActionRecord,
    build_accepted_action_record,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FRAGMENT_CONTRACT_DOMAIN = b"ptcg-rl/stateless-fragment-contract/v1\x00"
_STATIC_CONTRACT_DOMAIN = b"ptcg-rl/stateless-fragment-static-contract/v1\x00"
_FRAGMENT_ID_DOMAIN = b"ptcg-rl/stateless-fragment-id/v1\x00"


class StatelessFragmentIdentity(BaseModel):
    """Behavior snapshot and every input contract for one whole fragment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 1
    horizon: int = Field(ge=1)
    behavior_policy_version: int = Field(ge=0)
    behavior_policy_fingerprint: str
    model_config_fingerprint: str
    action_schema_fingerprint: str
    public_context_fingerprint: str
    card_catalog_fingerprint: str
    public_deck_catalog_fingerprint: str
    exact_registry_fingerprint: str
    belief_target_semantics_fingerprint: str
    input_contract_fingerprint: str
    resolved_config_fingerprint: str
    sequence_contract_fingerprint: str | None = None

    @field_validator(
        "behavior_policy_fingerprint",
        "model_config_fingerprint",
        "action_schema_fingerprint",
        "public_context_fingerprint",
        "card_catalog_fingerprint",
        "public_deck_catalog_fingerprint",
        "exact_registry_fingerprint",
        "belief_target_semantics_fingerprint",
        "input_contract_fingerprint",
        "resolved_config_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable full content identities."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("fragment identity components must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_sequence_contract(self) -> Self:
        """Require the raw sequence contract only for schema V2 fragments."""
        if self.schema_version == 2:
            fingerprint = self.sequence_contract_fingerprint
            if (
                fingerprint is None
                or _SHA256_PATTERN.fullmatch(fingerprint) is None
            ):
                raise ValueError("sequence fragment requires contract fingerprint")
        elif self.sequence_contract_fingerprint is not None:
            raise ValueError("schema V1 fragment cannot declare sequence contract")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the exact behavior fragment contract."""
        return _fingerprint(
            _FRAGMENT_CONTRACT_DOMAIN,
            self.model_dump(mode="json"),
        )

    @property
    def static_contract_fingerprint(self) -> str:
        """Return the run contract excluding moving behavior publication."""
        return _fingerprint(
            _STATIC_CONTRACT_DOMAIN,
            self.model_dump(
                mode="json",
                exclude={
                    "behavior_policy_version",
                    "behavior_policy_fingerprint",
                },
            ),
        )


class StatelessFragmentContext(BaseModel):
    """Per-game learner side channel stored once for a fragment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    game_id: str
    seat: Literal[0, 1]
    start_decision_index: int = Field(ge=0)
    own_deck: tuple[int, ...]
    own_deck_digest: str
    opponent_deck: tuple[int, ...]
    opponent_deck_digest: str
    curriculum_generation: int = Field(ge=0)
    assignment_id: str
    opponent_artifact_fingerprint: str

    @field_validator("game_id", "assignment_id")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Require stable game and assignment identities."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("fragment game and assignment IDs must be non-empty")
        return normalized

    @field_validator(
        "own_deck_digest",
        "opponent_deck_digest",
        "opponent_artifact_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable exact deck and opponent identities."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("fragment context identity must be SHA-256")
        return normalized

    @model_validator(mode="after")
    def coherent_decks(self) -> Self:
        """Validate both 60-card contexts and their canonical digests."""
        own = canonicalize_deck(self.own_deck)
        opponent = canonicalize_deck(self.opponent_deck)
        if own.deck_digest != self.own_deck_digest:
            raise ValueError("fragment own deck fingerprint mismatch")
        if opponent.deck_digest != self.opponent_deck_digest:
            raise ValueError("fragment opponent deck fingerprint mismatch")
        return self


@dataclass(frozen=True)
class StatelessFragmentDecision:
    """One candidate decision with public actor input and compact PPO evidence."""

    decision_index: int
    actor_row: SimpleStatelessActorRow
    action: tuple[int, ...]
    action_logprob: float
    token_logprobs: tuple[float, ...]
    prefix_values: tuple[float, ...]
    root_value: float
    stop_sampled: bool
    known_opponent_counts: tuple[tuple[int, int], ...]
    public_event_delta: PublicEventDelta | None = None
    accepted_action: AcceptedActionRecord | None = None
    reward: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed action traces and learner public evidence."""
        if self.decision_index < 0:
            raise ValueError("decision index must be non-negative")
        if not self.token_logprobs:
            raise ValueError("decode-token trace must be non-empty")
        if len(self.token_logprobs) != len(self.prefix_values):
            raise ValueError("token log-probs and prefix values must align")
        if any(not math.isfinite(value) for value in self.token_logprobs):
            raise ValueError("token log-probs must be finite")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.prefix_values
        ):
            raise ValueError("prefix values must be finite and in [-1, 1]")
        if not math.isfinite(self.action_logprob) or not math.isclose(
            self.action_logprob,
            sum(self.token_logprobs),
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("action log-prob must equal the decode-token sum")
        if not math.isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError("root value must be finite and in [-1, 1]")
        if not math.isfinite(self.reward):
            raise ValueError("decision reward must be finite")
        option_count = len(self.actor_row.options)
        if (
            len(self.action) < self.actor_row.min_count
            or len(self.action) > self.actor_row.max_count
            or len(set(self.action)) != len(self.action)
            or any(index < 0 or index >= option_count for index in self.action)
        ):
            raise ValueError("fragment action violates the engine legal option set")
        count_first = _actor_row_uses_count_first(self.actor_row)
        if count_first and self.action != tuple(sorted(self.action)):
            raise ValueError("count-first fragment action is not canonical")
        expected_tokens = len(self.action) + (
            1 if count_first else int(self.stop_sampled)
        )
        if len(self.token_logprobs) != expected_tokens:
            raise ValueError("decode-token trace differs from action termination")
        expected_stop = (
            not count_first and len(self.action) < self.actor_row.max_count
        )
        if self.stop_sampled != expected_stop:
            raise ValueError("fragment STOP flag differs from action termination")
        if tuple(sorted(self.known_opponent_counts)) != self.known_opponent_counts:
            raise ValueError("known opponent counts must be sorted")
        if len({card_id for card_id, _count in self.known_opponent_counts}) != len(
            self.known_opponent_counts
        ):
            raise ValueError("known opponent card IDs must be unique")
        if any(
            card_id <= 0 or count <= 0
            for card_id, count in self.known_opponent_counts
        ):
            raise ValueError("known opponent counts must be positive")
        if (
            self.accepted_action is not None
            and self.accepted_action.stable_identity
            != build_accepted_action_record(
                state=self.actor_row.state,
                options=self.actor_row.options,
                action=self.action,
                min_count=self.actor_row.min_count,
                max_count=self.actor_row.max_count,
                stop_sampled=self.stop_sampled,
                fallback=self.accepted_action.fallback,
            ).stable_identity
        ):
            raise ValueError("accepted action differs from fragment action")


@dataclass(frozen=True)
class StatelessFragment:
    """One terminal or behavior-bootstrapped fixed-horizon GAE unit."""

    identity: StatelessFragmentIdentity
    context: StatelessFragmentContext
    decisions: tuple[StatelessFragmentDecision, ...]
    terminal: bool
    truncated: bool
    bootstrap_value: float
    terminal_reward: float

    def __post_init__(self) -> None:
        """Validate whole-fragment behavior, indexing, and endpoint semantics."""
        if not self.decisions:
            raise ValueError("fragment must contain at least one decision")
        if self.terminal == self.truncated:
            raise ValueError("fragment must be exactly terminal or truncated")
        if len(self.decisions) > self.identity.horizon:
            raise ValueError("fragment exceeds its fixed horizon")
        expected = tuple(
            range(
                self.context.start_decision_index,
                self.context.start_decision_index + len(self.decisions),
            )
        )
        if tuple(decision.decision_index for decision in self.decisions) != expected:
            raise ValueError("fragment decision identities are not contiguous")
        if any(
            decision.actor_row.input_contract_fingerprint
            != self.identity.input_contract_fingerprint
            for decision in self.decisions
        ):
            raise ValueError("fragment mixes actor input contracts")
        sequence_rows = self.identity.schema_version == 2
        if sequence_rows and any(
            decision.public_event_delta is None
            or decision.accepted_action is None
            or decision.actor_row.sequence_identity is None
            for decision in self.decisions
        ):
            raise ValueError("sequence fragment is missing raw EVENT/STATE/ACTION truth")
        if not sequence_rows and any(
            decision.public_event_delta is not None
            or decision.accepted_action is not None
            for decision in self.decisions
        ):
            raise ValueError("stateless fragment cannot carry sequence-only fields")
        if any(
            decision.actor_row.catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
            for decision in self.decisions
        ):
            raise ValueError("fragment mixes public catalog identities")
        if any(
            decision.actor_row.own_deck.deck_digest
            != self.context.own_deck_digest
            for decision in self.decisions
        ):
            raise ValueError("fragment actor rows changed exact own deck")
        _validate_public_subtraction(self.context, self.decisions)
        if self.terminal:
            if self.bootstrap_value != 0.0:
                raise ValueError("terminal fragments cannot carry a bootstrap")
            if self.terminal_reward not in (-1.0, 0.0, 1.0):
                raise ValueError("terminal reward must be engine win/loss/draw")
            if self.decisions[-1].reward != self.terminal_reward:
                raise ValueError("terminal reward must be assigned to the final decision")
        else:
            if self.terminal_reward != 0.0:
                raise ValueError("truncated fragments cannot carry terminal reward")
            if (
                not math.isfinite(self.bootstrap_value)
                or not -1.0 <= self.bootstrap_value <= 1.0
            ):
                raise ValueError("fragment bootstrap must be finite and in [-1, 1]")
            if any(decision.reward != 0.0 for decision in self.decisions):
                raise ValueError("non-terminal fragment rewards must remain zero")

    @property
    def fragment_id(self) -> str:
        """Return a stable identity for duplicate/recovery checks."""
        return _fingerprint(
            _FRAGMENT_ID_DOMAIN,
            {
                "identity": self.identity.fingerprint,
                "game_id": self.context.game_id,
                "seat": self.context.seat,
                "start_decision_index": self.context.start_decision_index,
                "decisions": len(self.decisions),
                "terminal": self.terminal,
                "truncated": self.truncated,
            },
        )


class FixedHorizonFragmentBuilder:
    """Build one game-seat fragment without learned cross-callback state."""

    def __init__(
        self,
        identity: StatelessFragmentIdentity,
        context: StatelessFragmentContext,
    ) -> None:
        """Bind one behavior snapshot and a learner-only per-game context."""
        self.identity = identity
        self.context = context
        self._decisions: list[StatelessFragmentDecision] = []
        self._closed = False

    @property
    def full(self) -> bool:
        """Return whether the next public state must close via bootstrap."""
        return len(self._decisions) == self.identity.horizon

    @property
    def decision_count(self) -> int:
        """Return buffered candidate decisions."""
        return len(self._decisions)

    def append(self, decision: StatelessFragmentDecision) -> None:
        """Append one contiguous decision under the same behavior contract."""
        if self._closed:
            raise RuntimeError("fragment builder is closed")
        if self.full:
            raise RuntimeError("full fragment requires bootstrap before more rows")
        expected = self.context.start_decision_index + len(self._decisions)
        if decision.decision_index != expected:
            raise ValueError("fragment decision index is not contiguous")
        if (
            decision.actor_row.input_contract_fingerprint
            != self.identity.input_contract_fingerprint
        ):
            raise ValueError("decision input contract differs from behavior")
        if (
            decision.actor_row.catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("decision catalog differs from behavior")
        if decision.reward != 0.0:
            raise ValueError("collector decisions start with zero reward")
        self._decisions.append(decision)

    def truncate(
        self,
        *,
        bootstrap_value: float,
        bootstrap_policy_version: int,
        bootstrap_policy_fingerprint: str,
    ) -> StatelessFragment:
        """Close a full non-terminal fragment using the same behavior snapshot."""
        if self._closed:
            raise RuntimeError("fragment builder is closed")
        if not self.full:
            raise ValueError("truncated fragment has not reached fixed horizon")
        if (
            bootstrap_policy_version != self.identity.behavior_policy_version
            or bootstrap_policy_fingerprint
            != self.identity.behavior_policy_fingerprint
        ):
            raise ValueError("fragment bootstrap behavior identity changed")
        self._closed = True
        return StatelessFragment(
            identity=self.identity,
            context=self.context,
            decisions=tuple(self._decisions),
            terminal=False,
            truncated=True,
            bootstrap_value=float(bootstrap_value),
            terminal_reward=0.0,
        )

    def finish_terminal(self, *, engine_reward: float) -> StatelessFragment:
        """Close at a real engine terminal without any bootstrap."""
        if self._closed:
            raise RuntimeError("fragment builder is closed")
        if not self._decisions:
            raise ValueError("terminal fragment has no candidate decision")
        if engine_reward not in (-1.0, 0.0, 1.0):
            raise ValueError("engine reward must be win/loss/draw")
        self._closed = True
        decisions = list(self._decisions)
        decisions[-1] = replace(decisions[-1], reward=float(engine_reward))
        return StatelessFragment(
            identity=self.identity,
            context=self.context,
            decisions=tuple(decisions),
            terminal=True,
            truncated=False,
            bootstrap_value=0.0,
            terminal_reward=float(engine_reward),
        )


def next_fragment_context(
    fragment: StatelessFragment,
    *,
    assignment_id: str | None = None,
) -> StatelessFragmentContext:
    """Continue the same game after a truncated boundary."""
    if not fragment.truncated:
        raise ValueError("only a truncated fragment can continue a game")
    return fragment.context.model_copy(
        update={
            "start_decision_index": (
                fragment.context.start_decision_index + len(fragment.decisions)
            ),
            "assignment_id": assignment_id or fragment.context.assignment_id,
        }
    )


def fragment_is_trainable(
    fragment: StatelessFragment,
    *,
    current_policy_version: int,
    maximum_version_age: int,
) -> bool:
    """Apply staleness to the entire GAE unit without slicing middle rows."""
    if current_policy_version < fragment.identity.behavior_policy_version:
        raise ValueError("current policy version cannot trail behavior")
    if maximum_version_age < 0:
        raise ValueError("maximum version age must be non-negative")
    return (
        current_policy_version - fragment.identity.behavior_policy_version
        <= maximum_version_age
    )


def _validate_public_subtraction(
    context: StatelessFragmentContext,
    decisions: tuple[StatelessFragmentDecision, ...],
) -> None:
    exact = Counter(context.opponent_deck)
    for decision in decisions:
        known = Counter(dict(decision.known_opponent_counts))
        if any(exact[card_id] < count for card_id, count in known.items()):
            raise ValueError(
                "public evidence cannot be subtracted from learner opponent deck"
            )
        if sum(known.values()) > DECK_SIZE:
            raise ValueError("public evidence exceeds exact opponent deck")


def _actor_row_uses_count_first(row: SimpleStatelessActorRow) -> bool:
    if row.min_count >= row.max_count:
        return False
    return any(
        int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        for context in row.options.contexts
    )


def _fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


__all__ = [
    "FixedHorizonFragmentBuilder",
    "StatelessFragment",
    "StatelessFragmentContext",
    "StatelessFragmentDecision",
    "StatelessFragmentIdentity",
    "fragment_is_trainable",
    "next_fragment_context",
]
