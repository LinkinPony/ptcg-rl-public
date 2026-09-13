"""Transactional per-game temporal KV and raw-tape ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ptcg_rl.context.public_events import PublicEventDelta
from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.model.sequence.core import TemporalKvCache
from ptcg_rl.model.sequence.network import TemporalPreparedDecision
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SequenceCacheIdentity:
    """Complete isolation identity for one actor temporal cache."""

    game_id: str
    seat: Literal[0, 1]
    exact_deck_digest: str
    policy_artifact_fingerprint: str
    model_config_fingerprint: str
    input_contract_fingerprint: str
    sequence_contract_fingerprint: str
    max_context_blocks: int
    _hash_value: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize immutable fields and cache the frequently reused hash."""
        game_id = self.game_id.strip()
        if not game_id:
            raise ValueError("sequence cache game ID cannot be empty")
        if self.seat not in (0, 1):
            raise ValueError("sequence cache seat must be zero or one")
        if self.max_context_blocks < 2:
            raise ValueError("sequence cache context must contain at least two blocks")
        object.__setattr__(self, "game_id", game_id)
        fingerprint_names = (
            "exact_deck_digest",
            "policy_artifact_fingerprint",
            "model_config_fingerprint",
            "input_contract_fingerprint",
            "sequence_contract_fingerprint",
        )
        for name in fingerprint_names:
            normalized = getattr(self, name).strip().lower()
            if _SHA256_PATTERN.fullmatch(normalized) is None:
                raise ValueError("sequence cache identity component must be SHA-256")
            object.__setattr__(self, name, normalized)
        object.__setattr__(
            self,
            "_hash_value",
            hash(
                (
                    self.game_id,
                    self.seat,
                    self.exact_deck_digest,
                    self.policy_artifact_fingerprint,
                    self.model_config_fingerprint,
                    self.input_contract_fingerprint,
                    self.sequence_contract_fingerprint,
                    self.max_context_blocks,
                )
            ),
        )

    def __hash__(self) -> int:
        """Return the construction-time hash for hot transactional lookups."""
        return self._hash_value


@dataclass(frozen=True)
class SequenceRawBlock:
    """Append-only learner truth for one accepted policy decision."""

    block_index: int
    public_events: PublicEventDelta
    snapshot: SimpleStatelessActorRow
    accepted_action: AcceptedActionRecord
    engine_fact_producer_fingerprint: str | None

    def __post_init__(self) -> None:
        """Keep block coordinates and snapshot metadata coherent."""
        if self.block_index < 0:
            raise ValueError("sequence raw block index must be non-negative")
        identity = self.snapshot.sequence_identity
        if identity is not None and identity.decision_index != self.block_index:
            raise ValueError("sequence snapshot and raw block clocks differ")
        if (
            self.engine_fact_producer_fingerprint
            != self.snapshot.engine_fact_producer_fingerprint
        ):
            raise ValueError("sequence engine fact producer identity differs")


@dataclass(frozen=True)
class SequenceCacheFork:
    """Immutable snapshot from which a provisional branch is computed."""

    identity: SequenceCacheIdentity
    request_id: str
    block_index: int
    generation: int
    committed_cache: TemporalKvCache | None


@dataclass(frozen=True)
class StagedSequenceProposal:
    """Provisional EVENT/STATE branch retained for duplicate retries."""

    fork: SequenceCacheFork
    prepared: TemporalPreparedDecision
    accepted_action: AcceptedActionRecord | None


@dataclass
class _CacheEntry:
    committed_cache: TemporalKvCache | None = None
    raw_blocks: tuple[SequenceRawBlock, ...] = ()
    next_block_index: int = 0
    generation: int = 0
    staged: StagedSequenceProposal | None = None


class TransactionalSequenceCache:
    """Fail-closed prepare/stage/commit/abort cache state machine."""

    def __init__(self) -> None:
        """Create an empty cache registry."""
        self._entries: dict[SequenceCacheIdentity, _CacheEntry] = {}
        self._game_seat_identities: dict[
            tuple[str, int], set[SequenceCacheIdentity]
        ] = {}

    def prepare(
        self,
        identity: SequenceCacheIdentity,
        *,
        request_id: str,
        block_index: int,
    ) -> SequenceCacheFork | StagedSequenceProposal:
        """Fork committed KV or return the exact staged duplicate proposal."""
        normalized = request_id.strip()
        if not normalized:
            raise ValueError("sequence request ID cannot be empty")
        entry = self._entries.get(identity)
        if entry is None:
            entry = _CacheEntry()
            self._entries[identity] = entry
            self._game_seat_identities.setdefault(
                (identity.game_id, identity.seat),
                set(),
            ).add(identity)
        staged = entry.staged
        if staged is not None:
            if (
                staged.fork.request_id == normalized
                and staged.fork.block_index == block_index
            ):
                return staged
            raise RuntimeError("sequence cache already has a provisional proposal")
        if block_index != entry.next_block_index:
            raise RuntimeError("sequence decision clock is discontinuous")
        return SequenceCacheFork(
            identity=identity,
            request_id=normalized,
            block_index=block_index,
            generation=entry.generation,
            committed_cache=entry.committed_cache,
        )

    def stage(
        self,
        fork: SequenceCacheFork,
        *,
        prepared: TemporalPreparedDecision,
        accepted_action: AcceptedActionRecord | None,
    ) -> StagedSequenceProposal:
        """Publish one duplicate-safe provisional proposal."""
        entry = self._entry_for_fork(fork)
        if entry.staged is not None:
            raise RuntimeError("sequence proposal was already staged")
        if prepared.block_index != fork.block_index:
            raise ValueError("prepared temporal block differs from cache fork")
        proposal = StagedSequenceProposal(
            fork=fork,
            prepared=prepared,
            accepted_action=accepted_action,
        )
        entry.staged = proposal
        return proposal

    def commit(
        self,
        proposal: StagedSequenceProposal,
        *,
        committed_cache: TemporalKvCache,
        raw_block: SequenceRawBlock | None,
        retain_raw_block: bool = True,
    ) -> None:
        """Atomically publish ACTION KV and its raw complete block."""
        entry = self._entry_for_proposal(proposal)
        if raw_block is None:
            if retain_raw_block:
                raise ValueError("retained sequence history requires a raw block")
        else:
            if raw_block.block_index != proposal.fork.block_index:
                raise ValueError("raw block differs from staged decision clock")
            accepted_action = proposal.accepted_action
            if accepted_action is None or (
                raw_block.accepted_action.stable_identity
                != accepted_action.stable_identity
            ):
                raise ValueError("served action differs from staged accepted action")
        entry.committed_cache = committed_cache
        if retain_raw_block and raw_block is not None:
            entry.raw_blocks = (*entry.raw_blocks, raw_block)
        entry.next_block_index += 1
        entry.generation += 1
        entry.staged = None

    def abort(self, proposal: StagedSequenceProposal) -> None:
        """Discard a provisional branch without advancing cache or raw tape."""
        entry = self._entry_for_proposal(proposal)
        entry.staged = None

    def release(self, identity: SequenceCacheIdentity) -> None:
        """Release all cache and raw-tape residency after terminal/reset."""
        if self._entries.pop(identity, None) is None:
            return
        game_seat = (identity.game_id, identity.seat)
        identities = self._game_seat_identities[game_seat]
        identities.remove(identity)
        if not identities:
            del self._game_seat_identities[game_seat]

    def identities_for_game_seat(
        self,
        *,
        game_id: str,
        seat: int,
    ) -> tuple[SequenceCacheIdentity, ...]:
        """Return resident identities without scanning unrelated games."""
        return tuple(self._game_seat_identities.get((game_id, seat), ()))

    def has_staged(self, identity: SequenceCacheIdentity) -> bool:
        """Return whether an identity owns a provisional proposal."""
        entry = self._entries.get(identity)
        return entry is not None and entry.staged is not None

    def raw_blocks(
        self,
        identity: SequenceCacheIdentity,
    ) -> tuple[SequenceRawBlock, ...]:
        """Return immutable raw truth retained for same-weight rebuild."""
        entry = self._entries.get(identity)
        return () if entry is None else entry.raw_blocks

    def committed_cache(
        self,
        identity: SequenceCacheIdentity,
    ) -> TemporalKvCache | None:
        """Return current actor KV for diagnostics and rebuild parity."""
        entry = self._entries.get(identity)
        return None if entry is None else entry.committed_cache

    def identities(self) -> tuple[SequenceCacheIdentity, ...]:
        """Return current cache identities for lifecycle diagnostics."""
        return tuple(self._entries)

    def _entry_for_fork(self, fork: SequenceCacheFork) -> _CacheEntry:
        entry = self._entries.get(fork.identity)
        if entry is None:
            raise RuntimeError("sequence cache fork identity was released")
        if (
            fork.generation != entry.generation
            or fork.block_index != entry.next_block_index
        ):
            raise RuntimeError("sequence cache fork is stale")
        return entry

    def _entry_for_proposal(
        self,
        proposal: StagedSequenceProposal,
    ) -> _CacheEntry:
        entry = self._entry_for_fork(proposal.fork)
        if entry.staged is not proposal:
            raise RuntimeError("sequence proposal is stale or not owned")
        return entry


__all__ = [
    "SequenceCacheFork",
    "SequenceCacheIdentity",
    "SequenceRawBlock",
    "StagedSequenceProposal",
    "TransactionalSequenceCache",
]
