"""Observation-free inference for frozen legacy historical policies."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.context import OpponentBeliefFeatureConfig, OpponentBeliefFeatureProducer
from ptcg_rl.decks import (
    CanonicalDeck,
    DeckBatch,
    canonicalize_deck,
    concatenate_deck_batches,
)
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_public_history import SUCCESS_STATUSES
from ptcg_rl.engine.native_rollout import (
    NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT,
)
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.native_legacy_belief import (
    LegacyBelief,
    append_legacy_belief_tokens,
    legacy_known_counts,
)
from ptcg_rl.rl.native_policy_context import NativePublicContextBatch
from ptcg_rl.rl.native_policy_options import encode_native_option_batch
from ptcg_rl.rl.native_policy_state import encode_native_state_batch
from ptcg_rl.rl.stateless_training_config import StatelessHistoricalAnchorConfig

_DEFAULT_BELIEF_CACHE_CAPACITY = 32768
_Batch = TypeVar("_Batch", StateBatch, OptionBatch)
_KnownIdentity = tuple[tuple[int, int], ...]
_ROW_TUPLE_FIELDS = frozenset(
    (
        "root_input_fingerprints",
        "sequence_lengths",
        "option_lengths",
        "maximum_counts",
    )
)


@dataclass(frozen=True, slots=True)
class _ArtifactBinding:
    checkpoint_path: Path
    checkpoint_sha256: str
    device: str


@dataclass(slots=True)
class _ArtifactRuntime:
    policy: CheckpointPolicy


@dataclass(slots=True)
class _MemberRuntime:
    config: StatelessHistoricalAnchorConfig
    own_deck: CanonicalDeck
    producer: OpponentBeliefFeatureProducer | None
    belief_cache: OrderedDict[_KnownIdentity, LegacyBelief]
    deck_batches: dict[str, DeckBatch]


class NativeHistoricalPolicyPool:
    """Batch frozen legacy checkpoints directly from native public columns."""

    def __init__(
        self,
        resources: Mapping[
            str,
            tuple[StatelessHistoricalAnchorConfig, Sequence[int]],
        ],
        *,
        belief_cache_capacity: int = _DEFAULT_BELIEF_CACHE_CAPACITY,
    ) -> None:
        """Bind immutable members while deferring each checkpoint load."""
        if belief_cache_capacity <= 0:
            raise ValueError("historical belief cache capacity must be positive")
        self._belief_cache_capacity = int(belief_cache_capacity)
        self._members: dict[str, _MemberRuntime] = {}
        self._bindings: dict[str, _ArtifactBinding] = {}
        self._artifacts: dict[str, _ArtifactRuntime] = {}
        self.ensure_resources(resources)

    def ensure_resources(
        self,
        resources: Mapping[
            str,
            tuple[StatelessHistoricalAnchorConfig, Sequence[int]],
        ],
    ) -> None:
        """Add immutable member bindings without evicting loaded artifacts."""
        for member_id, (config, raw_deck) in resources.items():
            if member_id != config.member_id:
                raise ValueError("historical resource key differs from member ID")
            deck = canonicalize_deck(raw_deck)
            if deck.deck_digest != config.exact_deck_digest:
                raise ValueError("historical native deck fingerprint mismatch")
            existing = self._members.get(member_id)
            if existing is not None:
                if existing.config != config or existing.own_deck.card_ids != deck.card_ids:
                    raise ValueError(
                        "historical member ID has conflicting immutable resources"
                    )
                self._bind_artifact(config)
                continue
            producer = (
                None
                if config.belief_summary_path is None
                else OpponentBeliefFeatureProducer.from_config(
                    OpponentBeliefFeatureConfig(
                        enabled=True,
                        deck_signature_summary_path=config.belief_summary_path,
                        deck_signature_summary_sha256=(config.belief_summary_sha256),
                    )
                )
            )
            self._members[member_id] = _MemberRuntime(
                config=config,
                own_deck=deck,
                producer=producer,
                belief_cache=OrderedDict(),
                deck_batches={},
            )
            self._bind_artifact(config)

    @property
    def loaded_artifact_count(self) -> int:
        """Return the number of unique checkpoint models loaded in this process."""
        return len(self._artifacts)

    @property
    def loaded_artifact_ids(self) -> tuple[str, ...]:
        """Return loaded checkpoint identities in stable binding order."""
        return tuple(
            artifact_id
            for artifact_id in self._bindings
            if artifact_id in self._artifacts
        )

    def preload_artifacts(self) -> None:
        """Load each immutable checkpoint once before the rollout hot path."""
        self.retain_artifacts(tuple(self._bindings))

    def ensure_artifacts(self, checkpoint_sha256s: Sequence[str]) -> None:
        """Load requested immutable checkpoints without evicting warm runtimes."""
        required = tuple(dict.fromkeys(checkpoint_sha256s))
        missing = set(required) - set(self._bindings)
        if missing:
            raise ValueError(
                "historical lease references unknown checkpoint artifacts: "
                + ", ".join(sorted(missing))
            )
        for artifact_id in required:
            self._artifact(artifact_id)

    def retain_artifacts(self, checkpoint_sha256s: Sequence[str]) -> None:
        """Load exactly the declared lease set and evict every other runtime."""
        required = tuple(dict.fromkeys(checkpoint_sha256s))
        missing = set(required) - set(self._bindings)
        if missing:
            raise ValueError(
                "historical lease references unknown checkpoint artifacts: "
                + ", ".join(sorted(missing))
            )
        required_set = set(required)
        for artifact_id in tuple(self._artifacts):
            if artifact_id in required_set:
                continue
            runtime = self._artifacts.pop(artifact_id)
            runtime.policy.close()
        for member in self._members.values():
            if member.config.checkpoint_sha256 not in required_set:
                member.belief_cache.clear()
                member.deck_batches.clear()
        self.ensure_artifacts(required)

    def close(self) -> None:
        """Release all loaded legacy checkpoints and device-side deck caches."""
        self.retain_artifacts(())

    def act_many(
        self,
        view: NativeTrainingBatchView,
        context: NativePublicContextBatch,
        known_opponent: NativeKnownOpponentBatch,
        *,
        member_ids: Sequence[str],
        forced_actions: Sequence[tuple[int, ...] | None] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return aligned forced or greedy actions without observation objects."""
        rows = view.batch_size
        context.validate(batch_size=rows)
        if len(member_ids) != rows:
            raise ValueError("historical member IDs must align with native rows")
        for row, member_id in enumerate(member_ids):
            try:
                member = self._members[member_id]
            except KeyError as error:
                raise KeyError(
                    f"unknown native historical member: {member_id}"
                ) from error
            _validate_member_deck(context, row=row, member=member)
        base_states, lookup = encode_native_state_batch(
            view,
            context,
            device="cpu",
        )
        base_options = encode_native_option_batch(
            view,
            lookup,
            device="cpu",
        )
        return self.act_many_preencoded(
            view,
            base_states,
            base_options,
            known_opponent,
            member_ids=member_ids,
            deck_signatures=context.deck_signatures,
            model_encoding_fingerprint=(
                NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT
            ),
            forced_actions=forced_actions,
        )

    def act_many_preencoded(
        self,
        view: NativeTrainingBatchView,
        base_states: StateBatch,
        base_options: OptionBatch,
        known_opponent: NativeKnownOpponentBatch,
        *,
        member_ids: Sequence[str],
        deck_signatures: Sequence[str],
        model_encoding_fingerprint: str,
        forced_actions: Sequence[tuple[int, ...] | None] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Run frozen policies from native model-ready state/option tensors."""
        if (
            model_encoding_fingerprint
            != NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT
        ):
            raise ValueError(
                "historical base tensors use an incompatible model encoding"
            )
        rows = view.batch_size
        if len(member_ids) != rows or len(deck_signatures) != rows:
            raise ValueError("historical identities must align with native rows")
        if (
            base_states.card_ids.shape[0] != rows
            or base_options.option_types.shape[0] != rows
        ):
            raise ValueError(
                "historical preencoded tensors must align with native rows"
            )
        forced = (
            tuple(None for _ in range(rows))
            if forced_actions is None
            else tuple(forced_actions)
        )
        if len(forced) != rows:
            raise ValueError("historical forced actions must align with native rows")
        _validate_known_batch(known_opponent, batch_size=rows)

        members: list[_MemberRuntime] = []
        actions: list[tuple[int, ...] | None] = [None] * rows
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for row, (member_id, deck_signature, forced_action) in enumerate(
            zip(member_ids, deck_signatures, forced, strict=True)
        ):
            try:
                member = self._members[member_id]
            except KeyError as error:
                raise KeyError(
                    f"unknown native historical member: {member_id}"
                ) from error
            members.append(member)
            if deck_signature != member.own_deck.signature:
                raise ValueError("historical native row deck signature mismatch")
            if int(view.status[row]) not in SUCCESS_STATUSES:
                raise ValueError("historical native inference received a failed row")
            if forced_action is not None:
                actions[row] = _validate_forced_action(
                    view,
                    row=row,
                    action=forced_action,
                )
            else:
                grouped[member.config.checkpoint_sha256].append(row)

        if not grouped:
            return cast(
                tuple[tuple[int, ...], ...],
                tuple(actions),
            )

        for artifact_sha256, artifact_rows in grouped.items():
            artifact = self._artifact(artifact_sha256)
            row_indices = torch.tensor(artifact_rows, dtype=torch.long)
            group_members = tuple(members[row] for row in artifact_rows)
            group_states = _select_batch(base_states, row_indices)
            group_beliefs = tuple(
                self._belief_for_row(
                    member,
                    known_opponent,
                    view,
                    row=row,
                )
                for row, member in zip(
                    artifact_rows,
                    group_members,
                    strict=True,
                )
            )
            group_states = append_legacy_belief_tokens(
                group_states,
                group_beliefs,
            )
            device = torch.device(artifact.policy.device)
            device_states = _move_batch(group_states, device=device)
            device_options = _move_batch(
                _select_batch(base_options, row_indices),
                device=device,
            )
            decks = concatenate_deck_batches(
                tuple(
                    self._deck_batch(member, device=device) for member in group_members
                )
            )
            decoded = artifact.policy.select_preencoded_actions(
                device_states,
                device_options,
                decks,
                max_select_steps=max(
                    (int(view.select_max[row]) for row in artifact_rows),
                    default=0,
                ),
            )
            for row, action in zip(artifact_rows, decoded, strict=True):
                actions[row] = _normalize_native_action(view, row=row, action=action)

        if any(action is None for action in actions):
            raise RuntimeError("historical native inference omitted an action")
        return cast(tuple[tuple[int, ...], ...], tuple(actions))

    def _bind_artifact(self, config: StatelessHistoricalAnchorConfig) -> None:
        binding = _ArtifactBinding(
            checkpoint_path=config.checkpoint_path.resolve(),
            checkpoint_sha256=config.checkpoint_sha256,
            device=config.device,
        )
        existing = self._bindings.get(config.checkpoint_sha256)
        if existing is not None and existing != binding:
            raise ValueError(
                "one historical checkpoint artifact has conflicting bindings"
            )
        self._bindings[config.checkpoint_sha256] = binding

    def _artifact(self, checkpoint_sha256: str) -> _ArtifactRuntime:
        existing = self._artifacts.get(checkpoint_sha256)
        if existing is not None:
            return existing
        binding = self._bindings[checkpoint_sha256]
        policy = CheckpointPolicy(
            binding.checkpoint_path,
            device=binding.device,
        )
        if policy.checkpoint_sha256 != binding.checkpoint_sha256:
            raise ValueError("historical native checkpoint SHA-256 mismatch")
        if policy.recurrent_enabled:
            raise ValueError("native historical pool cannot load recurrent checkpoints")
        expected_registries = {
            member.config.exact_registry_fingerprint
            for member in self._members.values()
            if member.config.checkpoint_sha256 == checkpoint_sha256
        }
        actual_registry = policy.checkpoint_registry_sha256
        if actual_registry is not None and expected_registries != {actual_registry}:
            raise ValueError("historical native checkpoint registry mismatch")
        policy.configure_inference_cache(enabled=False)
        runtime = _ArtifactRuntime(policy=policy)
        self._artifacts[checkpoint_sha256] = runtime
        return runtime

    @staticmethod
    def _deck_batch(
        member: _MemberRuntime,
        *,
        device: torch.device,
    ) -> DeckBatch:
        key = str(device)
        cached = member.deck_batches.get(key)
        if cached is not None:
            return cached
        batch = DeckBatch.from_decks((member.own_deck,), device=device)
        member.deck_batches[key] = batch
        return batch

    def _belief_for_row(
        self,
        member: _MemberRuntime,
        known: NativeKnownOpponentBatch,
        view: NativeTrainingBatchView,
        *,
        row: int,
    ) -> LegacyBelief:
        known_counts = legacy_known_counts(view, known, row=row)
        identity = tuple(known_counts.items())
        cached = member.belief_cache.get(identity)
        if cached is not None:
            member.belief_cache.move_to_end(identity)
            return cached
        if member.producer is None:
            features: LegacyBelief = ((), 0.0, True)
        else:
            features = member.producer.features_from_known_counts(known_counts)
        member.belief_cache[identity] = features
        if len(member.belief_cache) > self._belief_cache_capacity:
            member.belief_cache.popitem(last=False)
        return features


def _select_batch(batch: _Batch, indices: Tensor) -> _Batch:
    values: dict[str, Any] = {}
    host_indices = tuple(int(index) for index in indices.tolist())
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, Tensor):
            values[field.name] = value.index_select(0, indices)
        elif field.name in _ROW_TUPLE_FIELDS and value:
            values[field.name] = tuple(value[index] for index in host_indices)
        else:
            values[field.name] = value
    return type(batch)(**values)


def _move_batch(batch: _Batch, *, device: torch.device) -> _Batch:
    values: dict[str, Any] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = (
            value.to(device=device) if isinstance(value, Tensor) else value
        )
    return type(batch)(**values)


def _validate_member_deck(
    context: NativePublicContextBatch,
    *,
    row: int,
    member: _MemberRuntime,
) -> None:
    actual_cards = tuple(int(card_id) for card_id in context.own_decks[row].tolist())
    if actual_cards != member.own_deck.card_ids:
        raise ValueError("historical native row uses the wrong exact own deck")
    if context.deck_signatures[row] != member.own_deck.signature:
        raise ValueError("historical native row deck signature mismatch")


def _validate_known_batch(
    known: NativeKnownOpponentBatch,
    *,
    batch_size: int,
) -> None:
    offsets: npt.NDArray[np.int64] = known.offsets.astype(
        np.int64,
        copy=False,
    )
    if (
        offsets.shape != (batch_size + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(known.card_ids.shape[0])
        or np.any(offsets[1:] < offsets[:-1])
        or known.card_ids.shape != known.counts.shape
    ):
        raise ValueError("historical known-opponent CSR is invalid")
    if np.any(known.card_ids <= 0) or np.any(known.counts <= 0):
        raise ValueError("historical known-opponent entries must be positive")
    for row in range(batch_size):
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        pairs = tuple(
            (int(card_id), int(count))
            for card_id, count in zip(
                known.card_ids[start:stop],
                known.counts[start:stop],
                strict=True,
            )
        )
        if pairs != tuple(sorted(pairs)) or len({pair[0] for pair in pairs}) != len(
            pairs
        ):
            raise ValueError("historical known-opponent rows must be canonical")


def _validate_forced_action(
    view: NativeTrainingBatchView,
    *,
    row: int,
    action: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in action)
    option_count = int(view.option_offsets[row + 1] - view.option_offsets[row])
    minimum = min(option_count, max(0, int(view.select_min[row])))
    maximum = min(option_count, max(minimum, int(view.select_max[row])))
    if (
        len(normalized) < minimum
        or len(normalized) > maximum
        or len(set(normalized)) != len(normalized)
        or any(index < 0 or index >= option_count for index in normalized)
    ):
        raise ValueError("historical forced action is not legal")
    return normalized


def _normalize_native_action(
    view: NativeTrainingBatchView,
    *,
    row: int,
    action: Sequence[int],
) -> tuple[int, ...]:
    """Mirror legacy normalization without reconstructing a select mapping."""
    normalized = tuple(int(index) for index in action)
    # Legacy normalization preserves every multi-select sequence. Sorting the
    # remaining zero/singleton case is intentionally a semantic no-op.
    if int(view.select_max[row]) > 1:
        return normalized
    return tuple(sorted(normalized))


__all__ = ["NativeHistoricalPolicyPool"]
