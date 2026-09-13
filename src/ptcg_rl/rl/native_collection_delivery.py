"""Bounded in-memory native trajectory-part buffering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.stateless_array_replay import _validate_source_semantics
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart

Array = npt.NDArray[np.generic]

_FRAGMENT_FIELDS = (
    "fragment_ids",
    "game_ids",
    "seats",
    "start_decision_indices",
    "own_decks",
    "opponent_decks",
    "own_deck_digests",
    "opponent_deck_digests",
    "curriculum_generations",
    "assignment_ids",
    "opponent_artifact_fingerprints",
    "terminal",
    "truncated",
    "bootstrap_values",
    "terminal_rewards",
    "horizons",
    "behavior_policy_versions",
    "behavior_policy_fingerprints",
    "model_config_fingerprints",
    "action_schema_fingerprints",
    "public_context_fingerprints",
    "card_catalog_fingerprints",
    "public_deck_catalog_fingerprints",
    "exact_registry_fingerprints",
    "belief_target_semantics_fingerprints",
    "input_contract_fingerprints",
    "resolved_config_fingerprints",
)
_DECISION_FIELDS = (
    "decision_indices",
    "action_logprobs",
    "root_values",
    "rewards",
    "stop_sampled",
    "min_counts",
    "max_counts",
    "belief_scalars",
)
_RAGGED_FIELDS = (
    (
        "state_offsets",
        (
            "state_card_ids",
            "state_areas",
            "state_owner_roles",
            "state_token_kinds",
            "state_scalars",
            "state_last_attack_ids",
            "state_entity_slots",
        ),
    ),
    (
        "attachment_offsets",
        (
            "attachment_card_ids",
            "attachment_parent_indices",
            "attachment_kinds",
        ),
    ),
    (
        "option_offsets",
        (
            "option_types",
            "option_contexts",
            "option_entity_slots",
            "option_entity_slot_mask",
            "option_attack_ids",
            "option_card_ids",
            "option_scalars",
            "option_dynamic_effect_features",
            "option_dynamic_effect_masks",
        ),
    ),
    (
        "belief_offsets",
        (
            "belief_card_ids",
            "belief_expected_counts",
        ),
    ),
    (
        "known_offsets",
        (
            "known_card_ids",
            "known_counts",
        ),
    ),
    ("action_offsets", ("action_choices",)),
    (
        "token_offsets",
        (
            "token_logprobs",
            "prefix_values",
        ),
    ),
)
_SEQUENCE_FRAGMENT_FIELDS = (
    "fragment_schema_versions",
    "sequence_contract_fingerprints",
)
_SEQUENCE_DECISION_FIELDS = (
    "engine_fact_producer_fingerprints",
    "sequence_request_ids",
    "accepted_action_stable_ids",
    "accepted_action_prompt_contexts",
    "accepted_action_ordered",
    "accepted_action_fallback",
)
_SEQUENCE_RAGGED_FIELDS = (
    (
        "event_offsets",
        (
            "event_types",
            "event_actor_roles",
            "event_from_areas",
            "event_to_areas",
            "event_card_ids",
            "event_serials",
            "event_entity_mask",
            "event_attack_ids",
            "event_attack_id_mask",
            "event_values",
            "event_value_mask",
            "event_categorical_values",
        ),
    ),
    (
        "event_overflow_offsets",
        (
            "event_overflow_types",
            "event_overflow_actor_roles",
            "event_overflow_counts",
        ),
    ),
)
_SEQUENCE_ACTION_FIELDS = (
    "accepted_action_option_types",
    "accepted_action_option_contexts",
    "accepted_action_card_ids",
    "accepted_action_attack_ids",
    "accepted_action_option_scalars",
    "accepted_action_entity_card_ids",
    "accepted_action_entity_areas",
    "accepted_action_entity_owner_roles",
    "accepted_action_entity_token_kinds",
    "accepted_action_entity_scalars",
)


class NativePartDelivery:
    """Retain compact parts under one rollback-capable optimizer transaction."""

    def __init__(
        self,
        *,
        part_sink: Callable[[CompactFragmentPart], None] | None = None,
    ) -> None:
        self._retained: list[CompactFragmentPart] = []
        self._part_sink = part_sink
        self.fragment_count = 0
        self.decision_count = 0

    @property
    def retained(self) -> tuple[CompactFragmentPart, ...]:
        """Return local parts in production order."""
        return tuple(self._retained)

    def accept(self, parts: Sequence[CompactFragmentPart]) -> None:
        """Retain complete parts and update aligned window totals."""
        for part in parts:
            if part.path is not None:
                raise ValueError("native hot-path parts must remain memory-only")
            self._retained.append(part)
            self.fragment_count += part.fragment_count
            self.decision_count += part.decision_count
            if self._part_sink is not None:
                self._part_sink(part)

    def discard_games(self, game_ids: Sequence[str]) -> None:
        """Revoke every published fragment for an explicit whole-game rollback.

        This is deliberately an array-native cold path. Normal full-horizon
        publication can therefore release encoder chunks immediately, while a
        cutoff, watchdog step limit, or failed collection can still remove all
        data from affected games.
        """
        rows = tuple(game_ids)
        targets = frozenset(rows)
        if not targets:
            raise ValueError("native delivery discard requires game IDs")
        if len(targets) != len(rows) or any(
            not value or value.strip() != value for value in targets
        ):
            raise ValueError("native delivery discard game IDs are invalid")
        retained = [
            filtered
            for part in self._retained
            if (filtered := _discard_field_values(part, "game_ids", targets))
            is not None
        ]
        self._retained = retained
        self.fragment_count = sum(part.fragment_count for part in retained)
        self.decision_count = sum(part.decision_count for part in retained)


def discard_compact_assignments(
    part: CompactFragmentPart,
    assignment_ids: Sequence[str],
) -> CompactFragmentPart | None:
    """Remove every fragment owned by explicit curriculum assignments."""
    targets = frozenset(assignment_ids)
    if not targets or any(not value or value.strip() != value for value in targets):
        raise ValueError("native assignment discard IDs are invalid")
    return _discard_field_values(part, "assignment_ids", targets)


def _discard_field_values(
    part: CompactFragmentPart,
    field: str,
    targets: frozenset[str],
) -> CompactFragmentPart | None:
    """Return a compact part with matching whole fragments removed."""
    keep = np.asarray(
        [str(value) not in targets for value in part.arrays[field]],
        dtype=np.bool_,
    )
    if np.all(keep):
        return part
    if not np.any(keep):
        return None
    return _select_fragments(part, keep)


def _select_fragments(
    part: CompactFragmentPart,
    keep: npt.NDArray[np.bool_],
) -> CompactFragmentPart:
    """Copy selected compact fragments while rebuilding every ragged offset."""
    arrays = part.arrays
    fragments = part.fragment_count
    if keep.shape != (fragments,) or not np.any(keep) or np.all(keep):
        raise ValueError("native fragment selection must be a proper non-empty subset")
    selected_fragments = np.flatnonzero(keep)
    fragment_offsets = np.asarray(
        arrays["fragment_decision_offsets"],
        dtype=np.int64,
    )
    fragment_lengths = np.diff(fragment_offsets)[selected_fragments]
    decision_rows = np.concatenate(
        [
            np.arange(
                fragment_offsets[row],
                fragment_offsets[row + 1],
                dtype=np.int64,
            )
            for row in selected_fragments
        ]
    )
    result: dict[str, Array] = {
        "schema_version": np.ascontiguousarray(arrays["schema_version"]),
        "fragment_decision_offsets": _offsets(fragment_lengths),
        "decision_fragment_indices": np.repeat(
            np.arange(selected_fragments.size, dtype=np.int32),
            fragment_lengths,
        ),
    }
    fragment_fields = _FRAGMENT_FIELDS + tuple(
        field for field in _SEQUENCE_FRAGMENT_FIELDS if field in arrays
    )
    decision_fields = _DECISION_FIELDS + tuple(
        field for field in _SEQUENCE_DECISION_FIELDS if field in arrays
    )
    for field in fragment_fields:
        result[field] = _select_rows(arrays[field], selected_fragments)
    for field in decision_fields:
        result[field] = _select_rows(arrays[field], decision_rows)
    ragged_fields = _RAGGED_FIELDS + tuple(
        item for item in _SEQUENCE_RAGGED_FIELDS if item[0] in arrays
    )
    for offsets_field, value_fields in ragged_fields:
        source_offsets = np.asarray(arrays[offsets_field], dtype=np.int64)
        value_lengths = (
            source_offsets[decision_rows + 1] - source_offsets[decision_rows]
        )
        value_rows = np.concatenate(
            [
                np.arange(
                    source_offsets[row],
                    source_offsets[row + 1],
                    dtype=np.int64,
                )
                for row in decision_rows
            ]
        )
        result[offsets_field] = _offsets(value_lengths)
        selected_value_fields: tuple[str, ...] = tuple(value_fields)
        if offsets_field == "action_offsets":
            selected_value_fields += tuple(
                field for field in _SEQUENCE_ACTION_FIELDS if field in arrays
            )
        for field in selected_value_fields:
            result[field] = _select_rows(arrays[field], value_rows)
    if set(result) != set(arrays):
        raise RuntimeError("native fragment selection changed the compact schema")
    static_fingerprints = _validate_source_semantics(result)
    if len(set(static_fingerprints)) != 1:
        raise RuntimeError("native fragment selection mixed static contracts")
    return CompactFragmentPart(path=None, arrays=result)


def _select_rows(values: Array, rows: npt.NDArray[np.int64]) -> Array:
    """Return contiguous rows with canonical string width."""
    selected = np.asarray(values)[rows]
    if selected.dtype.kind == "U":
        strings = [str(value) for value in selected]
        width = max(1, *(len(value) for value in strings))
        return np.asarray(strings, dtype=f"U{width}")
    return np.ascontiguousarray(selected)


def _offsets(lengths: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return canonical CSR offsets for selected row lengths."""
    result = np.zeros(lengths.size + 1, dtype=np.int64)
    result[1:] = np.cumsum(lengths, dtype=np.int64)
    return result


__all__ = ["NativePartDelivery"]
