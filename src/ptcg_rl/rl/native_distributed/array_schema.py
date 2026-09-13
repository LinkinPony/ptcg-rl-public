"""Canonical raw-array schema for memory-only native trajectory parts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.stateless_fragment_io import (
    SEQUENCE_FRAGMENT_ARRAY_KEYS,
    STATELESS_FRAGMENT_ARRAY_KEYS,
    _validate_fragment_arrays,
)

Array = npt.NDArray[np.generic]

COMPACT_FRAGMENT_ARRAY_FIELDS = (
    "schema_version",
    "fragment_ids",
    "fragment_decision_offsets",
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
    "decision_fragment_indices",
    "decision_indices",
    "action_logprobs",
    "root_values",
    "rewards",
    "stop_sampled",
    "state_offsets",
    "state_card_ids",
    "state_areas",
    "state_owner_roles",
    "state_token_kinds",
    "state_scalars",
    "state_last_attack_ids",
    "state_entity_slots",
    "attachment_offsets",
    "attachment_card_ids",
    "attachment_parent_indices",
    "attachment_kinds",
    "option_offsets",
    "option_types",
    "option_contexts",
    "option_entity_slots",
    "option_entity_slot_mask",
    "option_attack_ids",
    "option_card_ids",
    "option_scalars",
    "option_dynamic_effect_features",
    "option_dynamic_effect_masks",
    "min_counts",
    "max_counts",
    "belief_offsets",
    "belief_card_ids",
    "belief_expected_counts",
    "belief_scalars",
    "known_offsets",
    "known_card_ids",
    "known_counts",
    "action_offsets",
    "action_choices",
    "token_offsets",
    "token_logprobs",
    "prefix_values",
)
SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS = (
    *COMPACT_FRAGMENT_ARRAY_FIELDS,
    *tuple(
        sorted(SEQUENCE_FRAGMENT_ARRAY_KEYS - STATELESS_FRAGMENT_ARRAY_KEYS)
    ),
)

_STRING_FIELDS = frozenset(
    {
        "fragment_ids",
        "game_ids",
        "own_deck_digests",
        "opponent_deck_digests",
        "assignment_ids",
        "opponent_artifact_fingerprints",
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
        "sequence_contract_fingerprints",
        "engine_fact_producer_fingerprints",
        "sequence_request_ids",
        "accepted_action_stable_ids",
    }
)

_DTYPES: dict[str, np.dtype[np.generic]] = {
    "schema_version": np.dtype("<i2"),
    "fragment_decision_offsets": np.dtype("<i8"),
    "seats": np.dtype("|i1"),
    "start_decision_indices": np.dtype("<i8"),
    "own_decks": np.dtype("<i4"),
    "opponent_decks": np.dtype("<i4"),
    "curriculum_generations": np.dtype("<i8"),
    "terminal": np.dtype("|b1"),
    "truncated": np.dtype("|b1"),
    "bootstrap_values": np.dtype("<f4"),
    "terminal_rewards": np.dtype("<f4"),
    "horizons": np.dtype("<i4"),
    "behavior_policy_versions": np.dtype("<i8"),
    "decision_fragment_indices": np.dtype("<i4"),
    "decision_indices": np.dtype("<i8"),
    "action_logprobs": np.dtype("<f4"),
    "root_values": np.dtype("<f4"),
    "rewards": np.dtype("<f4"),
    "stop_sampled": np.dtype("|b1"),
    "state_offsets": np.dtype("<i8"),
    "state_card_ids": np.dtype("<i4"),
    "state_areas": np.dtype("<i2"),
    "state_owner_roles": np.dtype("|i1"),
    "state_token_kinds": np.dtype("|i1"),
    "state_scalars": np.dtype("<f4"),
    "state_last_attack_ids": np.dtype("<i4"),
    "state_entity_slots": np.dtype("|u1"),
    "attachment_offsets": np.dtype("<i8"),
    "attachment_card_ids": np.dtype("<i4"),
    "attachment_parent_indices": np.dtype("<i4"),
    "attachment_kinds": np.dtype("|i1"),
    "option_offsets": np.dtype("<i8"),
    "option_types": np.dtype("<i2"),
    "option_contexts": np.dtype("<i2"),
    "option_entity_slots": np.dtype("<i4"),
    "option_entity_slot_mask": np.dtype("|b1"),
    "option_attack_ids": np.dtype("<i4"),
    "option_card_ids": np.dtype("<i4"),
    "option_scalars": np.dtype("<f4"),
    "option_dynamic_effect_features": np.dtype("<f4"),
    "option_dynamic_effect_masks": np.dtype("|b1"),
    "min_counts": np.dtype("<i2"),
    "max_counts": np.dtype("<i2"),
    "belief_offsets": np.dtype("<i8"),
    "belief_card_ids": np.dtype("<i4"),
    "belief_expected_counts": np.dtype("<f4"),
    "belief_scalars": np.dtype("<f4"),
    "known_offsets": np.dtype("<i8"),
    "known_card_ids": np.dtype("<i4"),
    "known_counts": np.dtype("<i2"),
    "action_offsets": np.dtype("<i8"),
    "action_choices": np.dtype("<i4"),
    "token_offsets": np.dtype("<i8"),
    "token_logprobs": np.dtype("<f4"),
    "prefix_values": np.dtype("<f4"),
    "fragment_schema_versions": np.dtype("<i2"),
    "event_offsets": np.dtype("<i8"),
    "event_types": np.dtype("|u1"),
    "event_actor_roles": np.dtype("|u1"),
    "event_from_areas": np.dtype("|u1"),
    "event_to_areas": np.dtype("|u1"),
    "event_card_ids": np.dtype("<u2"),
    "event_serials": np.dtype("<i4"),
    "event_entity_mask": np.dtype("|b1"),
    "event_attack_ids": np.dtype("<i4"),
    "event_attack_id_mask": np.dtype("|b1"),
    "event_values": np.dtype("<f4"),
    "event_value_mask": np.dtype("|b1"),
    "event_categorical_values": np.dtype("|u1"),
    "event_overflow_offsets": np.dtype("<i8"),
    "event_overflow_types": np.dtype("|u1"),
    "event_overflow_actor_roles": np.dtype("|u1"),
    "event_overflow_counts": np.dtype("<i4"),
    "accepted_action_prompt_contexts": np.dtype("<i2"),
    "accepted_action_ordered": np.dtype("|b1"),
    "accepted_action_fallback": np.dtype("|b1"),
    "accepted_action_option_types": np.dtype("<i2"),
    "accepted_action_option_contexts": np.dtype("<i2"),
    "accepted_action_card_ids": np.dtype("<i4"),
    "accepted_action_attack_ids": np.dtype("<i4"),
    "accepted_action_option_scalars": np.dtype("<f4"),
    "accepted_action_entity_card_ids": np.dtype("<i4"),
    "accepted_action_entity_areas": np.dtype("<i2"),
    "accepted_action_entity_owner_roles": np.dtype("|i1"),
    "accepted_action_entity_token_kinds": np.dtype("|i1"),
    "accepted_action_entity_scalars": np.dtype("<f4"),
}

_MATRIX_FIELDS = frozenset(
    {
        "own_decks",
        "opponent_decks",
        "state_scalars",
        "option_entity_slots",
        "option_entity_slot_mask",
        "option_scalars",
        "option_dynamic_effect_features",
        "belief_scalars",
        "event_card_ids",
        "event_serials",
        "event_entity_mask",
        "event_categorical_values",
        "accepted_action_option_scalars",
        "accepted_action_entity_card_ids",
        "accepted_action_entity_areas",
        "accepted_action_entity_owner_roles",
        "accepted_action_entity_token_kinds",
    }
)
_TENSOR_FIELDS = frozenset({"accepted_action_entity_scalars"})


def expected_wire_dtype(name: str) -> np.dtype[np.generic] | None:
    """Return the fixed numeric dtype, or ``None`` for canonical Unicode."""
    if name in _STRING_FIELDS:
        return None
    return _DTYPES[name]


def expected_wire_ndim(name: str) -> int:
    """Return the fixed rank for a compact fragment column."""
    if name in _TENSOR_FIELDS:
        return 3
    return 2 if name in _MATRIX_FIELDS else 1


def compact_fragment_array_fields(
    arrays: Mapping[str, Array],
) -> tuple[str, ...]:
    """Select the authoritative V1 or V2 column inventory for one part."""
    raw_version = np.asarray(arrays.get("schema_version"))
    if raw_version.shape != (1,):
        raise ValueError("compact fragment schema version is invalid")
    schema_version = int(raw_version[0])
    if schema_version == 1:
        return COMPACT_FRAGMENT_ARRAY_FIELDS
    if schema_version == 2:
        return SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS
    raise ValueError("compact fragment schema version is unsupported")


def prepare_compact_fragment_arrays(
    arrays: Mapping[str, Array],
) -> tuple[Array, ...]:
    """Validate a full part and expose C-contiguous columns in wire order.

    Existing contiguous arrays remain zero-copy. A non-contiguous source column
    receives only the one unavoidable normalization copy.
    """
    fields = compact_fragment_array_fields(arrays)
    _validate_exact_schema(arrays, fields=fields)
    normalized_columns: list[Array] = []
    for name in fields:
        array = np.asarray(arrays[name])
        if name in _STRING_FIELDS and array.dtype.kind == "U":
            width = max((len(str(value)) for value in array.flat), default=1)
            array = array.astype(
                np.dtype(f"<U{max(width, 1)}"),
                copy=False,
            )
        normalized_columns.append(np.ascontiguousarray(array))
    normalized = tuple(normalized_columns)
    canonical = dict(zip(fields, normalized, strict=True))
    validate_compact_fragment_arrays(canonical)
    return normalized


def validate_compact_fragment_arrays(arrays: Mapping[str, Array]) -> None:
    """Validate exact wire dtypes, ranks, contiguity, and fragment semantics."""
    fields = compact_fragment_array_fields(arrays)
    _validate_exact_schema(arrays, fields=fields)
    for name in fields:
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise TypeError(f"compact fragment field {name} is not an ndarray")
        if array.ndim != expected_wire_ndim(name):
            raise ValueError(f"compact fragment field {name} has the wrong rank")
        if not array.flags.c_contiguous:
            raise ValueError(f"compact fragment field {name} is not C-contiguous")
        expected = expected_wire_dtype(name)
        if expected is not None:
            if array.dtype != expected:
                raise ValueError(
                    f"compact fragment field {name} has dtype "
                    f"{array.dtype.str}, expected {expected.str}"
                )
            continue
        if array.dtype.kind != "U":
            raise ValueError(f"compact fragment field {name} must be Unicode")
        width = max((len(str(value)) for value in array.flat), default=1)
        canonical = np.dtype(f"<U{max(width, 1)}")
        if array.dtype != canonical:
            raise ValueError(
                f"compact fragment field {name} has non-canonical Unicode dtype"
            )
    _validate_fragment_arrays(arrays)


def _validate_exact_schema(
    arrays: Mapping[str, Array],
    *,
    fields: tuple[str, ...],
) -> None:
    actual = set(arrays)
    expected = set(fields)
    if actual != expected:
        raise ValueError(
            "compact fragment wire schema mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
