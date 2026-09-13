"""Compact raw EVENT/ACTION columns for native sequence trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.context.public_event_arrays import build_public_event_array_block
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS
from ptcg_rl.model.sequence.action import AcceptedActionRecord
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow

Array = npt.NDArray[np.generic]


def build_native_sequence_decision_arrays(
    rows: Sequence[SimpleStatelessActorRow],
    actions: Sequence[AcceptedActionRecord],
) -> dict[str, Array]:
    """Flatten sequence truth aligned with one native decision chunk."""
    actor_rows = tuple(rows)
    accepted = tuple(actions)
    if not actor_rows or len(actor_rows) != len(accepted):
        raise ValueError("native sequence trajectory rows are misaligned")
    if any(row.sequence_identity is None for row in actor_rows):
        raise ValueError("native sequence trajectory row has no clock identity")
    events = build_public_event_array_block(
        tuple(row.public_event_delta for row in actor_rows)
    )
    accepted_count = sum(len(action.option_types) for action in accepted)

    def vector(field: str, *, dtype: npt.DTypeLike) -> Array:
        return np.asarray(
            [value for action in accepted for value in getattr(action, field)],
            dtype=dtype,
        )

    def matrix(
        field: str,
        *,
        width: int,
        dtype: npt.DTypeLike,
    ) -> Array:
        values = [row for action in accepted for row in getattr(action, field)]
        return np.asarray(values, dtype=dtype).reshape(accepted_count, width)

    return {
        "engine_fact_producer_fingerprints": _strings(
            tuple(row.engine_fact_producer_fingerprint or "" for row in actor_rows)
        ),
        "sequence_request_ids": _strings(
            tuple(
                str(row.sequence_identity.request_id)
                for row in actor_rows
                if row.sequence_identity is not None
            )
        ),
        "event_offsets": events.event_offsets,
        "event_types": events.event_types,
        "event_actor_roles": events.actor_roles,
        "event_from_areas": events.from_areas,
        "event_to_areas": events.to_areas,
        "event_card_ids": events.card_ids,
        "event_serials": events.serials,
        "event_entity_mask": events.entity_mask,
        "event_attack_ids": events.attack_ids,
        "event_attack_id_mask": events.attack_id_mask,
        "event_values": events.values,
        "event_value_mask": events.value_mask,
        "event_categorical_values": events.categorical_values,
        "event_overflow_offsets": events.overflow_offsets,
        "event_overflow_types": events.overflow_event_types,
        "event_overflow_actor_roles": events.overflow_actor_roles,
        "event_overflow_counts": events.overflow_counts,
        "accepted_action_stable_ids": _strings(
            tuple(action.stable_identity for action in accepted)
        ),
        "accepted_action_prompt_contexts": np.asarray(
            [action.prompt_context for action in accepted],
            dtype=np.int16,
        ),
        "accepted_action_ordered": np.asarray(
            [action.ordered for action in accepted],
            dtype=np.bool_,
        ),
        "accepted_action_fallback": np.asarray(
            [action.fallback for action in accepted],
            dtype=np.bool_,
        ),
        "accepted_action_option_types": vector(
            "option_types",
            dtype=np.int16,
        ),
        "accepted_action_option_contexts": vector(
            "option_contexts",
            dtype=np.int16,
        ),
        "accepted_action_card_ids": vector("card_ids", dtype=np.int32),
        "accepted_action_attack_ids": vector("attack_ids", dtype=np.int32),
        "accepted_action_option_scalars": matrix(
            "option_scalars",
            width=SCALAR_FEATURE_SIZE,
            dtype=np.float32,
        ),
        "accepted_action_entity_card_ids": matrix(
            "entity_card_ids",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int32,
        ),
        "accepted_action_entity_areas": matrix(
            "entity_areas",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int16,
        ),
        "accepted_action_entity_owner_roles": matrix(
            "entity_owner_roles",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int8,
        ),
        "accepted_action_entity_token_kinds": matrix(
            "entity_token_kinds",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int8,
        ),
        "accepted_action_entity_scalars": np.asarray(
            [
                row
                for action in accepted
                for row in action.entity_scalars
            ],
            dtype=np.float32,
        ).reshape(
            accepted_count,
            MAX_ENTITY_SLOTS,
            TOKEN_SCALAR_SIZE,
        ),
    }


def _strings(values: Sequence[str]) -> npt.NDArray[np.str_]:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


__all__ = ["build_native_sequence_decision_arrays"]
