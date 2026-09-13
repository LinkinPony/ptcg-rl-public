"""Canonical CPU identity for one planner root model input."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.encoding import EncodedOptionArrayFeatures
from ptcg_rl.model.state_encoder import StateTokenArrayFeatures


def canonical_planner_root_input_fingerprint(
    state: StateTokenArrayFeatures,
    options: EncodedOptionArrayFeatures,
    *,
    min_count: int,
    max_count: int,
) -> str:
    """Hash one canonical CPU collation row before accelerator transfer."""
    digest = hashlib.sha256(b"ptcg-rl/planner-context-root/v3\x00")
    for values in (
        state.card_ids,
        state.areas,
        state.owner_roles,
        state.token_kinds,
        state.scalars,
        state.last_attack_ids,
        state.attachment_card_ids,
        state.attachment_parent_indices,
        state.attachment_kinds,
        state.entity_slots,
        options.option_types,
        options.contexts,
        options.entity_slots,
        options.entity_slot_mask,
        options.attack_ids,
        options.card_ids,
        options.scalars,
        options.dynamic_effect_features,
        options.dynamic_effect_masks,
    ):
        _update_numpy_digest(digest, values)
    digest.update(int(min_count).to_bytes(8, "big", signed=True))
    digest.update(int(max_count).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _update_numpy_digest(digest: Any, values: npt.NDArray[Any]) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


__all__ = ["canonical_planner_root_input_fingerprint"]
