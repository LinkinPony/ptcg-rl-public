"""Stable identity for public policy inputs shared by train and serve paths."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from ptcg_rl.actions.encoding import LEGACY_SCALAR_FEATURE_SIZE
from ptcg_rl.model.state_encoder import LEGACY_TOKEN_SCALAR_SIZE

POLICY_INPUT_SCHEMA_VERSION = 2

_POLICY_INPUT_SCHEMA_DESCRIPTOR = {
    "name": "dccr_public_policy_input",
    "version": POLICY_INPUT_SCHEMA_VERSION,
    "state": {
        "legacy_token_scalars": {
            "contract": "dccr_state_token_scalars_v1",
            "width": LEGACY_TOKEN_SCALAR_SIZE,
        },
        "public_state_scalars": (
            "own_bench_capacity_div_8",
            "own_bench_capacity_present",
            "opponent_bench_capacity_div_8",
            "opponent_bench_capacity_present",
        ),
        "attachments": ("card_id", "parent_token", "attachment_kind"),
    },
    "option": {
        "legacy_scalars": {
            "contract": "complete_action_option_scalars_v1",
            "width": LEGACY_SCALAR_FEATURE_SIZE,
        },
        "attachment_identity_scalars": (
            "energy_index_present",
            "tool_index_present",
            "attachment_serial_div_128",
            "attachment_serial_present",
        ),
        "attachment_card_id_in_card_id": True,
        "attachment_parent_in_entity_slot": True,
        "attachment_kind_in_option_type": True,
    },
}


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"ptcg-rl/policy-input-schema/v1\x00" + encoded).hexdigest()


POLICY_INPUT_SCHEMA_FINGERPRINT = _fingerprint(_POLICY_INPUT_SCHEMA_DESCRIPTOR)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def policy_input_schema_metadata() -> dict[str, int | str]:
    """Return portable metadata for the current train/serve input contract."""
    return {
        "version": POLICY_INPUT_SCHEMA_VERSION,
        "fingerprint": POLICY_INPUT_SCHEMA_FINGERPRINT,
    }


def validate_policy_input_schema_metadata(value: Any) -> None:
    """Reject missing or incompatible current policy-input metadata."""
    if not isinstance(value, Mapping):
        raise ValueError("policy input schema metadata is missing")
    version = value.get("version")
    fingerprint = value.get("fingerprint")
    if version != POLICY_INPUT_SCHEMA_VERSION:
        raise ValueError("policy input schema version mismatch")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        raise ValueError("policy input schema fingerprint is invalid")
    if fingerprint != POLICY_INPUT_SCHEMA_FINGERPRINT:
        raise ValueError("policy input schema fingerprint mismatch")


__all__ = [
    "POLICY_INPUT_SCHEMA_FINGERPRINT",
    "POLICY_INPUT_SCHEMA_VERSION",
    "policy_input_schema_metadata",
    "validate_policy_input_schema_metadata",
]
