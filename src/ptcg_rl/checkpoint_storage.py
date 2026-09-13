"""Portable lossless storage transforms for deployment checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch

BYTE_SHUFFLE_FORMAT = "byte_shuffle_v1"
DIRECT_RECURRENT_POLICY_FORMAT = "recurrent_ppo_only_v1"

DIRECT_RECURRENT_POLICY_UNUSED_PREFIXES = (
    "search_reranker.",
    "planner_reranker.",
    "root_perspective_value_adapter.",
    "macro_outcome_heads.",
    "value_head.",
    "prefix_value_delta_head.",
    "prize_diff_head.",
    "opponent_card_head.",
    "opponent_hand_head.",
    "effect_head.",
    "policy_head.factual_presence_head.",
    "policy_head.factual_magnitude_head.",
    "policy_head.factual_actor_relation_head.",
    "policy_head.factual_next_context_head.",
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def strip_direct_recurrent_policy_state(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove heads that the planner-free recurrent Kaggle actor cannot call."""
    removed = tuple(
        sorted(
            str(key)
            for key in state_dict
            if str(key).startswith(DIRECT_RECURRENT_POLICY_UNUSED_PREFIXES)
        )
    )
    if not removed:
        raise ValueError("direct recurrent policy export removed no runtime-unused state")
    return (
        {str(key): value for key, value in state_dict.items() if str(key) not in removed},
        removed,
    )


def direct_recurrent_policy_metadata(removed_keys: tuple[str, ...]) -> dict[str, Any]:
    """Return an exact allowlist for intentionally omitted direct-policy state."""
    if not removed_keys or tuple(sorted(removed_keys)) != removed_keys:
        raise ValueError("removed direct-policy state keys must be nonempty and sorted")
    if any(
        not key.startswith(DIRECT_RECURRENT_POLICY_UNUSED_PREFIXES)
        for key in removed_keys
    ):
        raise ValueError("direct-policy export attempted to remove required model state")
    return {
        "format": DIRECT_RECURRENT_POLICY_FORMAT,
        "removed_state_keys": list(removed_keys),
    }


def direct_recurrent_policy_missing_keys(checkpoint: Any) -> frozenset[str]:
    """Read and validate the exact missing-key allowlist from an export payload."""
    export = checkpoint.get("export") if isinstance(checkpoint, Mapping) else None
    raw = export.get("direct_policy_only") if isinstance(export, Mapping) else None
    if raw is None:
        return frozenset()
    if not isinstance(raw, Mapping) or raw.get("format") != DIRECT_RECURRENT_POLICY_FORMAT:
        raise ValueError("unsupported direct recurrent policy export format")
    keys = raw.get("removed_state_keys")
    if (
        not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) for key in keys)
    ):
        raise ValueError("direct recurrent policy export has invalid removed keys")
    normalized = tuple(cast(list[str], keys))
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise ValueError("direct recurrent policy removed keys are not canonical")
    if any(
        not key.startswith(DIRECT_RECURRENT_POLICY_UNUSED_PREFIXES)
        for key in normalized
    ):
        raise ValueError("direct recurrent policy export omits required model state")
    return frozenset(normalized)


def byte_shuffle_state_dict(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Losslessly group tensor byte lanes so the outer gzip can compress them."""
    packed: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            packed[key] = value
            continue
        tensor = value.detach().cpu().contiguous()
        width = tensor.element_size()
        if width <= 1:
            packed[key] = tensor
            continue
        packed[key] = (
            tensor.view(torch.uint8)
            .reshape(tensor.numel(), width)
            .transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )
        metadata[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "element_size": width,
        }
    if not metadata:
        raise ValueError("byte-shuffle checkpoint storage packed no floating tensors")
    return packed, {"format": BYTE_SHUFFLE_FORMAT, "tensors": metadata}


def unpack_checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Return native tensors from a regular or byte-shuffled checkpoint payload."""
    state_dict = _checkpoint_state_mapping(checkpoint)
    export = checkpoint.get("export") if isinstance(checkpoint, Mapping) else None
    storage = export.get("tensor_storage") if isinstance(export, Mapping) else None
    if storage is None:
        return state_dict
    if not isinstance(storage, Mapping) or storage.get("format") != BYTE_SHUFFLE_FORMAT:
        raise ValueError("unsupported checkpoint tensor storage format")
    raw_metadata = storage.get("tensors")
    if not isinstance(raw_metadata, Mapping) or not raw_metadata:
        raise ValueError("byte-shuffle checkpoint has no tensor metadata")
    metadata = cast(Mapping[str, Any], raw_metadata)
    state_keys = {str(key) for key in state_dict}
    if not set(metadata).issubset(state_keys):
        raise ValueError("byte-shuffle metadata names missing checkpoint tensors")
    return {
        str(key): _unpack_tensor(str(key), value, metadata.get(str(key)))
        for key, value in state_dict.items()
    }


def _checkpoint_state_mapping(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return cast(Mapping[str, Any], value)
        if checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return cast(Mapping[str, Any], checkpoint)
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _unpack_tensor(key: str, value: Any, raw_metadata: Any) -> Any:
    if raw_metadata is None:
        return value
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"byte-shuffle tensor metadata is invalid: {key}")
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError(f"byte-shuffle tensor payload is invalid: {key}")
    dtype_name = raw_metadata.get("dtype")
    raw_shape = raw_metadata.get("shape")
    width = raw_metadata.get("element_size")
    if dtype_name not in _DTYPES or not isinstance(raw_shape, list):
        raise ValueError(f"byte-shuffle tensor identity is invalid: {key}")
    if not all(isinstance(size, int) and size >= 0 for size in raw_shape):
        raise ValueError(f"byte-shuffle tensor shape is invalid: {key}")
    dtype = _DTYPES[cast(str, dtype_name)]
    if (
        not isinstance(width, int)
        or width != torch.empty((), dtype=dtype).element_size()
    ):
        raise ValueError(f"byte-shuffle tensor element size is invalid: {key}")
    shape = tuple(cast(list[int], raw_shape))
    numel = math.prod(shape)
    if value.numel() != numel * width:
        raise ValueError(f"byte-shuffle tensor payload size is invalid: {key}")
    raw = value.reshape(width, numel).transpose(0, 1).contiguous()
    return raw.reshape(-1).view(dtype).reshape(shape)


__all__ = [
    "BYTE_SHUFFLE_FORMAT",
    "DIRECT_RECURRENT_POLICY_UNUSED_PREFIXES",
    "byte_shuffle_state_dict",
    "direct_recurrent_policy_metadata",
    "direct_recurrent_policy_missing_keys",
    "strip_direct_recurrent_policy_state",
    "unpack_checkpoint_state_dict",
]
