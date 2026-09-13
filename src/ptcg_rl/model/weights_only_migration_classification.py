"""Raw tensor classification for weights-only checkpoint migrations."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from ptcg_rl.model.network import AgentNetworkConfig

HASH_CHUNK_BYTES = 8 * 1024 * 1024
ANCHOR_EXCLUDED_PREFIX = "search_reranker."

REINITIALIZED_ZERO_PARAMETERS = frozenset(
    {
        "policy_head.proposal_query_projection.weight",
        "planner_reranker.residual_head.2.weight",
        "planner_reranker.residual_head.2.bias",
        "root_perspective_value_adapter.residual.2.weight",
        "root_perspective_value_adapter.residual.2.bias",
    }
)
REINITIALIZED_ZERO_OUTPUT_INPUTS = frozenset(
    {
        "planner_reranker.residual_head.0.weight",
        "planner_reranker.residual_head.0.bias",
        "root_perspective_value_adapter.residual.0.weight",
        "root_perspective_value_adapter.residual.0.bias",
    }
)
EXPECTED_REINITIALIZED = (
    REINITIALIZED_ZERO_PARAMETERS | REINITIALIZED_ZERO_OUTPUT_INPUTS
)

TensorClassification = Literal[
    "loaded",
    "reinitialized",
    "loaded_but_anchor_excluded",
    "unexpected",
    "shape_mismatch",
]


def stream_file_fingerprint(path: Path) -> tuple[int, str]:
    """Return size and SHA256 while reading a checkpoint in bounded chunks."""
    hasher = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(HASH_CHUNK_BYTES):
            size_bytes += len(chunk)
            hasher.update(chunk)
    return size_bytes, hasher.hexdigest()


def classify_tensors(
    *,
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[TensorClassification, list[dict[str, Any]]]:
    """Compare raw source and pre-hook target tensors by name, shape, and dtype."""
    categories: dict[TensorClassification, list[dict[str, Any]]] = {
        "loaded": [],
        "reinitialized": [],
        "loaded_but_anchor_excluded": [],
        "unexpected": [],
        "shape_mismatch": [],
    }
    for name in sorted(set(source_state) | set(target_state)):
        source = source_state.get(name)
        target = target_state.get(name)
        if source is None:
            if target is None:
                raise RuntimeError("tensor union contained no source or target")
            descriptor = tensor_descriptor(
                name,
                target,
                action="reinitialized",
            )
            descriptor.update(_reinitialization_descriptor(name, target))
            categories["reinitialized"].append(descriptor)
            continue
        if target is None:
            categories["unexpected"].append(
                tensor_descriptor(name, source, action="unexpected")
            )
            continue
        if source.shape != target.shape or source.dtype != target.dtype:
            categories["shape_mismatch"].append(
                {
                    "name": name,
                    "action": "shape_mismatch",
                    "source": tensor_descriptor(
                        name,
                        source,
                        action="shape_mismatch",
                        include_name=False,
                        include_action=False,
                    ),
                    "target": tensor_descriptor(
                        name,
                        target,
                        action="shape_mismatch",
                        include_name=False,
                        include_action=False,
                    ),
                }
            )
            continue
        category: TensorClassification = (
            "loaded_but_anchor_excluded"
            if name.startswith(ANCHOR_EXCLUDED_PREFIX)
            else "loaded"
        )
        categories[category].append(tensor_descriptor(name, source, action=category))
    return categories


def comparison_contract_valid(
    classifications: Mapping[TensorClassification, Sequence[Mapping[str, Any]]],
) -> bool:
    """Require exactly the reviewed schema-9 additions and no incompatibilities."""
    reinitialized = {str(record["name"]) for record in classifications["reinitialized"]}
    if reinitialized != EXPECTED_REINITIALIZED:
        return False
    if classifications["unexpected"] or classifications["shape_mismatch"]:
        return False
    for record in classifications["reinitialized"]:
        name = str(record["name"])
        initialization = record["initialization"]
        if name in REINITIALIZED_ZERO_PARAMETERS and initialization != "zero":
            return False
        if name in REINITIALIZED_ZERO_OUTPUT_INPUTS and initialization != "nonzero":
            return False
    excluded_names = {
        str(record["name"]) for record in classifications["loaded_but_anchor_excluded"]
    }
    return bool(excluded_names) and all(
        name.startswith(ANCHOR_EXCLUDED_PREFIX) for name in excluded_names
    )


def classification_summary(
    classifications: Mapping[TensorClassification, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate tensor and element counts without losing the descriptor digest."""
    summary: dict[str, Any] = {}
    for category, records in classifications.items():
        if category == "shape_mismatch":
            summary[category] = {
                "tensors": len(records),
                "source_numel": sum(
                    int(cast(Mapping[str, Any], record["source"])["numel"])
                    for record in records
                ),
                "target_numel": sum(
                    int(cast(Mapping[str, Any], record["target"])["numel"])
                    for record in records
                ),
            }
        else:
            summary[category] = {
                "tensors": len(records),
                "numel": sum(int(record["numel"]) for record in records),
            }
    summary["source"] = {
        "tensors": sum(
            int(summary[name]["tensors"])
            for name in (
                "loaded",
                "loaded_but_anchor_excluded",
                "unexpected",
                "shape_mismatch",
            )
        ),
        "numel": sum(
            int(summary[name]["numel"])
            for name in ("loaded", "loaded_but_anchor_excluded", "unexpected")
        )
        + int(summary["shape_mismatch"]["source_numel"]),
    }
    summary["target"] = {
        "tensors": sum(
            int(summary[name]["tensors"])
            for name in (
                "loaded",
                "loaded_but_anchor_excluded",
                "reinitialized",
                "shape_mismatch",
            )
        ),
        "numel": sum(
            int(summary[name]["numel"])
            for name in (
                "loaded",
                "loaded_but_anchor_excluded",
                "reinitialized",
            )
        )
        + int(summary["shape_mismatch"]["target_numel"]),
    }
    summary["descriptor_sha256"] = canonical_json_sha256(classifications)
    return summary


def checkpoint_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """Return an untouched copy of the raw checkpoint tensor mapping."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state: Mapping[str, Any] | None = None
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            state = cast(Mapping[str, Any], value)
            break
    if state is None:
        state = cast(Mapping[str, Any], checkpoint)
    if state and all(str(name).startswith("model.") for name in state):
        state = {
            str(name).removeprefix("model."): value for name, value in state.items()
        }
    tensors: dict[str, Tensor] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint state value is not a tensor: {name}")
        tensors[name] = value
    if not tensors:
        raise ValueError("checkpoint state dict must not be empty")
    return tensors


def checkpoint_model_config_payload(checkpoint: Any) -> Mapping[str, Any]:
    """Return the portable raw model-config payload before current defaults."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return cast(Mapping[str, Any], value.model_dump(mode="python"))
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        value = full_config.get("model")
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
    raise ValueError("checkpoint is missing a portable model configuration")


def checkpoint_publish_version(checkpoint: Any) -> int | None:
    """Return a checkpoint's immutable policy version when present."""
    if not isinstance(checkpoint, Mapping):
        return None
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("publish_version")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = checkpoint.get("version")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def tensor_descriptor(
    name: str,
    tensor: Tensor,
    *,
    action: TensorClassification,
    include_name: bool = True,
    include_action: bool = True,
) -> dict[str, Any]:
    """Return a content-bound tensor record without materializing all bytes."""
    descriptor: dict[str, Any] = {
        "shape": [int(dimension) for dimension in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": tensor.numel(),
        "sha256": tensor_sha256(tensor),
    }
    if include_name:
        descriptor["name"] = name
    if include_action:
        descriptor["action"] = action
    return descriptor


def tensor_sha256(tensor: Tensor) -> str:
    """Hash one tensor in bounded byte views using canonical contiguous order."""
    contiguous = tensor.detach().cpu().contiguous()
    byte_view = contiguous.view(torch.uint8).reshape(-1)
    hasher = hashlib.sha256()
    for start in range(0, byte_view.numel(), HASH_CHUNK_BYTES):
        stop = min(start + HASH_CHUNK_BYTES, byte_view.numel())
        hasher.update(byte_view[start:stop].numpy().tobytes())
    return hasher.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-compatible semantic object canonically."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically publish compact JSON only after every record is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reinitialization_descriptor(name: str, tensor: Tensor) -> dict[str, Any]:
    all_zero = bool(torch.count_nonzero(tensor).item() == 0)
    if name in REINITIALIZED_ZERO_PARAMETERS:
        role = "exact_zero"
    elif name in REINITIALIZED_ZERO_OUTPUT_INPUTS:
        role = "zero_output_module_input"
    else:
        role = "unapproved"
    return {
        "initialization": "zero" if all_zero else "nonzero",
        "role": role,
    }


__all__ = [
    "ANCHOR_EXCLUDED_PREFIX",
    "TensorClassification",
    "canonical_json_sha256",
    "checkpoint_model_config_payload",
    "checkpoint_publish_version",
    "checkpoint_state_dict",
    "classification_summary",
    "classify_tensors",
    "comparison_contract_valid",
    "stream_file_fingerprint",
    "tensor_sha256",
    "write_manifest",
]
