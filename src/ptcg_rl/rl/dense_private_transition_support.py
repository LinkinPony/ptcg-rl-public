"""Validation and tensor algebra for the dense-private transition."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
    DeckConditioningConfig,
)
from ptcg_rl.model.deck_lora import (
    RoutedLoRAMergeSpec,
    routed_lora_merge_specs,
)

TRANSFORMER_TARGET_BY_SUFFIX = {
    "self_attn.in_proj_weight": "attention_qkv",
    "self_attn.out_proj.weight": "attention_output",
    "linear1.weight": "ffn_input",
    "linear2.weight": "ffn_output",
}
POLICY_TARGETS = (
    "scalar_input",
    "scalar_output",
    "dynamic_input",
    "dynamic_output",
    "option_projection",
    "selected_projection",
    "ordered_history_projection",
    "decoder_cardinality_projection",
    "query_input",
    "query_output",
)
TARGET_PRIVATE_PREFIXES = (
    "state_encoder.private_strategy_stacks.",
    "policy_head.private_strategies.",
    "dense_private_root_value_heads.",
    "dense_private_prefix_value_heads.",
)
FORBIDDEN_TARGET_PREFIXES = (
    "state_encoder.private_adapters.",
    "state_encoder.private_lora.",
    "policy_head.private_lora.",
    "private_policy_adapters.",
    "private_root_value_heads.",
    "private_prefix_value_heads.",
)
_SOURCE_LORA_PREFIXES = (
    "state_encoder.private_lora.",
    "policy_head.private_lora.",
)


@dataclass(frozen=True)
class TransitionContext:
    """Validated source and target architecture details."""

    source_conditioning: DeckConditioningConfig
    target_conditioning: DeckConditioningConfig
    target_to_source: Mapping[str, str]
    shared_layer_count: int
    transformer_layer_count: int
    rank: int
    alpha: float
    scaling: float
    specs: tuple[RoutedLoRAMergeSpec, ...]
    specs_by_target: Mapping[str, RoutedLoRAMergeSpec]


def build_transition_context(
    *,
    target_model: AgentPolicyValueNet,
    source_config: AgentNetworkConfig,
    target_to_source_experts: Mapping[str, str],
) -> TransitionContext:
    """Validate architecture declarations and normalize the expert map."""
    target_config = target_model.config
    source_conditioning = source_config.deck_conditioning
    target_conditioning = target_config.deck_conditioning
    if (
        source_conditioning is None
        or source_conditioning.architecture_version
        != DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        or source_conditioning.lora is None
        or source_conditioning.lora.export_mode != "routed"
    ):
        raise ValueError("dense-private migration requires a routed v2 source")
    if (
        target_conditioning is None
        or target_conditioning.architecture_version
        != DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        or target_conditioning.dense_private is None
        or target_conditioning.lora is not None
    ):
        raise ValueError("dense-private migration requires a LoRA-free v3 target")
    source_layers = source_config.state_encoder.num_layers
    target_layers = target_config.state_encoder.num_layers
    if source_layers != target_layers:
        raise ValueError("source and target Transformer depths must match")
    shared_layers = target_conditioning.dense_private.shared_transformer_layers
    if any(index < shared_layers for index in source_conditioning.adapter_layer_indices):
        raise ValueError(
            "legacy private residual below the dense-private split cannot be omitted"
        )
    lora = source_conditioning.lora
    if set(lora.transformer_targets) != set(TRANSFORMER_TARGET_BY_SUFFIX.values()):
        raise ValueError("source LoRA must cover all Transformer linear targets")
    resolved_layers = set(lora.resolved_transformer_layers(num_layers=source_layers))
    if not set(range(shared_layers, source_layers)).issubset(resolved_layers):
        raise ValueError("source LoRA does not cover every private upper layer")
    if set(lora.policy_targets) != set(POLICY_TARGETS):
        raise ValueError("source LoRA must cover all ten private policy targets")
    scaling = float(lora.alpha) / int(lora.rank)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("source LoRA scaling must be finite and positive")

    normalized_map = _normalize_expert_map(
        target_to_source_experts,
        target_routes=target_conditioning.active_routes,
        source_routes=source_conditioning.active_routes,
    )
    target_by_module = {
        str(route.module_key): route for route in target_conditioning.active_routes
    }
    source_by_module = {
        str(route.module_key): route for route in source_conditioning.active_routes
    }
    for target_module, source_module in normalized_map.items():
        target_route = target_by_module[target_module]
        source_route = source_by_module[source_module]
        if (
            target_route.deck_digest != source_route.deck_digest
            or target_route.signature != source_route.signature
            or target_route.canonical_card_ids != source_route.canonical_card_ids
        ):
            raise ValueError(
                "dense-private donor must represent the exact same immutable deck: "
                f"{target_module} -> {source_module}"
            )
    specs = routed_lora_merge_specs(
        transformer_layer_indices=tuple(sorted(resolved_layers)),
        transformer_targets=lora.transformer_targets,
        policy_targets=lora.policy_targets,
    )
    specs_by_target = {spec.target_id: spec for spec in specs}
    if len(specs_by_target) != len(specs):
        raise ValueError("source LoRA merge targets are not unique")
    return TransitionContext(
        source_conditioning=source_conditioning,
        target_conditioning=target_conditioning,
        target_to_source=normalized_map,
        shared_layer_count=shared_layers,
        transformer_layer_count=source_layers,
        rank=int(lora.rank),
        alpha=float(lora.alpha),
        scaling=scaling,
        specs=specs,
        specs_by_target=specs_by_target,
    )


def reject_invalid_target_state(target_state: Mapping[str, Tensor]) -> None:
    """Ensure v3 has neither LoRA tensors nor retired private banks."""
    invalid = sorted(
        key
        for key in target_state
        if key.startswith(FORBIDDEN_TARGET_PREFIXES)
        or ".lora_a." in key
        or ".lora_b." in key
        or any("lora" in segment.lower() for segment in key.split("."))
    )
    if invalid:
        raise ValueError(f"v3 target contains retired private state: {invalid}")


def validate_source_lora_inventory(
    source_state: Mapping[str, Any],
    *,
    context: TransitionContext,
) -> None:
    """Require the checkpoint factors to match the declared v2 schema."""
    source_modules = {
        str(route.module_key) for route in context.source_conditioning.active_routes
    }
    expected = {
        spec.factor_key(module_key, factor)
        for spec in context.specs
        for module_key in source_modules
        for factor in ("lora_a", "lora_b")
    }
    actual = {
        str(key)
        for key in source_state
        if str(key).startswith(_SOURCE_LORA_PREFIXES)
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"source LoRA inventory mismatch: missing={missing}, extra={extra}"
        )
    if any(not isinstance(source_state[key], Tensor) for key in actual):
        raise ValueError("source LoRA inventory contains a non-tensor")


def merged_lora_weight(
    source_state: Mapping[str, Any],
    *,
    spec: RoutedLoRAMergeSpec,
    module_key: str,
    rank: int,
    scaling: float,
    target: Tensor,
) -> Tensor:
    """Validate and evaluate ``W + scaling * B @ A`` in FP32."""
    base, lora_a, lora_b = _lora_tensors(
        source_state,
        spec=spec,
        module_key=module_key,
        rank=rank,
    )
    delta = scaling * torch.matmul(
        lora_b.detach().cpu().float(),
        lora_a.detach().cpu().float(),
    )
    merged = base.detach().cpu().float() + delta
    if not torch.isfinite(delta).all() or not torch.isfinite(merged).all():
        raise ValueError(f"LoRA merge overflowed FP32: {spec.target_id}")
    if tuple(merged.shape) != tuple(target.shape) or not target.is_floating_point():
        raise ValueError(f"dense LoRA target is incompatible: {spec.target_id}")
    return merged.to(device=target.device, dtype=target.dtype)


def lower_lora_inventory(
    *,
    source_state: Mapping[str, Any],
    context: TransitionContext,
) -> list[dict[str, Any]]:
    """Fingerprint unmergeable lower deltas without materializing full deltas."""
    lower_specs = tuple(
        spec
        for spec in context.specs
        if spec.target_id.startswith("transformer.")
        and int(spec.target_id.split(".", maxsplit=2)[1])
        < context.shared_layer_count
    )
    target_by_source: dict[str, list[str]] = defaultdict(list)
    for target_key, source_key in context.target_to_source.items():
        target_by_source[source_key].append(target_key)
    source_modules = sorted(
        str(route.module_key) for route in context.source_conditioning.active_routes
    )
    inventory: list[dict[str, Any]] = []
    for source_module in source_modules:
        for spec in lower_specs:
            base, lora_a, lora_b = _lora_tensors(
                source_state,
                spec=spec,
                module_key=source_module,
                rank=context.rank,
            )
            a_fp32 = lora_a.detach().cpu().float()
            b_fp32 = lora_b.detach().cpu().float()
            gram_a = torch.matmul(a_fp32, a_fp32.transpose(0, 1))
            gram_b = torch.matmul(b_fp32.transpose(0, 1), b_fp32)
            squared_norm = float((gram_a * gram_b).sum().item())
            delta_norm = context.scaling * math.sqrt(max(0.0, squared_norm))
            base_norm = float(
                torch.linalg.vector_norm(base.detach().cpu().float()).item()
            )
            a_key = spec.factor_key(source_module, "lora_a")
            b_key = spec.factor_key(source_module, "lora_b")
            identity = {
                "target_id": spec.target_id,
                "base_weight_key": spec.base_weight_key,
                "base_weight_sha256": _tensor_sha256(base),
                "lora_a_key": a_key,
                "lora_a_sha256": _tensor_sha256(lora_a),
                "lora_a_frobenius_norm": float(
                    torch.linalg.vector_norm(a_fp32).item()
                ),
                "lora_b_key": b_key,
                "lora_b_sha256": _tensor_sha256(lora_b),
                "lora_b_frobenius_norm": float(
                    torch.linalg.vector_norm(b_fp32).item()
                ),
                "scaling": context.scaling,
            }
            inventory.append(
                {
                    "source_module_key": source_module,
                    "mapped_target_module_keys": sorted(
                        target_by_source.get(source_module, [])
                    ),
                    **identity,
                    "delta_identity_sha256": fingerprint_payload(identity),
                    "delta_frobenius_norm": delta_norm,
                    "base_frobenius_norm": base_norm,
                    "delta_to_base_ratio": (
                        delta_norm / base_norm if base_norm > 0.0 else None
                    ),
                    "factor_numel": int(lora_a.numel() + lora_b.numel()),
                    "delta_numel": int(base.numel()),
                    "handling": "retired_without_merge",
                }
            )
    return inventory


def _normalize_expert_map(
    requested: Mapping[str, str],
    *,
    target_routes: Sequence[Any],
    source_routes: Sequence[Any],
) -> dict[str, str]:
    target_aliases = _route_aliases(target_routes, label="target")
    source_aliases = _route_aliases(source_routes, label="source")
    normalized: dict[str, str] = {}
    for raw_target, raw_source in requested.items():
        target_key = target_aliases.get(str(raw_target))
        source_key = source_aliases.get(str(raw_source))
        if target_key is None or source_key is None:
            raise ValueError(f"unknown expert map entry: {raw_target} -> {raw_source}")
        if target_key in normalized:
            raise ValueError(f"target expert is mapped twice: {target_key}")
        normalized[target_key] = source_key
    expected_targets = {str(route.module_key) for route in target_routes}
    if set(normalized) != expected_targets:
        missing = sorted(expected_targets - set(normalized))
        extra = sorted(set(normalized) - expected_targets)
        raise ValueError(
            f"expert map must exactly cover target routes: missing={missing}, extra={extra}"
        )
    return dict(sorted(normalized.items()))


def _route_aliases(routes: Sequence[Any], *, label: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    module_keys: set[str] = set()
    for route in routes:
        module_key = str(route.module_key)
        if not module_key or "." in module_key or module_key in module_keys:
            raise ValueError(f"{label} route module keys must be unique state segments")
        module_keys.add(module_key)
        for alias in (module_key, str(route.expert_id)):
            previous = aliases.get(alias)
            if previous is not None and previous != module_key:
                raise ValueError(f"ambiguous {label} expert alias: {alias}")
            aliases[alias] = module_key
    if not module_keys:
        raise ValueError(f"{label} registry must contain at least one expert")
    return aliases


def _lora_tensors(
    source_state: Mapping[str, Any],
    *,
    spec: RoutedLoRAMergeSpec,
    module_key: str,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    base = _source_tensor(source_state, spec.base_weight_key)
    lora_a = _source_tensor(source_state, spec.factor_key(module_key, "lora_a"))
    lora_b = _source_tensor(source_state, spec.factor_key(module_key, "lora_b"))
    if not all(value.is_floating_point() for value in (base, lora_a, lora_b)):
        raise ValueError(f"LoRA tensors must be floating point: {spec.target_id}")
    if not all(torch.isfinite(value).all() for value in (base, lora_a, lora_b)):
        raise ValueError(f"LoRA tensors must be finite: {spec.target_id}")
    if base.ndim != 2:
        raise ValueError(f"LoRA base weight must be rank two: {spec.target_id}")
    if tuple(lora_a.shape) != (rank, int(base.shape[1])):
        raise ValueError(f"LoRA A shape mismatch: {spec.target_id}")
    if tuple(lora_b.shape) != (int(base.shape[0]), rank):
        raise ValueError(f"LoRA B shape mismatch: {spec.target_id}")
    return base, lora_a, lora_b


def _source_tensor(source_state: Mapping[str, Any], key: str) -> Tensor:
    value = source_state.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"transition source tensor is missing: {key}")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(cast(Any, tensor.view(torch.uint8).numpy()).tobytes())
    return digest.hexdigest()


__all__ = [
    "FORBIDDEN_TARGET_PREFIXES",
    "POLICY_TARGETS",
    "TARGET_PRIVATE_PREFIXES",
    "TRANSFORMER_TARGET_BY_SUFFIX",
    "TransitionContext",
    "build_transition_context",
    "lower_lora_inventory",
    "merged_lora_weight",
    "reject_invalid_target_state",
    "validate_source_lora_inventory",
]
