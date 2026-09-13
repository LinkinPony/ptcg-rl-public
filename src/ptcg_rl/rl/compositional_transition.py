"""Source-bound dense-v3 to compositional-v4 model-state transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model.compositional_projection import SharedCompositionalLinear
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
)
from ptcg_rl.model.network import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.rl import compositional_transition_ops as transition_ops
from ptcg_rl.rl.compositional_factorization import TruncatedSvdConfig

_TRANSFORMER_SUFFIXES = {
    "attention_qkv": "self_attn.in_proj_weight",
    "attention_output": "self_attn.out_proj.weight",
    "ffn_input": "linear1.weight",
    "ffn_output": "linear2.weight",
}
_POLICY_BASE_KEYS = {
    "scalar_input": "policy_head.scalar_projection.0",
    "scalar_output": "policy_head.scalar_projection.3",
    "dynamic_input": "policy_head.dynamic_effect_projection.0",
    "dynamic_output": "policy_head.dynamic_effect_projection.3",
    "option_projection": "policy_head.option_projection",
    "selected_projection": "policy_head.selected_projection",
    "ordered_history_projection": "policy_head.ordered_history_projection",
    "decoder_cardinality_projection": ("policy_head.decoder_cardinality_projection"),
    "query_input": "policy_head.query_projection.0",
    "query_output": "policy_head.query_projection.3",
    "count_state_projection": "policy_head.count_state_projection",
    "count_feature_projection": "policy_head.count_feature_projection",
    "count_output_projection": "policy_head.count_output_projection",
}


@dataclass(frozen=True)
class CompositionalTransitionResult:
    """Converted target model state and its complete migration manifest."""

    state_dict: dict[str, Tensor]
    manifest: dict[str, Any]


def migrate_dense_v3_to_compositional_v4(
    target_model: AgentPolicyValueNet,
    source_state: Mapping[str, Tensor],
    *,
    source_config: AgentNetworkConfig,
    sampling_weights: Mapping[str, float] | None = None,
    factorization_device: torch.device | str = "cpu",
    factorization_config: TruncatedSvdConfig | None = None,
    source_policy_sha256: str | None = None,
) -> CompositionalTransitionResult:
    """Convert one immutable routed dense-v3 state into a routed v4 state."""
    source_conditioning, target_conditioning = (
        transition_ops._validate_transition_configs(
            source_config,
            target_model.config,
        )
    )
    source_by_digest = {
        route.deck_digest: route for route in source_conditioning.expert_routes
    }
    target_routes = target_conditioning.expert_routes
    if set(source_by_digest) != {route.deck_digest for route in target_routes}:
        raise ValueError("v4 transition requires identical exact source/target decks")
    source_routes = tuple(
        source_by_digest[route.deck_digest] for route in target_routes
    )
    probabilities = transition_ops._sampling_weight_tensor(
        target_routes,
        sampling_weights,
    )
    factor_device = torch.device(factorization_device)
    fit_config = factorization_config or TruncatedSvdConfig()
    target_state = dict(target_model.state_dict())
    source_key_actions: dict[str, dict[str, Any]] = {}
    transition_ops._copy_unchanged_shared_state(
        target_state,
        source_state,
        source_key_actions=source_key_actions,
    )
    target_model.load_state_dict(target_state, strict=True)
    deck_embeddings = transition_ops._target_deck_embeddings(
        target_model,
        target_conditioning,
    )
    compositional = cast(Any, target_conditioning.compositional)
    dense = cast(Any, source_conditioning.dense_private)
    projection_records: list[dict[str, Any]] = []

    for layer_index in compositional.transformer_layer_indices:
        relative_index = layer_index - dense.shared_transformer_layers
        if relative_index < 0:
            raise ValueError("v4 target layers must lie in the dense-v3 private stack")
        layer_key = f"layer_{layer_index:02d}"
        source_prefixes = tuple(
            "state_encoder.private_strategy_stacks."
            f"{route.module_key}.layers.{relative_index}.layer."
            for route in source_routes
        )
        for target, suffix in _TRANSFORMER_SUFFIXES.items():
            source_keys = tuple(f"{prefix}{suffix}" for prefix in source_prefixes)
            base_key = f"state_encoder.transformer.layers.{layer_index}.{suffix}"
            projection_prefix = (
                f"state_encoder.compositional_projections.{layer_key}.{target}"
            )
            projection_bank = cast(
                nn.ModuleDict,
                target_model.state_encoder.compositional_projections[layer_key],
            )
            projection = cast(
                SharedCompositionalLinear,
                projection_bank[target],
            )
            record = transition_ops._factorize_projection(
                target_state,
                source_state,
                source_keys=source_keys,
                base_weight_key=base_key,
                projection_prefix=projection_prefix,
                projection=projection,
                target_routes=target_routes,
                target_domain="transformer",
                target_name=f"{layer_key}_{target}",
                capsules=target_model.exact_capsules,
                deck_embeddings=deck_embeddings,
                sampling_weights=probabilities,
                shared_rank=compositional.transformer_shared_rank,
                exact_rank=compositional.transformer_exact_rank,
                factorization_device=factor_device,
                factorization_config=fit_config,
                source_key_actions=source_key_actions,
            )
            projection_records.append(record)
            bias_suffix = suffix.removesuffix("weight") + "bias"
            bias_keys = tuple(f"{prefix}{bias_suffix}" for prefix in source_prefixes)
            if all(key in source_state for key in bias_keys):
                transition_ops._migrate_vector_mean_and_offsets(
                    target_state,
                    source_state,
                    source_keys=bias_keys,
                    base_key=base_key.removesuffix("weight") + "bias",
                    target_routes=target_routes,
                    target_offset_keys=tuple(
                        "exact_capsules."
                        f"{route.module_key}.transformer_bias_offsets."
                        f"{layer_key}_{target}"
                        for route in target_routes
                    ),
                    sampling_weights=probabilities,
                    source_key_actions=source_key_actions,
                )
        for norm_name in ("norm1", "norm2"):
            for field in ("weight", "bias"):
                source_keys = tuple(
                    f"{prefix}{norm_name}.{field}" for prefix in source_prefixes
                )
                transition_ops._migrate_vector_mean_and_offsets(
                    target_state,
                    source_state,
                    source_keys=source_keys,
                    base_key=(
                        f"state_encoder.transformer.layers.{layer_index}."
                        f"{norm_name}.{field}"
                    ),
                    target_routes=target_routes,
                    target_offset_keys=tuple(
                        "exact_capsules."
                        f"{route.module_key}.transformer_norm_offsets."
                        f"{layer_key}_{norm_name}_{field}"
                        for route in target_routes
                    ),
                    sampling_weights=probabilities,
                    source_key_actions=source_key_actions,
                )

    transition_ops._migrate_output_norm(
        target_state,
        source_state,
        source_routes=source_routes,
        target_routes=target_routes,
        sampling_weights=probabilities,
        source_key_actions=source_key_actions,
    )
    transition_ops._migrate_shared_state_adapters(
        target_state,
        source_state,
        source_routes=source_routes,
        target_conditioning=target_conditioning,
        source_shared_layers=dense.shared_transformer_layers,
        sampling_weights=probabilities,
        source_key_actions=source_key_actions,
    )

    for target, base_prefix in _POLICY_BASE_KEYS.items():
        source_keys = tuple(
            f"policy_head.private_strategies.{route.module_key}.linears.{target}.weight"
            for route in source_routes
        )
        projection_prefix = f"policy_head.compositional_linears.{target}"
        projection = cast(
            SharedCompositionalLinear,
            target_model.policy_head.compositional_linears[target],
        )
        projection_records.append(
            transition_ops._factorize_projection(
                target_state,
                source_state,
                source_keys=source_keys,
                base_weight_key=f"{base_prefix}.weight",
                projection_prefix=projection_prefix,
                projection=projection,
                target_routes=target_routes,
                target_domain="policy",
                target_name=target,
                capsules=target_model.exact_capsules,
                deck_embeddings=deck_embeddings,
                sampling_weights=probabilities,
                shared_rank=compositional.policy_shared_rank,
                exact_rank=compositional.policy_exact_rank,
                factorization_device=factor_device,
                factorization_config=fit_config,
                source_key_actions=source_key_actions,
            )
        )
        bias_keys = tuple(key.removesuffix("weight") + "bias" for key in source_keys)
        if all(key in source_state for key in bias_keys):
            transition_ops._migrate_vector_mean_and_offsets(
                target_state,
                source_state,
                source_keys=bias_keys,
                base_key=f"{base_prefix}.bias",
                target_routes=target_routes,
                target_offset_keys=tuple(
                    f"exact_capsules.{route.module_key}.policy_bias_offsets.{target}"
                    for route in target_routes
                ),
                sampling_weights=probabilities,
                source_key_actions=source_key_actions,
            )

    transition_ops._migrate_policy_structures(
        target_state,
        source_state,
        source_routes=source_routes,
        target_routes=target_routes,
        sampling_weights=probabilities,
        source_key_actions=source_key_actions,
    )
    transition_ops._migrate_value_bases(
        target_state,
        source_state,
        source_routes=source_routes,
        sampling_weights=probabilities,
        source_key_actions=source_key_actions,
    )
    for key in source_state:
        if (
            key.startswith(transition_ops.SOURCE_PRIVATE_PREFIXES)
            and key not in source_key_actions
        ):
            source_key_actions[key] = {"action": "retired_into_distillation"}

    target_model.load_state_dict(target_state, strict=True)
    converted = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in target_model.state_dict().items()
    }
    transition_ops._validate_finite_state(converted)
    manifest = {
        "schema": "dense-v3-to-dccr-v4-transition-v1",
        "source_architecture_version": (
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ),
        "target_architecture_version": (
            DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        ),
        "source_policy_sha256": source_policy_sha256,
        "source_model_config_sha256": fingerprint_payload(
            source_config.model_dump(mode="json")
        ),
        "target_model_config_sha256": fingerprint_payload(
            target_model.config.model_dump(mode="json")
        ),
        "source_registry_sha256": source_conditioning.resolved_registry_sha256,
        "target_registry_sha256": target_conditioning.resolved_registry_sha256,
        "factorization": {
            "fitting_version": "dccr-v4-weight-factorization-v1",
            **fit_config.manifest(),
        },
        "sampling_weights": {
            route.deck_digest: float(weight)
            for route, weight in zip(target_routes, probabilities.tolist(), strict=True)
        },
        "lineages": [
            {
                "deck_digest": target.deck_digest,
                "source_strategy_id": source.expert_id,
                "source_strategy_retired": True,
                "target_strategy_id": target.expert_id,
            }
            for source, target in zip(source_routes, target_routes, strict=True)
        ],
        "projections": projection_records,
        "source_key_actions": source_key_actions,
        "source_private_key_count": sum(
            key.startswith(transition_ops.SOURCE_PRIVATE_PREFIXES)
            for key in source_state
        ),
        "target_parameter_count": sum(tensor.numel() for tensor in converted.values()),
    }
    if manifest["source_private_key_count"] != sum(
        key.startswith(transition_ops.SOURCE_PRIVATE_PREFIXES)
        for key in source_key_actions
    ):
        raise RuntimeError("transition manifest did not classify every private key")
    return CompositionalTransitionResult(state_dict=converted, manifest=manifest)


__all__ = [
    "CompositionalTransitionResult",
    "migrate_dense_v3_to_compositional_v4",
]
