"""FP32 materialization for one fixed DCCR-v4 deployment route."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.decks.registry import DeckExpertRoute
from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model.compositional_capsule import ExactDeckCapsule, exact_capsule
from ptcg_rl.model.compositional_projection import (
    DeckConditionedFiLM,
    SharedCompositionalLinear,
)
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DeckEncoder,
    adapter_layer_key,
)
from ptcg_rl.model.network import (
    AgentNetworkConfig,
    AgentPolicyValueNet,
    build_agent_policy_value_net,
)

_TRANSFORMER_TARGETS = {
    "attention_qkv": ("self_attn.in_proj_weight", "self_attn.in_proj_bias"),
    "attention_output": ("self_attn.out_proj.weight", "self_attn.out_proj.bias"),
    "ffn_input": ("linear1.weight", "linear1.bias"),
    "ffn_output": ("linear2.weight", "linear2.bias"),
}
_POLICY_TARGETS = {
    "scalar_input": "scalar_projection.0",
    "scalar_output": "scalar_projection.3",
    "dynamic_input": "dynamic_effect_projection.0",
    "dynamic_output": "dynamic_effect_projection.3",
    "option_projection": "option_projection",
    "selected_projection": "selected_projection",
    "ordered_history_projection": "ordered_history_projection",
    "decoder_cardinality_projection": "decoder_cardinality_projection",
    "query_input": "query_projection.0",
    "query_output": "query_projection.3",
    "count_state_projection": "count_state_projection",
    "count_feature_projection": "count_feature_projection",
    "count_output_projection": "count_output_projection",
}
_FIXED_CAPSULE_PREFIXES = (
    "option_adapter.",
    "policy_global_residual.",
    "value_calibrators.",
)


def fix_selected_compositional_strategy(
    *,
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
    route: DeckExpertRoute,
) -> tuple[Mapping[str, Tensor], AgentNetworkConfig, Mapping[str, Any]]:
    """Merge one routed-v4 strategy and prune all dynamic composition state."""
    conditioning = model_config.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or conditioning.compositional is None
    ):
        raise ValueError("fixed compositional export requires architecture v4")
    compositional = conditioning.compositional
    if compositional.export_mode != "routed":
        raise ValueError("source compositional checkpoint must use routed export mode")
    if len(conditioning.expert_routes) != 1:
        raise ValueError("fixed compositional export requires one pruned exact route")
    if conditioning.expert_routes[0] != route:
        raise ValueError("selected compositional route differs from pruned registry")

    source_model = build_agent_policy_value_net(model_config).cpu().float().eval()
    source_model.load_state_dict(dict(state_dict), strict=True)
    capsule = exact_capsule(source_model.exact_capsules, route.module_key)
    if not isinstance(capsule, ExactDeckCapsule):
        raise TypeError("routed v4 checkpoint did not build a full exact capsule")
    deck_embedding = _selected_deck_embedding(source_model, route)
    source_state = source_model.state_dict()
    overrides: dict[str, Tensor] = {}

    _materialize_transformer(
        source_model,
        capsule=capsule,
        deck_embedding=deck_embedding,
        overrides=overrides,
    )
    _materialize_policy(
        source_model,
        capsule=capsule,
        deck_embedding=deck_embedding,
        overrides=overrides,
    )

    fixed_compositional = compositional.model_copy(update={"export_mode": "fixed"})
    fixed_conditioning = conditioning.model_copy(
        update={"compositional": fixed_compositional}
    )
    fixed_config = model_config.model_copy(
        update={"deck_conditioning": fixed_conditioning}
    )
    fixed_model = build_agent_policy_value_net(fixed_config).cpu().float().eval()
    fixed_state: dict[str, Tensor] = {}
    missing: list[str] = []
    for key, expected in fixed_model.state_dict().items():
        value = overrides.get(key)
        if value is None:
            raw = source_state.get(key)
            if not isinstance(raw, Tensor) or tuple(raw.shape) != tuple(expected.shape):
                missing.append(key)
                continue
            value = raw
        fixed_state[key] = value.detach().cpu().to(dtype=expected.dtype).clone()
    if missing:
        raise ValueError(f"fixed compositional state is incomplete: {sorted(missing)}")
    if set(overrides).difference(fixed_state):
        extra = sorted(set(overrides).difference(fixed_state))
        raise ValueError(f"fixed compositional overrides are unused: {extra}")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all().item())
        for value in fixed_state.values()
    ):
        raise FloatingPointError("fixed compositional export produced non-finite state")
    fixed_model.load_state_dict(fixed_state, strict=True)

    schema_sha256 = fixed_compositional_schema_sha256(
        fixed_config,
        fixed_state,
        module_key=route.module_key,
    )
    merged_projection_count, fixed_film_site_count = (
        compositional_materialization_inventory(fixed_config)
    )
    return (
        fixed_state,
        fixed_config,
        {
            "selected_strategy_id": route.expert_id,
            "fixed_strategy": True,
            "fixed_compositional": True,
            "strategy_schema_sha256": schema_sha256,
            "merged_projection_count": merged_projection_count,
            "fixed_film_site_count": fixed_film_site_count,
            "dynamic_router_pruned": True,
            "shared_basis_pruned": True,
            "non_selected_capsules_pruned": True,
            "lora_runtime_absent": True,
        },
    )


def _selected_deck_embedding(
    model: AgentPolicyValueNet,
    route: DeckExpertRoute,
) -> Tensor:
    deck_encoder = model.deck_encoder
    if not isinstance(deck_encoder, DeckEncoder):
        raise ValueError("routed v4 source has no deck encoder")
    cards = torch.tensor((route.canonical_card_ids,), dtype=torch.long)
    with torch.no_grad():
        embedding = deck_encoder(cards, card_encoder=model.card_encoder)
    if tuple(embedding.shape) != (1, model.config.state_encoder.d_model):
        raise ValueError("selected v4 deck embedding has an invalid shape")
    if not bool(torch.isfinite(embedding).all().item()):
        raise FloatingPointError("selected v4 deck embedding is non-finite")
    return cast(Tensor, embedding.squeeze(0).detach().float().clone())


def _materialize_transformer(
    model: AgentPolicyValueNet,
    *,
    capsule: ExactDeckCapsule,
    deck_embedding: Tensor,
    overrides: dict[str, Tensor],
) -> None:
    conditioning = cast(Any, model.config.deck_conditioning)
    compositional = cast(Any, conditioning.compositional)
    state = model.state_dict()
    for layer_index in compositional.transformer_layer_indices:
        layer_key = adapter_layer_key(layer_index)
        projections = cast(
            nn.ModuleDict,
            model.state_encoder.compositional_projections[layer_key],
        )
        prefix = f"state_encoder.transformer.layers.{layer_index}."
        for target, (weight_suffix, bias_suffix) in _TRANSFORMER_TARGETS.items():
            projection = cast(SharedCompositionalLinear, projections[target])
            weight_key = prefix + weight_suffix
            bias_key = prefix + bias_suffix
            base_weight = _required_tensor(state, weight_key).float()
            overrides[weight_key] = (
                base_weight
                + projection.materialized_delta_weight(
                    deck_embedding,
                    capsule,
                ).cpu()
            )
            bias_offset = capsule.bias_offset("transformer", projection.target)
            if bias_offset is None:
                raise ValueError(f"transformer target has no bias offset: {target}")
            overrides[bias_key] = _required_tensor(state, bias_key).float() + (
                bias_offset.detach().cpu().float()
            )
        for norm_name in ("norm1", "norm2"):
            for affine_name in ("weight", "bias"):
                state_key = prefix + f"{norm_name}.{affine_name}"
                offset_key = f"{layer_key}_{norm_name}_{affine_name}"
                overrides[state_key] = _required_tensor(state, state_key).float() + (
                    capsule.transformer_norm_offsets[offset_key].detach().cpu().float()
                )
        for site in ("attention", "feedforward"):
            film_bank = cast(
                nn.ModuleDict,
                model.state_encoder.compositional_film[layer_key],
            )
            film = cast(
                DeckConditionedFiLM,
                film_bank[site],
            )
            gamma, beta = film.materialized_affine(deck_embedding, capsule)
            film_prefix = f"state_encoder.compositional_film.{layer_key}.{site}."
            overrides[film_prefix + "gamma"] = gamma.cpu()
            overrides[film_prefix + "beta"] = beta.cpu()
    for affine_name in ("weight", "bias"):
        state_key = f"state_encoder.layer_norm.{affine_name}"
        offset_key = f"output_{affine_name}"
        overrides[state_key] = _required_tensor(state, state_key).float() + (
            capsule.transformer_norm_offsets[offset_key].detach().cpu().float()
        )


def _materialize_policy(
    model: AgentPolicyValueNet,
    *,
    capsule: ExactDeckCapsule,
    deck_embedding: Tensor,
    overrides: dict[str, Tensor],
) -> None:
    state = model.state_dict()
    for target, module_path in _POLICY_TARGETS.items():
        projection = cast(
            SharedCompositionalLinear,
            model.policy_head.compositional_linears[target],
        )
        weight_key = f"policy_head.{module_path}.weight"
        overrides[weight_key] = _required_tensor(state, weight_key).float() + (
            projection.materialized_delta_weight(deck_embedding, capsule).cpu()
        )
        bias_key = f"policy_head.{module_path}.bias"
        bias_offset = capsule.bias_offset("policy", target)
        if bias_offset is not None:
            overrides[bias_key] = _required_tensor(state, bias_key).float() + (
                bias_offset.detach().cpu().float()
            )
    for site in ("attention", "feedforward"):
        film = cast(
            DeckConditionedFiLM,
            model.policy_head.compositional_option_film[site],
        )
        gamma, beta = film.materialized_affine(deck_embedding, capsule)
        prefix = f"policy_head.compositional_option_film.{site}."
        overrides[prefix + "gamma"] = gamma.cpu()
        overrides[prefix + "beta"] = beta.cpu()
    overrides["policy_head.stop_embedding"] = (
        model.policy_head.stop_embedding.detach().cpu().float()
        + capsule.stop_delta.detach().cpu().float()
    )
    count_projection = model.policy_head.compositional_count_set_projection
    if not isinstance(count_projection, nn.Linear):
        raise TypeError("routed v4 policy has no shared count-set projection")
    overrides["policy_head.compositional_count_set_projection.weight"] = (
        count_projection.weight.detach().cpu().float()
        + capsule.count_set_projection.weight.detach().cpu().float()
    )
    if count_projection.bias is None or capsule.count_set_projection.bias is None:
        raise ValueError("v4 count-set projections require bias")
    overrides["policy_head.compositional_count_set_projection.bias"] = (
        count_projection.bias.detach().cpu().float()
        + capsule.count_set_projection.bias.detach().cpu().float()
    )


def fixed_compositional_schema_sha256(
    model_config: AgentNetworkConfig,
    state_dict: Mapping[str, Tensor],
    *,
    module_key: str,
) -> str:
    """Fingerprint the fixed v4 topology independently of tensor values."""
    prefix = f"exact_capsules.{module_key}."
    capsule_tensors = [
        {
            "key": key.removeprefix(prefix),
            "shape": list(value.shape),
        }
        for key, value in state_dict.items()
        if key.startswith(prefix)
        and key.removeprefix(prefix).startswith(_FIXED_CAPSULE_PREFIXES)
    ]
    if not capsule_tensors:
        raise ValueError("fixed compositional model has no retained capsule tensors")
    conditioning = model_config.deck_conditioning
    if conditioning is None or conditioning.compositional is None:
        raise ValueError("fixed compositional schema has no v4 conditioning")
    canonical_conditioning = conditioning.model_copy(
        update={"deck_context_mode": "folded"}
    )
    canonical_model_config = model_config.model_copy(
        update={"deck_conditioning": canonical_conditioning}
    )
    return fingerprint_payload(
        {
            "schema": "compositional-fixed-v1",
            "model": canonical_model_config.model_dump(mode="json"),
            "capsule_tensors": sorted(capsule_tensors, key=lambda item: item["key"]),
        }
    )


def compositional_materialization_inventory(
    model_config: AgentNetworkConfig,
) -> tuple[int, int]:
    """Return the exact projection and FiLM counts materialized for DCCR-v4."""
    conditioning = model_config.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or conditioning.compositional is None
    ):
        raise ValueError("compositional inventory requires architecture v4")
    layer_count = len(conditioning.compositional.transformer_layer_indices)
    return 4 * layer_count + len(_POLICY_TARGETS), 2 * layer_count + 2


def _required_tensor(state: Mapping[str, Tensor], key: str) -> Tensor:
    value = state.get(key)
    if not isinstance(value, Tensor):
        raise KeyError(f"compositional export state is missing {key!r}")
    return value.detach().cpu()


__all__ = [
    "compositional_materialization_inventory",
    "fix_selected_compositional_strategy",
    "fixed_compositional_schema_sha256",
]
