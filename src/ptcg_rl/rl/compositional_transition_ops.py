"""Tensor-level operations for dense-v3 to DCCR-v4 conversion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.model.compositional_capsule import exact_capsule
from ptcg_rl.model.compositional_projection import SharedCompositionalLinear
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DeckConditioningConfig,
)
from ptcg_rl.model.network import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.rl.compositional_factorization import (
    TruncatedSvdConfig,
    factorize_dense_expert_weights,
    fit_compositional_router,
)

SOURCE_PRIVATE_PREFIXES = (
    "state_encoder.private_strategy_stacks.",
    "policy_head.private_strategies.",
    "dense_private_root_value_heads.",
    "dense_private_prefix_value_heads.",
)


def _validate_transition_configs(
    source: AgentNetworkConfig,
    target: AgentNetworkConfig,
) -> tuple[DeckConditioningConfig, DeckConditioningConfig]:
    source_conditioning = source.deck_conditioning
    target_conditioning = target.deck_conditioning
    if (
        source_conditioning is None
        or not source_conditioning.enabled
        or source_conditioning.architecture_version
        != DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        or source_conditioning.dense_private is None
        or source_conditioning.dense_private.export_mode != "routed"
    ):
        raise ValueError("compositional transition source must be routed dense-v3")
    if (
        target_conditioning is None
        or not target_conditioning.enabled
        or target_conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or target_conditioning.compositional is None
        or target_conditioning.compositional.export_mode != "routed"
    ):
        raise ValueError("compositional transition target must be routed v4")
    if source.state_encoder != target.state_encoder or source.policy != target.policy:
        raise ValueError("transition cannot change state or policy base geometry")
    return source_conditioning, target_conditioning


def _sampling_weight_tensor(
    routes: Sequence[Any],
    configured: Mapping[str, float] | None,
) -> Tensor:
    if configured is None:
        return torch.full((len(routes),), 1.0 / len(routes))
    if set(configured) != {route.deck_digest for route in routes}:
        raise ValueError("sampling weights must cover every exact target deck")
    values = torch.tensor(
        [float(configured[route.deck_digest]) for route in routes],
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(values).all().item()) or bool(
        (values <= 0.0).any().item()
    ):
        raise ValueError("sampling weights must be finite and positive")
    return values / values.sum()


def _copy_unchanged_shared_state(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    for key, target in tuple(target_state.items()):
        source = source_state.get(key)
        if (
            source is None
            or key.startswith(SOURCE_PRIVATE_PREFIXES)
            or source.shape != target.shape
        ):
            continue
        target_state[key] = source.detach().to(dtype=target.dtype).clone()
        source_key_actions[key] = {"action": "copied_shared", "target": key}


def _target_deck_embeddings(
    target_model: AgentPolicyValueNet,
    conditioning: DeckConditioningConfig,
) -> Tensor:
    if target_model.deck_encoder is None:
        raise RuntimeError("v4 transition target has no DeckEncoder")
    device = next(target_model.parameters()).device
    card_ids = torch.tensor(
        [route.canonical_card_ids for route in conditioning.expert_routes],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        return cast(
            Tensor,
            target_model.deck_encoder(
                card_ids,
                card_encoder=target_model.card_encoder,
            ).detach(),
        )


def _factorize_projection(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_keys: Sequence[str],
    base_weight_key: str,
    projection_prefix: str,
    projection: SharedCompositionalLinear,
    target_routes: Sequence[Any],
    target_domain: str,
    target_name: str,
    capsules: nn.ModuleDict,
    deck_embeddings: Tensor,
    sampling_weights: Tensor,
    shared_rank: int,
    exact_rank: int,
    factorization_device: torch.device,
    factorization_config: TruncatedSvdConfig,
    source_key_actions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _require_source_keys(source_state, source_keys)
    expert_weights = torch.stack(
        [source_state[key].detach().to(factorization_device) for key in source_keys]
    )
    result = factorize_dense_expert_weights(
        expert_weights,
        sampling_weights=sampling_weights.to(factorization_device),
        basis_count=projection.basis_count,
        shared_rank=shared_rank,
        exact_rank=exact_rank,
        svd_config=_projection_svd_config(
            factorization_config,
            domain=target_domain,
            target=target_name,
        ),
    )
    _assign_tensor(target_state, base_weight_key, result.mean)
    _assign_tensor(target_state, f"{projection_prefix}.shared_a", result.shared_a)
    _assign_tensor(target_state, f"{projection_prefix}.shared_b", result.shared_b)
    router_fit = fit_compositional_router(
        projection.router_norm,
        projection.router,
        deck_embeddings,
        result.coefficients.to(deck_embeddings.device),
    )
    for name, tensor in projection.state_dict().items():
        if name.startswith(("router.", "router_norm.")):
            _assign_tensor(target_state, f"{projection_prefix}.{name}", tensor)
    for index, route in enumerate(target_routes):
        capsule = exact_capsule(capsules, route.module_key)
        residual = capsule.residual(cast(Any, target_domain), target_name)
        residual_prefix = (
            f"exact_capsules.{route.module_key}.{target_domain}_residuals.{target_name}"
        )
        _assign_tensor(
            target_state,
            f"{residual_prefix}.lora_a.weight",
            result.exact_a[index],
        )
        _assign_tensor(
            target_state,
            f"{residual_prefix}.lora_b.weight",
            result.exact_b[index],
        )
        if residual.rank != exact_rank:
            raise RuntimeError("target exact residual rank changed during transition")
        route_bias_key = (
            f"exact_capsules.{route.module_key}.{target_domain}_route_biases."
            f"{target_name}"
        )
        _assign_tensor(
            target_state,
            route_bias_key,
            router_fit.route_biases[index],
        )
    for key in source_keys:
        source_key_actions[key] = {
            "action": "mean_shared_basis_exact_residual",
            "target": base_weight_key,
            "source_tensor_sha256": _tensor_sha256(source_state[key]),
        }
    del expert_weights
    return {
        "target": target_name,
        "domain": target_domain,
        "source_keys": list(source_keys),
        "source_shapes": [list(source_state[key].shape) for key in source_keys],
        "shared_rank": shared_rank,
        "exact_rank": exact_rank,
        "basis_count": projection.basis_count,
        "singular_values": [float(value) for value in result.singular_values.cpu()],
        "weighted_relative_error": result.weighted_relative_error,
        "router_fit_rmse": router_fit.root_mean_squared_error,
        "router_leave_one_out_rmse": (router_fit.leave_one_out_root_mean_squared_error),
        "svd": result.svd_config.manifest(),
        "coefficient_min": float(result.coefficients.min().item()),
        "coefficient_max": float(result.coefficients.max().item()),
    }


def _projection_svd_config(
    config: TruncatedSvdConfig,
    *,
    domain: str,
    target: str,
) -> TruncatedSvdConfig:
    """Derive one stable projection-local random stream from the formal seed."""
    payload = f"{config.seed}\0{domain}\0{target}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    return TruncatedSvdConfig(
        seed=seed,
        oversampling=config.oversampling,
        power_iterations=config.power_iterations,
        algorithm=config.algorithm,
    )


def _migrate_vector_mean_and_offsets(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_keys: Sequence[str],
    base_key: str,
    target_routes: Sequence[Any],
    target_offset_keys: Sequence[str],
    sampling_weights: Tensor,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    _require_source_keys(source_state, source_keys)
    if len(source_keys) != len(target_routes) or len(source_keys) != len(
        target_offset_keys
    ):
        raise ValueError("vector transition rows do not align")
    stacked = torch.stack([source_state[key].detach().float() for key in source_keys])
    weights = sampling_weights.to(device=stacked.device)
    mean = torch.einsum("d,d...->...", weights, stacked)
    _assign_tensor(target_state, base_key, mean)
    for key, offset_key, value in zip(
        source_keys,
        target_offset_keys,
        stacked,
        strict=True,
    ):
        _assign_tensor(target_state, offset_key, value - mean)
        source_key_actions[key] = {
            "action": "shared_mean_exact_vector_offset",
            "target": base_key,
            "offset_target": offset_key,
        }


def _migrate_output_norm(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_routes: Sequence[Any],
    target_routes: Sequence[Any],
    sampling_weights: Tensor,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    for field in ("weight", "bias"):
        source_keys = tuple(
            "state_encoder.private_strategy_stacks."
            f"{route.module_key}.output_norm.{field}"
            for route in source_routes
        )
        _migrate_vector_mean_and_offsets(
            target_state,
            source_state,
            source_keys=source_keys,
            base_key=f"state_encoder.layer_norm.{field}",
            target_routes=target_routes,
            target_offset_keys=tuple(
                "exact_capsules."
                f"{route.module_key}.transformer_norm_offsets.output_{field}"
                for route in target_routes
            ),
            sampling_weights=sampling_weights,
            source_key_actions=source_key_actions,
        )


def _migrate_shared_state_adapters(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_routes: Sequence[Any],
    target_conditioning: DeckConditioningConfig,
    source_shared_layers: int,
    sampling_weights: Tensor,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    for layer_index in target_conditioning.adapter_layer_indices:
        target_prefix = (
            f"state_encoder.compositional_shared_adapters.layer_{layer_index:02d}."
        )
        relative = layer_index - source_shared_layers
        if relative < 0 or not any(
            key.startswith(target_prefix) for key in target_state
        ):
            continue
        source_prefixes = tuple(
            "state_encoder.private_strategy_stacks."
            f"{route.module_key}.layers.{relative}.residual."
            for route in source_routes
        )
        suffixes = tuple(
            key.removeprefix(source_prefixes[0])
            for key in source_state
            if key.startswith(source_prefixes[0])
        )
        for suffix in suffixes:
            source_keys = tuple(f"{prefix}{suffix}" for prefix in source_prefixes)
            _migrate_shared_mean(
                target_state,
                source_state,
                source_keys=source_keys,
                target_key=f"{target_prefix}{suffix}",
                sampling_weights=sampling_weights,
                action="shared_mean",
                source_key_actions=source_key_actions,
            )


def _migrate_policy_structures(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_routes: Sequence[Any],
    target_routes: Sequence[Any],
    sampling_weights: Tensor,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    source_strategy_prefixes = tuple(
        f"policy_head.private_strategies.{route.module_key}." for route in source_routes
    )
    option_prefixes = tuple(
        f"{prefix}option_set." for prefix in source_strategy_prefixes
    )
    option_suffixes = tuple(
        key.removeprefix(option_prefixes[0])
        for key in source_state
        if key.startswith(option_prefixes[0])
    )
    for suffix in option_suffixes:
        source_keys = tuple(f"{prefix}{suffix}" for prefix in option_prefixes)
        _migrate_shared_mean(
            target_state,
            source_state,
            source_keys=source_keys,
            target_key=f"policy_head.compositional_option_set.{suffix}",
            sampling_weights=sampling_weights,
            action="shared_mean_option_core",
            source_key_actions=source_key_actions,
        )

    for field in ("weight", "bias"):
        source_keys = tuple(
            f"{prefix}count_set_projection.{field}"
            for prefix in source_strategy_prefixes
        )
        stacked, mean = _stacked_mean(
            source_state,
            source_keys,
            sampling_weights=sampling_weights,
        )
        target_key = f"policy_head.compositional_count_set_projection.{field}"
        _assign_tensor(target_state, target_key, mean)
        for source_key, route, value in zip(
            source_keys,
            target_routes,
            stacked,
            strict=True,
        ):
            exact_key = (
                f"exact_capsules.{route.module_key}.count_set_projection.{field}"
            )
            _assign_tensor(target_state, exact_key, value - mean)
            source_key_actions[source_key] = {
                "action": "shared_mean_exact_count_calibration",
                "target": target_key,
                "offset_target": exact_key,
            }

    stop_keys = tuple(f"{prefix}stop_embedding" for prefix in source_strategy_prefixes)
    stacked, stop_mean = _stacked_mean(
        source_state,
        stop_keys,
        sampling_weights=sampling_weights,
    )
    _assign_tensor(target_state, "policy_head.stop_embedding", stop_mean)
    for source_key, route, value in zip(
        stop_keys,
        target_routes,
        stacked,
        strict=True,
    ):
        exact_key = f"exact_capsules.{route.module_key}.stop_delta"
        _assign_tensor(target_state, exact_key, value - stop_mean)
        source_key_actions[source_key] = {
            "action": "shared_mean_exact_stop_delta",
            "target": "policy_head.stop_embedding",
            "offset_target": exact_key,
        }

    for source_route, target_route in zip(
        source_routes,
        target_routes,
        strict=True,
    ):
        for field in ("down.weight", "down.bias", "up.weight", "up.bias"):
            source_key = (
                f"policy_head.private_strategies.{source_route.module_key}."
                f"global_adapter.{field}"
            )
            target_key = (
                f"exact_capsules.{target_route.module_key}."
                f"policy_global_residual.{field}"
            )
            _assign_tensor(target_state, target_key, source_state[source_key])
            source_key_actions[source_key] = {
                "action": "moved_exact_global_residual",
                "target": target_key,
            }


def _migrate_value_bases(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_routes: Sequence[Any],
    sampling_weights: Tensor,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    definitions = (
        (
            "dense_private_root_value_heads",
            "value_head",
        ),
        (
            "dense_private_prefix_value_heads",
            "prefix_value_delta_head",
        ),
    )
    base_fields = {
        "base_hidden.weight": "0.weight",
        "base_hidden.bias": "0.bias",
        "base_output.weight": "2.weight",
        "base_output.bias": "2.bias",
    }
    for source_bank, target_bank in definitions:
        for source_field, target_field in base_fields.items():
            source_keys = tuple(
                f"{source_bank}.{route.module_key}.{source_field}"
                for route in source_routes
            )
            _migrate_shared_mean(
                target_state,
                source_state,
                source_keys=source_keys,
                target_key=f"{target_bank}.{target_field}",
                sampling_weights=sampling_weights,
                action="shared_mean_value_base",
                source_key_actions=source_key_actions,
            )


def _migrate_shared_mean(
    target_state: dict[str, Tensor],
    source_state: Mapping[str, Tensor],
    *,
    source_keys: Sequence[str],
    target_key: str,
    sampling_weights: Tensor,
    action: str,
    source_key_actions: dict[str, dict[str, Any]],
) -> None:
    _stacked, mean = _stacked_mean(
        source_state,
        source_keys,
        sampling_weights=sampling_weights,
    )
    _assign_tensor(target_state, target_key, mean)
    for key in source_keys:
        source_key_actions[key] = {"action": action, "target": target_key}


def _stacked_mean(
    source_state: Mapping[str, Tensor],
    source_keys: Sequence[str],
    *,
    sampling_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    _require_source_keys(source_state, source_keys)
    stacked = torch.stack([source_state[key].detach().float() for key in source_keys])
    weights = sampling_weights.to(device=stacked.device)
    return stacked, torch.einsum("d,d...->...", weights, stacked)


def _assign_tensor(
    target_state: dict[str, Tensor],
    key: str,
    value: Tensor,
) -> None:
    if key not in target_state:
        raise KeyError(f"transition target state is missing {key!r}")
    target = target_state[key]
    if value.shape != target.shape:
        raise ValueError(
            f"transition tensor shape mismatch for {key}: "
            f"{tuple(value.shape)} != {tuple(target.shape)}"
        )
    target_state[key] = (
        value.detach().to(device=target.device, dtype=target.dtype).clone()
    )


def _require_source_keys(
    source_state: Mapping[str, Tensor],
    keys: Sequence[str],
) -> None:
    missing = [key for key in keys if key not in source_state]
    if missing:
        raise KeyError(f"transition source state is missing keys: {missing}")


def _validate_finite_state(state: Mapping[str, Tensor]) -> None:
    non_finite = [
        name
        for name, tensor in state.items()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item())
    ]
    if non_finite:
        raise ValueError(
            f"converted v4 state contains non-finite tensors: {non_finite}"
        )


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
