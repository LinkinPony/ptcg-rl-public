"""Fused inference-only execution for simple-stateless v2 adapters."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, TypeVar, cast
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.model.simple_stateless.exact_residual_grouped import (
    apply_grouped_rollout_modules,
    can_use_grouped_rollout_modules,
)
from ptcg_rl.model.simple_stateless.rollout_cache import (
    ImmutableRolloutCacheGeneration,
    immutable_rollout_cache_generation,
)
from ptcg_rl.model.simple_stateless.routing import SimpleExactRoutePlan

if TYPE_CHECKING:
    from ptcg_rl.model.simple_stateless.v2 import (
        CompositionalCapsuleStage,
        ExactCompositionalCapsule,
    )

_ParameterSignature = tuple[
    int,
    int,
    int,
    tuple[int, ...],
    torch.dtype,
    torch.device,
]
_RolloutCacheIdentity = (
    tuple[_ParameterSignature, ...] | ImmutableRolloutCacheGeneration
)
_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True)
class _BasisGeometry:
    """Shared geometry required to fuse one basis bank."""

    d_model: int
    bottleneck_dim: int
    basis_count: int
    gelu_approximate: str
    layer_norm_shape: tuple[int, ...]
    layer_norm_eps: float


@dataclass(frozen=True)
class _FusedBasisWeights:
    """Detached BF16 projections for one immutable rollout snapshot."""

    cache_identity: _RolloutCacheIdentity
    down_weight: Tensor
    down_bias: Tensor
    up_weights: Tensor
    up_biases: Tensor
    geometry: _BasisGeometry


@dataclass(frozen=True)
class _PromptWeights:
    """Route-major semantic prompts for one rollout snapshot."""

    cache_identity: _RolloutCacheIdentity
    prompts: Tensor


@dataclass(frozen=True)
class _CapsuleCoefficients:
    """Route-major policy and value basis mixtures."""

    cache_identity: _RolloutCacheIdentity
    policy: Tensor
    value: Tensor


_BASIS_WEIGHT_CACHE: WeakKeyDictionary[
    nn.Module,
    dict[str, _FusedBasisWeights],
] = WeakKeyDictionary()
_PROMPT_WEIGHT_CACHE: WeakKeyDictionary[
    nn.Module,
    OrderedDict[tuple[str, ...], _PromptWeights],
] = WeakKeyDictionary()
_COEFFICIENT_CACHE: WeakKeyDictionary[
    nn.Module,
    OrderedDict[tuple[str, ...], _CapsuleCoefficients],
] = WeakKeyDictionary()
_ROLLOUT_CACHE_LOCK = RLock()
_MAX_CACHED_ROUTE_LAYOUTS = 32


def apply_prompt_rollout(
    semantic_tokens: Tensor,
    *,
    route_plan: SimpleExactRoutePlan,
    prompts: nn.ModuleDict,
) -> Tensor | None:
    """Return a vectorized exact prompt, or ``None`` for the safe fallback."""
    if (
        route_plan.allow_unrouted_rows
        or semantic_tokens.ndim != 3
        or int(semantic_tokens.shape[0]) != route_plan.batch_size
        or int(semantic_tokens.shape[1]) != 10
        or not _can_use_rollout_modules(
            semantic_tokens,
            tuple(prompts[key] for key in route_plan.module_keys),
            prompts,
        )
    ):
        return None
    dispatch = route_plan.grouped_dispatch
    if dispatch is None or int(dispatch.row_indices.numel()) != route_plan.batch_size:
        return None
    full_module_keys = tuple(sorted(prompts))
    full_dispatch = dispatch.with_empty_routes(full_module_keys)
    weights = _prompt_weights(prompts, full_module_keys, semantic_tokens)
    if weights is None:
        return None
    repeats = full_dispatch.element_counts(1)
    route_major = torch.repeat_interleave(
        weights.prompts,
        repeats,
        dim=0,
        output_size=route_plan.batch_size,
    )
    prompt = semantic_tokens.new_empty(semantic_tokens.shape)
    return prompt.index_copy(0, full_dispatch.row_indices, route_major)


def apply_capsule_stage_rollout(
    semantic_tokens: Tensor,
    *,
    route_plan: SimpleExactRoutePlan,
    stage: CompositionalCapsuleStage,
) -> Tensor | None:
    """Return a fused capsule-stage result, or ``None`` for the fallback."""
    if (
        route_plan.allow_unrouted_rows
        or semantic_tokens.ndim != 3
        or int(semantic_tokens.shape[0]) != route_plan.batch_size
        or int(semantic_tokens.shape[1]) != 10
    ):
        return None
    dispatch = route_plan.grouped_dispatch
    if dispatch is None or int(dispatch.row_indices.numel()) != route_plan.batch_size:
        return None
    full_module_keys = tuple(sorted(stage.exact_capsules))
    full_dispatch = dispatch.with_empty_routes(full_module_keys)
    capsules = tuple(
        cast("ExactCompositionalCapsule", stage.exact_capsules[key])
        for key in full_module_keys
    )
    policy_residuals = tuple(
        cast(nn.Module, capsule.policy_residual) for capsule in capsules
    )
    value_residuals = tuple(
        cast(nn.Module, capsule.value_residual) for capsule in capsules
    )
    policy_tokens = torch.cat(
        (semantic_tokens[:, :1], semantic_tokens[:, 2:]),
        dim=1,
    )
    value_token = semantic_tokens[:, 1:2]
    if (
        _basis_geometry(tuple(stage.policy_bases)) is None
        or _basis_geometry(tuple(stage.value_bases)) is None
        or not _can_use_rollout_modules(
            semantic_tokens,
            (*tuple(stage.policy_bases), *tuple(stage.value_bases)),
            stage,
        )
        or not can_use_grouped_rollout_modules(
            policy_tokens,
            policy_residuals,
            stage,
        )
        or not can_use_grouped_rollout_modules(
            value_token,
            value_residuals,
            stage,
        )
    ):
        return None
    coefficients = _capsule_coefficients(
        stage,
        full_module_keys,
        capsules,
        semantic_tokens,
    )
    if coefficients is None:
        return None

    policy_bases = _apply_fused_basis_bank(
        policy_tokens,
        modules=tuple(stage.policy_bases),
        cache_owner=stage,
        cache_key="policy",
    )
    value_bases = _apply_fused_basis_bank(
        value_token,
        modules=tuple(stage.value_bases),
        cache_owner=stage,
        cache_key="value",
    )
    repeats = full_dispatch.element_counts(1)
    policy_coefficients = _scatter_route_values(
        torch.repeat_interleave(
            coefficients.policy,
            repeats,
            dim=0,
            output_size=route_plan.batch_size,
        ),
        full_dispatch.row_indices,
    )
    value_coefficients = _scatter_route_values(
        torch.repeat_interleave(
            coefficients.value,
            repeats,
            dim=0,
            output_size=route_plan.batch_size,
        ),
        full_dispatch.row_indices,
    )
    policy_delta = torch.einsum(
        "btkd,bk->btd",
        policy_bases,
        policy_coefficients,
    )
    policy_delta = policy_delta + apply_grouped_rollout_modules(
        policy_tokens,
        batch_size=route_plan.batch_size,
        dispatch=full_dispatch,
        modules=policy_residuals,
        cache_owner=stage,
        cache_key=("policy_residual", *full_module_keys),
    )
    value_delta = torch.einsum(
        "btkd,bk->btd",
        value_bases,
        value_coefficients,
    )
    value_delta = value_delta + apply_grouped_rollout_modules(
        value_token,
        batch_size=route_plan.batch_size,
        dispatch=full_dispatch,
        modules=value_residuals,
        cache_owner=stage,
        cache_key=("value_residual", *full_module_keys),
    )
    delta = torch.cat(
        (
            policy_delta[:, :1],
            value_delta,
            policy_delta[:, 1:],
        ),
        dim=1,
    )
    return semantic_tokens + delta


def _can_use_rollout_modules(
    inputs: Tensor,
    modules: tuple[nn.Module, ...],
    cache_owner: nn.Module | None = None,
) -> bool:
    """Check the common safety contract for a detached BF16 rollout cache."""
    if (
        torch.is_grad_enabled()
        or inputs.device.type != "cuda"
        or inputs.dtype != torch.bfloat16
        or len(modules) < 4
        or any(module.training for module in modules)
    ):
        return False
    generation = (
        None
        if cache_owner is None
        else immutable_rollout_cache_generation(cache_owner)
    )
    if generation is not None:
        return (
            generation.device == inputs.device
            and generation.dtype == inputs.dtype
        )
    parameter_dtypes: set[torch.dtype] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.device != inputs.device or parameter.dtype not in (
                torch.float32,
                torch.bfloat16,
            ):
                return False
            parameter_dtypes.add(parameter.dtype)
    return torch.float32 not in parameter_dtypes or (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


def _apply_fused_basis_bank(
    inputs: Tensor,
    *,
    modules: tuple[nn.Module, ...],
    cache_owner: nn.Module,
    cache_key: str,
) -> Tensor:
    """Evaluate same-input basis residuals with one down GEMM and one BMM."""
    weights = _fused_basis_weights(
        modules,
        cache_owner=cache_owner,
        cache_key=cache_key,
        inputs=inputs,
    )
    geometry = weights.geometry
    normalized = functional.layer_norm(
        inputs,
        geometry.layer_norm_shape,
        eps=geometry.layer_norm_eps,
    ).to(dtype=inputs.dtype)
    flattened = normalized.reshape(-1, geometry.d_model)
    hidden = functional.linear(
        flattened,
        weights.down_weight,
        weights.down_bias,
    )
    hidden = functional.gelu(
        hidden,
        approximate=geometry.gelu_approximate,
    ).reshape(-1, geometry.basis_count, geometry.bottleneck_dim)
    outputs = torch.bmm(
        hidden.permute(1, 0, 2),
        weights.up_weights,
    )
    outputs = outputs + weights.up_biases
    return outputs.permute(1, 0, 2).reshape(
        *inputs.shape[:-1],
        geometry.basis_count,
        geometry.d_model,
    )


def _fused_basis_weights(
    modules: tuple[nn.Module, ...],
    *,
    cache_owner: nn.Module,
    cache_key: str,
    inputs: Tensor,
) -> _FusedBasisWeights:
    """Build or reuse the detached projection stack for one basis bank."""
    # The marker is an explicit lifetime contract: marked rollout trees are
    # never mutated. Rewalking every descendant and parameter binding on every
    # inference stage made a cache hit linear in model size. The strict
    # validator remains available at publication/debug boundaries.
    generation = immutable_rollout_cache_generation(cache_owner)
    cache_identity: _RolloutCacheIdentity = (
        generation
        if generation is not None
        else tuple(
            _parameter_signature(parameter)
            for module in modules
            for parameter in module.parameters()
        )
    )
    geometry = _basis_geometry(modules)
    if geometry is None:
        raise RuntimeError("fused capsule basis geometry changed unexpectedly")
    with _ROLLOUT_CACHE_LOCK:
        owner_cache = _BASIS_WEIGHT_CACHE.setdefault(cache_owner, {})
        cached = owner_cache.get(cache_key)
        if (
            cached is not None
            and cached.cache_identity == cache_identity
            and cached.down_weight.device == inputs.device
            and cached.geometry == geometry
        ):
            return cached
        down_weight = torch.stack(
            [_linear(module, "down").weight.detach() for module in modules]
        ).reshape(
            geometry.basis_count * geometry.bottleneck_dim,
            geometry.d_model,
        )
        down_bias = torch.stack(
            [_linear(module, "down").bias.detach() for module in modules]
        ).reshape(geometry.basis_count * geometry.bottleneck_dim)
        up_weights = torch.stack(
            [
                _linear(module, "up").weight.detach().transpose(0, 1)
                for module in modules
            ]
        )
        up_biases = torch.stack(
            [_linear(module, "up").bias.detach() for module in modules]
        ).unsqueeze(1)
        cached = _FusedBasisWeights(
            cache_identity=cache_identity,
            down_weight=down_weight.to(dtype=torch.bfloat16).contiguous(),
            down_bias=down_bias.to(dtype=torch.bfloat16).contiguous(),
            up_weights=up_weights.to(dtype=torch.bfloat16).contiguous(),
            up_biases=up_biases.to(dtype=torch.bfloat16).contiguous(),
            geometry=geometry,
        )
        owner_cache[cache_key] = cached
        return cached


def _prompt_weights(
    prompts: nn.ModuleDict,
    module_keys: tuple[str, ...],
    inputs: Tensor,
) -> _PromptWeights | None:
    """Build or reuse one vectorized semantic prompt table."""
    prompt_parameters: list[tuple[nn.Parameter, nn.Parameter]] = []
    for module_key in module_keys:
        prompt = prompts[module_key]
        policy = getattr(prompt, "policy_and_scratch", None)
        value = getattr(prompt, "value", None)
        if (
            not isinstance(policy, nn.Parameter)
            or not isinstance(value, nn.Parameter)
            or policy.ndim != 2
            or tuple(value.shape) != (1, int(policy.shape[1]))
            or int(policy.shape[0]) != 9
        ):
            return None
        prompt_parameters.append((policy, value))
    generation = immutable_rollout_cache_generation(prompts)
    cache_identity: _RolloutCacheIdentity = (
        generation
        if generation is not None
        else tuple(
            _parameter_signature(parameter)
            for pair in prompt_parameters
            for parameter in pair
        )
    )
    with _ROLLOUT_CACHE_LOCK:
        bank_cache = _PROMPT_WEIGHT_CACHE.setdefault(prompts, OrderedDict())
        cached = bank_cache.get(module_keys)
        if (
            cached is not None
            and cached.cache_identity == cache_identity
            and cached.prompts.device == inputs.device
        ):
            bank_cache.move_to_end(module_keys)
            return cached
        stacked = torch.stack(
            [
                torch.cat((policy[:1], value, policy[1:]), dim=0)
                for policy, value in prompt_parameters
            ]
        )
        cached = _PromptWeights(
            cache_identity=cache_identity,
            prompts=stacked.detach().to(dtype=torch.bfloat16).contiguous(),
        )
        bank_cache[module_keys] = cached
        bank_cache.move_to_end(module_keys)
        _evict_route_layouts(bank_cache)
        return cached


def _capsule_coefficients(
    cache_owner: nn.Module,
    module_keys: tuple[str, ...],
    capsules: tuple[nn.Module, ...],
    inputs: Tensor,
) -> _CapsuleCoefficients | None:
    """Build or reuse softmaxed route coefficients for a capsule stage."""
    parameters: list[tuple[nn.Parameter, nn.Parameter]] = []
    coefficient_count: int | None = None
    for capsule in capsules:
        policy = getattr(capsule, "policy_coefficients", None)
        value = getattr(capsule, "value_coefficients", None)
        if (
            not isinstance(policy, nn.Parameter)
            or not isinstance(value, nn.Parameter)
            or policy.ndim != 1
            or value.shape != policy.shape
            or policy.device != inputs.device
            or value.device != inputs.device
            or policy.dtype not in (torch.float32, torch.bfloat16)
            or value.dtype not in (torch.float32, torch.bfloat16)
        ):
            return None
        if coefficient_count is None:
            coefficient_count = int(policy.numel())
        elif int(policy.numel()) != coefficient_count:
            return None
        parameters.append((policy, value))
    generation = immutable_rollout_cache_generation(cache_owner)
    cache_identity: _RolloutCacheIdentity = (
        generation
        if generation is not None
        else tuple(
            _parameter_signature(parameter)
            for pair in parameters
            for parameter in pair
        )
    )
    with _ROLLOUT_CACHE_LOCK:
        owner_cache = _COEFFICIENT_CACHE.setdefault(
            cache_owner,
            OrderedDict(),
        )
        cached = owner_cache.get(module_keys)
        if (
            cached is not None
            and cached.cache_identity == cache_identity
            and cached.policy.device == inputs.device
        ):
            owner_cache.move_to_end(module_keys)
            return cached
        policy_logits = torch.stack([policy for policy, _value in parameters])
        value_logits = torch.stack([value for _policy, value in parameters])
        cached = _CapsuleCoefficients(
            cache_identity=cache_identity,
            policy=torch.softmax(policy_logits.float(), dim=1)
            .to(dtype=torch.bfloat16)
            .contiguous(),
            value=torch.softmax(value_logits.float(), dim=1)
            .to(dtype=torch.bfloat16)
            .contiguous(),
        )
        owner_cache[module_keys] = cached
        owner_cache.move_to_end(module_keys)
        _evict_route_layouts(owner_cache)
        return cached


def _basis_geometry(modules: tuple[nn.Module, ...]) -> _BasisGeometry | None:
    """Validate the homogeneous no-affine pre-LN residual basis contract."""
    if not modules:
        return None
    first_down = getattr(modules[0], "down", None)
    first_up = getattr(modules[0], "up", None)
    first_norm = getattr(modules[0], "norm", None)
    first_activation = getattr(modules[0], "activation", None)
    if (
        not isinstance(first_down, nn.Linear)
        or first_down.bias is None
        or not isinstance(first_up, nn.Linear)
        or first_up.bias is None
        or first_up.in_features != first_down.out_features
        or first_up.out_features != first_down.in_features
        or not isinstance(first_norm, nn.LayerNorm)
        or first_norm.elementwise_affine
        or not isinstance(first_activation, nn.GELU)
    ):
        return None
    geometry = _BasisGeometry(
        d_model=first_down.in_features,
        bottleneck_dim=first_down.out_features,
        basis_count=len(modules),
        gelu_approximate=first_activation.approximate,
        layer_norm_shape=tuple(int(value) for value in first_norm.normalized_shape),
        layer_norm_eps=first_norm.eps,
    )
    for module in modules:
        down = getattr(module, "down", None)
        up = getattr(module, "up", None)
        norm = getattr(module, "norm", None)
        activation = getattr(module, "activation", None)
        if (
            not isinstance(down, nn.Linear)
            or down.bias is None
            or down.in_features != geometry.d_model
            or down.out_features != geometry.bottleneck_dim
            or not isinstance(up, nn.Linear)
            or up.bias is None
            or up.in_features != geometry.bottleneck_dim
            or up.out_features != geometry.d_model
            or not isinstance(norm, nn.LayerNorm)
            or norm.elementwise_affine
            or tuple(norm.normalized_shape) != geometry.layer_norm_shape
            or norm.eps != geometry.layer_norm_eps
            or not isinstance(activation, nn.GELU)
            or activation.approximate != geometry.gelu_approximate
        ):
            return None
    return geometry


def _scatter_route_values(route_major: Tensor, row_indices: Tensor) -> Tensor:
    """Restore route-major rows to original batch order."""
    output = route_major.new_empty(route_major.shape)
    return output.index_copy(0, row_indices, route_major)


def _linear(module: nn.Module, attribute: str) -> nn.Linear:
    """Return a linear projection after basis geometry validation."""
    return cast(nn.Linear, getattr(module, attribute))


def _parameter_signature(parameter: nn.Parameter) -> _ParameterSignature:
    """Identify in-place mutations, device moves, and parameter replacement."""
    return (
        id(parameter),
        parameter._version,  # noqa: SLF001 - detects in-place state copies.
        parameter.data_ptr(),
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


def _evict_route_layouts(
    cache: OrderedDict[tuple[str, ...], _CacheValue],
) -> None:
    """Bound route-layout caches for long-lived rollout processes."""
    while len(cache) > _MAX_CACHED_ROUTE_LAYOUTS:
        cache.popitem(last=False)
