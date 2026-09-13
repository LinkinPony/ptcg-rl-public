"""H200 grouped-GEMM execution for exact-deck residual banks."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from threading import RLock
from typing import cast
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn
from torch.nn import functional

from ptcg_rl.model.deck_lora import RoutedLoRADispatch
from ptcg_rl.model.simple_stateless.rollout_cache import (
    ImmutableRolloutCacheGeneration,
    immutable_rollout_cache_generation,
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


@dataclass(frozen=True)
class _ExactResidualGeometry:
    """Common two-layer geometry accepted by the grouped rollout kernel."""

    in_features: int
    hidden_features: int
    out_features: int
    gelu_approximate: str
    layer_norm_shape: tuple[int, ...] | None
    layer_norm_eps: float
    output_attribute: str


@dataclass(frozen=True)
class _GroupedResidualWeights:
    """Detached BF16 weights cached for one immutable rollout generation."""

    cache_identity: _RolloutCacheIdentity
    down_weights: Tensor
    down_biases: Tensor
    output_weights: Tensor
    output_biases: Tensor
    geometry: _ExactResidualGeometry


_ROLLOUT_WEIGHT_CACHE: WeakKeyDictionary[
    nn.Module,
    OrderedDict[tuple[str, ...], _GroupedResidualWeights],
] = WeakKeyDictionary()
_ROLLOUT_WEIGHT_CACHE_LOCK = RLock()
_MAX_CACHED_ROUTE_LAYOUTS = 32


def can_use_grouped_rollout_residual(
    inputs: Tensor,
    module_keys: tuple[str, ...],
    modules: nn.ModuleDict,
) -> bool:
    """Return whether this inference call fits the H200 grouped fast path."""
    if any(module_key not in modules for module_key in module_keys):
        return False
    active_modules = tuple(modules[module_key] for module_key in module_keys)
    return can_use_grouped_rollout_modules(
        inputs,
        active_modules,
        modules,
    )


def can_use_grouped_rollout_modules(
    inputs: Tensor,
    modules: tuple[nn.Module, ...],
    cache_owner: nn.Module | None = None,
) -> bool:
    """Return whether a residual bank fits a CUDA batched rollout path."""
    if (
        torch.is_grad_enabled()
        or any(module.training for module in modules)
        or inputs.device.type != "cuda"
        or inputs.dtype != torch.bfloat16
        # Small route counts remain cheaper as independent modules.
        or len(modules) < 4
        or not _batched_mm_device_supported(inputs.get_device())
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
            and _exact_residual_geometry(modules) is not None
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
    # Frozen checkpoint state remains FP32 and is converted only for the GEMM,
    # exactly where CUDA autocast would cast nn.Linear operands.  Requiring an
    # active autocast context prevents this rollout-only shortcut from changing
    # the semantics of explicit FP32 inference.
    if torch.float32 in parameter_dtypes and (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        return False
    return _exact_residual_geometry(modules) is not None


def apply_grouped_rollout_residual(
    inputs: Tensor,
    *,
    batch_size: int,
    dispatch: RoutedLoRADispatch,
    modules: nn.ModuleDict,
) -> Tensor:
    """Evaluate a route-major exact bank with two jagged grouped GEMMs."""
    full_module_keys = tuple(sorted(modules))
    full_dispatch = dispatch.with_empty_routes(full_module_keys)
    full_modules = tuple(modules[module_key] for module_key in full_module_keys)
    return apply_grouped_rollout_modules(
        inputs,
        batch_size=batch_size,
        dispatch=full_dispatch,
        modules=full_modules,
        cache_owner=modules,
        cache_key=full_module_keys,
    )


def apply_grouped_rollout_modules(
    inputs: Tensor,
    *,
    batch_size: int,
    dispatch: RoutedLoRADispatch,
    modules: tuple[nn.Module, ...],
    cache_owner: nn.Module,
    cache_key: tuple[str, ...],
) -> Tensor:
    """Evaluate an ordered residual bank with two batched CUDA GEMMs."""
    if len(modules) != len(dispatch.module_keys):
        raise ValueError("grouped residual modules and dispatch routes differ")
    weights = _grouped_rollout_weights(
        cache_owner,
        cache_key,
        modules,
        inputs,
    )
    geometry = weights.geometry
    selected = inputs.index_select(0, dispatch.row_indices)
    if geometry.layer_norm_shape is not None:
        selected = functional.layer_norm(
            selected,
            geometry.layer_norm_shape,
            eps=geometry.layer_norm_eps,
        ).to(dtype=inputs.dtype)
    elements_per_row = _elements_per_batch_row(inputs)
    flattened = selected.reshape(-1, geometry.in_features)
    aligned_in = int(weights.down_weights.shape[1])
    if aligned_in != geometry.in_features:
        flattened = functional.pad(
            flattened,
            (0, aligned_in - geometry.in_features),
        )
    if inputs.device.type == "cuda" and not _grouped_mm_device_supported(
        inputs.get_device()
    ):
        return _apply_padded_batched_rollout_modules(
            inputs,
            batch_size=batch_size,
            dispatch=dispatch,
            weights=weights,
            flattened=flattened,
            elements_per_row=elements_per_row,
        )
    offsets = dispatch.element_offsets(elements_per_row)
    element_counts = dispatch.element_counts(elements_per_row)
    hidden = functional.grouped_mm(
        flattened,
        weights.down_weights,
        offs=offsets,
    )
    hidden = hidden + torch.repeat_interleave(
        weights.down_biases,
        element_counts,
        dim=0,
        output_size=int(flattened.shape[0]),
    )
    hidden = functional.gelu(
        hidden,
        approximate=geometry.gelu_approximate,
    )
    contributions = functional.grouped_mm(
        hidden,
        weights.output_weights,
        offs=offsets,
    )
    contributions = contributions + torch.repeat_interleave(
        weights.output_biases,
        element_counts,
        dim=0,
        output_size=int(flattened.shape[0]),
    )
    if int(contributions.shape[1]) != geometry.out_features:
        contributions = contributions.narrow(1, 0, geometry.out_features)
    residual = contributions.reshape(
        int(dispatch.row_indices.numel()),
        *inputs.shape[1:-1],
        geometry.out_features,
    )
    output = inputs.new_zeros((batch_size, *residual.shape[1:]))
    return output.index_copy(0, dispatch.row_indices, residual)


def _apply_padded_batched_rollout_modules(
    inputs: Tensor,
    *,
    batch_size: int,
    dispatch: RoutedLoRADispatch,
    weights: _GroupedResidualWeights,
    flattened: Tensor,
    elements_per_row: int,
) -> Tensor:
    """Run route-major residuals as two padded BMMs on pre-SM90 CUDA."""
    geometry = weights.geometry
    max_rows, valid_indices = dispatch.padded_element_indices(elements_per_row)
    elements_per_route = max_rows * elements_per_row
    route_count = len(dispatch.module_keys)
    aligned_in = int(weights.down_weights.shape[1])
    padded = flattened.new_zeros(
        (route_count * elements_per_route, aligned_in),
    )
    padded.index_copy_(0, valid_indices, flattened)
    hidden = torch.bmm(
        padded.reshape(route_count, elements_per_route, aligned_in),
        weights.down_weights,
    )
    hidden = functional.gelu(
        hidden + weights.down_biases.unsqueeze(1),
        approximate=geometry.gelu_approximate,
    )
    contributions = torch.bmm(
        hidden,
        weights.output_weights,
    )
    contributions = contributions + weights.output_biases.unsqueeze(1)
    if int(contributions.shape[2]) != geometry.out_features:
        contributions = contributions.narrow(2, 0, geometry.out_features)
    residual = contributions.reshape(
        -1,
        geometry.out_features,
    ).index_select(0, valid_indices)
    residual = residual.reshape(
        int(dispatch.row_indices.numel()),
        *inputs.shape[1:-1],
        geometry.out_features,
    )
    output = inputs.new_zeros((batch_size, *residual.shape[1:]))
    return output.index_copy(0, dispatch.row_indices, residual)


def _grouped_rollout_weights(
    cache_owner: nn.Module,
    cache_key: tuple[str, ...],
    modules: tuple[nn.Module, ...],
    inputs: Tensor,
) -> _GroupedResidualWeights:
    """Return an automatically invalidated, detached BF16 rollout stack."""
    # A marked tree is immutable for the remainder of its lifetime. Trust the
    # generation token on cache hits instead of rescanning every descendant;
    # strict binding validation remains a boundary diagnostic.
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
    with _ROLLOUT_WEIGHT_CACHE_LOCK:
        bank_cache = _ROLLOUT_WEIGHT_CACHE.setdefault(cache_owner, OrderedDict())
        cached_weights = bank_cache.get(cache_key)
        if (
            cached_weights is not None
            and cached_weights.cache_identity == cache_identity
            and cached_weights.down_weights.device == inputs.device
        ):
            bank_cache.move_to_end(cache_key)
            return cached_weights
        geometry = _exact_residual_geometry(modules)
        if geometry is None:
            raise RuntimeError("grouped exact residual geometry changed unexpectedly")
        aligned_in = _aligned_width(geometry.in_features)
        aligned_hidden = _aligned_width(geometry.hidden_features)
        aligned_out = _aligned_width(geometry.out_features)
        down_weights = torch.stack(
            [
                _linear(module, "down").weight.detach().transpose(0, 1)
                for module in modules
            ]
        ).to(dtype=torch.bfloat16)
        down_biases = torch.stack(
            [_linear(module, "down").bias.detach() for module in modules]
        ).to(dtype=torch.bfloat16)
        output_weights = torch.stack(
            [
                _linear(module, geometry.output_attribute)
                .weight.detach()
                .transpose(0, 1)
                for module in modules
            ]
        ).to(dtype=torch.bfloat16)
        output_biases = torch.stack(
            [
                _linear(
                    module,
                    geometry.output_attribute,
                ).bias.detach()
                for module in modules
            ]
        ).to(dtype=torch.bfloat16)
        down_weights = functional.pad(
            down_weights,
            (
                0,
                aligned_hidden - geometry.hidden_features,
                0,
                aligned_in - geometry.in_features,
            ),
        ).contiguous()
        down_biases = functional.pad(
            down_biases,
            (0, aligned_hidden - geometry.hidden_features),
        ).contiguous()
        output_weights = functional.pad(
            output_weights,
            (
                0,
                aligned_out - geometry.out_features,
                0,
                aligned_hidden - geometry.hidden_features,
            ),
        ).contiguous()
        output_biases = functional.pad(
            output_biases,
            (0, aligned_out - geometry.out_features),
        ).contiguous()
        cached_weights = _GroupedResidualWeights(
            cache_identity=cache_identity,
            down_weights=down_weights,
            down_biases=down_biases,
            output_weights=output_weights,
            output_biases=output_biases,
            geometry=geometry,
        )
        bank_cache[cache_key] = cached_weights
        bank_cache.move_to_end(cache_key)
        while len(bank_cache) > _MAX_CACHED_ROUTE_LAYOUTS:
            bank_cache.popitem(last=False)
        return cached_weights


def _parameter_signature(parameter: nn.Parameter) -> _ParameterSignature:
    """Identify mutations, device moves, and whole-parameter replacement."""
    return (
        id(parameter),
        parameter._version,  # noqa: SLF001 - detects in-place state copies.
        parameter.data_ptr(),
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


def _exact_residual_geometry(
    modules: tuple[nn.Module, ...],
) -> _ExactResidualGeometry | None:
    """Recognize the exact policy/value two-layer residual contract."""
    if not modules:
        return None
    first = modules[0]
    down = getattr(first, "down", None)
    activation = getattr(first, "activation", None)
    output_attribute = "up" if hasattr(first, "up") else "output"
    output = getattr(first, output_attribute, None)
    norm = getattr(first, "norm", None)
    if (
        not isinstance(down, nn.Linear)
        or down.bias is None
        or not isinstance(activation, nn.GELU)
        or not isinstance(output, nn.Linear)
        or output.bias is None
        or (
            norm is not None
            and (not isinstance(norm, nn.LayerNorm) or norm.elementwise_affine)
        )
    ):
        return None
    layer_norm_shape = (
        tuple(int(value) for value in norm.normalized_shape)
        if isinstance(norm, nn.LayerNorm)
        else None
    )
    geometry = _ExactResidualGeometry(
        in_features=down.in_features,
        hidden_features=down.out_features,
        out_features=output.out_features,
        gelu_approximate=activation.approximate,
        layer_norm_shape=layer_norm_shape,
        layer_norm_eps=(norm.eps if isinstance(norm, nn.LayerNorm) else 0.0),
        output_attribute=output_attribute,
    )
    if output.in_features != geometry.hidden_features:
        return None
    for module in modules:
        candidate_down = getattr(module, "down", None)
        candidate_activation = getattr(module, "activation", None)
        candidate_output = getattr(module, output_attribute, None)
        candidate_norm = getattr(module, "norm", None)
        if (
            not isinstance(candidate_down, nn.Linear)
            or candidate_down.bias is None
            or candidate_down.in_features != geometry.in_features
            or candidate_down.out_features != geometry.hidden_features
            or not isinstance(candidate_activation, nn.GELU)
            or candidate_activation.approximate != geometry.gelu_approximate
            or not isinstance(candidate_output, nn.Linear)
            or candidate_output.bias is None
            or candidate_output.in_features != geometry.hidden_features
            or candidate_output.out_features != geometry.out_features
            or not _matching_layer_norm(candidate_norm, geometry)
        ):
            return None
    return geometry


def _matching_layer_norm(
    candidate: object,
    geometry: _ExactResidualGeometry,
) -> bool:
    if geometry.layer_norm_shape is None:
        return candidate is None
    return (
        isinstance(candidate, nn.LayerNorm)
        and not candidate.elementwise_affine
        and tuple(candidate.normalized_shape) == geometry.layer_norm_shape
        and candidate.eps == geometry.layer_norm_eps
    )


def _linear(module: nn.Module, attribute: str) -> nn.Linear:
    """Return a previously validated linear projection."""
    return cast(nn.Linear, getattr(module, attribute))


def _elements_per_batch_row(inputs: Tensor) -> int:
    elements = 1
    for dimension in inputs.shape[1:-1]:
        elements *= int(dimension)
    return elements


def _aligned_width(width: int) -> int:
    alignment = 16 // torch.bfloat16.itemsize
    return ((width + alignment - 1) // alignment) * alignment


@cache
def _grouped_mm_device_supported(device_index: int) -> bool:
    return hasattr(functional, "grouped_mm") and torch.cuda.get_device_capability(
        device_index
    ) >= (9, 0)


@cache
def _batched_mm_device_supported(device_index: int) -> bool:
    """Return whether the measured Ada path or grouped-MM path is available."""
    capability = torch.cuda.get_device_capability(device_index)
    return capability == (8, 9) or _grouped_mm_device_supported(device_index)
