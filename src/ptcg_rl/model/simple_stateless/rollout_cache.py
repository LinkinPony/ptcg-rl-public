"""Explicit immutable generations for detached rollout-only weight caches."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

_ROLLOUT_GENERATION_ATTRIBUTE = "_ptcg_immutable_rollout_cache_generation"


@dataclass(frozen=True, slots=True, eq=False)
class ImmutableRolloutCacheGeneration:
    """Opaque identity and tensor contract for one immutable rollout model."""

    device: torch.device
    dtype: torch.dtype
    parameter_bindings: dict[
        int,
        tuple[tuple[str, nn.Parameter], ...],
    ] = field(repr=False)


def mark_immutable_rollout_cache_generation(
    module: nn.Module,
) -> ImmutableRolloutCacheGeneration:
    """Declare an eval-only BF16 module tree immutable for its remaining life.

    Callers must publish a new module tree, rather than mutate this one, for a
    later weight generation. The marker deliberately stays outside state dicts.
    """
    descendants = tuple(module.modules())
    existing = tuple(
        immutable_rollout_cache_generation(descendant)
        for descendant in descendants
    )
    present = tuple(marker for marker in existing if marker is not None)
    if present:
        generation = present[0]
        if len(present) != len(descendants) or any(
            marker is not generation for marker in present
        ):
            raise ValueError("rollout module tree mixes immutable generations")
        return generation
    if any(descendant.training for descendant in descendants):
        raise ValueError("immutable rollout cache generation requires eval mode")
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("immutable rollout cache generation requires parameters")
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError(
            "immutable rollout cache generation requires gradient-free parameters"
        )
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("immutable rollout parameters must share one device")
    floating_dtypes = {
        parameter.dtype
        for parameter in parameters
        if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise ValueError("immutable rollout parameters must be pure BF16")
    generation = ImmutableRolloutCacheGeneration(
        device=next(iter(devices)),
        dtype=torch.bfloat16,
        parameter_bindings={
            id(descendant): tuple(
                descendant.named_parameters(recurse=False)
            )
            for descendant in descendants
        },
    )
    for descendant in descendants:
        setattr(descendant, _ROLLOUT_GENERATION_ATTRIBUTE, generation)
    return generation


def immutable_rollout_cache_generation(
    module: nn.Module,
) -> ImmutableRolloutCacheGeneration | None:
    """Return the explicit immutable generation attached to ``module``."""
    marker = getattr(module, _ROLLOUT_GENERATION_ATTRIBUTE, None)
    if marker is None:
        return None
    if not isinstance(marker, ImmutableRolloutCacheGeneration):
        raise TypeError("rollout cache generation marker has an invalid type")
    return marker


def validated_immutable_rollout_cache_generation(
    cache_owner: nn.Module,
    modules: tuple[nn.Module, ...],
) -> ImmutableRolloutCacheGeneration | None:
    """Return a generation only while relevant parameter bindings stay frozen."""
    generation = immutable_rollout_cache_generation(cache_owner)
    if generation is None:
        return None
    visited: set[int] = set()
    for module in modules:
        for descendant in module.modules():
            identity = id(descendant)
            if identity in visited:
                continue
            visited.add(identity)
            if descendant.training:
                return None
            expected = generation.parameter_bindings.get(identity)
            if expected is None:
                return None
            actual = tuple(descendant.named_parameters(recurse=False))
            if len(actual) != len(expected) or any(
                actual_name != expected_name or actual_parameter is not expected_parameter
                for (actual_name, actual_parameter), (
                    expected_name,
                    expected_parameter,
                ) in zip(actual, expected, strict=True)
            ):
                return None
            if any(parameter.requires_grad for _name, parameter in actual):
                return None
    return generation


__all__ = [
    "ImmutableRolloutCacheGeneration",
    "immutable_rollout_cache_generation",
    "mark_immutable_rollout_cache_generation",
    "validated_immutable_rollout_cache_generation",
]
