"""Deterministic exact-deck routing for small output residuals."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Sequence, Set
from dataclasses import dataclass, field
from functools import cached_property
from threading import RLock
from typing import Self

import torch
from torch import Tensor, nn

from ptcg_rl.decks.registry import DeckExpertRoute
from ptcg_rl.model.deck_conditioning import DeckRouteGroup
from ptcg_rl.model.deck_lora import RoutedLoRADispatch
from ptcg_rl.model.simple_stateless.config import SimpleStatelessModelConfig
from ptcg_rl.model.simple_stateless.exact_residual_grouped import (
    apply_grouped_rollout_residual as _apply_grouped_rollout_residual,
)
from ptcg_rl.model.simple_stateless.exact_residual_grouped import (
    can_use_grouped_rollout_residual as _can_use_grouped_rollout_residual,
)

_RESOLVER_PARTITION_PROOF = object()
_ROUTE_LOOKUP_CACHE: OrderedDict[str, dict[str, DeckExpertRoute]] = OrderedDict()
_ROUTE_LOOKUP_CACHE_LOCK = RLock()
_MAX_ROUTE_LOOKUP_CACHE_ENTRIES = 32

_RouteRowSignature = tuple[
    int,
    int,
    int,
    tuple[int, ...],
    torch.dtype,
    torch.device,
]


@dataclass(frozen=True)
class _ValidatedExactPartition:
    """Cached structural evidence for one route plan."""

    module_keys: frozenset[str]
    row_device: torch.device | None
    row_signatures: tuple[_RouteRowSignature, ...]


@dataclass(frozen=True)
class _ResolverPartitionEvidence:
    """Host-validated evidence available only to the official resolvers."""

    proof: object
    partition: _ValidatedExactPartition
    grouped_dispatch: RoutedLoRADispatch | None
    host_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SimpleExactRoutePlan:
    """Disjoint exact-route row groups with no active generic fallback."""

    batch_size: int
    resolved_registry_sha256: str
    groups: tuple[DeckRouteGroup, ...]
    allow_unrouted_rows: bool = False
    _resolver_evidence: _ResolverPartitionEvidence | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @cached_property
    def module_keys(self) -> tuple[str, ...]:
        """Return the immutable active route order."""
        return tuple(group.module_key for group in self.groups)

    @cached_property
    def grouped_dispatch(self) -> RoutedLoRADispatch | None:
        """Return the reusable route-major layout for grouped inference."""
        resolver_evidence = self._resolver_evidence
        if resolver_evidence is not None:
            if resolver_evidence.proof is not _RESOLVER_PARTITION_PROOF:
                raise RuntimeError("route plan resolver evidence is invalid")
            return resolver_evidence.grouped_dispatch
        return RoutedLoRADispatch.from_routes(
            self.groups,
            disjoint_rows=True,
        )

    @cached_property
    def host_groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Return route rows without synchronizing official CUDA plans."""
        resolver_evidence = self._resolver_evidence
        if resolver_evidence is not None:
            if resolver_evidence.proof is not _RESOLVER_PARTITION_PROOF:
                raise RuntimeError("route plan resolver evidence is invalid")
            rows = resolver_evidence.host_rows
        else:
            rows = tuple(
                tuple(int(row) for row in group.row_indices.detach().cpu().tolist())
                for group in self.groups
            )
        return tuple(
            (group.module_key, group_rows)
            for group, group_rows in zip(self.groups, rows, strict=True)
        )

    @classmethod
    def _from_resolver(
        cls,
        *,
        batch_size: int,
        resolved_registry_sha256: str,
        groups: tuple[DeckRouteGroup, ...],
        host_rows: tuple[tuple[int, ...], ...],
        allow_unrouted_rows: bool,
        grouped_dispatch: RoutedLoRADispatch | None,
    ) -> Self:
        """Construct a plan whose partition was proven before device transfer."""
        if len(groups) != len(host_rows):
            raise RuntimeError("resolver route groups and host evidence differ")
        module_keys = tuple(group.module_key for group in groups)
        if len(module_keys) != len(set(module_keys)):
            raise RuntimeError("resolver produced duplicate route groups")
        flattened: list[int] = []
        row_device: torch.device | None = None
        for group, rows in zip(groups, host_rows, strict=True):
            if not rows or int(group.row_indices.numel()) != len(rows):
                raise RuntimeError("resolver produced inconsistent route rows")
            if group.row_indices.ndim != 1 or group.row_indices.dtype != torch.long:
                raise RuntimeError("resolver produced invalid route row storage")
            if row_device is None:
                row_device = group.row_indices.device
            elif group.row_indices.device != row_device:
                raise RuntimeError("resolver route rows use different devices")
            flattened.extend(rows)
        if any(row < 0 or row >= batch_size for row in flattened):
            raise RuntimeError("resolver produced an out-of-range route row")
        if len(flattened) != len(set(flattened)):
            raise RuntimeError("resolver produced duplicate route rows")
        if not allow_unrouted_rows and sorted(flattened) != list(range(batch_size)):
            raise RuntimeError("resolver did not cover the exact route batch")
        plan = cls(
            batch_size=batch_size,
            resolved_registry_sha256=resolved_registry_sha256,
            groups=groups,
            allow_unrouted_rows=allow_unrouted_rows,
        )
        object.__setattr__(
            plan,
            "_resolver_evidence",
            _ResolverPartitionEvidence(
                proof=_RESOLVER_PARTITION_PROOF,
                partition=_ValidatedExactPartition(
                    module_keys=frozenset(module_keys),
                    row_device=row_device,
                    row_signatures=tuple(
                        _route_row_signature(group.row_indices) for group in groups
                    ),
                ),
                grouped_dispatch=grouped_dispatch,
                host_rows=host_rows,
            ),
        )
        return plan

    def validate_exact_partition(
        self,
        *,
        expected_module_keys: Set[str] | None,
        device: torch.device | str,
        allow_unrouted_rows: bool = False,
    ) -> None:
        """Validate one immutable route partition against its model bank."""
        if self.allow_unrouted_rows != allow_unrouted_rows:
            raise ValueError("route plan unrouted-row mode is not authorized")
        partition = self._validated_partition
        if partition.row_signatures != tuple(
            _route_row_signature(group.row_indices) for group in self.groups
        ):
            raise ValueError("route plan row storage changed after validation")
        if expected_module_keys is not None and not partition.module_keys.issubset(
            expected_module_keys
        ):
            raise ValueError(
                "route plan references a module outside the model registry"
            )
        expected_device = torch.device(device)
        if partition.row_device is not None and partition.row_device != expected_device:
            raise ValueError("route plan row indices and model inputs differ in device")

    @cached_property
    def _validated_partition(self) -> _ValidatedExactPartition:
        """Materialize and validate row coverage once per immutable plan."""
        resolver_evidence = self._resolver_evidence
        if resolver_evidence is not None:
            if resolver_evidence.proof is not _RESOLVER_PARTITION_PROOF:
                raise RuntimeError("route plan resolver evidence is invalid")
            return resolver_evidence.partition
        if self.batch_size < 0:
            raise ValueError("route plan batch size cannot be negative")
        module_keys = tuple(group.module_key for group in self.groups)
        if len(module_keys) != len(set(module_keys)):
            raise ValueError("route plan module groups must be unique")
        row_device: torch.device | None = None
        row_groups: list[Tensor] = []
        for group in self.groups:
            rows = group.row_indices
            if rows.ndim != 1 or rows.dtype != torch.long:
                raise ValueError("route plan row indices must be one-dimensional int64")
            if int(rows.numel()) == 0:
                raise ValueError("route plan cannot contain an empty route group")
            if rows.is_meta:
                raise ValueError("route plan row indices cannot use the meta device")
            if row_device is None:
                row_device = rows.device
            elif rows.device != row_device:
                raise ValueError("route plan row groups must share one device")
            row_groups.append(rows)
        if row_groups:
            rows = torch.cat(row_groups)
            out_of_range = (rows < 0) | (rows >= self.batch_size)
            if bool(out_of_range.any().item()):
                raise ValueError("route plan row index is outside the batch")
            counts = torch.bincount(rows, minlength=self.batch_size)
            if self.allow_unrouted_rows:
                valid_partition = counts.le(1).all()
            else:
                valid_partition = counts.eq(1).all()
            if not bool(valid_partition.item()):
                raise ValueError(
                    "route plan rows must form the authorized disjoint partition"
                )
        elif not self.allow_unrouted_rows and self.batch_size != 0:
            raise ValueError("exact route plan does not cover the batch")
        return _ValidatedExactPartition(
            module_keys=frozenset(module_keys),
            row_device=row_device,
            row_signatures=tuple(
                _route_row_signature(group.row_indices) for group in self.groups
            ),
        )


def resolve_simple_exact_routes(
    deck_signatures: Sequence[str],
    config: SimpleStatelessModelConfig,
    *,
    device: torch.device | str,
) -> SimpleExactRoutePlan:
    """Resolve every row to its dedicated immutable exact strategy."""
    if config.resolved_registry_sha256 is None or not config.exact_routes:
        raise ValueError("simple stateless model requires a resolved exact registry")
    by_signature = _routes_by_signature(config)
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for row, signature in enumerate(deck_signatures):
        route = by_signature.get(signature)
        if route is None:
            raise ValueError(
                "active exact deck is absent from the simple stateless registry"
            )
        grouped[route.module_key].append(row)
    ordered_groups = tuple(sorted(grouped.items()))
    row_groups, grouped_dispatch = _resolved_route_layout(
        ordered_groups,
        device=device,
    )
    covered = sum(int(group.row_indices.numel()) for group in row_groups)
    if covered != len(deck_signatures):
        raise RuntimeError("exact route plan does not cover every batch row")
    return SimpleExactRoutePlan._from_resolver(
        batch_size=len(deck_signatures),
        resolved_registry_sha256=config.resolved_registry_sha256,
        groups=row_groups,
        host_rows=tuple(tuple(rows) for _module_key, rows in ordered_groups),
        allow_unrouted_rows=False,
        grouped_dispatch=grouped_dispatch,
    )


def resolve_simple_pretraining_routes(
    deck_signatures: Sequence[str],
    config: SimpleStatelessModelConfig,
    *,
    device: torch.device | str,
) -> SimpleExactRoutePlan:
    """Route matching offline rows while leaving foreign decks shared-only.

    Public replay corpora contain exact decks outside the active RL registry.
    Those rows remain useful for the shared model, but assigning them to an
    unrelated private residual would corrupt the exact-strategy identity.  This
    explicitly offline route plan therefore applies private modules only to
    exact signature matches and emits a zero residual for every other row.
    """
    if config.resolved_registry_sha256 is None or not config.exact_routes:
        raise ValueError("simple stateless model requires a resolved exact registry")
    by_signature = _routes_by_signature(config)
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for row, signature in enumerate(deck_signatures):
        route = by_signature.get(signature)
        if route is not None:
            grouped[route.module_key].append(row)
    ordered_groups = tuple(sorted(grouped.items()))
    row_groups, grouped_dispatch = _resolved_route_layout(
        ordered_groups,
        device=device,
    )
    return SimpleExactRoutePlan._from_resolver(
        batch_size=len(deck_signatures),
        resolved_registry_sha256=config.resolved_registry_sha256,
        groups=row_groups,
        host_rows=tuple(tuple(rows) for _module_key, rows in ordered_groups),
        allow_unrouted_rows=True,
        grouped_dispatch=grouped_dispatch,
    )


def _resolved_route_layout(
    ordered_groups: Sequence[tuple[str, Sequence[int]]],
    *,
    device: torch.device | str,
) -> tuple[tuple[DeckRouteGroup, ...], RoutedLoRADispatch | None]:
    """Create route-group views over one eagerly transferred flat layout."""
    dispatch = RoutedLoRADispatch.from_host_routes(
        ordered_groups,
        device=device,
        disjoint_rows=True,
    )
    if dispatch is None:
        return (), None
    row_views = torch.split(dispatch.row_indices, list(dispatch.row_counts))
    groups = tuple(
        DeckRouteGroup(
            module_key=module_key,
            row_indices=rows,
        )
        for module_key, rows in zip(
            dispatch.module_keys,
            row_views,
            strict=True,
        )
    )
    return groups, dispatch


def apply_exact_residual(
    inputs: Tensor,
    plan: SimpleExactRoutePlan,
    modules: nn.ModuleDict,
) -> Tensor:
    """Apply each private module only to the rows assigned to that route."""
    if inputs.shape[0] != plan.batch_size:
        raise ValueError("exact route plan and input batch differ")
    plan.validate_exact_partition(
        expected_module_keys=None,
        device=inputs.device,
        allow_unrouted_rows=plan.allow_unrouted_rows,
    )
    if any(module_key not in modules for module_key in plan.module_keys):
        raise ValueError("exact route module is missing from the model")
    covered_rows = sum(int(group.row_indices.numel()) for group in plan.groups)
    can_use_grouped = plan.allow_unrouted_rows or covered_rows == plan.batch_size
    if can_use_grouped and _can_use_grouped_rollout_residual(
        inputs,
        plan.module_keys,
        modules,
    ):
        dispatch = plan.grouped_dispatch
        if dispatch is None:
            raise RuntimeError("grouped exact residual requires active routes")
        return _apply_grouped_rollout_residual(
            inputs,
            batch_size=plan.batch_size,
            dispatch=dispatch,
            modules=modules,
        )
    return _apply_exact_residual_fallback(inputs, plan, modules)


def _apply_exact_residual_fallback(
    inputs: Tensor,
    plan: SimpleExactRoutePlan,
    modules: nn.ModuleDict,
) -> Tensor:
    """Preserve the portable, differentiable per-route implementation."""
    output: Tensor | None = None
    covered_rows = 0
    for group in plan.groups:
        if group.module_key not in modules:
            raise ValueError("exact route module is missing from the model")
        residual = modules[group.module_key](
            inputs.index_select(0, group.row_indices)
        ).to(dtype=inputs.dtype)
        if output is None:
            output = inputs.new_zeros((plan.batch_size, *residual.shape[1:]))
        output = output.index_copy(0, group.row_indices, residual)
        covered_rows += int(group.row_indices.numel())
    if output is None:
        if not plan.allow_unrouted_rows:
            raise ValueError("exact route plan contains no route groups")
        try:
            template_module = next(iter(modules.values()))
        except StopIteration as error:
            raise ValueError("exact route modules are empty") from error
        template = template_module(inputs[:0]).to(dtype=inputs.dtype)
        return inputs.new_zeros((plan.batch_size, *template.shape[1:]))
    if not plan.allow_unrouted_rows and covered_rows != plan.batch_size:
        raise ValueError("exact route plan does not cover every batch row")
    return output


def _routes_by_signature(
    config: SimpleStatelessModelConfig,
) -> dict[str, DeckExpertRoute]:
    """Cache immutable registry lookup tables across learner microbatches."""
    registry_sha256 = config.resolved_registry_sha256
    if registry_sha256 is None:
        raise ValueError("route lookup requires a resolved registry")
    with _ROUTE_LOOKUP_CACHE_LOCK:
        cached = _ROUTE_LOOKUP_CACHE.get(registry_sha256)
        if cached is not None:
            _ROUTE_LOOKUP_CACHE.move_to_end(registry_sha256)
            return cached
        resolved = {route.signature: route for route in config.exact_routes}
        _ROUTE_LOOKUP_CACHE[registry_sha256] = resolved
        while len(_ROUTE_LOOKUP_CACHE) > _MAX_ROUTE_LOOKUP_CACHE_ENTRIES:
            _ROUTE_LOOKUP_CACHE.popitem(last=False)
        return resolved


def _route_row_signature(rows: Tensor) -> _RouteRowSignature:
    """Identify route-row replacement and in-place mutation without device sync."""
    try:
        version = rows._version  # noqa: SLF001 - detect in-place route mutation.
    except RuntimeError:
        # Inference tensors omit version counters; immutable storage metadata
        # still detects whole-tensor replacement without synchronizing values.
        version = -1
    return (
        id(rows),
        version,
        rows.data_ptr(),
        tuple(rows.shape),
        rows.dtype,
        rows.device,
    )
