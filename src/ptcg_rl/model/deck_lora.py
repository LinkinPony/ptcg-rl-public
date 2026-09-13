"""Low-rank primitives for exact-deck routed specialization."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Protocol, cast
from weakref import ref

import torch
from torch import Tensor, nn
from torch.nn import functional


class ModuleKeyLike(Protocol):
    """Structural input carrying a stable private module key."""

    @property
    def module_key(self) -> str:
        """Return the stable module-bank key."""
        ...


class RowRouteLike(ModuleKeyLike, Protocol):
    """Structural route selecting batch rows for one private module."""

    @property
    def row_indices(self) -> Tensor:
        """Return one-dimensional batch-row indices."""
        ...


@dataclass(frozen=True)
class LowRankMergeMetadata:
    """Shape and scaling information needed for offline weight merging."""

    in_features: int
    out_features: int
    rank: int
    alpha: float
    scaling: float


@dataclass(frozen=True)
class RoutedLoRAMergeSpec:
    """One portable routed-bank target and its unchanged base weight key."""

    target_id: str
    base_weight_key: str
    adapter_bank_prefix: str

    def factor_key(self, module_key: str, factor: str) -> str:
        """Return the state-dict key for one selected A or B factor."""
        if factor not in {"lora_a", "lora_b"}:
            raise ValueError("LoRA factor must be lora_a or lora_b")
        return f"{self.adapter_bank_prefix}.{module_key}.{factor}.weight"


@dataclass(frozen=True)
class RoutedLoRADispatch:
    """Reusable active-row layout shared by every bank in one model pass."""

    module_keys: tuple[str, ...]
    row_indices: Tensor
    row_counts: tuple[int, ...]
    device_row_counts: Tensor
    cumulative_row_counts: Tensor
    disjoint_rows: bool = True
    contiguous_prefix: bool = False
    _element_offsets: dict[int, Tensor] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _element_counts: dict[int, Tensor] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _padded_element_indices: dict[int, tuple[int, Tensor]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _stacked_weight_cache: dict[int, tuple[Tensor, Tensor]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _expanded_dispatch_cache: dict[
        tuple[str, ...],
        RoutedLoRADispatch,
    ] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_routes(
        cls,
        routes: Sequence[RowRouteLike],
        *,
        disjoint_rows: bool,
    ) -> RoutedLoRADispatch | None:
        """Build one route-major layout without touching inactive experts."""
        for route in routes:
            if route.row_indices.ndim != 1:
                raise ValueError("LoRA route row indices must be one-dimensional")
        active = tuple(route for route in routes if route.row_indices.numel() > 0)
        if not active:
            return None
        device = active[0].row_indices.device
        for route in active:
            if route.row_indices.device != device:
                raise ValueError("LoRA route row indices must share one device")
        row_counts = tuple(int(route.row_indices.numel()) for route in active)
        row_count_tensor = _copy_host_integers(
            row_counts,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            module_keys=tuple(route.module_key for route in active),
            row_indices=torch.cat([route.row_indices for route in active]),
            row_counts=row_counts,
            device_row_counts=row_count_tensor,
            cumulative_row_counts=row_count_tensor.cumsum(0, dtype=torch.int32),
            disjoint_rows=disjoint_rows,
        )

    @classmethod
    def from_host_routes(
        cls,
        routes: Sequence[tuple[str, Sequence[int]]],
        *,
        device: torch.device | str,
        disjoint_rows: bool,
    ) -> RoutedLoRADispatch | None:
        """Build one route layout with a single flat host-index transfer."""
        active = tuple(
            (module_key, tuple(int(row) for row in rows))
            for module_key, rows in routes
            if len(rows) > 0
        )
        if not active:
            return None
        module_keys = tuple(module_key for module_key, _rows in active)
        if len(module_keys) != len(set(module_keys)):
            raise ValueError("LoRA host routes must use unique module keys")
        row_counts = tuple(len(rows) for _module_key, rows in active)
        flat_rows = tuple(row for _module_key, rows in active for row in rows)
        resolved_device = torch.device(device)
        row_indices = _copy_host_integers(
            flat_rows,
            dtype=torch.long,
            device=resolved_device,
        )
        device_row_counts = _copy_host_integers(
            row_counts,
            dtype=torch.int32,
            device=resolved_device,
        )
        return cls(
            module_keys=module_keys,
            row_indices=row_indices,
            row_counts=row_counts,
            device_row_counts=device_row_counts,
            cumulative_row_counts=device_row_counts.cumsum(
                0,
                dtype=torch.int32,
            ),
            disjoint_rows=disjoint_rows,
        )

    def element_offsets(self, elements_per_row: int) -> Tensor:
        """Return cached cumulative grouped-GEMM offsets for one leading shape."""
        if elements_per_row <= 0:
            raise ValueError("LoRA elements_per_row must be positive")
        cached = self._element_offsets.get(elements_per_row)
        if cached is None:
            cached = self.cumulative_row_counts * elements_per_row
            self._element_offsets[elements_per_row] = cached
        return cached

    def element_counts(self, elements_per_row: int) -> Tensor:
        """Return cached int64 element counts without a host tensor transfer."""
        if elements_per_row <= 0:
            raise ValueError("LoRA elements_per_row must be positive")
        cached = self._element_counts.get(elements_per_row)
        if cached is None:
            cached = self.device_row_counts.to(dtype=torch.long)
            if elements_per_row != 1:
                cached = cached * elements_per_row
            self._element_counts[elements_per_row] = cached
        return cached

    def padded_element_indices(
        self,
        elements_per_row: int,
    ) -> tuple[int, Tensor]:
        """Return a cached route-padded layout for portable batched GEMMs."""
        if elements_per_row <= 0:
            raise ValueError("LoRA elements_per_row must be positive")
        cached = self._padded_element_indices.get(elements_per_row)
        if cached is not None:
            return cached
        max_rows = max(self.row_counts)
        indices = tuple(
            (route_index * max_rows + route_row) * elements_per_row + element_index
            for route_index, row_count in enumerate(self.row_counts)
            for route_row in range(row_count)
            for element_index in range(elements_per_row)
        )
        padded_indices = _copy_host_integers(
            indices,
            dtype=torch.long,
            device=self.row_indices.device,
        )
        cached = (max_rows, padded_indices)
        self._padded_element_indices[elements_per_row] = cached
        return cached

    def with_empty_routes(
        self,
        module_keys: Sequence[str],
    ) -> RoutedLoRADispatch:
        """Expand to one stable bank order while retaining inactive zero groups."""
        target_keys = tuple(module_keys)
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("expanded LoRA routes must use unique module keys")
        if target_keys == self.module_keys:
            return self
        cached = self._expanded_dispatch_cache.get(target_keys)
        if cached is not None:
            return cached
        active_keys = frozenset(self.module_keys)
        if not active_keys.issubset(target_keys):
            raise ValueError("expanded LoRA routes omitted an active module key")
        target_active_keys = tuple(
            module_key for module_key in target_keys if module_key in active_keys
        )
        row_views = torch.split(self.row_indices, list(self.row_counts))
        rows_by_key = dict(zip(self.module_keys, row_views, strict=True))
        if target_active_keys == self.module_keys:
            row_indices = self.row_indices
        else:
            if self.contiguous_prefix:
                raise ValueError(
                    "expanded contiguous LoRA routes cannot reorder active groups"
                )
            row_indices = torch.cat(
                tuple(rows_by_key[module_key] for module_key in target_active_keys)
            )
        counts_by_key = dict(zip(self.module_keys, self.row_counts, strict=True))
        row_counts = tuple(
            counts_by_key.get(module_key, 0) for module_key in target_keys
        )
        device_row_counts = _copy_host_integers(
            row_counts,
            dtype=torch.int32,
            device=self.row_indices.device,
        )
        expanded = RoutedLoRADispatch(
            module_keys=target_keys,
            row_indices=row_indices,
            row_counts=row_counts,
            device_row_counts=device_row_counts,
            cumulative_row_counts=device_row_counts.cumsum(
                0,
                dtype=torch.int32,
            ),
            disjoint_rows=self.disjoint_rows,
            contiguous_prefix=self.contiguous_prefix,
        )
        self._expanded_dispatch_cache[target_keys] = expanded
        return expanded

    def stacked_weights(
        self,
        bank: nn.Module,
        factory: Callable[[], tuple[Tensor, Tensor]],
        *,
        cache_for_backward: bool,
    ) -> tuple[Tensor, Tensor]:
        """Return one bank's stacked factors, cached only for this backward."""
        if not cache_for_backward or not bank.training or not torch.is_grad_enabled():
            return factory()
        cache_key = id(bank)
        cached = self._stacked_weight_cache.get(cache_key)
        if cached is not None:
            return cached
        stacked = factory()
        eviction_tensor = next(
            (factor for factor in stacked if factor.requires_grad),
            None,
        )
        if eviction_tensor is None:
            return stacked
        self._stacked_weight_cache[cache_key] = stacked

        dispatch_ref = ref(self)

        def evict_after_backward(gradient: Tensor) -> Tensor:
            dispatch = dispatch_ref()
            if dispatch is not None:
                dispatch._stacked_weight_cache.pop(cache_key, None)
            return gradient

        eviction_tensor.register_hook(  # type: ignore[no-untyped-call]
            evict_after_backward
        )
        return stacked

    def as_contiguous_prefix(self) -> RoutedLoRADispatch:
        """Return the equivalent layout after a route-major batch permutation."""
        return RoutedLoRADispatch(
            module_keys=self.module_keys,
            row_indices=torch.arange(
                int(self.row_indices.numel()),
                dtype=torch.long,
                device=self.row_indices.device,
            ),
            row_counts=self.row_counts,
            device_row_counts=self.device_row_counts,
            cumulative_row_counts=self.cumulative_row_counts,
            disjoint_rows=True,
            contiguous_prefix=True,
            _element_offsets=self._element_offsets,
            _element_counts=self._element_counts,
            _padded_element_indices=self._padded_element_indices,
            _stacked_weight_cache=self._stacked_weight_cache,
        )


def _copy_host_integers(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    """Copy one integer vector, using pinned asynchronous CUDA staging."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=resolved_device)
    host = torch.tensor(
        values,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    return host.to(device=resolved_device, non_blocking=True)


@dataclass(frozen=True)
class _PreparedRoute:
    """Internal protocol adapter used by the portable fallback path."""

    module_key: str
    row_indices: Tensor


class LowRankAdapter(nn.Module):
    """Zero-output ``B(A(x))`` update for an external linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: float | None = None,
    ) -> None:
        """Initialize a normal A projection and an inert B projection."""
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("LoRA feature sizes and rank must be positive")
        resolved_alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(resolved_alpha) or resolved_alpha <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = resolved_alpha
        self.scaling = resolved_alpha / float(rank)
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    @property
    def merge_metadata(self) -> LowRankMergeMetadata:
        """Return immutable metadata describing the mergeable update."""
        return LowRankMergeMetadata(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            scaling=self.scaling,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a low-rank delta over arbitrary leading dimensions."""
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.in_features:
            raise ValueError(f"LoRA inputs must end in width {self.in_features}")
        output = cast(Tensor, self.lora_b(self.lora_a(inputs)))
        if self.scaling == 1.0:
            return output
        return output * self.scaling

    def delta_weight(self) -> Tensor:
        """Return the merge-ready ``[out_features, in_features]`` update."""
        return torch.matmul(self.lora_b.weight, self.lora_a.weight) * self.scaling


class RoutedLinearLoRA(nn.Module):
    """Active-only exact-route LoRA bank for an external base linear layer."""

    def __init__(
        self,
        routes: Sequence[ModuleKeyLike],
        *,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        """Build one physically independent low-rank adapter per route key."""
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("LoRA feature sizes and rank must be positive")
        resolved_alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(resolved_alpha) or resolved_alpha <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        module_keys = tuple(sorted({route.module_key for route in routes}))
        if any(not module_key for module_key in module_keys):
            raise ValueError("LoRA route module keys must be non-empty")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = resolved_alpha
        self.adapters = nn.ModuleDict(
            {
                module_key: LowRankAdapter(
                    in_features,
                    out_features,
                    rank,
                    alpha=resolved_alpha,
                )
                for module_key in module_keys
            }
        )

    def forward(
        self,
        inputs: Tensor,
        routes: Sequence[RowRouteLike],
        *,
        base: nn.Linear,
        dispatch: RoutedLoRADispatch | None = None,
        cache_stacked_weights: bool = False,
    ) -> Tensor:
        """Apply an external base linear plus active exact-route deltas."""
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"routed LoRA inputs must have shape [B, ..., {self.in_features}]"
            )
        if (
            base.in_features != self.in_features
            or base.out_features != self.out_features
        ):
            raise ValueError("external base linear shape does not match LoRA bank")
        output = cast(Tensor, base(inputs))
        return self.add_routed_delta_(
            output,
            inputs,
            routes,
            dispatch=dispatch,
            cache_stacked_weights=cache_stacked_weights,
        )

    def add_routed_delta_(
        self,
        output: Tensor,
        inputs: Tensor,
        routes: Sequence[RowRouteLike],
        *,
        dispatch: RoutedLoRADispatch | None = None,
        cache_stacked_weights: bool = False,
    ) -> Tensor:
        """Add active route deltas directly to an aligned output tensor."""
        expected_shape = (*inputs.shape[:-1], self.out_features)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                "routed LoRA output must have shape "
                f"{expected_shape}, got {tuple(output.shape)}"
            )
        routed = self._routed_contributions(
            inputs,
            routes,
            dispatch=dispatch,
            cache_stacked_weights=cache_stacked_weights,
        )
        if routed is None:
            return output
        row_indices, contributions = routed
        if dispatch is not None and dispatch.contiguous_prefix:
            output.narrow(0, 0, int(row_indices.numel())).add_(
                contributions.to(dtype=output.dtype)
            )
            return output
        if dispatch is not None and dispatch.disjoint_rows:
            return output.index_add_(
                0,
                row_indices,
                contributions.to(dtype=output.dtype),
            )
        selected_output = output.index_select(0, row_indices)
        updated_output = selected_output + contributions.to(dtype=output.dtype)
        return output.index_copy_(
            0,
            row_indices,
            updated_output,
        )

    def routed_delta(
        self,
        inputs: Tensor,
        routes: Sequence[RowRouteLike],
        *,
        output_dtype: torch.dtype | None = None,
        dispatch: RoutedLoRADispatch | None = None,
        cache_stacked_weights: bool = False,
    ) -> Tensor:
        """Return active route deltas without evaluating an external base."""
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"routed LoRA inputs must have shape [B, ..., {self.in_features}]"
            )
        residual = inputs.new_zeros(
            (*inputs.shape[:-1], self.out_features),
            dtype=inputs.dtype if output_dtype is None else output_dtype,
        )
        return self.add_routed_delta_(
            residual,
            inputs,
            routes,
            dispatch=dispatch,
            cache_stacked_weights=cache_stacked_weights,
        )

    def _routed_contributions(
        self,
        inputs: Tensor,
        routes: Sequence[RowRouteLike],
        *,
        dispatch: RoutedLoRADispatch | None = None,
        cache_stacked_weights: bool = False,
    ) -> tuple[Tensor, Tensor] | None:
        """Return row indices and deltas ready for one scatter operation."""
        if dispatch is not None:
            return self._prepared_routed_contributions(
                inputs,
                dispatch,
                cache_stacked_weights=cache_stacked_weights,
            )
        selected_inputs: list[Tensor] = []
        selected_routes: list[RowRouteLike] = []
        adapters: list[LowRankAdapter] = []
        for route in routes:
            if route.module_key not in self.adapters:
                raise KeyError(f"unknown LoRA route module: {route.module_key}")
            if route.row_indices.ndim != 1:
                raise ValueError("LoRA route row indices must be one-dimensional")
            if route.row_indices.numel() == 0:
                continue
            selected_inputs.append(inputs.index_select(0, route.row_indices))
            selected_routes.append(route)
            adapters.append(self.adapter(route.module_key))
        if not selected_inputs:
            return None
        if len(selected_inputs) == 1:
            return (
                selected_routes[0].row_indices,
                adapters[0](selected_inputs[0]),
            )

        flattened = [
            selected.reshape(-1, self.in_features) for selected in selected_inputs
        ]
        element_counts = [int(selected.shape[0]) for selected in flattened]
        max_elements = max(element_counts)
        if max_elements * len(flattened) > 2 * sum(element_counts):
            return (
                torch.cat([route.row_indices for route in selected_routes]),
                torch.cat(
                    [
                        adapter(selected)
                        for selected, adapter in zip(
                            selected_inputs,
                            adapters,
                            strict=True,
                        )
                    ]
                ),
            )

        padded_inputs = torch.stack(
            [
                functional.pad(
                    selected,
                    (0, 0, 0, max_elements - element_count),
                )
                for selected, element_count in zip(
                    flattened,
                    element_counts,
                    strict=True,
                )
            ]
        )
        lora_a = torch.stack([adapter.lora_a.weight for adapter in adapters])
        lora_b = torch.stack([adapter.lora_b.weight for adapter in adapters])
        hidden = torch.bmm(padded_inputs, lora_a.transpose(1, 2))
        contributions = torch.bmm(hidden, lora_b.transpose(1, 2))
        if adapters[0].scaling != 1.0:
            contributions = contributions * adapters[0].scaling
        row_counts = [int(selected.shape[0]) for selected in selected_inputs]
        max_rows = max(row_counts)
        shaped_contributions = contributions.reshape(
            len(selected_inputs),
            max_rows,
            *inputs.shape[1:-1],
            self.out_features,
        )
        return (
            torch.cat([route.row_indices for route in selected_routes]),
            torch.cat(
                [
                    route_contributions[:row_count]
                    for route_contributions, row_count in zip(
                        shaped_contributions,
                        row_counts,
                        strict=True,
                    )
                ]
            ),
        )

    def _prepared_routed_contributions(
        self,
        inputs: Tensor,
        dispatch: RoutedLoRADispatch,
        *,
        cache_stacked_weights: bool,
    ) -> tuple[Tensor, Tensor] | None:
        """Evaluate a cached route layout with one gather and no padded rows."""
        adapters = tuple(
            self.adapter(module_key) for module_key in dispatch.module_keys
        )
        if not adapters:
            return None
        return _prepared_low_rank_contributions(
            inputs,
            dispatch,
            adapters,
            cache_owner=self,
            cache_stacked_weights=cache_stacked_weights,
        )

    def adapter(self, module_key: str) -> LowRankAdapter:
        """Return one route's adapter for inspection or offline merging."""
        if module_key not in self.adapters:
            raise KeyError(f"unknown LoRA route module: {module_key}")
        return cast(LowRankAdapter, self.adapters[module_key])

    def delta_weight(self, module_key: str) -> Tensor:
        """Return one route's merge-ready weight delta."""
        return self.adapter(module_key).delta_weight()

    def merge_metadata(self, module_key: str) -> LowRankMergeMetadata:
        """Return one route's immutable merge metadata."""
        return self.adapter(module_key).merge_metadata


def routed_low_rank_contributions(
    inputs: Tensor,
    dispatch: RoutedLoRADispatch,
    adapters: Sequence[LowRankAdapter],
    *,
    cache_owner: nn.Module,
    cache_stacked_weights: bool = False,
) -> tuple[Tensor, Tensor]:
    """Evaluate externally owned exact factors with a reusable route layout."""
    if len(adapters) != len(dispatch.module_keys) or not adapters:
        raise ValueError("routed exact factors must align with active route keys")
    return _prepared_low_rank_contributions(
        inputs,
        dispatch,
        adapters,
        cache_owner=cache_owner,
        cache_stacked_weights=cache_stacked_weights,
    )


def _prepared_low_rank_contributions(
    inputs: Tensor,
    dispatch: RoutedLoRADispatch,
    adapters: Sequence[LowRankAdapter],
    *,
    cache_owner: nn.Module,
    cache_stacked_weights: bool,
) -> tuple[Tensor, Tensor]:
    """Run one gathered low-rank bank through grouped or padded GEMMs."""
    first = adapters[0]
    if any(
        adapter.in_features != first.in_features
        or adapter.out_features != first.out_features
        or adapter.rank != first.rank
        or adapter.scaling != first.scaling
        for adapter in adapters
    ):
        raise ValueError("routed exact factors must share one projection geometry")
    if inputs.ndim < 2 or int(inputs.shape[-1]) != first.in_features:
        raise ValueError("routed exact-factor inputs have an invalid shape")
    selected = (
        inputs.narrow(0, 0, int(dispatch.row_indices.numel()))
        if dispatch.contiguous_prefix
        else inputs.index_select(0, dispatch.row_indices)
    )
    if len(adapters) == 1:
        return dispatch.row_indices, adapters[0](selected)
    if not _can_use_native_grouped_mm(selected, adapters):
        flattened_rows = tuple(
            selected_rows.reshape(-1, first.in_features)
            for selected_rows in torch.split(selected, list(dispatch.row_counts))
        )
        element_counts = tuple(int(row.shape[0]) for row in flattened_rows)
        max_elements = max(element_counts)
        padded = torch.stack(
            tuple(
                functional.pad(
                    row,
                    (0, 0, 0, max_elements - element_count),
                )
                for row, element_count in zip(
                    flattened_rows,
                    element_counts,
                    strict=True,
                )
            )
        )
        lora_a = torch.stack([adapter.lora_a.weight for adapter in adapters])
        lora_b = torch.stack([adapter.lora_b.weight for adapter in adapters])
        hidden = torch.bmm(padded, lora_a.transpose(1, 2))
        result = torch.bmm(hidden, lora_b.transpose(1, 2))
        if first.scaling != 1.0:
            result = result * first.scaling
        contributions = torch.cat(
            tuple(
                row[:element_count]
                for row, element_count in zip(
                    result,
                    element_counts,
                    strict=True,
                )
            )
        )
        return (
            dispatch.row_indices,
            contributions.reshape(
                int(dispatch.row_indices.numel()),
                *inputs.shape[1:-1],
                first.out_features,
            ),
        )

    elements_per_row = math.prod(inputs.shape[1:-1])
    aligned_in_features = _aligned_grouped_mm_width(first.in_features)
    aligned_rank = _aligned_grouped_mm_width(first.rank)
    aligned_out_features = _aligned_grouped_mm_width(first.out_features)
    flattened = selected.reshape(-1, first.in_features).to(dtype=torch.bfloat16)
    if aligned_in_features != first.in_features:
        flattened = functional.pad(
            flattened,
            (0, aligned_in_features - first.in_features),
        )

    def stack_weights() -> tuple[Tensor, Tensor]:
        lora_a = torch.stack(
            [adapter.lora_a.weight.transpose(0, 1) for adapter in adapters]
        ).to(dtype=torch.bfloat16)
        lora_b = torch.stack(
            [adapter.lora_b.weight.transpose(0, 1) for adapter in adapters]
        ).to(dtype=torch.bfloat16)
        if aligned_in_features != first.in_features or aligned_rank != first.rank:
            lora_a = functional.pad(
                lora_a,
                (
                    0,
                    aligned_rank - first.rank,
                    0,
                    aligned_in_features - first.in_features,
                ),
            )
        if aligned_rank != first.rank or aligned_out_features != first.out_features:
            lora_b = functional.pad(
                lora_b,
                (
                    0,
                    aligned_out_features - first.out_features,
                    0,
                    aligned_rank - first.rank,
                ),
            )
        return lora_a, lora_b

    lora_a, lora_b = dispatch.stacked_weights(
        cache_owner,
        stack_weights,
        cache_for_backward=cache_stacked_weights,
    )
    offsets = dispatch.element_offsets(elements_per_row)
    grouped_mm = cast(Any, torch._grouped_mm)
    hidden = grouped_mm(flattened, lora_a, offs=offsets)
    contributions = grouped_mm(hidden, lora_b, offs=offsets)
    if aligned_out_features != first.out_features:
        contributions = contributions.narrow(1, 0, first.out_features)
    if first.scaling != 1.0:
        contributions = contributions * first.scaling
    return (
        dispatch.row_indices,
        contributions.reshape(
            int(dispatch.row_indices.numel()),
            *inputs.shape[1:-1],
            first.out_features,
        ),
    )


def _split_dispatch_rows(dispatch: RoutedLoRADispatch) -> tuple[Tensor, ...]:
    """Recover per-expert row views only for the non-grouped fallback."""
    return tuple(torch.split(dispatch.row_indices, list(dispatch.row_counts)))


def _can_use_native_grouped_mm(
    inputs: Tensor,
    adapters: Sequence[LowRankAdapter],
) -> bool:
    """Return whether this call can use the native sm90 BF16 grouped kernel."""
    if inputs.device.type != "cuda" or not hasattr(torch, "_grouped_mm"):
        return False
    if not _grouped_mm_device_supported(inputs.get_device()):
        return False
    autocast_bf16 = torch.is_autocast_enabled("cuda") and (
        torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    explicit_bf16 = inputs.dtype == torch.bfloat16 and all(
        adapter.lora_a.weight.dtype == torch.bfloat16
        and adapter.lora_b.weight.dtype == torch.bfloat16
        for adapter in adapters
    )
    return autocast_bf16 or explicit_bf16


def _aligned_grouped_mm_width(width: int) -> int:
    """Round one BF16 matrix width up to the grouped-kernel alignment."""
    alignment = 16 // torch.bfloat16.itemsize
    return math.ceil(width / alignment) * alignment


@cache
def _grouped_mm_device_supported(device_index: int) -> bool:
    """Cache the immutable CUDA capability check outside the projection loop."""
    return torch.cuda.get_device_capability(device_index) >= (9, 0)


def routed_lora_merge_specs(
    *,
    transformer_layer_indices: Sequence[int],
    transformer_targets: Sequence[str],
    policy_targets: Sequence[str],
) -> tuple[RoutedLoRAMergeSpec, ...]:
    """Return the complete deterministic merge manifest for architecture v2."""
    transformer_base_suffixes = {
        "attention_qkv": "self_attn.in_proj_weight",
        "attention_output": "self_attn.out_proj.weight",
        "ffn_input": "linear1.weight",
        "ffn_output": "linear2.weight",
    }
    policy_base_keys = {
        "scalar_input": "policy_head.scalar_projection.0.weight",
        "scalar_output": "policy_head.scalar_projection.3.weight",
        "dynamic_input": "policy_head.dynamic_effect_projection.0.weight",
        "dynamic_output": "policy_head.dynamic_effect_projection.3.weight",
        "option_projection": "policy_head.option_projection.weight",
        "selected_projection": "policy_head.selected_projection.weight",
        "ordered_history_projection": ("policy_head.ordered_history_projection.weight"),
        "decoder_cardinality_projection": (
            "policy_head.decoder_cardinality_projection.weight"
        ),
        "query_input": "policy_head.query_projection.0.weight",
        "query_output": "policy_head.query_projection.3.weight",
    }
    specs: list[RoutedLoRAMergeSpec] = []
    for layer_index in transformer_layer_indices:
        layer_key = f"layer_{layer_index:02d}"
        base_prefix = f"state_encoder.transformer.layers.{layer_index}"
        bank_prefix = f"state_encoder.private_lora.{layer_key}"
        for target in transformer_targets:
            specs.append(
                RoutedLoRAMergeSpec(
                    target_id=f"transformer.{layer_index}.{target}",
                    base_weight_key=(
                        f"{base_prefix}.{transformer_base_suffixes[target]}"
                    ),
                    adapter_bank_prefix=f"{bank_prefix}.{target}.adapters",
                )
            )
    for target in policy_targets:
        specs.append(
            RoutedLoRAMergeSpec(
                target_id=f"policy.{target}",
                base_weight_key=policy_base_keys[target],
                adapter_bank_prefix=f"policy_head.private_lora.{target}.adapters",
            )
        )
    return tuple(specs)


__all__ = [
    "LowRankAdapter",
    "LowRankMergeMetadata",
    "ModuleKeyLike",
    "RoutedLinearLoRA",
    "RoutedLoRADispatch",
    "RoutedLoRAMergeSpec",
    "RowRouteLike",
    "routed_lora_merge_specs",
]
