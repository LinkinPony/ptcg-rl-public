"""Exact-deck compositional prompt and capsule adapters for architecture v2."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
from torch import Tensor, nn

from ptcg_rl.model.simple_stateless.layers import (
    initialize_simple_stateless_module,
)
from ptcg_rl.model.simple_stateless.packed import (
    PackedTokenBatch,
    SpecialTokenPositions,
)
from ptcg_rl.model.simple_stateless.routing import (
    SimpleExactRoutePlan,
    apply_exact_residual,
)
from ptcg_rl.model.simple_stateless.v2_rollout import (
    apply_capsule_stage_rollout,
    apply_prompt_rollout,
)

# These values are part of the immutable ``simple_stateless_v2`` topology.
# Changing one requires a new architecture discriminator.
SIMPLE_STATELESS_V2_CAPSULE_STAGES = (8, 17, 25, 34)
GENERALIST_SEQUENCE_V2_CAPSULE_STAGES = (5, 10, 15, 20)
SIMPLE_STATELESS_V2_SHARED_BASES = 4
SIMPLE_STATELESS_V2_BASIS_RANK = 16
SIMPLE_STATELESS_V2_EXACT_BOTTLENECK = 16

_POLICY_AND_SCRATCH_TOKENS = 9
_SEMANTIC_TOKENS = 10


class ZeroOutputBottleneckResidual(nn.Module):
    """Small pre-LN residual whose output projection starts at exactly zero."""

    def __init__(self, *, d_model: int, bottleneck_dim: int) -> None:
        """Initialize the reusable low-rank nonlinear residual."""
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one shape-preserving residual contribution."""
        return cast(Tensor, self.up(self.activation(self.down(self.norm(inputs)))))

    def zero_output(self) -> None:
        """Make the residual exactly inert while retaining its input features."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


class ExactSemanticPrompt(nn.Module):
    """One route's independent policy/scratch and value prompt deltas."""

    def __init__(self, *, d_model: int) -> None:
        """Create an exactly inert prompt without adding sequence tokens."""
        super().__init__()
        self.policy_and_scratch = nn.Parameter(
            torch.zeros(_POLICY_AND_SCRATCH_TOKENS, d_model)
        )
        self.value = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, inputs: Tensor) -> Tensor:
        """Expand this route's prompt over its assigned batch rows."""
        if inputs.ndim != 3 or int(inputs.shape[1]) != _SEMANTIC_TOKENS:
            raise ValueError("semantic prompt inputs must have shape [batch, 10, d]")
        prompt = torch.cat(
            (
                self.policy_and_scratch[:1],
                self.value,
                self.policy_and_scratch[1:],
            ),
            dim=0,
        )
        return prompt.to(dtype=inputs.dtype).unsqueeze(0).expand(
            int(inputs.shape[0]),
            -1,
            -1,
        )

    def zero_output(self) -> None:
        """Reset both independent prompt channels to exact identity."""
        nn.init.zeros_(self.policy_and_scratch)
        nn.init.zeros_(self.value)


class ExactCompositionalCapsule(nn.Module):
    """One route's basis mixture and exact policy/value bottlenecks."""

    def __init__(
        self,
        *,
        d_model: int,
        shared_bases: int,
        bottleneck_dim: int,
    ) -> None:
        """Initialize independent policy and value capsule parameters."""
        super().__init__()
        if shared_bases <= 0:
            raise ValueError("a compositional capsule requires shared bases")
        self.policy_coefficients = nn.Parameter(torch.zeros(shared_bases))
        self.value_coefficients = nn.Parameter(torch.zeros(shared_bases))
        self.policy_residual = ZeroOutputBottleneckResidual(
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
        )
        self.value_residual = ZeroOutputBottleneckResidual(
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Mix shared bases and add the route's exact residual.

        The input layout is ``[batch, semantic, source, d_model]``. Source zero
        contains the current semantic token and the remaining sources contain
        outputs from the shared low-rank bases.
        """
        if (
            inputs.ndim != 4
            or int(inputs.shape[1]) != _SEMANTIC_TOKENS
            or int(inputs.shape[2]) != int(self.policy_coefficients.numel()) + 1
        ):
            raise ValueError(
                "capsule inputs must have shape [batch, 10, bases + 1, d]"
            )
        semantic = inputs[:, :, 0]
        basis_outputs = inputs[:, :, 1:]
        policy_tokens = torch.cat((semantic[:, :1], semantic[:, 2:]), dim=1)
        policy_bases = torch.cat(
            (basis_outputs[:, :1], basis_outputs[:, 2:]),
            dim=1,
        )
        policy_delta = torch.einsum(
            "btkd,k->btd",
            policy_bases,
            torch.softmax(self.policy_coefficients.float(), dim=0).to(
                dtype=policy_bases.dtype
            ),
        )
        policy_delta = policy_delta + self.policy_residual(policy_tokens)
        value_token = semantic[:, 1:2]
        value_delta = torch.einsum(
            "btkd,k->btd",
            basis_outputs[:, 1:2],
            torch.softmax(self.value_coefficients.float(), dim=0).to(
                dtype=basis_outputs.dtype
            ),
        )
        value_delta = value_delta + self.value_residual(value_token)
        return torch.cat(
            (
                policy_delta[:, :1],
                value_delta,
                policy_delta[:, 1:],
            ),
            dim=1,
        )

    def zero_output(self) -> None:
        """Reset route-private bottleneck outputs to exact identity."""
        self.policy_residual.zero_output()
        self.value_residual.zero_output()


class CompositionalCapsuleStage(nn.Module):
    """One fixed trunk boundary with shared bases and exact route capsules."""

    def __init__(
        self,
        *,
        d_model: int,
        shared_bases: int,
        basis_rank: int,
        exact_bottleneck_dim: int,
        module_keys: tuple[str, ...],
    ) -> None:
        """Build policy/value basis banks and one capsule per exact route."""
        super().__init__()
        self.policy_bases = nn.ModuleList(
            ZeroOutputBottleneckResidual(
                d_model=d_model,
                bottleneck_dim=basis_rank,
            )
            for _ in range(shared_bases)
        )
        self.value_bases = nn.ModuleList(
            ZeroOutputBottleneckResidual(
                d_model=d_model,
                bottleneck_dim=basis_rank,
            )
            for _ in range(shared_bases)
        )
        self.exact_capsules = nn.ModuleDict(
            {
                module_key: ExactCompositionalCapsule(
                    d_model=d_model,
                    shared_bases=shared_bases,
                    bottleneck_dim=exact_bottleneck_dim,
                )
                for module_key in module_keys
            }
        )

    def forward(
        self,
        semantic_tokens: Tensor,
        *,
        route_plan: SimpleExactRoutePlan,
    ) -> Tensor:
        """Apply the shared basis bank and only each row's exact capsule."""
        if (
            semantic_tokens.ndim != 3
            or int(semantic_tokens.shape[1]) != _SEMANTIC_TOKENS
        ):
            raise ValueError("semantic tokens must have shape [batch, 10, d_model]")
        route_plan.validate_exact_partition(
            expected_module_keys=frozenset(self.exact_capsules),
            device=semantic_tokens.device,
            allow_unrouted_rows=route_plan.allow_unrouted_rows,
        )
        rollout = apply_capsule_stage_rollout(
            semantic_tokens,
            route_plan=route_plan,
            stage=self,
        )
        if rollout is not None:
            return rollout
        policy_tokens = torch.cat(
            (semantic_tokens[:, :1], semantic_tokens[:, 2:]),
            dim=1,
        )
        value_token = semantic_tokens[:, 1:2]
        policy_basis_outputs = torch.stack(
            tuple(basis(policy_tokens) for basis in self.policy_bases),
            dim=2,
        )
        value_basis_outputs = torch.stack(
            tuple(basis(value_token) for basis in self.value_bases),
            dim=2,
        )
        basis_outputs = torch.cat(
            (
                policy_basis_outputs[:, :1],
                value_basis_outputs,
                policy_basis_outputs[:, 1:],
            ),
            dim=1,
        )
        capsule_inputs = torch.cat(
            (semantic_tokens.unsqueeze(2), basis_outputs),
            dim=2,
        )
        delta = apply_exact_residual(
            capsule_inputs,
            route_plan,
            self.exact_capsules,
        )
        return semantic_tokens + delta

    def zero_output(self) -> None:
        """Make every shared basis and exact capsule output-inert."""
        for basis in (*self.policy_bases, *self.value_bases):
            cast(ZeroOutputBottleneckResidual, basis).zero_output()
        for capsule in self.exact_capsules.values():
            cast(ExactCompositionalCapsule, capsule).zero_output()


class SimpleStatelessV2Adapters(nn.Module):
    """All prompt and staged exact-deck capacity added by architecture v2."""

    def __init__(
        self,
        *,
        d_model: int,
        module_keys: tuple[str, ...],
        stage_layers: tuple[int, ...] = SIMPLE_STATELESS_V2_CAPSULE_STAGES,
        shared_bases: int = SIMPLE_STATELESS_V2_SHARED_BASES,
        basis_rank: int = SIMPLE_STATELESS_V2_BASIS_RANK,
        exact_bottleneck_dim: int = SIMPLE_STATELESS_V2_EXACT_BOTTLENECK,
    ) -> None:
        """Build exact prompts and four compositional capsule stages."""
        super().__init__()
        if (
            not stage_layers
            or tuple(sorted(set(stage_layers))) != stage_layers
            or any(layer <= 0 for layer in stage_layers)
        ):
            raise ValueError("capsule stage layers must be positive and ordered")
        self.stage_layers = stage_layers
        self._exact_module_keys = frozenset(module_keys)
        self.prompts = nn.ModuleDict(
            {
                module_key: ExactSemanticPrompt(d_model=d_model)
                for module_key in module_keys
            }
        )
        self.stages = nn.ModuleDict(
            {
                self._stage_key(layer): CompositionalCapsuleStage(
                    d_model=d_model,
                    shared_bases=shared_bases,
                    basis_rank=basis_rank,
                    exact_bottleneck_dim=exact_bottleneck_dim,
                    module_keys=module_keys,
                )
                for layer in stage_layers
            }
        )

    def apply_prompt(
        self,
        batch: PackedTokenBatch,
        positions: SpecialTokenPositions,
        *,
        route_plan: SimpleExactRoutePlan,
        allow_unrouted_rows: bool = False,
    ) -> PackedTokenBatch:
        """Add each route's prompt to policy/value/scratch packed rows."""
        route_plan.validate_exact_partition(
            expected_module_keys=self._exact_module_keys,
            device=batch.tokens.device,
            allow_unrouted_rows=allow_unrouted_rows,
        )
        semantic = gather_semantic_tokens(batch, positions)
        prompt = apply_prompt_rollout(
            semantic,
            route_plan=route_plan,
            prompts=self.prompts,
        )
        if prompt is None:
            prompt = apply_exact_residual(semantic, route_plan, self.prompts)
        return replace_semantic_tokens(batch, positions, semantic + prompt)

    def apply_stage(
        self,
        completed_layers: int,
        batch: PackedTokenBatch,
        positions: SpecialTokenPositions,
        *,
        route_plan: SimpleExactRoutePlan,
        allow_unrouted_rows: bool = False,
    ) -> PackedTokenBatch:
        """Apply one configured stage, leaving all other boundaries untouched."""
        key = self._stage_key(completed_layers)
        if key not in self.stages:
            return batch
        semantic = gather_semantic_tokens(batch, positions)
        stage = cast(CompositionalCapsuleStage, self.stages[key])
        route_plan.validate_exact_partition(
            expected_module_keys=self._exact_module_keys,
            device=batch.tokens.device,
            allow_unrouted_rows=allow_unrouted_rows,
        )
        transformed = stage(semantic, route_plan=route_plan)
        return replace_semantic_tokens(batch, positions, transformed)

    def route_private_banks(self) -> tuple[tuple[str, nn.ModuleDict], ...]:
        """Enumerate every route-keyed bank for transitions and validation."""
        banks: list[tuple[str, nn.ModuleDict]] = [("prompts", self.prompts)]
        for layer in self.stage_layers:
            key = self._stage_key(layer)
            stage = cast(CompositionalCapsuleStage, self.stages[key])
            banks.append((f"stages.{key}.exact_capsules", stage.exact_capsules))
        return tuple(banks)

    def inert_output_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield parameters that must be zero for an output-inert transition."""
        for name, parameter in self.named_parameters():
            if name.startswith("prompts.") or name.endswith(
                (".up.weight", ".up.bias")
            ):
                yield (name, parameter)

    def assert_output_inert(self) -> None:
        """Fail when any prompt, shared basis, or exact residual can change output."""
        for name, parameter in self.inert_output_named_parameters():
            if parameter.is_meta:
                raise ValueError("cannot validate inert v2 parameters on meta device")
            if torch.count_nonzero(parameter).item() != 0:
                raise ValueError(f"v2 adapter output parameter is not inert: {name}")

    def zero_output(self) -> None:
        """Reset all additive v2 paths to exact v1-equivalent output."""
        for prompt in self.prompts.values():
            cast(ExactSemanticPrompt, prompt).zero_output()
        for stage in self.stages.values():
            cast(CompositionalCapsuleStage, stage).zero_output()

    def initialize_exact_routes(self, module_keys: tuple[str, ...]) -> None:
        """Fresh-initialize selected exact leaves and keep their outputs inert."""
        if len(module_keys) != len(set(module_keys)):
            raise ValueError("exact routes to initialize must be unique")
        unknown = set(module_keys) - self._exact_module_keys
        if unknown:
            raise ValueError("cannot initialize unknown exact routes")
        for module_key in module_keys:
            prompt = self.prompts[module_key]
            if not isinstance(prompt, ExactSemanticPrompt):
                raise TypeError("exact prompt bank contains an unexpected module")
            prompt.zero_output()
            for stage in self.stages.values():
                if not isinstance(stage, CompositionalCapsuleStage):
                    raise TypeError("exact stage bank contains an unexpected module")
                capsule = stage.exact_capsules[module_key]
                if not isinstance(capsule, ExactCompositionalCapsule):
                    raise TypeError(
                        "exact capsule bank contains an unexpected module"
                    )
                initialize_simple_stateless_module(capsule)
                capsule.zero_output()

    @staticmethod
    def _stage_key(layer: int) -> str:
        """Return the stable state-dict key for one layer boundary."""
        return f"after_layer_{layer:02d}"


def gather_semantic_tokens(
    batch: PackedTokenBatch,
    positions: SpecialTokenPositions,
) -> Tensor:
    """Gather policy, value, and eight scratch tokens as ``[batch, 10, d]``."""
    indices = _semantic_indices(batch, positions)
    gathered = batch.tokens.index_select(0, indices.reshape(-1))
    return gathered.reshape(batch.batch_size, _SEMANTIC_TOKENS, batch.d_model)


def replace_semantic_tokens(
    batch: PackedTokenBatch,
    positions: SpecialTokenPositions,
    semantic_tokens: Tensor,
) -> PackedTokenBatch:
    """Differentiably replace only policy/value/scratch packed token rows."""
    expected = (batch.batch_size, _SEMANTIC_TOKENS, batch.d_model)
    if tuple(semantic_tokens.shape) != expected:
        raise ValueError(f"semantic_tokens must have shape {expected}")
    indices = _semantic_indices(batch, positions)
    tokens = batch.tokens.index_copy(
        0,
        indices.reshape(-1),
        semantic_tokens.reshape(-1, batch.d_model),
    )
    return batch.with_tokens(tokens)


def _semantic_indices(
    batch: PackedTokenBatch,
    positions: SpecialTokenPositions,
) -> Tensor:
    """Return the stable row-major policy/value/scratch packed indices."""
    expected_rows = batch.batch_size
    if (
        positions.policy.shape != (expected_rows,)
        or positions.value.shape != (expected_rows,)
        or positions.scratch.shape != (expected_rows, 8)
    ):
        raise ValueError("v2 requires policy/value and exactly eight scratch positions")
    return torch.cat(
        (
            positions.policy.unsqueeze(1),
            positions.value.unsqueeze(1),
            positions.scratch,
        ),
        dim=1,
    )


__all__ = [
    "GENERALIST_SEQUENCE_V2_CAPSULE_STAGES",
    "CompositionalCapsuleStage",
    "ExactCompositionalCapsule",
    "ExactSemanticPrompt",
    "SIMPLE_STATELESS_V2_BASIS_RANK",
    "SIMPLE_STATELESS_V2_CAPSULE_STAGES",
    "SIMPLE_STATELESS_V2_EXACT_BOTTLENECK",
    "SIMPLE_STATELESS_V2_SHARED_BASES",
    "SimpleStatelessV2Adapters",
    "ZeroOutputBottleneckResidual",
    "gather_semantic_tokens",
    "replace_semantic_tokens",
]
