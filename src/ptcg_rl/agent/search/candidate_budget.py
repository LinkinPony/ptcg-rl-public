"""Validated candidate and global planning work budgets."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SeedSource = Literal[
    "base",
    "proposal",
    "cardinality",
    "stochastic",
    "structural",
    "random",
]
ExpansionSource = Literal["mutation", "novelty"]
MandatoryAnchor = Literal["base_greedy"]

SEED_SOURCE_ORDER: tuple[SeedSource, ...] = (
    "base",
    "proposal",
    "cardinality",
    "stochastic",
    "structural",
    "random",
)
EXPANSION_SOURCE_ORDER: tuple[ExpansionSource, ...] = (
    "mutation",
    "novelty",
)


class SeedSourceQuotas(BaseModel):
    """Reserved seed slots for independent candidate sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base: int = 1
    proposal: int = 0
    cardinality: int = 0
    stochastic: int = 0
    structural: int = 0
    random: int = 0

    @field_validator(*SEED_SOURCE_ORDER)
    @classmethod
    def non_negative(cls, value: int) -> int:
        """Reject negative source reservations."""
        if value < 0:
            raise ValueError("seed source quotas must be non-negative")
        return value

    def as_dict(self) -> dict[SeedSource, int]:
        """Return quotas with a stable, typed source order."""
        return {source: int(getattr(self, source)) for source in SEED_SOURCE_ORDER}


class ExpansionSourceQuotas(BaseModel):
    """Reserved post-evaluation expansion slots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation: int = 0
    novelty: int = 0

    @field_validator(*EXPANSION_SOURCE_ORDER)
    @classmethod
    def non_negative(cls, value: int) -> int:
        """Reject negative expansion reservations."""
        if value < 0:
            raise ValueError("expansion source quotas must be non-negative")
        return value

    def as_dict(self) -> dict[ExpansionSource, int]:
        """Return quotas with a stable, typed source order."""
        return {source: int(getattr(self, source)) for source in EXPANSION_SOURCE_ORDER}


class GlobalPlanningBudget(BaseModel):
    """One request's non-renewable node, transition, and wall-time ceilings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_transition_limit: int
    prefix_node_limit: int
    wall_clock_limit_ms: int

    @field_validator(
        "engine_transition_limit",
        "prefix_node_limit",
        "wall_clock_limit_ms",
    )
    @classmethod
    def positive(cls, value: int) -> int:
        """Require a usable global budget."""
        if value <= 0:
            raise ValueError("global planning budget limits must be positive")
        return value


class CandidateConstructorConfig(BaseModel):
    """Versioned two-stage candidate-construction policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = 1
    exhaustive_action_cap: int
    k_seed: int
    k_total: int
    mandatory_anchors: tuple[MandatoryAnchor, ...] = ("base_greedy",)
    seed_quotas: SeedSourceQuotas
    expansion_quotas: ExpansionSourceQuotas
    seed_refill_priority: tuple[SeedSource, ...] = SEED_SOURCE_ORDER
    expansion_refill_priority: tuple[ExpansionSource, ...] = EXPANSION_SOURCE_ORDER
    work_budget: GlobalPlanningBudget

    @field_validator(
        "architecture_version",
        "exhaustive_action_cap",
        "k_seed",
        "k_total",
    )
    @classmethod
    def positive(cls, value: int) -> int:
        """Reject non-positive constructor limits or versions."""
        if value <= 0:
            raise ValueError("constructor limits and version must be positive")
        return value

    @model_validator(mode="after")
    def valid_two_stage_budget(self) -> CandidateConstructorConfig:
        """Validate anchors, separate quotas, and deterministic refill tables."""
        if self.k_seed > self.k_total:
            raise ValueError("k_seed must not exceed k_total")
        if tuple(dict.fromkeys(self.mandatory_anchors)) != self.mandatory_anchors:
            raise ValueError("mandatory anchors must be unique")
        if "base_greedy" not in self.mandatory_anchors:
            raise ValueError("base_greedy is a mandatory constructor anchor")
        seed = self.seed_quotas.as_dict()
        if seed["base"] < len(self.mandatory_anchors):
            raise ValueError("base quota must cover every mandatory anchor")
        if sum(seed.values()) > self.k_seed:
            raise ValueError("seed reserved slots must not exceed k_seed")
        expansion = self.expansion_quotas.as_dict()
        if sum(expansion.values()) > self.k_total - self.k_seed:
            raise ValueError("expansion reserved slots must fit k_total - k_seed")
        _validate_priority(
            self.seed_refill_priority,
            expected=SEED_SOURCE_ORDER,
            label="seed_refill_priority",
        )
        _validate_priority(
            self.expansion_refill_priority,
            expected=EXPANSION_SOURCE_ORDER,
            label="expansion_refill_priority",
        )
        return self


class CandidateBudgetRequest(BaseModel):
    """Root-observable shape and profiled cost inputs for one prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    legal_action_count: int
    option_count: int
    min_count: int
    max_count: int
    ordered: bool
    scenario_count: int
    estimated_engine_steps_per_cell: int
    estimated_prefix_nodes_per_candidate: int
    estimated_cell_time_us: int
    semantic_boundary: Literal["terminal_handoff_or_same_main"]

    @field_validator(
        "legal_action_count",
        "scenario_count",
        "estimated_engine_steps_per_cell",
        "estimated_prefix_nodes_per_candidate",
        "estimated_cell_time_us",
    )
    @classmethod
    def positive(cls, value: int) -> int:
        """Require positive counts and profiled costs."""
        if value <= 0:
            raise ValueError("candidate budget request costs must be positive")
        return value

    @field_validator("option_count", "min_count", "max_count")
    @classmethod
    def non_negative(cls, value: int) -> int:
        """Reject invalid prompt cardinalities."""
        if value < 0:
            raise ValueError("prompt cardinalities must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_prompt_shape(self) -> CandidateBudgetRequest:
        """Validate the generic selection grammar shape."""
        if self.min_count > self.max_count:
            raise ValueError("min_count must not exceed max_count")
        if self.max_count > self.option_count:
            raise ValueError("max_count must not exceed option_count")
        return self


class CandidateBudgetPlan(BaseModel):
    """Resolved support and global-work contract for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int
    exhaustive: bool
    k_seed: int
    k_total: int
    seed_quotas: SeedSourceQuotas
    expansion_quotas: ExpansionSourceQuotas
    baseline_grid_cells: int
    baseline_engine_steps: int
    baseline_prefix_nodes: int
    baseline_wall_clock_us: int
    engine_transition_limit: int
    prefix_node_limit: int
    wall_clock_limit_ms: int
    feasible: bool
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def valid_resolved_budget(self) -> CandidateBudgetPlan:
        """Keep every resolved bound internally consistent."""
        if self.k_seed < 0 or self.k_total < self.k_seed:
            raise ValueError("resolved candidate counts are inconsistent")
        if sum(self.seed_quotas.as_dict().values()) > self.k_seed:
            raise ValueError("resolved seed quotas exceed k_seed")
        if sum(self.expansion_quotas.as_dict().values()) > self.k_total - self.k_seed:
            raise ValueError("resolved expansion quotas exceed expansion capacity")
        if self.baseline_engine_steps > self.engine_transition_limit:
            raise ValueError("baseline grid exceeds the transition limit")
        if self.baseline_prefix_nodes > self.prefix_node_limit:
            raise ValueError("baseline grid exceeds the prefix-node limit")
        if self.baseline_wall_clock_us > self.wall_clock_limit_ms * 1_000:
            raise ValueError("baseline grid exceeds the wall-clock limit")
        if self.feasible != (self.fallback_reason is None):
            raise ValueError("feasible and fallback_reason must agree")
        return self


class CandidateBudgetPolicy:
    """Derive candidate limits without reading card or deck identity."""

    def __init__(self, config: CandidateConstructorConfig) -> None:
        self.config = config

    def resolve(self, request: CandidateBudgetRequest) -> CandidateBudgetPlan:
        """Resolve exhaustive/support sizes under one non-renewable work budget."""
        budget = self.config.work_budget
        cell_steps = request.scenario_count * request.estimated_engine_steps_per_cell
        cell_time_us = request.scenario_count * request.estimated_cell_time_us
        node_cost = request.estimated_prefix_nodes_per_candidate
        capacity = min(
            budget.engine_transition_limit // cell_steps,
            budget.prefix_node_limit // node_cost,
            (budget.wall_clock_limit_ms * 1_000) // cell_time_us,
            request.legal_action_count,
        )
        exhaustive = (
            request.legal_action_count <= self.config.exhaustive_action_cap
            and request.legal_action_count <= self.config.k_seed
            and request.legal_action_count <= capacity
        )
        if exhaustive:
            k_seed = request.legal_action_count
            k_total = request.legal_action_count
            seed_quotas = SeedSourceQuotas(base=0)
            expansion_quotas = ExpansionSourceQuotas()
            return self._plan(
                request=request,
                exhaustive=True,
                k_seed=k_seed,
                k_total=k_total,
                seed_quotas=seed_quotas,
                expansion_quotas=expansion_quotas,
            )

        mandatory_count = len(self.config.mandatory_anchors)
        reserved_seed_count = sum(self.config.seed_quotas.as_dict().values())
        if capacity < max(mandatory_count, reserved_seed_count):
            return self._fallback_plan("baseline_grid_budget")

        k_seed = min(self.config.k_seed, capacity)
        k_total = min(self.config.k_total, capacity)
        seed_quotas = _truncate_seed_quotas(
            self.config.seed_quotas,
            k_seed,
        )
        expansion_quotas = _truncate_expansion_quotas(
            self.config.expansion_quotas,
            k_total - k_seed,
        )
        return self._plan(
            request=request,
            exhaustive=False,
            k_seed=k_seed,
            k_total=k_total,
            seed_quotas=seed_quotas,
            expansion_quotas=expansion_quotas,
        )

    def _plan(
        self,
        *,
        request: CandidateBudgetRequest,
        exhaustive: bool,
        k_seed: int,
        k_total: int,
        seed_quotas: SeedSourceQuotas,
        expansion_quotas: ExpansionSourceQuotas,
    ) -> CandidateBudgetPlan:
        """Build one feasible plan with consistent baseline accounting."""
        budget = self.config.work_budget
        baseline_cells = k_seed * request.scenario_count
        return CandidateBudgetPlan(
            architecture_version=self.config.architecture_version,
            exhaustive=exhaustive,
            k_seed=k_seed,
            k_total=k_total,
            seed_quotas=seed_quotas,
            expansion_quotas=expansion_quotas,
            baseline_grid_cells=baseline_cells,
            baseline_engine_steps=(
                baseline_cells * request.estimated_engine_steps_per_cell
            ),
            baseline_prefix_nodes=(
                k_seed * request.estimated_prefix_nodes_per_candidate
            ),
            baseline_wall_clock_us=(baseline_cells * request.estimated_cell_time_us),
            engine_transition_limit=budget.engine_transition_limit,
            prefix_node_limit=budget.prefix_node_limit,
            wall_clock_limit_ms=budget.wall_clock_limit_ms,
            feasible=True,
        )

    def _fallback_plan(self, reason: str) -> CandidateBudgetPlan:
        budget = self.config.work_budget
        return CandidateBudgetPlan(
            architecture_version=self.config.architecture_version,
            exhaustive=False,
            k_seed=0,
            k_total=0,
            seed_quotas=SeedSourceQuotas(base=0),
            expansion_quotas=ExpansionSourceQuotas(),
            baseline_grid_cells=0,
            baseline_engine_steps=0,
            baseline_prefix_nodes=0,
            baseline_wall_clock_us=0,
            engine_transition_limit=budget.engine_transition_limit,
            prefix_node_limit=budget.prefix_node_limit,
            wall_clock_limit_ms=budget.wall_clock_limit_ms,
            feasible=False,
            fallback_reason=reason,
        )


def _validate_priority(
    priority: tuple[str, ...],
    *,
    expected: tuple[str, ...],
    label: str,
) -> None:
    if len(priority) != len(set(priority)) or set(priority) != set(expected):
        raise ValueError(f"{label} must contain every source exactly once")


def _truncate_seed_quotas(
    quotas: SeedSourceQuotas,
    capacity: int,
) -> SeedSourceQuotas:
    desired = quotas.as_dict()
    allocated = dict.fromkeys(SEED_SOURCE_ORDER, 0)
    remaining = capacity
    for source in SEED_SOURCE_ORDER:
        if remaining <= 0:
            break
        minimum = 1 if source == "base" else 0
        take = min(desired[source], remaining)
        if source == "base":
            take = max(minimum, take)
        allocated[source] = min(take, remaining)
        remaining -= allocated[source]
    return SeedSourceQuotas(**cast(dict[str, int], allocated))


def _truncate_expansion_quotas(
    quotas: ExpansionSourceQuotas,
    capacity: int,
) -> ExpansionSourceQuotas:
    desired = quotas.as_dict()
    allocated = dict.fromkeys(EXPANSION_SOURCE_ORDER, 0)
    remaining = capacity
    for source in EXPANSION_SOURCE_ORDER:
        if remaining <= 0:
            break
        allocated[source] = min(desired[source], remaining)
        remaining -= allocated[source]
    return ExpansionSourceQuotas(**cast(dict[str, int], allocated))


__all__ = [
    "CandidateBudgetPlan",
    "CandidateBudgetPolicy",
    "CandidateBudgetRequest",
    "CandidateConstructorConfig",
    "EXPANSION_SOURCE_ORDER",
    "ExpansionSource",
    "ExpansionSourceQuotas",
    "GlobalPlanningBudget",
    "SEED_SOURCE_ORDER",
    "SeedSource",
    "SeedSourceQuotas",
]
