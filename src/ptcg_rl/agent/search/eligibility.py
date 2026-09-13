"""Generic planner eligibility and fixed-single-select probe contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.agent.search.reusable_evidence import (
    ReusableCompactEvidenceEnvelope,
)
from ptcg_rl.engine.constants import SelectContext

if TYPE_CHECKING:
    from ptcg_rl.agent.search.planning_seed_reuse import ReusableV5SeedEvidence

EligibilityBranch = Literal["base_fast_path", "planner", "base_fallback"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SuccessorSemanticFingerprint(BaseModel):
    """Producer-only semantics needed for action-equivalence comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    successor: str
    chance_cursor: str
    belief_update: str
    endpoint: str
    unresolved_prompt: str

    @field_validator(
        "successor",
        "chance_cursor",
        "belief_update",
        "endpoint",
        "unresolved_prompt",
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        """Require an auditable semantic digest or endpoint label."""
        if not value:
            raise ValueError("successor semantic fingerprints must be non-empty")
        return value


class EquivalenceProbeCell(BaseModel):
    """One candidate/scenario cell from a reusable batched probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_index: int
    scenario_index: int
    semantics: SuccessorSemanticFingerprint

    @field_validator("action_index", "scenario_index")
    @classmethod
    def non_negative(cls, value: int) -> int:
        """Reject invalid ragged-grid coordinates."""
        if value < 0:
            raise ValueError("probe cell indices must be non-negative")
        return value


class FixedSingleSelectProbeResult(BaseModel):
    """Serializable summary of one complete paired-grid equivalence probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_version: int = 1
    action_count: int
    scenario_count: int
    transitions_used: int
    valid: bool
    paired_grid_complete: bool
    equivalent: bool
    stop_reason: str
    request_contract_fingerprint: str
    cells: tuple[EquivalenceProbeCell, ...] = ()

    @field_validator("request_contract_fingerprint")
    @classmethod
    def valid_request_fingerprint(cls, value: str) -> str:
        """Bind the summary to one exact native producer request."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("probe request identity must be lowercase SHA-256")
        return value

    @field_validator("probe_version", "action_count", "scenario_count")
    @classmethod
    def positive(cls, value: int) -> int:
        """Require positive schema and grid dimensions."""
        if value <= 0:
            raise ValueError("probe version and dimensions must be positive")
        return value

    @field_validator("transitions_used")
    @classmethod
    def non_negative(cls, value: int) -> int:
        """Reject negative engine work accounting."""
        if value < 0:
            raise ValueError("transitions_used must be non-negative")
        return value

    @model_validator(mode="after")
    def valid_grid_contract(self) -> FixedSingleSelectProbeResult:
        """Ensure a claimed complete result really contains one full grid."""
        coordinates = {(cell.action_index, cell.scenario_index) for cell in self.cells}
        if len(coordinates) != len(self.cells):
            raise ValueError("equivalence probe contains duplicate cells")
        in_bounds = all(
            action < self.action_count and scenario < self.scenario_count
            for action, scenario in coordinates
        )
        if not in_bounds:
            raise ValueError("equivalence probe cell is outside the declared grid")
        expected_coordinates = {
            (action_index, scenario_index)
            for action_index in range(self.action_count)
            for scenario_index in range(self.scenario_count)
        }
        if self.paired_grid_complete != (coordinates == expected_coordinates):
            raise ValueError("paired_grid_complete does not match probe cells")
        if self.transitions_used < len(self.cells):
            raise ValueError("probe transition accounting is smaller than its cells")
        derived_equivalent = self.valid and self.paired_grid_complete
        if derived_equivalent:
            indexed = {
                (cell.action_index, cell.scenario_index): cell.semantics
                for cell in self.cells
            }
            derived_equivalent = all(
                indexed[(action_index, scenario_index)] == indexed[(0, scenario_index)]
                for scenario_index in range(self.scenario_count)
                for action_index in range(1, self.action_count)
            )
        if self.equivalent != derived_equivalent:
            raise ValueError("equivalent does not match the probe cell semantics")
        return self


class FixedSingleSelectEquivalenceProbe(Protocol):
    """Engine-backed reusable one-step equivalence-probe interface."""

    def probe(
        self,
        select: Any,
        *,
        scenario_count: int,
        transition_budget: int,
    ) -> FixedSingleSelectProbeExecution:
        """Execute one shared candidate-by-scenario grid."""


@dataclass(frozen=True, slots=True)
class FixedSingleSelectProbeExecution:
    """Probe summary plus its optional request-local compact execution handle."""

    result: FixedSingleSelectProbeResult
    reusable_evidence: ReusableCompactEvidenceEnvelope | None
    reusable_v5_evidence: ReusableV5SeedEvidence[Any] | None = None

    def __post_init__(self) -> None:
        """Bind a reusable native grid to the probe's declared dimensions."""
        evidence = self.reusable_evidence
        v5_evidence = self.reusable_v5_evidence
        if evidence is not None and v5_evidence is not None:
            raise ValueError("probe execution cannot carry v4 and v5 evidence")
        if evidence is not None:
            _validate_probe_evidence_binding(self.result, evidence)
        if v5_evidence is not None:
            _validate_v5_probe_evidence_binding(self.result, v5_evidence)


class GenericEligibilityConfig(BaseModel):
    """Identity-free eligibility and small single-select probe limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = 1
    single_select_probe_cap: int
    probe_transition_limit: int

    @field_validator(
        "architecture_version",
        "single_select_probe_cap",
        "probe_transition_limit",
    )
    @classmethod
    def positive(cls, value: int) -> int:
        """Require usable eligibility bounds."""
        if value <= 0:
            raise ValueError("eligibility limits and version must be positive")
        return value


@dataclass(frozen=True)
class PlannerEligibilityDecision:
    """Eligibility branch plus reusable pre-action evidence, if any."""

    branch: EligibilityBranch
    reason: str
    probe_result: FixedSingleSelectProbeResult | None = None
    reuse_probe_as_seed_evidence: bool = False
    reusable_evidence: ReusableCompactEvidenceEnvelope | None = None
    reusable_v5_evidence: ReusableV5SeedEvidence[Any] | None = None

    def __post_init__(self) -> None:
        """Keep the reuse branch and its request-bound handle inseparable."""
        evidence = self.reusable_evidence
        v5_evidence = self.reusable_v5_evidence
        result = self.probe_result
        if evidence is not None and v5_evidence is not None:
            raise ValueError("eligibility cannot carry v4 and v5 seed evidence")
        has_evidence = evidence is not None or v5_evidence is not None
        if self.reuse_probe_as_seed_evidence != has_evidence:
            raise ValueError("probe reuse flag and evidence envelope must agree")
        if not has_evidence:
            return
        if self.branch != "planner" or result is None:
            raise ValueError("reusable probe evidence requires a planner decision")
        if not result.valid or not result.paired_grid_complete or result.equivalent:
            raise ValueError("reusable probe evidence requires differing valid cells")
        if evidence is not None:
            _validate_probe_evidence_binding(result, evidence)
        if v5_evidence is not None:
            _validate_v5_probe_evidence_binding(result, v5_evidence)

    @property
    def eligible(self) -> bool:
        """Return whether the bounded planner should run."""
        return self.branch == "planner"


def decide_planner_eligibility(
    select: Any,
    *,
    scenario_count: int,
    baseline_transition_budget: int,
    config: GenericEligibilityConfig,
    probe: FixedSingleSelectEquivalenceProbe | None,
    ordered: bool | None = None,
) -> PlannerEligibilityDecision:
    """Apply the fixed generic eligibility pipeline without identity rules."""
    if scenario_count <= 0 or baseline_transition_budget <= 0:
        raise ValueError("scenario count and baseline budget must be positive")
    space = describe_prompt_action_space(select)
    order_sensitive = space.ordered if ordered is None else bool(ordered)
    if forced_action(select) is not None or space.legal_action_count == 1:
        return PlannerEligibilityDecision("base_fast_path", "forced_or_single_action")

    context = _int_field(select, "context", -1)
    if (
        context == int(SelectContext.MAIN)
        or order_sensitive
        or space.max_count > 1
        or space.min_count != space.max_count
    ):
        return PlannerEligibilityDecision("planner", "strategic_prompt_shape")

    if space.legal_action_count > config.single_select_probe_cap:
        return PlannerEligibilityDecision("planner", "single_select_above_probe_cap")

    required_transitions = space.legal_action_count * scenario_count
    probe_budget = min(config.probe_transition_limit, baseline_transition_budget)
    if required_transitions > probe_budget:
        return PlannerEligibilityDecision(
            "base_fallback",
            "equivalence_probe_budget",
        )
    if probe is None:
        return PlannerEligibilityDecision(
            "base_fallback",
            "equivalence_probe_unavailable",
        )
    try:
        execution = probe.probe(
            select,
            scenario_count=scenario_count,
            transition_budget=probe_budget,
        )
        if not isinstance(execution, FixedSingleSelectProbeExecution):
            raise TypeError("probe must return FixedSingleSelectProbeExecution")
        result = execution.result
    except Exception:
        return PlannerEligibilityDecision(
            "base_fallback",
            "equivalence_probe_error",
        )
    if (
        result.action_count != space.legal_action_count
        or result.scenario_count != scenario_count
        or result.transitions_used > probe_budget
        or not result.valid
        or not result.paired_grid_complete
    ):
        return PlannerEligibilityDecision(
            "base_fallback",
            "equivalence_probe_invalid",
            probe_result=result,
        )
    if result.equivalent:
        return PlannerEligibilityDecision(
            "base_fast_path",
            "successors_equivalent",
            probe_result=result,
        )
    if execution.reusable_evidence is None and execution.reusable_v5_evidence is None:
        return PlannerEligibilityDecision(
            "base_fallback",
            "equivalence_probe_evidence_unavailable",
            probe_result=result,
        )
    return PlannerEligibilityDecision(
        "planner",
        "successors_differ",
        probe_result=result,
        reuse_probe_as_seed_evidence=True,
        reusable_evidence=execution.reusable_evidence,
        reusable_v5_evidence=execution.reusable_v5_evidence,
    )


def build_fixed_single_select_probe_result(
    *,
    action_count: int,
    scenario_count: int,
    cells: Sequence[EquivalenceProbeCell],
    transitions_used: int,
    request_contract_fingerprint: str,
    valid: bool = True,
    stop_reason: str = "complete",
) -> FixedSingleSelectProbeResult:
    """Validate a producer's full grid and derive semantic equivalence."""
    frozen_cells = tuple(cells)
    coordinates = {(cell.action_index, cell.scenario_index) for cell in frozen_cells}
    expected_coordinates = {
        (action_index, scenario_index)
        for action_index in range(action_count)
        for scenario_index in range(scenario_count)
    }
    complete = coordinates == expected_coordinates
    equivalent = bool(valid and complete)
    if equivalent:
        indexed = {
            (cell.action_index, cell.scenario_index): cell.semantics
            for cell in frozen_cells
        }
        for scenario_index in range(scenario_count):
            reference = indexed[(0, scenario_index)]
            if any(
                indexed[(action_index, scenario_index)] != reference
                for action_index in range(1, action_count)
            ):
                equivalent = False
                break
    return FixedSingleSelectProbeResult(
        action_count=action_count,
        scenario_count=scenario_count,
        transitions_used=transitions_used,
        valid=valid,
        paired_grid_complete=complete,
        equivalent=equivalent,
        stop_reason=stop_reason,
        request_contract_fingerprint=request_contract_fingerprint,
        cells=frozen_cells,
    )


def _validate_probe_evidence_binding(
    result: FixedSingleSelectProbeResult,
    evidence: ReusableCompactEvidenceEnvelope,
) -> None:
    if evidence.batch.candidate_count != result.action_count:
        raise ValueError("probe evidence action count differs from result")
    if evidence.batch.scenario_count != result.scenario_count:
        raise ValueError("probe evidence scenario count differs from result")
    if evidence.request_fingerprint != result.request_contract_fingerprint:
        raise ValueError("probe evidence request identity differs from result")


def _validate_v5_probe_evidence_binding(
    result: FixedSingleSelectProbeResult,
    evidence: ReusableV5SeedEvidence[Any],
) -> None:
    if len(evidence.actions) != result.action_count:
        raise ValueError("v5 probe evidence action count differs from result")
    if evidence.base_scenario_count != result.scenario_count:
        raise ValueError("v5 probe evidence scenario count differs from result")
    if evidence.request_fingerprint != result.request_contract_fingerprint:
        raise ValueError("v5 probe evidence request identity differs from result")


def _int_field(select: Any, name: str, default: int) -> int:
    value = (
        select.get(name, default)
        if isinstance(select, Mapping)
        else getattr(
            select,
            name,
            default,
        )
    )
    return int(value) if value is not None else default


__all__ = [
    "EligibilityBranch",
    "EquivalenceProbeCell",
    "FixedSingleSelectEquivalenceProbe",
    "FixedSingleSelectProbeExecution",
    "FixedSingleSelectProbeResult",
    "GenericEligibilityConfig",
    "PlannerEligibilityDecision",
    "SuccessorSemanticFingerprint",
    "build_fixed_single_select_probe_result",
    "decide_planner_eligibility",
]
