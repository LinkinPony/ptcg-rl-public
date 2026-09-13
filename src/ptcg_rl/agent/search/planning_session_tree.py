"""Bounded nonanticipative v5 continuation and manual-coin expansion."""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import orjson

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.search.hierarchical_contract import (
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planning_session_contract import (
    ContinuationActionProvider,
    ContinuationPrompt,
    HierarchicalOutcomeBatch,
    HierarchicalSearchConfig,
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.planning_session_outcomes import (
    CompletedPlanningBranch,
    build_hierarchical_outcome_batch,
)
from ptcg_rl.agent.search.root_information import (
    canonical_float32_vector_bytes,
)
from ptcg_rl.agent.search.root_information_context import (
    decode_root_information_observation,
)
from ptcg_rl.agent.search.root_information_producer import (
    RootInformationProducerBridge,
    RootInformationProducerState,
    context_snapshot_fingerprint,
    public_belief_producer_fingerprint,
)
from ptcg_rl.context import GameContextSnapshot
from ptcg_rl.engine.compact_consequence import (
    ScenarioSupport,
    ScenarioSupportMode,
    SemanticEndpoint,
)
from ptcg_rl.engine.consequence_identity import (
    MaterializedScenario,
    candidate_action_fingerprint,
    normalize_scenario_support,
    root_token_fingerprint,
)
from ptcg_rl.engine.consequence_request_identity import (
    root_observation_fingerprint,
)
from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.engine.native_planning_session import NativePlanningSessionBatchResult
from ptcg_rl.engine.native_planning_session_payload import (
    NativePlanningSessionHandle,
    NativePlanningSessionMetadataColumn,
)
from ptcg_rl.engine.native_planning_session_pool import (
    NativePlanningSessionContinueCall,
    NativePlanningSessionLanePool,
    NativePlanningSessionLease,
    NativePlanningSessionOpenCall,
    NativePlanningSessionPoolDeadlineError,
    NativePlanningSessionPoolError,
    NativePlanningSessionPoolSaturatedError,
)
from ptcg_rl.engine.probe_resolution import ProbeTransition
from ptcg_rl.runtime.planner_telemetry import (
    PlannerRequestTelemetry,
    PlannerStage,
    PlannerStageEvent,
)
from ptcg_rl.runtime.work_ledger import (
    PlannerRequestLedger,
    PlannerWorkReservation,
    PlannerWorkStopReason,
)

_ROOT_HISTORY_DOMAIN = b"ptcg-rl/root-planning-information-history/v3\x00"
_REACHED_HISTORY_DOMAIN = b"ptcg-rl/v5-reached-information-history/v3\x00"
_PRODUCER_CONTRACT_DOMAIN = b"ptcg-rl/v5-planner-producer-contract/v3\x00"
_NATIVE_UNSUPPORTED_CHANCE_ERROR = 92
_COMPARABLE = frozenset(
    {
        SemanticEndpoint.TERMINAL,
        SemanticEndpoint.SAME_SEAT_MAIN,
        SemanticEndpoint.TURN_HANDOFF,
    }
)


@dataclass(frozen=True, slots=True)
class _ActiveBranch:
    candidate_index: int
    belief_world_index: int
    chance_bits: tuple[int, ...]
    endpoint: SemanticEndpoint
    handle: NativePlanningSessionHandle | None
    last_observation: Mapping[str, Any]
    model_observation_bytes: bytes
    merged_observation_bytes: bytes
    controller_observation: Mapping[str, Any]
    producer_state: RootInformationProducerState
    information_history_fingerprint: str
    engine_result: int
    transition_steps: int
    transitions: tuple[ProbeTransition, ...]
    continuation_decisions: tuple[tuple[str, str], ...]
    continuation_depth: int


@dataclass(frozen=True, slots=True)
class _PendingStep:
    parent: _ActiveBranch
    action: tuple[int, ...]
    chance_bit: int | None
    decision: tuple[str, str] | None


class HierarchicalPlanningSessionExecutor:
    """Evaluate root options to comparable leaves on one pinned native lane."""

    def __init__(
        self,
        *,
        config: HierarchicalSearchConfig,
        pool: NativePlanningSessionLanePool,
        controller: ContinuationActionProvider,
        controller_identity: StableContinuationControllerIdentity,
        telemetry: PlannerRequestTelemetry | None = None,
    ) -> None:
        self.config = config
        self._pool = pool
        self._controller = controller
        self._controller_identity = controller_identity
        self._telemetry = telemetry
        self._prefix_reuse_count = 0
        self._native_transition_rows = 0

    @property
    def runtime_stats(self) -> tuple[int, int]:
        """Return request-local prefix reuse and executed native rows."""
        return (self._prefix_reuse_count, self._native_transition_rows)

    def evaluate(
        self,
        request: HierarchicalSearchRequest,
        *,
        ledger: PlannerRequestLedger,
    ) -> HierarchicalOutcomeBatch:
        """Run one complete candidate grid or raise one whole-request fallback."""
        base_support, scenarios, producer_bridge = self._validate_request(request)
        root_history = root_planning_information_fingerprint(request)
        contract = _producer_contract_fingerprint(
            request,
            support=base_support,
            controller=self._controller_identity,
            config=self.config,
        )
        root_rows = len(request.candidate_actions) * len(scenarios)
        node_reservation = _reserve_nodes(ledger, root_rows)
        opened = False
        native_started = time.perf_counter()
        try:
            lease = self._pool.open_session(
                NativePlanningSessionOpenCall.from_sequences(
                    request.state_token,
                    hidden_worlds=tuple(
                        scenario.hidden_information for scenario in scenarios
                    ),
                    candidate_actions=request.candidate_actions,
                    producer_contract_fingerprint=contract,
                    root_player=request.root_player,
                    manual_coin=True,
                    max_state_slots=self.config.max_state_slots,
                    caps=self.config.native_caps,
                ),
                ledger=ledger,
            )
            opened = True
        except Exception as exc:
            _complete_nodes(ledger, node_reservation, success=False)
            raise _as_planner_error(exc) from exc
        self._record_native_call(
            elapsed_seconds=time.perf_counter() - native_started,
            queue_wait_seconds=lease.open_queue_wait_seconds,
            rows=root_rows,
        )
        self._native_transition_rows += root_rows
        _complete_nodes(ledger, node_reservation, success=True)
        if not opened:
            raise AssertionError("planning-session OPEN state is inconsistent")
        with lease:
            frontier, completed = self._initial_frontier(
                request,
                lease.initial_result,
                root_history=root_history,
                producer_bridge=producer_bridge,
            )
            continuation_nodes = 0
            chance_nodes = 0
            while frontier:
                pending, added_chance_nodes = self._pending_steps(
                    frontier,
                    root_actions=request.candidate_actions,
                    ledger=ledger,
                )
                chance_nodes += added_chance_nodes
                continuation_nodes += len(pending)
                if chance_nodes > self.config.max_chance_nodes:
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.BUDGET_TRUNCATED,
                        "manual-coin tree exceeds its fixed node budget",
                    )
                children = self._continue_pending(
                    lease,
                    pending,
                    ledger=ledger,
                    request=request,
                    root_history=root_history,
                    producer_bridge=producer_bridge,
                )
                frontier = []
                for child in children:
                    if child.endpoint in _COMPARABLE:
                        completed.append(_complete_branch(child))
                    else:
                        frontier.append(child)
            return build_hierarchical_outcome_batch(
                request=request,
                base_support=base_support,
                root_information_history_fingerprint=root_history,
                branches=tuple(completed),
                controller=self._controller_identity,
                continuation_nodes=continuation_nodes,
                producer_contract_fingerprint=contract.hex(),
            )

    def producer_contract_fingerprint(
        self,
        request: HierarchicalSearchRequest,
    ) -> str:
        """Return the exact validated producer identity without native work."""
        support, _scenarios, _producer_bridge = self._validate_request(request)
        return _producer_contract_fingerprint(
            request,
            support=support,
            controller=self._controller_identity,
            config=self.config,
        ).hex()

    def _initial_frontier(
        self,
        request: HierarchicalSearchRequest,
        result: NativePlanningSessionBatchResult,
        *,
        root_history: str,
        producer_bridge: RootInformationProducerBridge,
    ) -> tuple[list[_ActiveBranch], list[CompletedPlanningBranch]]:
        world_count = len(request.scenarios)
        if (
            result.worlds != world_count
            or result.candidates != len(request.candidate_actions)
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "v5 OPEN response differs from the final union grid",
            )
        frontier: list[_ActiveBranch] = []
        completed: list[CompletedPlanningBranch] = []
        for candidate_index in range(len(request.candidate_actions)):
            for world_index in range(world_count):
                row = result.root_row_index(world_index, candidate_index)
                branch = _branch_from_row(
                    result,
                    row,
                    request=request,
                    root_history=root_history,
                    candidate_index=candidate_index,
                    world_index=world_index,
                    parent=None,
                    pending=None,
                    producer_bridge=producer_bridge,
                )
                if branch.endpoint in _COMPARABLE:
                    completed.append(_complete_branch(branch))
                else:
                    frontier.append(branch)
        return frontier, completed

    def _pending_steps(
        self,
        frontier: list[_ActiveBranch],
        *,
        root_actions: tuple[tuple[int, ...], ...],
        ledger: PlannerRequestLedger,
    ) -> tuple[tuple[_PendingStep, ...], int]:
        if any(
            branch.continuation_depth >= self.config.max_continuation_depth
            for branch in frontier
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.BUDGET_TRUNCATED,
                "continuation tree exceeds its fixed depth budget",
            )
        pending: list[_PendingStep] = []
        chance_nodes = 0
        strategic: dict[str, list[_ActiveBranch]] = {}
        for branch in frontier:
            if branch.endpoint is SemanticEndpoint.CHANCE_PROMPT:
                if len(branch.chance_bits) >= self.config.max_chance_depth:
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.BUDGET_TRUNCATED,
                        "manual-coin tree exceeds its fixed chance depth",
                    )
                for bit, action in _manual_coin_actions(
                    branch.controller_observation
                ):
                    pending.append(
                        _PendingStep(
                            parent=branch,
                            action=action,
                            chance_bit=bit,
                            decision=None,
                        )
                    )
                    chance_nodes += 1
            elif branch.endpoint is SemanticEndpoint.ROOT_STRATEGIC_PROMPT:
                strategic.setdefault(
                    branch.information_history_fingerprint,
                    [],
                ).append(branch)
            else:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                    "continuation frontier contains a non-continuable endpoint",
                )
        pending.extend(
            self._strategic_steps(
                strategic,
                root_actions=root_actions,
                ledger=ledger,
            )
        )
        return tuple(pending), chance_nodes

    def _strategic_steps(
        self,
        grouped: dict[str, list[_ActiveBranch]],
        *,
        root_actions: tuple[tuple[int, ...], ...],
        ledger: PlannerRequestLedger,
    ) -> tuple[_PendingStep, ...]:
        if not grouped:
            return ()
        prompts: list[ContinuationPrompt] = []
        histories = tuple(grouped)
        self._prefix_reuse_count += sum(len(rows) for rows in grouped.values()) - len(
            histories
        )
        for history in histories:
            representative = grouped[history][0]
            if any(
                prior_history == history
                for prior_history, _action in representative.continuation_decisions
            ):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.NONANTICIPATIVITY_VIOLATION,
                    "continuation controller revisited one information history",
                )
            prompts.append(
                ContinuationPrompt(
                    information_history_fingerprint=history,
                    observation=representative.controller_observation,
                    root_action=root_actions[representative.candidate_index],
                    continuation_depth=representative.continuation_depth,
                )
            )
        gpu_reservation = _reserve_gpu_rows(ledger, len(prompts))
        success = False
        try:
            selected = tuple(
                tuple(int(index) for index in action)
                for action in self._controller.select_actions(tuple(prompts))
            )
            success = True
        except Exception as exc:
            raise PlannerEvidenceError(
                PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                "continuation controller could not produce a leased action",
            ) from exc
        finally:
            _complete_nodes(ledger, gpu_reservation, success=success)
        if len(selected) != len(prompts):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "continuation controller result count differs from prompts",
            )
        pending: list[_PendingStep] = []
        for history, action in zip(histories, selected, strict=True):
            branches = grouped[history]
            select = branches[0].controller_observation.get("select")
            if not is_legal_action(select, action):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.LEGALITY_INCONSISTENT,
                    "continuation controller returned an illegal complete action",
                )
            action_fingerprint = candidate_action_fingerprint(action)
            for branch in branches:
                if not is_legal_action(
                    branch.controller_observation.get("select"),
                    action,
                ):
                    raise PlannerEvidenceError(
                        PlannerFallbackReason.LEGALITY_INCONSISTENT,
                        "one public information group has inconsistent legality",
                    )
                pending.append(
                    _PendingStep(
                        parent=branch,
                        action=action,
                        chance_bit=None,
                        decision=(history, action_fingerprint),
                    )
                )
        return tuple(pending)

    def _continue_pending(
        self,
        lease: NativePlanningSessionLease,
        pending: tuple[_PendingStep, ...],
        *,
        ledger: PlannerRequestLedger,
        request: HierarchicalSearchRequest,
        root_history: str,
        producer_bridge: RootInformationProducerBridge,
    ) -> tuple[_ActiveBranch, ...]:
        if not pending:
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "continuation frontier produced no work",
            )
        results: list[tuple[NativePlanningSessionBatchResult, _PendingStep]] = []
        for start in range(0, len(pending), self.config.max_continue_rows_per_call):
            chunk = pending[start : start + self.config.max_continue_rows_per_call]
            node_reservation = _reserve_nodes(ledger, len(chunk))
            success = False
            try:
                native_started = time.perf_counter()
                pool_result = lease.continue_batch(
                    NativePlanningSessionContinueCall.from_sequences(
                        tuple(_require_handle(item.parent) for item in chunk),
                        tuple(item.action for item in chunk),
                        caps=self.config.native_caps,
                    ),
                    ledger=ledger,
                )
                response = pool_result.batch
                self._record_native_call(
                    elapsed_seconds=time.perf_counter() - native_started,
                    queue_wait_seconds=pool_result.queue_wait_seconds,
                    rows=len(chunk),
                )
                self._native_transition_rows += len(chunk)
                success = True
            except Exception as exc:
                raise _as_planner_error(exc) from exc
            finally:
                _complete_nodes(ledger, node_reservation, success=success)
            if response.row_count != len(chunk):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                    "v5 continuation response differs from its request",
                )
            results.extend((response, item) for item in chunk)
        handles = tuple(
            dict.fromkeys(_require_handle(item.parent) for item in pending)
        )
        lease.release(handles)
        children: list[_ActiveBranch] = []
        row_by_response: dict[int, int] = {}
        for response, item in results:
            response_identity = id(response)
            row = row_by_response.get(response_identity, 0)
            row_by_response[response_identity] = row + 1
            children.append(
                _branch_from_row(
                    response,
                    row,
                    request=request,
                    root_history=root_history,
                    candidate_index=item.parent.candidate_index,
                    world_index=item.parent.belief_world_index,
                    parent=item.parent,
                    pending=item,
                    producer_bridge=producer_bridge,
                )
            )
        return tuple(children)

    def _record_native_call(
        self,
        *,
        elapsed_seconds: float,
        queue_wait_seconds: float,
        rows: int,
    ) -> None:
        if self._telemetry is None:
            return
        bounded_wait = min(max(0.0, queue_wait_seconds), elapsed_seconds)
        self._telemetry.record(
            PlannerStageEvent(
                stage=PlannerStage.NATIVE_QUEUE_WAIT,
                seconds=bounded_wait,
                rows=rows,
            )
        )
        self._telemetry.record(
            PlannerStageEvent(
                stage=PlannerStage.NATIVE_ENGINE,
                seconds=max(0.0, elapsed_seconds - bounded_wait),
                rows=rows,
            )
        )

    @staticmethod
    def _validate_request(
        request: HierarchicalSearchRequest,
    ) -> tuple[
        ScenarioSupport,
        tuple[MaterializedScenario, ...],
        RootInformationProducerBridge,
    ]:
        if request.root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        if not request.producer_context:
            raise ValueError("producer_context must not be empty")
        if request.belief_summary_width < 0:
            raise ValueError("belief_summary_width must be non-negative")
        if len(request.belief_summary) != request.belief_summary_width:
            raise ValueError("belief_summary width differs from its producer contract")
        _validate_root_actions(request)
        bridge = RootInformationProducerBridge(
            root_player=request.root_player,
            belief_summary_width=request.belief_summary_width,
            belief_feature_producer=request.belief_feature_producer,
        )
        root_visible = {
            "current": request.root_observation.get("current"),
            "logs": request.root_observation.get("logs", ()),
            "select": request.root_observation.get("select"),
        }
        try:
            bridge.validate_root(
                root_observable_state=orjson.dumps(
                    root_visible,
                    option=orjson.OPT_SORT_KEYS,
                ),
                context_snapshot=request.context_snapshot,
                producer_context=request.producer_context,
                belief_summary=request.belief_summary,
            )
        except (TypeError, ValueError) as exc:
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "root producer snapshot differs from its model inputs",
            ) from exc
        support, scenarios = normalize_scenario_support(
            request.scenarios,
            mode=ScenarioSupportMode.SAMPLED_BELIEF_NO_CHANCE,
        )
        return support, scenarios, bridge


def _branch_from_row(
    result: NativePlanningSessionBatchResult,
    row: int,
    *,
    request: HierarchicalSearchRequest,
    root_history: str,
    candidate_index: int,
    world_index: int,
    parent: _ActiveBranch | None,
    pending: _PendingStep | None,
    producer_bridge: RootInformationProducerBridge,
) -> _ActiveBranch:
    metadata = result.payload.metadata[row]
    column = NativePlanningSessionMetadataColumn
    error = int(metadata[int(column.ERROR)])
    if error != 0:
        reason = (
            PlannerFallbackReason.UNSUPPORTED_CHANCE
            if error == _NATIVE_UNSUPPORTED_CHANCE_ERROR
            else PlannerFallbackReason.ENGINE_ERROR
        )
        raise PlannerEvidenceError(reason, f"v5 transition failed with error {error}")
    if int(metadata[int(column.RULES_EXACT)]) != 1:
        raise PlannerEvidenceError(
            PlannerFallbackReason.RULES_INEXACT,
            "v5 transition is not rules-exact",
        )
    endpoint = SemanticEndpoint(int(metadata[int(column.ENDPOINT)]))
    if endpoint is SemanticEndpoint.INVALID:
        raise PlannerEvidenceError(
            PlannerFallbackReason.ENGINE_ERROR,
            "v5 transition returned an invalid endpoint",
        )
    raw = result.payload.decode_observation_row(row)
    if raw is None:
        raise PlannerEvidenceError(
            PlannerFallbackReason.EVIDENCE_ABSENT,
            "v5 transition returned no root-visible observation",
        )
    new_logs = _sequence(raw.get("logs"), "native observation logs")
    model_observation = {
        "current": raw.get("current"),
        "logs": new_logs,
        "select": raw.get("select"),
    }
    model_observation_bytes = orjson.dumps(
        model_observation,
        option=orjson.OPT_SORT_KEYS,
    )
    prior_logs = () if parent is None else _merged_logs(parent.merged_observation_bytes)
    merged = {
        "current": raw.get("current"),
        "logs": (*prior_logs, *new_logs),
        "select": raw.get("select"),
    }
    merged_bytes = orjson.dumps(merged, option=orjson.OPT_SORT_KEYS)
    parent_snapshot = (
        request.context_snapshot
        if parent is None
        else parent.producer_state.context_snapshot
    )
    try:
        producer_transition = producer_bridge.advance(
            parent_context_snapshot=parent_snapshot,
            transition_observable_state=model_observation_bytes,
            model_observable_state=model_observation_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            "v5 observation cannot advance the root-information producer",
        ) from exc
    controller_observation = producer_transition.model_observation
    producer_state = producer_transition.state
    prior_observation = (
        request.root_observation if parent is None else parent.last_observation
    )
    transition = ProbeTransition(
        before_observation=prior_observation,
        after_observation=model_observation,
        logs=new_logs,
    )
    chance_bits = () if parent is None else parent.chance_bits
    decisions = () if parent is None else parent.continuation_decisions
    depth = 0 if parent is None else parent.continuation_depth + 1
    steps = int(metadata[int(column.TRANSITION_STEPS)])
    transitions = (transition,) if parent is None else (*parent.transitions, transition)
    if parent is not None:
        steps += parent.transition_steps
    if pending is not None:
        if pending.chance_bit is not None:
            chance_bits = (*chance_bits, pending.chance_bit)
        if pending.decision is not None:
            decisions = (*decisions, pending.decision)
    prior_history = (
        root_history if parent is None else parent.information_history_fingerprint
    )
    executed_action = (
        request.candidate_actions[candidate_index]
        if pending is None
        else pending.action
    )
    transition_kind = (
        0 if pending is None else (2 if pending.chance_bit is not None else 1)
    )
    history = _reached_history_fingerprint(
        root_history=root_history,
        prior_history=prior_history,
        root_action=request.candidate_actions[candidate_index],
        executed_action=executed_action,
        transition_kind=transition_kind,
        chance_bits=chance_bits,
        continuation_decisions=decisions,
        observation=merged_bytes,
        producer_context=producer_state.producer_context,
        belief_summary=producer_state.belief_summary,
        context_snapshot=producer_state.context_snapshot,
        endpoint=endpoint,
    )
    handle = result.payload.handle_at(row)
    if (endpoint in _COMPARABLE) != (handle is None):
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "v5 endpoint and continuation handle disagree",
        )
    return _ActiveBranch(
        candidate_index=candidate_index,
        belief_world_index=world_index,
        chance_bits=chance_bits,
        endpoint=endpoint,
        handle=handle,
        last_observation=model_observation,
        model_observation_bytes=model_observation_bytes,
        merged_observation_bytes=merged_bytes,
        controller_observation=controller_observation,
        producer_state=producer_state,
        information_history_fingerprint=history,
        engine_result=int(metadata[int(column.RESULT)]),
        transition_steps=steps,
        transitions=transitions,
        continuation_decisions=decisions,
        continuation_depth=depth,
    )


def _complete_branch(branch: _ActiveBranch) -> CompletedPlanningBranch:
    if branch.endpoint not in _COMPARABLE or branch.handle is not None:
        raise ValueError("only comparable handle-free branches can complete")
    return CompletedPlanningBranch(
        candidate_index=branch.candidate_index,
        belief_world_index=branch.belief_world_index,
        chance_bits=branch.chance_bits,
        endpoint=branch.endpoint,
        final_observation_bytes=branch.model_observation_bytes,
        producer_context=branch.producer_state.producer_context,
        belief_summary=branch.producer_state.belief_summary,
        information_history_fingerprint=branch.information_history_fingerprint,
        engine_result=branch.engine_result,
        transition_steps=branch.transition_steps,
        transitions=branch.transitions,
        continuation_decisions=branch.continuation_decisions,
    )


def _manual_coin_actions(
    observation: Mapping[str, Any],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    select = observation.get("select")
    if not isinstance(select, Mapping) or int(select.get("context", -1)) != int(
        SelectContext.COIN_HEAD
    ):
        raise PlannerEvidenceError(
            PlannerFallbackReason.UNSUPPORTED_CHANCE,
            "chance endpoint is not an engine-exposed manual coin prompt",
        )
    options = _sequence(select.get("option"), "manual-coin options")
    yes: int | None = None
    no: int | None = None
    for index, option in enumerate(options):
        option_type = _field_int(option, "type", -1)
        if option_type == int(OptionType.YES):
            if yes is not None:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.UNSUPPORTED_CHANCE,
                    "manual-coin prompt contains duplicate YES options",
                )
            yes = index
        elif option_type == int(OptionType.NO):
            if no is not None:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.UNSUPPORTED_CHANCE,
                    "manual-coin prompt contains duplicate NO options",
                )
            no = index
    if yes is None or no is None:
        raise PlannerEvidenceError(
            PlannerFallbackReason.UNSUPPORTED_CHANCE,
            "manual-coin prompt does not expose one YES and one NO action",
        )
    actions = ((0, (no,)), (1, (yes,)))
    if any(not is_legal_action(select, action) for _bit, action in actions):
        raise PlannerEvidenceError(
            PlannerFallbackReason.LEGALITY_INCONSISTENT,
            "engine-exposed manual-coin choices are not legal selections",
        )
    return actions


def root_planning_information_fingerprint(
    request: HierarchicalSearchRequest,
) -> str:
    root_visible = {
        "current": request.root_observation.get("current"),
        "logs": request.root_observation.get("logs", ()),
        "select": request.root_observation.get("select"),
    }
    encoded = orjson.dumps(root_visible, option=orjson.OPT_SORT_KEYS)
    decoded = decode_root_information_observation(encoded, request.producer_context)
    current = decoded.get("current")
    if not isinstance(current, Mapping) or int(current.get("yourIndex", -1)) != (
        request.root_player
    ):
        raise ValueError("root player differs from the root-information context")
    digest = hashlib.sha256()
    digest.update(_ROOT_HISTORY_DOMAIN)
    _update_bytes(digest, encoded)
    _update_bytes(digest, request.producer_context)
    _update_bytes(
        digest,
        canonical_float32_vector_bytes(request.belief_summary),
    )
    digest.update(
        bytes.fromhex(context_snapshot_fingerprint(request.context_snapshot))
    )
    digest.update(struct.pack(">i", request.root_player))
    return digest.hexdigest()


def _reached_history_fingerprint(
    *,
    root_history: str,
    prior_history: str,
    root_action: tuple[int, ...],
    executed_action: tuple[int, ...],
    transition_kind: int,
    chance_bits: tuple[int, ...],
    continuation_decisions: tuple[tuple[str, str], ...],
    observation: bytes,
    producer_context: bytes,
    belief_summary: tuple[float, ...],
    context_snapshot: GameContextSnapshot,
    endpoint: SemanticEndpoint,
) -> str:
    if transition_kind not in (0, 1, 2):
        raise ValueError("transition_kind must identify root, strategic, or chance")
    if any(bit not in (0, 1) for bit in chance_bits):
        raise ValueError("chance history must contain only binary outcomes")
    digest = hashlib.sha256()
    digest.update(_REACHED_HISTORY_DOMAIN)
    digest.update(bytes.fromhex(root_history))
    digest.update(bytes.fromhex(prior_history))
    digest.update(bytes.fromhex(candidate_action_fingerprint(root_action)))
    digest.update(bytes.fromhex(candidate_action_fingerprint(executed_action)))
    digest.update(struct.pack(">BI", transition_kind, len(chance_bits)))
    digest.update(bytes(chance_bits))
    digest.update(struct.pack(">I", len(continuation_decisions)))
    for history, action in continuation_decisions:
        digest.update(bytes.fromhex(history))
        digest.update(bytes.fromhex(action))
    _update_bytes(digest, observation)
    _update_bytes(digest, producer_context)
    _update_bytes(digest, canonical_float32_vector_bytes(belief_summary))
    digest.update(bytes.fromhex(context_snapshot_fingerprint(context_snapshot)))
    digest.update(struct.pack(">i", int(endpoint)))
    return digest.hexdigest()


def _producer_contract_fingerprint(
    request: HierarchicalSearchRequest,
    *,
    support: ScenarioSupport,
    controller: StableContinuationControllerIdentity,
    config: HierarchicalSearchConfig,
) -> bytes:
    if len(request.producer_contract_fingerprint) != 32:
        raise ValueError("base producer contract fingerprint must contain 32 bytes")
    digest = hashlib.sha256()
    digest.update(_PRODUCER_CONTRACT_DOMAIN)
    digest.update(request.producer_contract_fingerprint)
    digest.update(
        bytes.fromhex(
            public_belief_producer_fingerprint(
                request.belief_feature_producer
            )
        )
    )
    digest.update(bytes.fromhex(root_token_fingerprint(request.state_token)))
    digest.update(bytes.fromhex(root_observation_fingerprint(request.root_observation)))
    digest.update(bytes.fromhex(root_planning_information_fingerprint(request)))
    digest.update(bytes.fromhex(support.support_fingerprint))
    digest.update(bytes.fromhex(controller.controller_fingerprint))
    config_bytes = orjson.dumps(
        config.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS,
    )
    _update_bytes(digest, config_bytes)
    digest.update(
        struct.pack(
            ">IIB",
            len(request.candidate_actions),
            request.legal_action_count,
            int(request.support_exhaustive),
        )
    )
    for action in request.candidate_actions:
        digest.update(bytes.fromhex(candidate_action_fingerprint(action)))
    return digest.digest()


def _validate_root_actions(request: HierarchicalSearchRequest) -> None:
    if not request.candidate_actions or len(set(request.candidate_actions)) != len(
        request.candidate_actions
    ):
        raise PlannerEvidenceError(
            PlannerFallbackReason.CONSTRUCTOR_INVALID,
            "final union candidate actions must be nonempty and unique",
        )
    if request.legal_action_count < len(request.candidate_actions):
        raise PlannerEvidenceError(
            PlannerFallbackReason.CONSTRUCTOR_INVALID,
            "final union exceeds the legal action count",
        )
    if request.support_exhaustive != (
        request.legal_action_count == len(request.candidate_actions)
    ):
        raise PlannerEvidenceError(
            PlannerFallbackReason.CONSTRUCTOR_INVALID,
            "root support exhaustiveness differs from the final union",
        )
    select = request.root_observation.get("select")
    if any(not is_legal_action(select, action) for action in request.candidate_actions):
        raise PlannerEvidenceError(
            PlannerFallbackReason.LEGALITY_INCONSISTENT,
            "final union contains an illegal root action",
        )


def _reserve_nodes(
    ledger: PlannerRequestLedger,
    count: int,
) -> PlannerWorkReservation:
    reservation = ledger.reserve(nodes=count)
    if reservation is None:
        raise PlannerEvidenceError(
            _fallback_for_stop_reason(ledger.snapshot().stop_reason),
            "planning-session node reservation exceeded the global budget",
        )
    return reservation


def _reserve_gpu_rows(
    ledger: PlannerRequestLedger,
    count: int,
) -> PlannerWorkReservation:
    reservation = ledger.reserve(gpu_rows=count)
    if reservation is None:
        raise PlannerEvidenceError(
            _fallback_for_stop_reason(ledger.snapshot().stop_reason),
            "planner inference rows exceed the global budget",
        )
    return reservation


def _complete_nodes(
    ledger: PlannerRequestLedger,
    reservation: PlannerWorkReservation,
    *,
    success: bool,
) -> None:
    ledger.complete(reservation, elapsed_seconds=0.0, success=success)


def _fallback_for_stop_reason(reason: PlannerWorkStopReason) -> PlannerFallbackReason:
    if reason is PlannerWorkStopReason.DEADLINE_GUARD:
        return PlannerFallbackReason.DEADLINE
    if reason is PlannerWorkStopReason.NATIVE_POOL_SATURATED:
        return PlannerFallbackReason.QUEUE_FULL
    return PlannerFallbackReason.BUDGET_TRUNCATED


def _as_planner_error(exc: Exception) -> PlannerEvidenceError:
    if isinstance(exc, PlannerEvidenceError):
        return exc
    if isinstance(exc, NativePlanningSessionPoolDeadlineError):
        return PlannerEvidenceError(PlannerFallbackReason.DEADLINE, str(exc))
    if isinstance(exc, NativePlanningSessionPoolSaturatedError):
        return PlannerEvidenceError(PlannerFallbackReason.QUEUE_FULL, str(exc))
    if isinstance(exc, NativePlanningSessionPoolError):
        return PlannerEvidenceError(PlannerFallbackReason.BUDGET_TRUNCATED, str(exc))
    return PlannerEvidenceError(PlannerFallbackReason.ENGINE_ERROR, str(exc))


def _require_handle(branch: _ActiveBranch) -> NativePlanningSessionHandle:
    if branch.handle is None:
        raise PlannerEvidenceError(
            PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
            "continuation branch is missing its lane-local handle",
        )
    return branch.handle


def _merged_logs(payload: bytes) -> tuple[Any, ...]:
    decoded = orjson.loads(payload)
    if not isinstance(decoded, Mapping):
        raise ValueError("merged observation is not an object")
    return _sequence(decoded.get("logs"), "merged observation logs")


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PlannerEvidenceError(
            PlannerFallbackReason.FINGERPRINT_MISMATCH,
            f"{name} must be a sequence",
        )
    return tuple(value)


def _field_int(value: Any, name: str, default: int) -> int:
    if isinstance(value, Mapping):
        raw = value.get(name, default)
    else:
        raw = getattr(value, name, default)
    return int(raw) if raw is not None else default


def _update_bytes(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


__all__ = [
    "HierarchicalPlanningSessionExecutor",
    "root_planning_information_fingerprint",
]
