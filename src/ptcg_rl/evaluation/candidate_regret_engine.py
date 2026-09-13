"""Exact paired engine execution through the shared planner scorer."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.consequence_flow import (
    RootInformationCellContext,
    build_direct_option_outcomes,
    build_root_information_transition_batch,
)
from ptcg_rl.agent.search.hierarchical_contract import (
    DeploymentContinuationControllerIdentity,
)
from ptcg_rl.agent.search.paired_scenarios import (
    CandidateScenarioAggregate,
    aggregate_paired_option_outcomes,
)
from ptcg_rl.agent.search.planner_scoring import (
    PlannerScoringConfig,
    SharedRootInformationLeafScorer,
    leaf_scoring_batch_from_compact,
)
from ptcg_rl.agent.search.root_information import (
    RootInformationLeaf,
    RootInformationStateTensorBatch,
)
from ptcg_rl.engine.compact_consequence import (
    CELL_ENDPOINT_COLUMN,
    CELL_ERROR_CODE_COLUMN,
    ScenarioSupportMode,
    SemanticEndpoint,
)
from ptcg_rl.engine.consequence_bridge import execute_compact_consequence_batch
from ptcg_rl.engine.consequence_identity import MaterializedScenario
from ptcg_rl.engine.consequence_request_identity import (
    root_observation_fingerprint,
)
from ptcg_rl.engine.native_consequence import NativeConsequenceLane
from ptcg_rl.engine.session import HiddenInformation
from ptcg_rl.evaluation.candidate_regret_corpus import CandidateAuditCase
from ptcg_rl.evaluation.candidate_regret_sources import Action

ObservationBatch = tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExactCandidateScores:
    """Comparable exhaustive scores and paired-integrity diagnostics."""

    scores: Mapping[Action, float]
    aggregates: Mapping[Action, CandidateScenarioAggregate]
    successor_fingerprints: Mapping[Action, str]
    scenario_count: int
    scenario_support_fingerprint: str
    scenario_support_mode: str
    scenario_grid_complete: bool
    rules_exact: bool
    paired_support_integrity: bool
    nonanticipativity_integrity: bool
    scorer_fingerprint: str
    controller_fingerprint: str
    leaf_bootstrapped: bool
    unique_value_rows: int
    native_pack_ms: float
    native_call_ms: float
    native_parse_ms: float
    native_payload_bytes: int
    endpoint_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class UnsupportedExactReference:
    """Explicit reason an exhaustive-fit root could not be ranked."""

    status: str
    native_error_counts: Mapping[str, int]
    endpoint_counts: Mapping[str, int]
    scenario_count: int
    scenario_support_fingerprint: str
    scenario_support_mode: str
    scenario_grid_complete: bool
    rules_exact: bool
    paired_support_integrity: bool
    nonanticipativity_integrity: bool
    native_pack_ms: float
    native_call_ms: float
    native_parse_ms: float
    native_payload_bytes: int


class _JsonLeafTensorizer:
    """Decode native root-visible JSON and reattach root-visible context."""

    def tensorize(
        self,
        leaves: tuple[RootInformationLeaf, ...],
    ) -> ObservationBatch:
        observations: list[Mapping[str, Any]] = []
        for leaf in leaves:
            raw = json.loads(leaf.root_observable_state)
            context = json.loads(leaf.producer_context)
            if not isinstance(raw, dict) or not isinstance(context, dict):
                raise ValueError("root-information JSON must decode to mappings")
            raw["gameContext"] = context.get("gameContext", {})
            observations.append(cast(Mapping[str, Any], raw))
        return tuple(observations)


class _CheckpointValueProvider:
    """Evaluate zero-adapter migration leaves with the immutable checkpoint."""

    def __init__(self, policy: CheckpointPolicy, *, root_player: int) -> None:
        self._policy = policy
        self._root_player = root_player

    def values(
        self,
        batch: RootInformationStateTensorBatch[ObservationBatch],
    ) -> npt.ArrayLike:
        """Return checkpoint values aligned with unique root-visible leaves."""
        return self._policy.values(batch.model_inputs, self._root_player)


def materialize_paired_scenarios(
    hidden: HiddenInformation,
    *,
    requested_count: int,
    seed: int,
) -> tuple[MaterializedScenario, ...]:
    """Create generic count-preserving belief particles without identity rules."""
    if requested_count <= 0:
        raise ValueError("requested scenario count must be positive")
    scenarios: list[MaterializedScenario] = []
    fingerprints: set[str] = set()
    maximum_attempts = max(16, requested_count * 16)
    for attempt in range(maximum_attempts):
        candidate = _rotated_hidden(hidden, offset=seed + attempt)
        scenario = MaterializedScenario.create(
            hidden_information=candidate,
            belief_world_handle=len(scenarios),
            chance_support_handle=0,
            chance_support_identity=b"manual-coin-unsupported-v1",
            weight=1.0,
        )
        fingerprint = scenario.handle.belief_world_fingerprint
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        scenarios.append(scenario)
        if len(scenarios) >= requested_count:
            break
    if not scenarios:
        raise RuntimeError("failed to construct any count-correct belief scenario")
    return tuple(scenarios)


def score_exhaustive_candidates(
    lane: NativeConsequenceLane,
    policy: CheckpointPolicy,
    case: CandidateAuditCase,
    actions: Sequence[Action],
    *,
    scenarios: Sequence[MaterializedScenario],
    scoring_config: PlannerScoringConfig,
    controller: DeploymentContinuationControllerIdentity,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> ExactCandidateScores | UnsupportedExactReference:
    """Execute a true exhaustive-fit grid and rank only comparable outcomes."""
    root = dict(case.observation)
    root["search_begin_input"] = case.consequence.state_token
    bridge = execute_compact_consequence_batch(
        lane,
        case.consequence.state_token,
        root_observation=root,
        scenarios=scenarios,
        candidate_actions=actions,
        root_player=case.consequence.player_index,
        support_mode=ScenarioSupportMode.SAMPLED_BELIEF_CHANCE_UNSUPPORTED,
        max_cells=max_cells,
        max_engine_steps=max_engine_steps,
        max_forced_steps=max_forced_steps,
        max_observation_bytes=max_observation_bytes,
    )
    if not bridge.exactness.support_exhaustive:
        raise ValueError("exhaustive reference request omitted a legal action")
    endpoint_counts = Counter(
        SemanticEndpoint(int(value)).name.lower()
        for value in bridge.batch.cell_metadata[:, CELL_ENDPOINT_COLUMN]
    )
    errors = Counter(
        str(int(value))
        for value in bridge.batch.cell_metadata[:, CELL_ERROR_CODE_COLUMN]
        if int(value) != 0
    )
    timings = bridge.native_timings
    if not bridge.rankable:
        endpoints = set(endpoint_counts)
        if "chance_prompt" in endpoints or "92" in errors:
            status = "unsupported_rng_or_chance"
        elif "root_strategic_prompt" in endpoints:
            status = "unsupported_strategic_continuation"
        elif errors:
            status = "native_engine_error"
        else:
            status = "noncomparable_endpoint"
        return UnsupportedExactReference(
            status=status,
            native_error_counts=dict(sorted(errors.items())),
            endpoint_counts=dict(sorted(endpoint_counts.items())),
            scenario_count=bridge.batch.scenario_count,
            scenario_support_fingerprint=(
                bridge.batch.contract.scenario_support.support_fingerprint
            ),
            scenario_support_mode=(
                bridge.batch.contract.scenario_support.mode.value
            ),
            scenario_grid_complete=(
                bridge.batch.contract.scenario_grid_complete
            ),
            rules_exact=bridge.exactness.rules_exact,
            paired_support_integrity=(
                bridge.batch.contract.scenario_grid_complete
            ),
            nonanticipativity_integrity=True,
            native_pack_ms=timings.pack_seconds * 1_000.0,
            native_call_ms=timings.native_call_seconds * 1_000.0,
            native_parse_ms=timings.parse_seconds * 1_000.0,
            native_payload_bytes=(
                int(bridge.batch.root_observation_bytes.nbytes)
                + int(bridge.batch.cell_metadata.nbytes)
                + int(bridge.batch.exact_effects.nbytes)
            ),
        )

    source_history = hashlib.sha256(
        b"ptcg-rl/candidate-regret/source-history/v1\x00"
        + bytes.fromhex(case.case_id)
        + bytes.fromhex(root_observation_fingerprint(root))
    ).hexdigest()
    context_payload = json.dumps(
        {"gameContext": root.get("gameContext", {})},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    contexts = tuple(
        RootInformationCellContext(
            producer_context=context_payload,
            belief_summary=(),
        )
        for _ in range(bridge.batch.cell_count)
    )
    transitions = build_root_information_transition_batch(
        bridge.batch,
        source_information_history_fingerprint=source_history,
        cell_contexts=contexts,
    )
    scorer = SharedRootInformationLeafScorer[ObservationBatch](
        config=scoring_config,
        tensorizer=_JsonLeafTensorizer(),
        value_provider=_CheckpointValueProvider(
            policy,
            root_player=case.consequence.player_index,
        ),
    )
    score_result = scorer.score(
        leaves=transitions.value_leaves,
        cells=leaf_scoring_batch_from_compact(
            bridge.batch,
            root_player=case.consequence.player_index,
        ),
    )
    outcomes = build_direct_option_outcomes(
        bridge.batch,
        transitions,
        scores=score_result,
        controller=controller,
        root_player=case.consequence.player_index,
    )
    aggregation = aggregate_paired_option_outcomes(
        outcomes,
        cell_scores=score_result.cell_scores,
        scorer_fingerprint=score_result.scorer_fingerprint,
        config=scoring_config,
    )
    aggregate_by_action = dict(zip(actions, aggregation.candidates, strict=True))
    scores = {
        action: float(aggregate_by_action[action].robust_score) for action in actions
    }
    successors = {
        action: _candidate_successor_fingerprint(
            bridge.batch,
            candidate_index=index,
        )
        for index, action in enumerate(actions)
    }
    return ExactCandidateScores(
        scores=scores,
        aggregates=aggregate_by_action,
        successor_fingerprints=successors,
        scenario_count=bridge.batch.scenario_count,
        scenario_support_fingerprint=(
            bridge.batch.contract.scenario_support.support_fingerprint
        ),
        scenario_support_mode=bridge.batch.contract.scenario_support.mode.value,
        scenario_grid_complete=bridge.batch.contract.scenario_grid_complete,
        rules_exact=bridge.exactness.rules_exact,
        paired_support_integrity=bridge.batch.contract.scenario_grid_complete,
        nonanticipativity_integrity=True,
        scorer_fingerprint=aggregation.scorer_fingerprint,
        controller_fingerprint=aggregation.continuation_controller_fingerprint,
        leaf_bootstrapped=score_result.leaf_bootstrapped,
        unique_value_rows=len(score_result.unique_leaf_values),
        native_pack_ms=timings.pack_seconds * 1_000.0,
        native_call_ms=timings.native_call_seconds * 1_000.0,
        native_parse_ms=timings.parse_seconds * 1_000.0,
        native_payload_bytes=(
            int(bridge.batch.root_observation_bytes.nbytes)
            + int(bridge.batch.cell_metadata.nbytes)
            + int(bridge.batch.exact_effects.nbytes)
        ),
        endpoint_counts=dict(sorted(endpoint_counts.items())),
    )


def semantic_config_fingerprint(value: Mapping[str, Any]) -> str:
    """Fingerprint one resolved JSON-safe semantic configuration mapping."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/candidate-regret/resolved-config/v1\x00" + payload
    ).hexdigest()


def _rotated_hidden(hidden: HiddenInformation, *, offset: int) -> HiddenInformation:
    your_deck, your_prize = _rotate_partition(
        (hidden.your_deck, hidden.your_prize),
        offset,
    )
    opponent_deck, opponent_prize, opponent_hand = _rotate_partition(
        (hidden.opponent_deck, hidden.opponent_prize, hidden.opponent_hand),
        offset * 3 + 1,
    )
    return HiddenInformation.from_sequences(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=hidden.opponent_active,
    )


def _rotate_partition(
    zones: Sequence[Sequence[int]],
    offset: int,
) -> tuple[tuple[int, ...], ...]:
    lengths = tuple(len(zone) for zone in zones)
    pool = tuple(card for zone in zones for card in zone)
    if pool:
        shift = offset % len(pool)
        pool = pool[shift:] + pool[:shift]
    result: list[tuple[int, ...]] = []
    cursor = 0
    for length in lengths:
        result.append(pool[cursor : cursor + length])
        cursor += length
    return tuple(result)


def _candidate_successor_fingerprint(
    batch: Any,
    *,
    candidate_index: int,
) -> str:
    digest = hashlib.sha256(
        b"ptcg-rl/candidate-regret/root-observable-successor/v1\x00"
    )
    for scenario_index in range(batch.scenario_count):
        row = batch.row_at(candidate_index, scenario_index)
        payload = row.root_observable_state.payload.tobytes(order="C")
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
        digest.update(np.asarray(row.exact_effect, dtype="<f4").tobytes())
        digest.update(struct.pack(">ii", int(row.endpoint), row.engine_result))
    return digest.hexdigest()


__all__ = [
    "ExactCandidateScores",
    "UnsupportedExactReference",
    "materialize_paired_scenarios",
    "score_exhaustive_candidates",
    "semantic_config_fingerprint",
]
