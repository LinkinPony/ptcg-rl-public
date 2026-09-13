"""Validated bridge from native decisions to compact consequences.

This module binds opaque scenario handles to the exact hidden information sent
to the engine, validates the complete root-action support, and converts the
candidate-major native wire payload into the stable Python consequence
contract.  It deliberately does not aggregate manual-coin branches: one native
decision payload is one transition grid, not an enumerated chance support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.selection import is_legal_action, normalize_action_order
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.engine.compact_consequence import (
    CELL_ENDPOINT_COLUMN,
    CELL_ENGINE_RESULT_COLUMN,
    CELL_ERROR_CODE_COLUMN,
    CELL_RULES_EXACT_MASK_COLUMN,
    CELL_TRANSITION_STEPS_COLUMN,
    CELL_VALID_MASK_COLUMN,
    COMPACT_CONSEQUENCE_METADATA_WIDTH,
    CompactConsequenceBatch,
    CompactConsequenceMetadata,
    ConsequenceExactnessFlags,
    ScenarioSupport,
    ScenarioSupportMode,
)
from ptcg_rl.engine.consequence_identity import (
    MaterializedScenario,
    candidate_action_fingerprint,
    canonical_chance_support_fingerprint,
    hidden_information_fingerprint,
    normalize_scenario_support,
    root_observation_schema_fingerprint,
    root_token_bytes,
    root_token_fingerprint,
    strict_int32,
)
from ptcg_rl.engine.consequence_request_identity import (
    PreparedRequestIdentity,
    build_prepared_request_identity,
    root_observation_fingerprint,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.forward_model import (
    dynamic_effect_feature_from_dict_resolution,
)
from ptcg_rl.engine.native_consequence import (
    NativeConsequenceBatchResult,
    NativeConsequenceTimings,
)
from ptcg_rl.engine.native_consequence_payload import (
    NativeConsequenceEndpoint,
    NativeConsequenceMetadataColumn,
)
from ptcg_rl.engine.native_consequence_request import (
    prepare_native_consequence_request,
)
from ptcg_rl.engine.session import HiddenInformation


class NativeConsequenceRunner(Protocol):
    """Structural interface implemented by ``NativeConsequenceLane``."""

    @property
    def engine_library_fingerprint(self) -> str:
        """Return the immutable native engine-library SHA-256 identity."""

    @property
    def native_abi_fingerprint(self) -> str:
        """Return the planner ABI and observation-layout SHA-256 identity."""

    def run(
        self,
        state_token: bytes | str,
        *,
        hidden_worlds: Sequence[HiddenInformation],
        candidate_actions: Sequence[Sequence[int]],
        producer_contract_fingerprint: bytes,
        root_player: int,
        manual_coin: bool,
        stochastic_seed: int,
        max_cells: int,
        max_engine_steps: int,
        max_forced_steps: int,
        max_observation_bytes: int,
    ) -> NativeConsequenceBatchResult:
        """Execute one candidate-by-scenario native grid."""


@dataclass(frozen=True, slots=True)
class CompactConsequenceBridgeResult:
    """Compact batch, independent exactness facts, and native stage timings."""

    batch: CompactConsequenceBatch
    exactness: ConsequenceExactnessFlags
    native_timings: NativeConsequenceTimings
    request_identity: PreparedRequestIdentity
    cell_comparable_mask: npt.NDArray[np.bool_]
    candidate_rankable_mask: npt.NDArray[np.bool_]

    @property
    def native_request_fingerprint(self) -> str:
        """Return the independently echoed raw native-request identity."""
        return self.request_identity.native_request_fingerprint

    @property
    def rankable(self) -> bool:
        """Whether every retained candidate has comparable scenario outcomes."""
        return bool(self.candidate_rankable_mask.all())

    def require_rankable(self) -> CompactConsequenceBatch:
        """Return the batch or reject partial/chance/strategic evidence."""
        if not self.rankable:
            raise ValueError(
                "compact consequence request has non-rankable candidate outcomes"
            )
        return self.batch


@dataclass(frozen=True, slots=True)
class _PreparedBridgeInput:
    """Validated immutable input shared by execution and conversion."""

    state_token: bytes
    root_observation: Mapping[str, Any]
    root_player: int
    candidate_actions: tuple[tuple[int, ...], ...]
    candidate_fingerprints: tuple[str, ...]
    legal_action_count: int
    scenarios: tuple[MaterializedScenario, ...]
    scenario_support: ScenarioSupport
    identity: PreparedRequestIdentity


def execute_compact_consequence_batch(
    lane: NativeConsequenceRunner,
    state_token: bytes | str,
    *,
    root_observation: Mapping[str, Any],
    scenarios: Sequence[MaterializedScenario],
    candidate_actions: Sequence[Sequence[int]],
    root_player: int,
    support_mode: ScenarioSupportMode,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> CompactConsequenceBridgeResult:
    """Execute and convert one validated candidate-by-scenario request."""
    prepared = _prepare_bridge_input(
        state_token,
        root_observation=root_observation,
        scenarios=scenarios,
        candidate_actions=candidate_actions,
        root_player=root_player,
        support_mode=support_mode,
        engine_library_fingerprint=lane.engine_library_fingerprint,
        native_abi_fingerprint=lane.native_abi_fingerprint,
        max_cells=max_cells,
        max_engine_steps=max_engine_steps,
        max_forced_steps=max_forced_steps,
        max_observation_bytes=max_observation_bytes,
    )
    native_result = lane.run(
        prepared.state_token,
        hidden_worlds=tuple(item.hidden_information for item in prepared.scenarios),
        candidate_actions=prepared.candidate_actions,
        producer_contract_fingerprint=bytes.fromhex(
            prepared.identity.contract_fingerprint
        ),
        root_player=prepared.root_player,
        manual_coin=True,
        stochastic_seed=0,
        max_cells=prepared.identity.max_cells,
        max_engine_steps=prepared.identity.max_engine_steps,
        max_forced_steps=prepared.identity.max_forced_steps,
        max_observation_bytes=prepared.identity.max_observation_bytes,
    )
    return _convert_native_result(native_result, prepared)


def _prepare_bridge_input(
    state_token: bytes | str,
    *,
    root_observation: Mapping[str, Any],
    scenarios: Sequence[MaterializedScenario],
    candidate_actions: Sequence[Sequence[int]],
    root_player: int,
    support_mode: ScenarioSupportMode,
    engine_library_fingerprint: str,
    native_abi_fingerprint: str,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> _PreparedBridgeInput:
    token = root_token_bytes(state_token)
    root = strict_int32(root_player, "root_player")
    if root not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    if not isinstance(root_observation, Mapping):
        raise TypeError("root_observation must be a mapping")
    _validate_root_token(root_observation, token)
    _validate_root_player(root_observation, root)
    candidates, fingerprints, legal_count = _validated_candidate_actions(
        root_observation,
        candidate_actions,
    )
    support, normalized_scenarios = normalize_scenario_support(
        scenarios,
        mode=support_mode,
    )
    native_request = prepare_native_consequence_request(
        token,
        hidden_worlds=tuple(item.hidden_information for item in normalized_scenarios),
        candidate_actions=candidates,
        root_player=root,
        manual_coin=True,
        stochastic_seed=0,
        max_cells=max_cells,
        max_engine_steps=max_engine_steps,
        max_forced_steps=max_forced_steps,
        max_observation_bytes=max_observation_bytes,
    )
    identity = build_prepared_request_identity(
        root_state_fingerprint=root_token_fingerprint(token),
        root_observation_fingerprint=root_observation_fingerprint(root_observation),
        root_observation_schema_fingerprint=(root_observation_schema_fingerprint()),
        engine_library_fingerprint=engine_library_fingerprint,
        native_abi_fingerprint=native_abi_fingerprint,
        candidate_fingerprints=fingerprints,
        scenario_support=support,
        legal_action_count=legal_count,
        root_player=root,
        manual_coin=True,
        max_cells=native_request.max_cells,
        max_engine_steps=native_request.max_engine_steps,
        max_forced_steps=native_request.max_forced_steps,
        max_observation_bytes=native_request.max_observation_bytes,
        native_request_fingerprint=native_request.request_fingerprint.hex(),
    )
    return _PreparedBridgeInput(
        state_token=token,
        root_observation=root_observation,
        root_player=root,
        candidate_actions=candidates,
        candidate_fingerprints=fingerprints,
        legal_action_count=legal_count,
        scenarios=normalized_scenarios,
        scenario_support=support,
        identity=identity,
    )


def _convert_native_result(
    native_result: NativeConsequenceBatchResult,
    prepared: _PreparedBridgeInput,
) -> CompactConsequenceBridgeResult:
    candidate_count = len(prepared.candidate_actions)
    scenario_count = len(prepared.scenarios)
    if native_result.candidate_count != candidate_count:
        raise ValueError("native result candidate count differs from the request")
    if native_result.world_count != scenario_count:
        raise ValueError("native result world count differs from the request")
    if (
        native_result.payload.raw_request_fingerprint
        != prepared.identity.native_request_fingerprint
    ):
        raise ValueError("native result request identity differs from prepared input")
    if (
        native_result.payload.producer_contract_fingerprint
        != prepared.identity.contract_fingerprint
    ):
        raise ValueError(
            "native result producer contract differs from prepared input"
        )

    native_metadata = native_result.metadata
    root_players = native_metadata[:, int(NativeConsequenceMetadataColumn.ROOT_PLAYER)]
    if bool(np.any(root_players != prepared.root_player)):
        raise ValueError("native result root player differs from the request")

    errors = native_metadata[:, int(NativeConsequenceMetadataColumn.ERROR)]
    rules_exact = native_metadata[:, int(NativeConsequenceMetadataColumn.RULES_EXACT)]
    endpoints = native_metadata[:, int(NativeConsequenceMetadataColumn.ENDPOINT)]
    if (
        prepared.scenario_support.mode is ScenarioSupportMode.SAMPLED_BELIEF_NO_CHANCE
        and bool(np.any(endpoints == int(NativeConsequenceEndpoint.CHANCE_PROMPT)))
    ):
        raise ValueError(
            "native result reached a chance prompt under a no-chance support claim"
        )
    transition_steps = native_metadata[
        :, int(NativeConsequenceMetadataColumn.TRANSITION_STEPS)
    ]
    engine_results = native_metadata[
        :, int(NativeConsequenceMetadataColumn.RESULT)
    ]
    observation_sizes = native_metadata[
        :, int(NativeConsequenceMetadataColumn.OBSERVATION_SIZE)
    ]
    valid = errors == 0
    comparable_endpoints = np.asarray(
        (
            int(NativeConsequenceEndpoint.TERMINAL),
            int(NativeConsequenceEndpoint.SAME_SEAT_MAIN),
            int(NativeConsequenceEndpoint.TURN_HANDOFF),
        ),
        dtype=np.int32,
    )
    cell_comparable_mask = (
        valid & (rules_exact == 1) & np.isin(endpoints, comparable_endpoints)
    )
    candidate_rankable_mask = cell_comparable_mask.reshape(
        candidate_count,
        scenario_count,
    ).all(axis=1)

    compact_metadata = np.empty(
        (native_result.cell_count, COMPACT_CONSEQUENCE_METADATA_WIDTH),
        dtype=np.int32,
    )
    compact_metadata[:, CELL_ENDPOINT_COLUMN] = endpoints
    compact_metadata[:, CELL_TRANSITION_STEPS_COLUMN] = transition_steps
    compact_metadata[:, CELL_ERROR_CODE_COLUMN] = errors
    compact_metadata[:, CELL_VALID_MASK_COLUMN] = valid
    compact_metadata[:, CELL_RULES_EXACT_MASK_COLUMN] = rules_exact
    compact_metadata[:, CELL_ENGINE_RESULT_COLUMN] = engine_results

    observation_offsets = np.empty(native_result.cell_count + 1, dtype=np.int32)
    observation_offsets[0] = 0
    np.cumsum(observation_sizes, dtype=np.int32, out=observation_offsets[1:])
    root_observation_blob = native_result.payload.root_observation_blob
    if int(observation_offsets[-1]) != int(root_observation_blob.size):
        raise ValueError("native observation sizes do not cover the payload blob")

    exact_effects = np.zeros(
        (native_result.cell_count, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    for row_index in np.flatnonzero(valid):
        row = int(row_index)
        leaf_observation = native_result.decode_observation_row(row)
        if leaf_observation is None:
            raise ValueError("valid native result row has no leaf observation")
        candidate_index = row // scenario_count
        exact_effects[row] = dynamic_effect_feature_from_dict_resolution(
            select=prepared.candidate_actions[candidate_index],
            before_observation=prepared.root_observation,
            after_observation=leaf_observation,
            perspective_player=prepared.root_player,
        ).to_numpy()

    compact_metadata.setflags(write=False)
    observation_offsets.setflags(write=False)
    exact_effects.setflags(write=False)
    cell_comparable_mask.setflags(write=False)
    candidate_rankable_mask.setflags(write=False)
    scenario_grid_complete = bool(valid.all())
    contract = CompactConsequenceMetadata(
        observation_schema_fingerprint=(
            prepared.identity.root_observation_schema_fingerprint
        ),
        root_state_fingerprint=prepared.identity.root_state_fingerprint,
        candidate_fingerprints=prepared.identity.candidate_fingerprints,
        legal_action_count=prepared.identity.legal_action_count,
        scenario_support=prepared.scenario_support,
        support_exhaustive=(candidate_count == prepared.legal_action_count),
        scenario_grid_complete=scenario_grid_complete,
    )
    batch = CompactConsequenceBatch(
        contract=contract,
        root_observation_offsets=observation_offsets,
        root_observation_bytes=root_observation_blob,
        exact_effects=exact_effects,
        cell_metadata=compact_metadata,
    )
    return CompactConsequenceBridgeResult(
        batch=batch,
        exactness=ConsequenceExactnessFlags(
            rules_exact=bool(np.asarray(rules_exact, dtype=np.bool_).all()),
            support_exhaustive=contract.support_exhaustive,
            scenario_grid_complete=scenario_grid_complete,
            leaf_bootstrapped=False,
        ),
        native_timings=native_result.timings,
        request_identity=prepared.identity,
        cell_comparable_mask=cell_comparable_mask,
        candidate_rankable_mask=candidate_rankable_mask,
    )


def _validated_candidate_actions(
    root_observation: Mapping[str, Any],
    candidate_actions: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...], int]:
    select = root_observation.get("select")
    if select is None:
        raise ValueError("root_observation must contain a live select prompt")
    space = describe_prompt_action_space(select)
    actions_input = tuple(candidate_actions)
    if not actions_input:
        raise ValueError("candidate_actions must not be empty")
    actions: list[tuple[int, ...]] = []
    for index, action in enumerate(actions_input):
        if not isinstance(action, Sequence) or isinstance(
            action, (str, bytes, bytearray)
        ):
            raise TypeError(f"candidate_actions[{index}] must be a sequence")
        parsed = tuple(
            strict_int32(value, f"candidate_actions[{index}]") for value in action
        )
        canonical = normalize_action_order(select, parsed)
        if parsed != canonical:
            raise ValueError(f"candidate_actions[{index}] is not canonical")
        if not is_legal_action(select, canonical):
            raise ValueError(f"candidate_actions[{index}] is not legal")
        actions.append(canonical)
    canonical_actions = tuple(actions)
    if len(set(canonical_actions)) != len(canonical_actions):
        raise ValueError("candidate_actions must be unique")
    fingerprints = tuple(
        candidate_action_fingerprint(action) for action in canonical_actions
    )
    return canonical_actions, fingerprints, space.legal_action_count


def _validate_root_player(
    root_observation: Mapping[str, Any],
    root_player: int,
) -> None:
    current = root_observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("root_observation.current must be a mapping")
    observed_player = strict_int32(current.get("yourIndex"), "current.yourIndex")
    if observed_player != root_player:
        raise ValueError("root_player differs from root_observation.current.yourIndex")


def _validate_root_token(
    root_observation: Mapping[str, Any],
    state_token: bytes,
) -> None:
    if "search_begin_input" not in root_observation:
        raise ValueError("root_observation must contain search_begin_input")
    observed_token = root_token_bytes(root_observation["search_begin_input"])
    if observed_token != state_token:
        raise ValueError("state_token differs from root_observation.search_begin_input")


__all__ = [
    "CompactConsequenceBridgeResult",
    "MaterializedScenario",
    "NativeConsequenceRunner",
    "PreparedRequestIdentity",
    "candidate_action_fingerprint",
    "canonical_chance_support_fingerprint",
    "execute_compact_consequence_batch",
    "hidden_information_fingerprint",
    "normalize_scenario_support",
    "root_observation_schema_fingerprint",
    "root_token_fingerprint",
]
