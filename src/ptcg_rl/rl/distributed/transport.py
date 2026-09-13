"""Message framing and compact trajectory serialization for distributed RL."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import msgpack
import numpy as np

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.context import (
    PUBLIC_EVENT_SCHEMA_FINGERPRINT,
    PublicEventArrayBlock,
    public_event_array_field_names,
    public_event_schema_metadata,
    validate_public_event_array_block,
    validate_public_event_schema_metadata,
)
from ptcg_rl.engine.factual_schema import (
    FACTUAL_NEXT_CONTEXT_OOV,
    FACTUAL_NEXT_CONTEXT_TERMINAL,
    FactualActorRelation,
)
from ptcg_rl.model.input_schema import (
    policy_input_schema_metadata,
    validate_policy_input_schema_metadata,
)
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE
from ptcg_rl.rl.distributed.compatibility import DistributedModelCompatibility
from ptcg_rl.rl.endpoint_value_arrays import (
    ExecutedEndpointValueArrayBlock,
    endpoint_value_array_field_names,
    validate_executed_endpoint_value_array_block,
)
from ptcg_rl.rl.experience import (
    GameMetadata,
    GameTrajectory,
    OptionArrayBlock,
    StateArrayBlock,
    TrajectoryArrayBlock,
    TrajectoryDeckContext,
    compact_game_trajectory,
    validate_game_trajectory_array_block,
    validate_trajectory_array_block,
)
from ptcg_rl.rl.macro_credit_arrays import (
    ExecutedMacroArrayBlock,
    executed_macro_array_field_names,
    validate_executed_macro_array_block,
)
from ptcg_rl.rl.macro_teacher_arrays import (
    MacroTeacherArrayBlock,
    macro_teacher_array_field_names,
    validate_macro_teacher_array_block,
)
from ptcg_rl.rl.planner_evidence_arrays import (
    PlannerEvidenceArrayBlock,
    planner_array_field_names,
    validate_planner_evidence_array_block,
)
from ptcg_rl.rl.recurrent_runtime import PolicyArtifactIdentity
from ptcg_rl.rl.search_evidence_arrays import (
    SearchEvidenceArrayBlock,
)


@dataclass(frozen=True)
class ReceivedTrajectoryBatch:
    """One decoded distributed trajectory batch."""

    worker_id: str
    sequence_id: int
    schema: int
    compatibility: DistributedModelCompatibility | None
    trajectories: tuple[GameTrajectory, ...]
    sent_at: float
    received_at: float
    payload_bytes: int

    @property
    def decisions(self) -> int:
        """Return total decisions contained in the batch."""
        return sum(trajectory.decision_count for trajectory in self.trajectories)


def serialize_trajectory_batch(
    trajectories: Sequence[GameTrajectory],
    *,
    worker_id: str = "",
    sequence_id: int = 0,
    sent_at: float | None = None,
    compatibility: DistributedModelCompatibility | None = None,
) -> tuple[bytes, tuple[memoryview, ...]]:
    """Serialize compact trajectories into a msgpack header and raw frames."""
    compact = tuple(compact_game_trajectory(trajectory) for trajectory in trajectories)
    schema = _trajectory_batch_schema(compact)
    if compatibility is not None and compatibility.trajectory_schema_version != schema:
        raise ValueError(
            "trajectory payload schema differs from its compatibility manifest"
        )
    _validate_event_compatibility(schema, compatibility)
    _validate_sequence_artifact_compatibility(
        schema,
        compact,
        compatibility,
    )
    frames: list[memoryview] = []
    header = {
        "type": "trajectory_batch",
        "schema": schema,
        "worker_id": worker_id,
        "sequence_id": int(sequence_id),
        "sent_at": time.time() if sent_at is None else float(sent_at),
        "trajectories": [
            _trajectory_to_header(trajectory, frames) for trajectory in compact
        ],
    }
    if compatibility is not None:
        header["compatibility"] = compatibility.model_dump(mode="json")
    return (
        msgpack.packb(header, use_bin_type=True),
        tuple(frames),
    )


def deserialize_trajectory_batch(
    header_bytes: bytes,
    frames: Sequence[bytes | memoryview],
    *,
    received_at: float | None = None,
) -> ReceivedTrajectoryBatch:
    """Deserialize a distributed trajectory batch from msgpack and raw frames."""
    header = msgpack.unpackb(header_bytes, raw=False)
    if not isinstance(header, Mapping):
        raise ValueError("trajectory batch header must be a mapping")
    if header.get("type") != "trajectory_batch":
        raise ValueError("unexpected distributed message type")
    schema = int(header.get("schema", 0))
    if schema not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
        raise ValueError("unsupported trajectory batch schema")
    frame_views = tuple(memoryview(frame) for frame in frames)
    compatibility_header = header.get("compatibility")
    compatibility = (
        None
        if compatibility_header is None
        else DistributedModelCompatibility.model_validate(compatibility_header)
    )
    if compatibility is not None and compatibility.trajectory_schema_version != schema:
        raise ValueError(
            "trajectory payload schema differs from its compatibility manifest"
        )
    _validate_event_compatibility(schema, compatibility)
    trajectories = tuple(
        _trajectory_from_header(
            cast(Mapping[str, Any], item),
            frame_views,
            schema=schema,
        )
        for item in _sequence(header.get("trajectories", ()))
    )
    _validate_sequence_artifact_compatibility(
        schema,
        trajectories,
        compatibility,
    )
    payload_bytes = len(header_bytes) + sum(
        _frame_nbytes(frame) for frame in frame_views
    )
    return ReceivedTrajectoryBatch(
        worker_id=str(header.get("worker_id", "")),
        sequence_id=int(header.get("sequence_id", 0)),
        schema=schema,
        compatibility=compatibility,
        trajectories=trajectories,
        sent_at=float(header.get("sent_at", 0.0)),
        received_at=time.time() if received_at is None else float(received_at),
        payload_bytes=payload_bytes,
    )


def _trajectory_to_header(
    trajectory: GameTrajectory,
    frames: list[memoryview],
) -> dict[str, Any]:
    validate_game_trajectory_array_block(trajectory)
    block = trajectory.array_block
    if block is None:
        raise ValueError("distributed trajectories must be array-backed")
    header = {
        "game_id": trajectory.game_id,
        "seats_reward": list(trajectory.seats_reward),
        "metadata": _metadata_to_header(trajectory.metadata),
        "array_block": _array_block_to_header(block, frames),
        "policy_input_schema": policy_input_schema_metadata(),
    }
    if trajectory.deck_context is not None:
        header["deck_context"] = _deck_context_to_header(
            trajectory.deck_context,
            frames,
        )
    if block.has_public_events:
        header["public_event_schema"] = public_event_schema_metadata()
    if trajectory.policy_artifacts is not None:
        header["policy_artifacts"] = [
            None if artifact is None else artifact.model_dump(mode="json")
            for artifact in trajectory.policy_artifacts
        ]
    return header


def _validate_event_compatibility(
    schema: int,
    compatibility: DistributedModelCompatibility | None,
) -> None:
    """Bind public-event wire payloads to the exact event contract."""
    if compatibility is None:
        return
    fingerprint = compatibility.public_event_schema_fingerprint
    if schema in {11, 12} and fingerprint != PUBLIC_EVENT_SCHEMA_FINGERPRINT:
        raise ValueError(
            "schema 11/12 compatibility is missing the public event fingerprint"
        )
    if schema not in {11, 12} and fingerprint:
        raise ValueError(
            "legacy trajectory compatibility cannot claim a public event schema"
        )


def _validate_sequence_artifact_compatibility(
    schema: int,
    trajectories: Sequence[GameTrajectory],
    compatibility: DistributedModelCompatibility | None,
) -> None:
    """Bind every schema-12 seat artifact to the batch model contract."""
    if schema != 12:
        return
    if compatibility is None:
        raise ValueError("schema 12 requires distributed compatibility metadata")
    for trajectory in trajectories:
        artifacts = trajectory.policy_artifacts
        if artifacts is None:
            raise ValueError("schema 12 trajectory is missing policy artifacts")
        if any(
            artifact is not None and artifact.compatibility != compatibility
            for artifact in artifacts
        ):
            raise ValueError(
                "trajectory policy artifact differs from batch compatibility"
            )


def _trajectory_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
    *,
    schema: int,
) -> GameTrajectory:
    input_schema = header.get("policy_input_schema")
    if input_schema is not None:
        validate_policy_input_schema_metadata(input_schema)
    elif _header_uses_current_policy_input_widths(header):
        raise ValueError("current policy input schema metadata is missing")
    event_schema = header.get("public_event_schema")
    if schema in {11, 12}:
        validate_public_event_schema_metadata(event_schema)
    elif event_schema is not None:
        raise ValueError("public event schema metadata requires trajectory schema 11")
    artifact_header = header.get("policy_artifacts")
    if schema == 12:
        if not isinstance(artifact_header, Sequence) or isinstance(
            artifact_header,
            (str, bytes),
        ):
            raise ValueError("schema 12 requires policy artifact metadata")
        artifact_values = tuple(artifact_header)
        if len(artifact_values) != 2:
            raise ValueError("schema 12 requires two policy artifact slots")
        policy_artifacts = cast(
            tuple[PolicyArtifactIdentity | None, PolicyArtifactIdentity | None],
            tuple(
                None
                if value is None
                else PolicyArtifactIdentity.model_validate(value)
                for value in artifact_values
            ),
        )
    else:
        if artifact_header is not None:
            raise ValueError("policy artifact metadata requires trajectory schema 12")
        policy_artifacts = None
    seats_reward = tuple(float(value) for value in _sequence(header["seats_reward"]))
    if len(seats_reward) != 2:
        raise ValueError("seats_reward must contain exactly two values")
    deck_context_header = header.get("deck_context")
    if schema >= 4 and not isinstance(deck_context_header, Mapping):
        raise ValueError("schema 4+ trajectories require deck context")
    return GameTrajectory(
        game_id=str(header["game_id"]),
        seats_reward=(seats_reward[0], seats_reward[1]),
        decisions=(),
        metadata=_metadata_from_header(cast(Mapping[str, Any], header["metadata"])),
        deck_context=(
            None
            if not isinstance(deck_context_header, Mapping)
            else _deck_context_from_header(
                cast(Mapping[str, Any], deck_context_header),
                frames,
            )
        ),
        array_block=_array_block_from_header(
            cast(Mapping[str, Any], header["array_block"]),
            frames,
            require_sampling_temperatures=schema >= 2,
            require_token_trace=schema == 3,
            require_factual_targets=schema in {6, 7, 9, 10},
            allow_factual_targets=schema in {6, 7, 8, 9, 10},
            legacy_factual_schema=schema == 6,
            allow_search_evidence=schema in {8, 10},
            require_planner_behavior=schema == 9,
            allow_planner_behavior=schema == 9,
            require_macro_credit=schema == 10,
            allow_macro_credit=schema == 10,
            require_public_events=schema in {11, 12},
            allow_public_events=schema in {11, 12},
        ),
        policy_artifacts=policy_artifacts,
    )


def _header_uses_current_policy_input_widths(header: Mapping[str, Any]) -> bool:
    """Distinguish stripped current metadata from readable historical rows."""
    array_block = header.get("array_block")
    if not isinstance(array_block, Mapping):
        return False
    states = array_block.get("states")
    options = array_block.get("options")
    if not isinstance(states, Mapping) or not isinstance(options, Mapping):
        return False
    state_scalars = states.get("scalars")
    option_scalars = options.get("scalars")
    if not isinstance(state_scalars, Mapping) or not isinstance(
        option_scalars, Mapping
    ):
        return False
    state_shape = state_scalars.get("shape")
    option_shape = option_scalars.get("shape")
    return (
        isinstance(state_shape, Sequence)
        and isinstance(option_shape, Sequence)
        and bool(state_shape)
        and bool(option_shape)
        and int(state_shape[-1]) == TOKEN_SCALAR_SIZE
        and int(option_shape[-1]) == SCALAR_FEATURE_SIZE
    )


def _metadata_to_header(metadata: GameMetadata) -> dict[str, Any]:
    return {
        "deck_signature": metadata.deck_signature,
        "opponent_deck_signature": metadata.opponent_deck_signature,
        "seat_0_signature": metadata.seat_0_signature,
        "seat_1_signature": metadata.seat_1_signature,
        "opponent_name": metadata.opponent_name,
        "opponent_tier": int(metadata.opponent_tier),
        "policy_version": int(metadata.policy_version),
        "episode_length": int(metadata.episode_length),
        "extra": dict(metadata.extra or {}),
    }


def _metadata_from_header(header: Mapping[str, Any]) -> GameMetadata:
    extra = header.get("extra")
    return GameMetadata(
        deck_signature=str(header.get("deck_signature", "")),
        opponent_deck_signature=str(header.get("opponent_deck_signature", "")),
        seat_0_signature=str(header.get("seat_0_signature", "")),
        seat_1_signature=str(header.get("seat_1_signature", "")),
        opponent_name=str(header.get("opponent_name", "")),
        opponent_tier=int(header.get("opponent_tier", -1)),
        policy_version=int(header.get("policy_version", 0)),
        episode_length=int(header.get("episode_length", 0)),
        extra=cast(Mapping[str, Any], extra) if isinstance(extra, Mapping) else None,
    )


def _array_block_to_header(
    block: TrajectoryArrayBlock,
    frames: list[memoryview],
) -> dict[str, Any]:
    if block.sampling_temperatures is None:
        raise ValueError("schema 2 trajectories require sampling_temperatures")
    state_header = {
        "card_ids": _array_to_header(block.states.card_ids, frames),
        "areas": _array_to_header(block.states.areas, frames),
        "owner_roles": _array_to_header(block.states.owner_roles, frames),
        "token_kinds": _array_to_header(block.states.token_kinds, frames),
        "scalars": _array_to_header(block.states.scalars, frames),
        "last_attack_ids": _array_to_header(
            block.states.last_attack_ids,
            frames,
        ),
        "padding_mask": _array_to_header(block.states.padding_mask, frames),
    }
    optional_state_arrays = {
        "attachment_card_ids": block.states.attachment_card_ids,
        "attachment_parent_indices": block.states.attachment_parent_indices,
        "attachment_kinds": block.states.attachment_kinds,
        "entity_slots": block.states.entity_slots,
    }
    for name, array in optional_state_arrays.items():
        if array is not None:
            state_header[name] = _array_to_header(array, frames)
    header = {
        "states": state_header,
        "options": {
            "option_types": _array_to_header(block.options.option_types, frames),
            "contexts": _array_to_header(block.options.contexts, frames),
            "entity_slots": _array_to_header(block.options.entity_slots, frames),
            "entity_slot_mask": _array_to_header(
                block.options.entity_slot_mask,
                frames,
            ),
            "attack_ids": _array_to_header(block.options.attack_ids, frames),
            "card_ids": _array_to_header(block.options.card_ids, frames),
            "scalars": _array_to_header(block.options.scalars, frames),
            "dynamic_effect_features": _array_to_header(
                block.options.dynamic_effect_features,
                frames,
            ),
            "dynamic_effect_masks": _array_to_header(
                block.options.dynamic_effect_masks,
                frames,
            ),
            "valid_options": _array_to_header(block.options.valid_options, frames),
            "min_counts": _array_to_header(block.options.min_counts, frames),
            "max_counts": _array_to_header(block.options.max_counts, frames),
        },
        "seats": _array_to_header(block.seats, frames),
        "decision_indices": _array_to_header(block.decision_indices, frames),
        "action_offsets": _array_to_header(block.action_offsets, frames),
        "action_indices": _array_to_header(block.action_indices, frames),
        "action_logprobs": _array_to_header(block.action_logprobs, frames),
        "value_preds": _array_to_header(block.value_preds, frames),
        "policy_versions": _array_to_header(block.policy_versions, frames),
        "sampling_temperatures": _array_to_header(
            block.sampling_temperatures,
            frames,
        ),
    }
    if block.behavior_kinds is not None:
        header["behavior_kinds"] = _array_to_header(
            block.behavior_kinds,
            frames,
        )
    if block.has_token_trace:
        header.update(
            {
                "token_offsets": _array_to_header(
                    cast(np.ndarray, block.token_offsets),
                    frames,
                ),
                "token_logprobs": _array_to_header(
                    cast(np.ndarray, block.token_logprobs),
                    frames,
                ),
                "prefix_value_preds": _array_to_header(
                    cast(np.ndarray, block.prefix_value_preds),
                    frames,
                ),
                "stop_sampled": _array_to_header(
                    cast(np.ndarray, block.stop_sampled),
                    frames,
                ),
            }
        )
    if block.has_engine_teacher_targets:
        header.update(
            {
                "engine_teacher_action_offsets": _array_to_header(
                    cast(np.ndarray, block.engine_teacher_action_offsets),
                    frames,
                ),
                "engine_teacher_action_indices": _array_to_header(
                    cast(np.ndarray, block.engine_teacher_action_indices),
                    frames,
                ),
                "engine_teacher_confidences": _array_to_header(
                    cast(np.ndarray, block.engine_teacher_confidences),
                    frames,
                ),
                "engine_teacher_weights": _array_to_header(
                    cast(np.ndarray, block.engine_teacher_weights),
                    frames,
                ),
                "engine_teacher_masks": _array_to_header(
                    cast(np.ndarray, block.engine_teacher_masks),
                    frames,
                ),
            }
        )
    if block.engine_teacher_search is not None:
        search = block.engine_teacher_search
        header["engine_teacher_search"] = {
            "decision_candidate_offsets": _array_to_header(
                search.decision_candidate_offsets,
                frames,
            ),
            "candidate_action_offsets": _array_to_header(
                search.candidate_action_offsets,
                frames,
            ),
            "candidate_action_indices": _array_to_header(
                search.candidate_action_indices,
                frames,
            ),
            "candidate_features": _array_to_header(
                search.candidate_features,
                frames,
            ),
            "legal_action_counts": _array_to_header(
                search.legal_action_counts,
                frames,
            ),
            "world_counts": _array_to_header(search.world_counts, frames),
            "exhaustive": _array_to_header(search.exhaustive, frames),
            "exact": _array_to_header(search.exact, frames),
            "masks": _array_to_header(search.masks, frames),
        }
    if block.has_factual_targets:
        header.update(
            {
                "factual_effect_targets": _array_to_header(
                    cast(np.ndarray, block.factual_effect_targets),
                    frames,
                ),
                "factual_actor_relations": _array_to_header(
                    cast(np.ndarray, block.factual_actor_relations),
                    frames,
                ),
                "factual_next_contexts": _array_to_header(
                    cast(np.ndarray, block.factual_next_contexts),
                    frames,
                ),
            }
        )
    if block.factual_transition_steps is not None:
        header["factual_transition_steps"] = _array_to_header(
            block.factual_transition_steps,
            frames,
        )
    if block.executed_macros is not None:
        macros = block.executed_macros
        header["executed_macros"] = {
            "aggregation_fingerprint": macros.aggregation_fingerprint,
            "continuation_summary_fingerprint": (
                macros.continuation_summary_fingerprint
            ),
            **{
                name: _array_to_header(getattr(macros, name), frames)
                for name in executed_macro_array_field_names()
            },
        }
    if block.macro_teacher is not None:
        teacher = block.macro_teacher
        header["macro_teacher"] = {
            name: _array_to_header(getattr(teacher, name), frames)
            for name in macro_teacher_array_field_names()
        }
    if block.planner_behavior is not None:
        planner = block.planner_behavior
        header["planner_behavior"] = {
            name: _array_to_header(getattr(planner, name), frames)
            for name in planner_array_field_names()
        }
    if (
        block.executed_endpoint_value_indices is not None
        or block.executed_endpoint_values is not None
    ):
        endpoint_values = cast(
            ExecutedEndpointValueArrayBlock,
            block.executed_endpoint_values,
        )
        header["executed_endpoint_value_indices"] = _array_to_header(
            cast(np.ndarray, block.executed_endpoint_value_indices),
            frames,
        )
        header["executed_endpoint_values"] = {
            name: _array_to_header(getattr(endpoint_values, name), frames)
            for name in endpoint_value_array_field_names()
        }
    if block.public_events is not None:
        validate_public_event_array_block(block.public_events)
        header["public_events"] = {
            name: _array_to_header(getattr(block.public_events, name), frames)
            for name in public_event_array_field_names()
        }
    return header


def _array_block_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
    *,
    require_sampling_temperatures: bool,
    require_token_trace: bool,
    require_factual_targets: bool,
    allow_factual_targets: bool,
    legacy_factual_schema: bool,
    allow_search_evidence: bool,
    require_planner_behavior: bool,
    allow_planner_behavior: bool,
    require_macro_credit: bool,
    allow_macro_credit: bool,
    require_public_events: bool,
    allow_public_events: bool,
) -> TrajectoryArrayBlock:
    if require_sampling_temperatures and "sampling_temperatures" not in header:
        raise ValueError("schema 2 trajectories require sampling_temperatures")
    public_event_header = header.get("public_events")
    if require_public_events and not isinstance(public_event_header, Mapping):
        raise ValueError("schema 11 trajectories require public event deltas")
    if public_event_header is not None and not allow_public_events:
        raise ValueError("public event deltas require trajectory schema 11")
    if public_event_header is not None and not isinstance(
        public_event_header,
        Mapping,
    ):
        raise ValueError("public event header must be a mapping")
    if isinstance(public_event_header, Mapping) and not all(
        name in public_event_header for name in public_event_array_field_names()
    ):
        raise ValueError("trajectory contains incomplete public event evidence")
    token_field_names = (
        "token_offsets",
        "token_logprobs",
        "prefix_value_preds",
        "stop_sampled",
    )
    token_fields_present = tuple(name in header for name in token_field_names)
    if require_token_trace and not all(token_fields_present):
        raise ValueError("schema 3 trajectories require token behavior traces")
    if any(token_fields_present) and not all(token_fields_present):
        raise ValueError("trajectory contains an incomplete token behavior trace")
    teacher_field_names = (
        "engine_teacher_action_offsets",
        "engine_teacher_action_indices",
        "engine_teacher_confidences",
        "engine_teacher_weights",
        "engine_teacher_masks",
    )
    teacher_fields_present = tuple(name in header for name in teacher_field_names)
    if any(teacher_fields_present) and not all(teacher_fields_present):
        raise ValueError("trajectory contains incomplete engine teacher evidence")
    search_header = header.get("engine_teacher_search")
    if search_header is not None and not allow_search_evidence:
        raise ValueError("complete-action search evidence requires schema 8")
    if search_header is not None and not isinstance(search_header, Mapping):
        raise ValueError("engine teacher search header must be a mapping")
    search_field_names = (
        "decision_candidate_offsets",
        "candidate_action_offsets",
        "candidate_action_indices",
        "candidate_features",
        "legal_action_counts",
        "world_counts",
        "exhaustive",
        "exact",
        "masks",
    )
    if isinstance(search_header, Mapping) and not all(
        name in search_header for name in search_field_names
    ):
        raise ValueError(
            "trajectory contains incomplete engine teacher search evidence"
        )
    planner_header = header.get("planner_behavior")
    if planner_header is not None and not allow_planner_behavior:
        raise ValueError("planner behavior evidence requires schema 9")
    if planner_header is not None and not isinstance(planner_header, Mapping):
        raise ValueError("planner behavior header must be a mapping")
    if require_planner_behavior and not isinstance(planner_header, Mapping):
        raise ValueError("schema 9 trajectories require planner behavior evidence")
    if isinstance(planner_header, Mapping) and (
        any(teacher_fields_present) or search_header is not None
    ):
        raise ValueError("schema 9 cannot contain legacy engine teacher evidence")
    if isinstance(planner_header, Mapping) and not all(
        name in planner_header for name in planner_array_field_names()
    ):
        raise ValueError("trajectory contains incomplete planner behavior evidence")
    endpoint_indices_present = "executed_endpoint_value_indices" in header
    endpoint_header = header.get("executed_endpoint_values")
    if endpoint_header is not None and not (
        allow_planner_behavior or allow_macro_credit
    ):
        raise ValueError("executed endpoint value rows require schema 9 or 10")
    if endpoint_indices_present != isinstance(endpoint_header, Mapping):
        raise ValueError("trajectory contains incomplete endpoint value evidence")
    if isinstance(endpoint_header, Mapping) and not all(
        name in endpoint_header for name in endpoint_value_array_field_names()
    ):
        raise ValueError("trajectory contains incomplete endpoint value table")
    macro_header = header.get("executed_macros")
    macro_teacher_header = header.get("macro_teacher")
    macro_steps_present = "factual_transition_steps" in header
    if require_macro_credit and not isinstance(macro_header, Mapping):
        raise ValueError("schema 10 trajectories require executed macro fields")
    if (macro_header is not None or macro_teacher_header is not None) and not (
        allow_macro_credit
    ):
        raise ValueError("macro evidence requires schema 10")
    if macro_header is not None and not isinstance(macro_header, Mapping):
        raise ValueError("executed macro header must be a mapping")
    if macro_teacher_header is not None and not isinstance(
        macro_teacher_header,
        Mapping,
    ):
        raise ValueError("macro teacher header must be a mapping")
    if isinstance(macro_header, Mapping) and (
        not macro_steps_present
        or not all(name in macro_header for name in executed_macro_array_field_names())
        or "aggregation_fingerprint" not in macro_header
        or "continuation_summary_fingerprint" not in macro_header
    ):
        raise ValueError("trajectory contains incomplete executed macro evidence")
    if isinstance(macro_teacher_header, Mapping) and not all(
        name in macro_teacher_header for name in macro_teacher_array_field_names()
    ):
        raise ValueError("trajectory contains incomplete macro teacher evidence")
    factual_field_names = (
        ("factual_effect_targets", "factual_endpoints")
        if legacy_factual_schema
        else (
            "factual_effect_targets",
            "factual_actor_relations",
            "factual_next_contexts",
        )
    )
    factual_fields_present = tuple(name in header for name in factual_field_names)
    if require_factual_targets and not all(factual_fields_present):
        raise ValueError("factual trajectory schema requires dense factual targets")
    if any(factual_fields_present) and not all(factual_fields_present):
        raise ValueError("trajectory contains incomplete dense factual targets")
    if any(factual_fields_present) and not allow_factual_targets:
        raise ValueError("dense factual targets require a factual trajectory schema")
    states = cast(Mapping[str, Any], header["states"])
    options = cast(Mapping[str, Any], header["options"])
    block = TrajectoryArrayBlock(
        states=StateArrayBlock(
            card_ids=_array_from_header(states["card_ids"], frames),
            areas=_array_from_header(states["areas"], frames),
            owner_roles=_array_from_header(states["owner_roles"], frames),
            token_kinds=_array_from_header(states["token_kinds"], frames),
            scalars=_array_from_header(states["scalars"], frames),
            last_attack_ids=_array_from_header(states["last_attack_ids"], frames),
            padding_mask=_array_from_header(states["padding_mask"], frames),
            attachment_card_ids=_optional_array_from_header(
                states,
                "attachment_card_ids",
                frames,
            ),
            attachment_parent_indices=_optional_array_from_header(
                states,
                "attachment_parent_indices",
                frames,
            ),
            attachment_kinds=_optional_array_from_header(
                states,
                "attachment_kinds",
                frames,
            ),
            entity_slots=_optional_array_from_header(
                states,
                "entity_slots",
                frames,
            ),
        ),
        options=OptionArrayBlock(
            option_types=_array_from_header(options["option_types"], frames),
            contexts=_array_from_header(options["contexts"], frames),
            entity_slots=_array_from_header(options["entity_slots"], frames),
            entity_slot_mask=_array_from_header(options["entity_slot_mask"], frames),
            attack_ids=_array_from_header(options["attack_ids"], frames),
            card_ids=_array_from_header(options["card_ids"], frames),
            scalars=_array_from_header(options["scalars"], frames),
            dynamic_effect_features=_array_from_header(
                options["dynamic_effect_features"],
                frames,
            ),
            dynamic_effect_masks=_array_from_header(
                options["dynamic_effect_masks"],
                frames,
            ),
            valid_options=_array_from_header(options["valid_options"], frames),
            min_counts=_array_from_header(options["min_counts"], frames),
            max_counts=_array_from_header(options["max_counts"], frames),
        ),
        seats=_array_from_header(header["seats"], frames),
        decision_indices=_array_from_header(header["decision_indices"], frames),
        action_offsets=_array_from_header(header["action_offsets"], frames),
        action_indices=_array_from_header(header["action_indices"], frames),
        action_logprobs=_array_from_header(header["action_logprobs"], frames),
        value_preds=_array_from_header(header["value_preds"], frames),
        policy_versions=_array_from_header(header["policy_versions"], frames),
        behavior_kinds=_optional_array_from_header(
            header,
            "behavior_kinds",
            frames,
        ),
        sampling_temperatures=_optional_array_from_header(
            header,
            "sampling_temperatures",
            frames,
        ),
        token_offsets=_optional_array_from_header(
            header,
            "token_offsets",
            frames,
        ),
        token_logprobs=_optional_array_from_header(
            header,
            "token_logprobs",
            frames,
        ),
        prefix_value_preds=_optional_array_from_header(
            header,
            "prefix_value_preds",
            frames,
        ),
        stop_sampled=_optional_array_from_header(
            header,
            "stop_sampled",
            frames,
        ),
        engine_teacher_action_offsets=_optional_array_from_header(
            header,
            "engine_teacher_action_offsets",
            frames,
        ),
        engine_teacher_action_indices=_optional_array_from_header(
            header,
            "engine_teacher_action_indices",
            frames,
        ),
        engine_teacher_confidences=_optional_array_from_header(
            header,
            "engine_teacher_confidences",
            frames,
        ),
        engine_teacher_weights=_optional_array_from_header(
            header,
            "engine_teacher_weights",
            frames,
        ),
        engine_teacher_masks=_optional_array_from_header(
            header,
            "engine_teacher_masks",
            frames,
        ),
        engine_teacher_search=(
            None
            if not isinstance(search_header, Mapping)
            else SearchEvidenceArrayBlock(
                decision_candidate_offsets=_array_from_header(
                    search_header["decision_candidate_offsets"],
                    frames,
                ),
                candidate_action_offsets=_array_from_header(
                    search_header["candidate_action_offsets"],
                    frames,
                ),
                candidate_action_indices=_array_from_header(
                    search_header["candidate_action_indices"],
                    frames,
                ),
                candidate_features=_array_from_header(
                    search_header["candidate_features"],
                    frames,
                ),
                legal_action_counts=_array_from_header(
                    search_header["legal_action_counts"],
                    frames,
                ),
                world_counts=_array_from_header(
                    search_header["world_counts"],
                    frames,
                ),
                exhaustive=_array_from_header(
                    search_header["exhaustive"],
                    frames,
                ),
                exact=_array_from_header(search_header["exact"], frames),
                masks=_array_from_header(search_header["masks"], frames),
            )
        ),
        factual_effect_targets=_optional_array_from_header(
            header,
            "factual_effect_targets",
            frames,
        ),
        factual_actor_relations=(
            _optional_array_from_header(header, "factual_endpoints", frames)
            if legacy_factual_schema
            else _optional_array_from_header(
                header,
                "factual_actor_relations",
                frames,
            )
        ),
        factual_next_contexts=(
            _legacy_factual_contexts(
                _optional_array_from_header(header, "factual_endpoints", frames)
            )
            if legacy_factual_schema
            else _optional_array_from_header(
                header,
                "factual_next_contexts",
                frames,
            )
        ),
        factual_transition_steps=_optional_array_from_header(
            header,
            "factual_transition_steps",
            frames,
        ),
        executed_macros=(
            None
            if not isinstance(macro_header, Mapping)
            else _executed_macros_from_header(macro_header, frames)
        ),
        macro_teacher=(
            None
            if not isinstance(macro_teacher_header, Mapping)
            else _macro_teacher_from_header(macro_teacher_header, frames)
        ),
        planner_behavior=(
            None
            if not isinstance(planner_header, Mapping)
            else _planner_evidence_from_header(planner_header, frames)
        ),
        executed_endpoint_value_indices=_optional_array_from_header(
            header,
            "executed_endpoint_value_indices",
            frames,
        ),
        executed_endpoint_values=(
            None
            if not isinstance(endpoint_header, Mapping)
            else _endpoint_values_from_header(endpoint_header, frames)
        ),
        public_events=(
            None
            if not isinstance(public_event_header, Mapping)
            else _public_events_from_header(public_event_header, frames)
        ),
    )
    validate_trajectory_array_block(block)
    return block


def _trajectory_batch_schema(trajectories: Sequence[GameTrajectory]) -> int:
    token_trace_presence: list[bool] = []
    deck_context_presence: list[bool] = []
    factual_target_presence: list[bool] = []
    search_evidence_presence: list[bool] = []
    planner_behavior_presence: list[bool] = []
    macro_credit_presence: list[bool] = []
    public_event_presence: list[bool] = []
    policy_artifact_presence: list[bool] = []
    for trajectory in trajectories:
        block = trajectory.array_block
        if block is None:
            raise ValueError("distributed trajectories must be array-backed")
        token_trace_presence.append(block.has_token_trace)
        deck_context_presence.append(trajectory.deck_context is not None)
        factual_target_presence.append(block.has_factual_targets)
        search_evidence_presence.append(block.engine_teacher_search is not None)
        planner_behavior_presence.append(block.planner_behavior is not None)
        macro_credit_presence.append(block.executed_macros is not None)
        public_event_presence.append(block.has_public_events)
        policy_artifact_presence.append(trajectory.policy_artifacts is not None)
    if any(deck_context_presence) and not all(deck_context_presence):
        raise ValueError("cannot mix trajectories with and without deck context")
    if any(factual_target_presence) and not all(factual_target_presence):
        raise ValueError("cannot mix factual and legacy trajectories in one batch")
    if any(public_event_presence):
        if not all(public_event_presence):
            raise ValueError(
                "cannot mix public-event and legacy trajectories in one batch"
            )
        if not all(deck_context_presence) or not all(token_trace_presence):
            raise ValueError(
                "schema 11 recurrent trajectories require deck and token context"
            )
        if (
            any(factual_target_presence)
            or any(search_evidence_presence)
            or any(planner_behavior_presence)
            or any(macro_credit_presence)
            or any(
                cast(
                    TrajectoryArrayBlock,
                    trajectory.array_block,
                ).has_engine_teacher_targets
                for trajectory in trajectories
            )
        ):
            raise ValueError("schema 11 is a PPO-only public-event trajectory")
        if any(policy_artifact_presence):
            if not all(policy_artifact_presence):
                raise ValueError(
                    "cannot mix schema-11 and schema-12 trajectories in one batch"
                )
            return 12
        return 11
    if any(policy_artifact_presence):
        raise ValueError("policy artifacts require public-event trajectories")
    if any(macro_credit_presence):
        if not all(macro_credit_presence):
            raise ValueError(
                "cannot mix schema-10 and legacy trajectories in one batch"
            )
        if any(planner_behavior_presence):
            raise ValueError("schema 10 cannot contain pre-action planner behavior")
        if not all(deck_context_presence) or not all(factual_target_presence):
            raise ValueError("schema 10 requires deck and factual context")
        return 10
    if any(planner_behavior_presence):
        if not all(planner_behavior_presence):
            raise ValueError("cannot mix schema-9 and legacy trajectories in one batch")
        if not all(deck_context_presence) or not all(factual_target_presence):
            raise ValueError(
                "schema 9 planner trajectories require deck and factual context"
            )
        if any(search_evidence_presence) or any(
            cast(
                TrajectoryArrayBlock, trajectory.array_block
            ).has_engine_teacher_targets
            for trajectory in trajectories
        ):
            raise ValueError("schema 9 cannot contain legacy engine teacher evidence")
        return 9
    if any(token_trace_presence) and not all(token_trace_presence):
        raise ValueError("cannot mix token-trace and legacy trajectories in one batch")
    if any(search_evidence_presence):
        if not deck_context_presence or not all(deck_context_presence):
            raise ValueError("schema 8 search evidence requires deck context")
        return 8
    if factual_target_presence and all(factual_target_presence):
        if not deck_context_presence or not all(deck_context_presence):
            raise ValueError("schema 8 factual trajectories require deck context")
        return 8
    if deck_context_presence and all(deck_context_presence):
        # Schema 5 is the current deck-context wire capability. Teacher fields
        # remain sparse and optional per trajectory, including all-zero batches.
        return 5
    return 3 if token_trace_presence and all(token_trace_presence) else 2


def _planner_evidence_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> PlannerEvidenceArrayBlock:
    arrays = {
        name: _array_from_header(header[name], frames)
        for name in planner_array_field_names()
    }
    block = PlannerEvidenceArrayBlock(**cast(Any, arrays))
    validate_planner_evidence_array_block(
        block,
        decision_count=block.decision_count,
    )
    return block


def _public_events_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> PublicEventArrayBlock:
    arrays = {
        name: _array_from_header(header[name], frames)
        for name in public_event_array_field_names()
    }
    block = PublicEventArrayBlock(**cast(Any, arrays))
    validate_public_event_array_block(block)
    return block


def _endpoint_values_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> ExecutedEndpointValueArrayBlock:
    arrays = {
        name: _array_from_header(header[name], frames)
        for name in endpoint_value_array_field_names()
    }
    block = ExecutedEndpointValueArrayBlock(**cast(Any, arrays))
    validate_executed_endpoint_value_array_block(block)
    return block


def _executed_macros_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> ExecutedMacroArrayBlock:
    arrays = {
        name: _array_from_header(header[name], frames)
        for name in executed_macro_array_field_names()
    }
    block = ExecutedMacroArrayBlock(
        **cast(Any, arrays),
        aggregation_fingerprint=str(header["aggregation_fingerprint"]),
        continuation_summary_fingerprint=str(
            header["continuation_summary_fingerprint"]
        ),
    )
    validate_executed_macro_array_block(
        block,
        decision_count=len(block.decision_macro_indices),
    )
    return block


def _macro_teacher_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> MacroTeacherArrayBlock:
    arrays = {
        name: _array_from_header(header[name], frames)
        for name in macro_teacher_array_field_names()
    }
    block = MacroTeacherArrayBlock(**cast(Any, arrays))
    validate_macro_teacher_array_block(block)
    return block


def _deck_context_to_header(
    context: TrajectoryDeckContext,
    frames: list[memoryview],
) -> dict[str, Any]:
    return {
        "seat_card_ids": _array_to_header(context.seat_card_ids, frames),
        "seat_signatures": list(context.seat_signatures),
    }


def _deck_context_from_header(
    header: Mapping[str, Any],
    frames: Sequence[memoryview],
) -> TrajectoryDeckContext:
    signatures = tuple(str(value) for value in _sequence(header["seat_signatures"]))
    if len(signatures) != 2:
        raise ValueError("trajectory deck context must contain two signatures")
    return TrajectoryDeckContext(
        seat_card_ids=_array_from_header(header["seat_card_ids"], frames),
        seat_signatures=(signatures[0], signatures[1]),
    )


def _array_to_header(array: np.ndarray, frames: list[memoryview]) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    frame_index = len(frames)
    frames.append(memoryview(cast(Any, contiguous)))
    return {
        "frame": frame_index,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }


def _optional_array_from_header(
    header: Mapping[str, Any],
    name: str,
    frames: Sequence[memoryview],
) -> np.ndarray | None:
    array_header = header.get(name)
    if array_header is None:
        return None
    return _array_from_header(array_header, frames)


def _legacy_factual_contexts(actor_relations: np.ndarray | None) -> np.ndarray | None:
    """Recover schema-6 MAIN/terminal context and mark handoff as unknown."""
    if actor_relations is None:
        return None
    contexts = np.full(actor_relations.shape, FACTUAL_NEXT_CONTEXT_OOV, dtype=np.uint8)
    contexts[actor_relations == int(FactualActorRelation.SAME_SEAT)] = 0
    contexts[actor_relations == int(FactualActorRelation.TERMINAL)] = (
        FACTUAL_NEXT_CONTEXT_TERMINAL
    )
    return contexts


def _array_from_header(header: Any, frames: Sequence[memoryview]) -> np.ndarray:
    if not isinstance(header, Mapping):
        raise ValueError("array header must be a mapping")
    frame_index = int(header["frame"])
    dtype = np.dtype(str(header["dtype"]))
    shape = tuple(int(value) for value in _sequence(header["shape"]))
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError("array frame index is out of range")
    array = np.frombuffer(frames[frame_index], dtype=dtype).reshape(shape)
    return array.copy()


def _frame_nbytes(frame: bytes | memoryview) -> int:
    """Return payload bytes for a bytes-like frame."""
    if isinstance(frame, memoryview):
        return int(frame.nbytes)
    return len(frame)


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("expected a sequence")
    return value
