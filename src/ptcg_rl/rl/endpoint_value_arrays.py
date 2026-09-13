"""Compact schema-9 rows for actual root-information value supervision."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ptcg_rl.agent.search.endpoint_value_rows import (
    ExecutedEndpointValueRow,
    ExecutedEndpointValueTable,
)
from ptcg_rl.agent.search.root_information import (
    RootActorRelation,
    RootInformationLeaf,
)
from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE


@dataclass(frozen=True, slots=True)
class ExecutedEndpointValueArrayBlock:
    """Deduplicated actual endpoints stored once per trajectory block."""

    state_offsets: np.ndarray
    state_bytes: np.ndarray
    producer_context_offsets: np.ndarray
    producer_context_bytes: np.ndarray
    belief_offsets: np.ndarray
    belief_values: np.ndarray
    exact_effects: np.ndarray
    actor_relations: np.ndarray
    endpoints: np.ndarray
    final_root_outcomes: np.ndarray
    root_players: np.ndarray
    game_fingerprints: np.ndarray
    semantic_endpoint_fingerprints: np.ndarray

    @property
    def row_count(self) -> int:
        """Return the number of unique actual endpoints."""
        return int(self.final_root_outcomes.shape[0])

    def row_at(self, index: int) -> ExecutedEndpointValueRow:
        """Reconstruct one privacy-safe root-information endpoint row."""
        if not 0 <= index < self.row_count:
            raise IndexError("executed endpoint value row is out of range")
        state_start = int(self.state_offsets[index])
        state_stop = int(self.state_offsets[index + 1])
        context_start = int(self.producer_context_offsets[index])
        context_stop = int(self.producer_context_offsets[index + 1])
        belief_start = int(self.belief_offsets[index])
        belief_stop = int(self.belief_offsets[index + 1])
        leaf = RootInformationLeaf(
            root_observable_state=self.state_bytes[state_start:state_stop].tobytes(
                order="C"
            ),
            producer_context=self.producer_context_bytes[
                context_start:context_stop
            ].tobytes(order="C"),
            belief_summary=tuple(
                float(value) for value in self.belief_values[belief_start:belief_stop]
            ),
            exact_effect=tuple(float(value) for value in self.exact_effects[index]),
            actor_relation=RootActorRelation(int(self.actor_relations[index])),
            endpoint=SemanticEndpoint(int(self.endpoints[index])),
        )
        return ExecutedEndpointValueRow(
            game_fingerprint=_fingerprint_at(self.game_fingerprints, index),
            root_player=int(self.root_players[index]),
            semantic_endpoint_fingerprint=_fingerprint_at(
                self.semantic_endpoint_fingerprints, index
            ),
            leaf=leaf,
            final_root_outcome=int(self.final_root_outcomes[index]),
        )


def build_executed_endpoint_value_array_block(
    table: ExecutedEndpointValueTable | None,
) -> ExecutedEndpointValueArrayBlock | None:
    """Flatten deduplicated actual endpoints into contiguous compact arrays."""
    if table is None or not table.rows:
        return None
    state_offsets = [0]
    state_bytes = bytearray()
    context_offsets = [0]
    context_bytes = bytearray()
    belief_offsets = [0]
    belief_values: list[float] = []
    for row in table.rows:
        state_bytes.extend(row.leaf.root_observable_state)
        state_offsets.append(len(state_bytes))
        context_bytes.extend(row.leaf.producer_context)
        context_offsets.append(len(context_bytes))
        belief_values.extend(row.leaf.belief_summary)
        belief_offsets.append(len(belief_values))
    int32_max = np.iinfo(np.int32).max
    if len(table.rows) > int32_max or any(
        value > int32_max
        for value in (len(state_bytes), len(context_bytes), len(belief_values))
    ):
        raise ValueError("executed endpoint table exceeds int32 wire capacity")
    block = ExecutedEndpointValueArrayBlock(
        state_offsets=np.asarray(state_offsets, dtype=np.int32),
        state_bytes=np.frombuffer(bytes(state_bytes), dtype=np.uint8).copy(),
        producer_context_offsets=np.asarray(context_offsets, dtype=np.int32),
        producer_context_bytes=np.frombuffer(
            bytes(context_bytes), dtype=np.uint8
        ).copy(),
        belief_offsets=np.asarray(belief_offsets, dtype=np.int32),
        belief_values=np.asarray(belief_values, dtype=np.float32),
        exact_effects=np.asarray(
            [row.leaf.exact_effect for row in table.rows], dtype=np.float32
        ),
        actor_relations=np.asarray(
            [int(row.leaf.actor_relation) for row in table.rows], dtype=np.uint8
        ),
        endpoints=np.asarray(
            [int(row.leaf.endpoint) for row in table.rows], dtype=np.uint8
        ),
        final_root_outcomes=np.asarray(
            [row.final_root_outcome for row in table.rows], dtype=np.int8
        ),
        root_players=np.asarray(
            [row.root_player for row in table.rows], dtype=np.uint8
        ),
        game_fingerprints=_fingerprint_array(
            [row.game_fingerprint for row in table.rows]
        ),
        semantic_endpoint_fingerprints=_fingerprint_array(
            [row.semantic_endpoint_fingerprint for row in table.rows]
        ),
    )
    validate_executed_endpoint_value_array_block(block)
    return block


def validate_executed_endpoint_value_array_block(
    block: ExecutedEndpointValueArrayBlock,
) -> None:
    """Reject malformed blobs, hidden duplicates, and invalid W/D/L labels."""
    row_count = block.row_count
    _require_array(block.state_offsets, "state_offsets", np.int32, (row_count + 1,))
    _require_array(
        block.state_bytes, "state_bytes", np.uint8, (int(block.state_offsets[-1]),)
    )
    _require_array(
        block.producer_context_offsets,
        "producer_context_offsets",
        np.int32,
        (row_count + 1,),
    )
    _require_array(
        block.producer_context_bytes,
        "producer_context_bytes",
        np.uint8,
        (int(block.producer_context_offsets[-1]),),
    )
    _require_array(
        block.belief_offsets,
        "belief_offsets",
        np.int32,
        (row_count + 1,),
    )
    _require_array(
        block.belief_values,
        "belief_values",
        np.float32,
        (int(block.belief_offsets[-1]),),
    )
    _require_array(
        block.exact_effects,
        "exact_effects",
        np.float32,
        (row_count, DYNAMIC_EFFECT_FEATURE_SIZE),
    )
    for name, dtype in (
        ("actor_relations", np.uint8),
        ("endpoints", np.uint8),
        ("final_root_outcomes", np.int8),
        ("root_players", np.uint8),
    ):
        _require_array(getattr(block, name), name, dtype, (row_count,))
    for name in ("game_fingerprints", "semantic_endpoint_fingerprints"):
        _require_array(getattr(block, name), name, np.uint8, (row_count, 32))
    for values, name in (
        (block.state_offsets, "state"),
        (block.producer_context_offsets, "producer context"),
        (block.belief_offsets, "belief"),
    ):
        if int(values[0]) != 0 or bool(np.any(values[1:] < values[:-1])):
            raise ValueError(f"executed endpoint {name} offsets are invalid")
    if not bool(np.isfinite(block.belief_values).all()) or not bool(
        np.isfinite(block.exact_effects).all()
    ):
        raise ValueError("executed endpoint numeric values must be finite")
    keys: set[tuple[str, int, str]] = set()
    for index in range(row_count):
        row = block.row_at(index)
        key = (
            row.game_fingerprint,
            row.root_player,
            row.semantic_endpoint_fingerprint,
        )
        if key in keys:
            raise ValueError("executed endpoint table contains a duplicate row")
        keys.add(key)


def endpoint_value_array_field_names() -> tuple[str, ...]:
    """Return the stable transport field order."""
    return tuple(field.name for field in fields(ExecutedEndpointValueArrayBlock))


def _fingerprint_array(values: list[str]) -> np.ndarray:
    return np.asarray([tuple(bytes.fromhex(value)) for value in values], dtype=np.uint8)


def _fingerprint_at(values: np.ndarray, index: int) -> str:
    return bytes(values[index]).hex()


def _require_array(
    value: np.ndarray,
    name: str,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} has the wrong NumPy dtype")
    if value.shape != shape:
        raise ValueError(f"{name} has the wrong shape")


__all__ = [
    "ExecutedEndpointValueArrayBlock",
    "build_executed_endpoint_value_array_block",
    "endpoint_value_array_field_names",
    "validate_executed_endpoint_value_array_block",
]
