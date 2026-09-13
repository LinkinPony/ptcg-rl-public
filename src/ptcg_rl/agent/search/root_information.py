"""Deduplicated root-information leaves for shared train/serve scoring."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generic, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEAF_DOMAIN = b"ptcg-rl/root-information-leaf/v2\x00"
_NONTERMINAL_ENDPOINTS = frozenset(
    {SemanticEndpoint.SAME_SEAT_MAIN, SemanticEndpoint.TURN_HANDOFF}
)

TensorInputs = TypeVar("TensorInputs")
TensorInputsCovariant = TypeVar("TensorInputsCovariant", covariant=True)
Int32Array = npt.NDArray[np.int32]


class RootActorRelation(IntEnum):
    """Relation between the root seat and the actor at an endpoint."""

    SAME_SEAT = 0
    OTHER_SEAT = 1


@dataclass(frozen=True, slots=True)
class RootInformationLeaf:
    """Root-visible, scenario-independent input for one nonterminal leaf.

    ``producer_context`` contains only request-local root-visible encoder input.
    Opaque scenario handles and sampled hidden identities are deliberately not
    fields of this record. ``exact_effect`` is retained as path-local factual
    evidence, but is not a value-model input and therefore does not participate
    in equality or the information-history fingerprint.
    """

    root_observable_state: bytes
    producer_context: bytes
    belief_summary: tuple[float, ...]
    exact_effect: tuple[float, ...] = field(compare=False)
    actor_relation: RootActorRelation
    endpoint: SemanticEndpoint

    def __post_init__(self) -> None:
        """Validate the model-facing public leaf contract."""
        if not isinstance(self.root_observable_state, bytes):
            raise TypeError("root_observable_state must be bytes")
        if not self.root_observable_state:
            raise ValueError("root_observable_state must not be empty")
        if not isinstance(self.producer_context, bytes):
            raise TypeError("producer_context must be bytes")
        if not self.producer_context:
            raise ValueError("producer_context must not be empty")
        if not isinstance(self.endpoint, SemanticEndpoint):
            raise TypeError("endpoint must be SemanticEndpoint")
        if not isinstance(self.actor_relation, RootActorRelation):
            raise TypeError("actor_relation must be RootActorRelation")
        if self.endpoint not in _NONTERMINAL_ENDPOINTS:
            raise ValueError("value leaves require SAME_SEAT_MAIN or TURN_HANDOFF")
        expected_relation = (
            RootActorRelation.SAME_SEAT
            if self.endpoint is SemanticEndpoint.SAME_SEAT_MAIN
            else RootActorRelation.OTHER_SEAT
        )
        if self.actor_relation is not expected_relation:
            raise ValueError("actor relation does not match semantic endpoint")
        belief_summary = _canonical_float32(self.belief_summary)
        exact_effect = _canonical_float32(self.exact_effect)
        object.__setattr__(self, "belief_summary", belief_summary)
        object.__setattr__(self, "exact_effect", exact_effect)
        if len(exact_effect) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("exact_effect has the wrong feature width")

    @property
    def model_input_fingerprint(self) -> str:
        """Return the content key used only for leaf/value deduplication."""
        digest = hashlib.sha256()
        digest.update(_LEAF_DOMAIN)
        _update_bytes(digest, self.root_observable_state)
        _update_bytes(digest, self.producer_context)
        _update_bytes(
            digest,
            canonical_float32_vector_bytes(self.belief_summary),
        )
        digest.update(struct.pack(">ii", int(self.actor_relation), int(self.endpoint)))
        return digest.hexdigest()

    @property
    def information_history_fingerprint(self) -> str:
        """Compatibility alias for the model-input fingerprint.

        Continuation/nonanticipativity histories have a separate path-aware
        identity and must not use this value-deduplication key.
        """
        return self.model_input_fingerprint


@dataclass(frozen=True, slots=True)
class RootInformationLeafCell:
    """One candidate/scenario cell referencing a public value leaf."""

    cell_index: int
    leaf: RootInformationLeaf

    def __post_init__(self) -> None:
        """Reject negative flattened grid positions."""
        if self.cell_index < 0:
            raise ValueError("cell_index must be non-negative")


@dataclass(frozen=True, slots=True)
class RootInformationLeafBatch:
    """Unique leaves plus a candidate-major cell-to-leaf gather map."""

    leaves: tuple[RootInformationLeaf, ...]
    cell_to_leaf: Int32Array

    def __post_init__(self) -> None:
        """Validate immutable gather indices."""
        if self.cell_to_leaf.dtype != np.dtype(np.int32):
            raise TypeError("cell_to_leaf must use int32")
        if self.cell_to_leaf.ndim != 1:
            raise ValueError("cell_to_leaf must be one-dimensional")
        if self.cell_to_leaf.flags.writeable:
            raise ValueError("cell_to_leaf must be read-only")
        if bool(np.any(self.cell_to_leaf < -1)):
            raise ValueError("cell_to_leaf may use only -1 as its terminal sentinel")
        if len({leaf.model_input_fingerprint for leaf in self.leaves}) != len(
            self.leaves
        ):
            raise ValueError("leaf batch contains duplicate information histories")
        mapped = self.cell_to_leaf[self.cell_to_leaf >= 0]
        if mapped.size and int(mapped.max()) >= len(self.leaves):
            raise ValueError("cell_to_leaf references an absent unique leaf")
        if {int(value) for value in mapped} != set(range(len(self.leaves))):
            raise ValueError("root-information leaf batch contains an unused leaf")

    def tensorize(
        self,
        tensorizer: RootInformationTensorizer[TensorInputs],
    ) -> RootInformationStateTensorBatch[TensorInputs]:
        """Decode/tensorize each unique information history exactly once."""
        return RootInformationStateTensorBatch(
            leaves=self.leaves,
            cell_to_leaf=self.cell_to_leaf,
            model_inputs=tensorizer.tensorize(self.leaves),
        )


@dataclass(frozen=True, slots=True)
class RootInformationStateTensorBatch(Generic[TensorInputs]):
    """Unique model inputs and the gather map back to scenario cells."""

    leaves: tuple[RootInformationLeaf, ...]
    cell_to_leaf: Int32Array
    model_inputs: TensorInputs


class RootInformationTensorizer(Protocol[TensorInputsCovariant]):
    """Shared root-visible decoder/encoder input builder."""

    def tensorize(
        self,
        leaves: tuple[RootInformationLeaf, ...],
    ) -> TensorInputsCovariant:
        """Build one model batch aligned with unique leaves."""


def deduplicate_root_information_leaves(
    *,
    cell_count: int,
    cells: tuple[RootInformationLeafCell, ...],
) -> RootInformationLeafBatch:
    """Deduplicate nonterminal cells before observation decode and inference."""
    if cell_count <= 0:
        raise ValueError("cell_count must be positive")
    gather = np.full(cell_count, -1, dtype=np.int32)
    leaves: list[RootInformationLeaf] = []
    index_by_fingerprint: dict[str, int] = {}
    for cell in cells:
        if cell.cell_index >= cell_count:
            raise ValueError("leaf cell index is outside the consequence grid")
        if int(gather[cell.cell_index]) >= 0:
            raise ValueError("multiple value leaves reference one consequence cell")
        fingerprint = cell.leaf.model_input_fingerprint
        leaf_index = index_by_fingerprint.get(fingerprint)
        if leaf_index is None:
            leaf_index = len(leaves)
            leaves.append(cell.leaf)
            index_by_fingerprint[fingerprint] = leaf_index
        elif leaves[leaf_index] != cell.leaf:
            raise ValueError(
                "one root information history has different scorer inputs"
            )
        gather[cell.cell_index] = leaf_index
    gather.setflags(write=False)
    return RootInformationLeafBatch(leaves=tuple(leaves), cell_to_leaf=gather)


def validate_information_history_fingerprint(value: str) -> str:
    """Validate one externally persisted root-information identity."""
    if _SHA256.fullmatch(value) is None:
        raise ValueError("expected a lowercase SHA-256 fingerprint")
    return value


def canonical_float32_vector_bytes(values: tuple[float, ...]) -> bytes:
    """Return one architecture-stable encoding of a finite float32 vector."""
    canonical = _canonical_float32(values)
    return np.asarray(canonical, dtype=np.dtype(">f4")).tobytes(order="C")


def _canonical_float32(values: tuple[float, ...]) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("root-information numeric inputs must be float32") from exc
    if array.ndim != 1 or not bool(np.isfinite(array).all()):
        raise ValueError("root-information numeric inputs must be finite vectors")
    if array.size:
        array = array.copy()
        array[array == np.float32(0.0)] = np.float32(0.0)
    return tuple(float(value) for value in array)


class _Digest(Protocol):
    def update(self, payload: bytes) -> None:
        """Add bytes to the incremental digest."""


def _update_bytes(digest: _Digest, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


__all__ = [
    "RootActorRelation",
    "RootInformationLeaf",
    "RootInformationLeafBatch",
    "RootInformationLeafCell",
    "RootInformationStateTensorBatch",
    "RootInformationTensorizer",
    "canonical_float32_vector_bytes",
    "deduplicate_root_information_leaves",
    "validate_information_history_fingerprint",
]
