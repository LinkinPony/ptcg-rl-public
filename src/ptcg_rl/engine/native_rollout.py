"""ctypes binding for the C++ stateful rollout encoder."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ptcg_rl.belief.public_catalog import PublicDeckCatalog
from ptcg_rl.context.game import default_supporter_card_ids
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.native_public_context import NativeKnownOpponentBatch
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingOutputBuffer,
    _CgTrainOutput,
)

_ABI_MAGIC = 0x43475245
_ABI_VERSION = 1
_REQUIRED_FEATURES = (1 << 5) - 1
_OK = 0
_INSUFFICIENT_CAPACITY = -2
_TOKEN_SCALAR_SIZE = 59
_OPTION_SCALAR_SIZE = 9
_DYNAMIC_EFFECT_SIZE = 33
_MAXIMUM_ENTITY_SLOTS = 2
_BELIEF_SCALAR_SIZE = 4
_HISTORY_SIZE = 8
_DECK_FLOW_SIZE = 14
NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT = (
    "cf1d0cba6050142d2ba40e9a09ef07251ac5285c1bbdf9cef8cfd8637fff264a"
)
_MODEL_ENCODING_FINGERPRINT = bytes.fromhex(
    NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT
)
_BIND_LOCK = threading.Lock()


class NativeRolloutLoadError(RuntimeError):
    """Raised when the native rollout ABI is absent or incompatible."""


class NativeRolloutCallError(RuntimeError):
    """Raised when a stateful native rollout operation fails."""


class NativeRolloutCapacityError(NativeRolloutCallError):
    """Raised when a caller-owned rollout tensor buffer is too small."""


class _CgTrainRolloutAbiDescriptor(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("descriptor_size", ctypes.c_uint32),
        ("catalog_size", ctypes.c_uint32),
        ("shape_size", ctypes.c_uint32),
        ("output_size", ctypes.c_uint32),
        ("token_scalar_size", ctypes.c_uint32),
        ("option_scalar_size", ctypes.c_uint32),
        ("dynamic_effect_size", ctypes.c_uint32),
        ("maximum_entity_slots", ctypes.c_uint32),
        ("belief_scalar_size", ctypes.c_uint32),
        ("history_size", ctypes.c_uint32),
        ("deck_flow_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
        ("model_encoding_fingerprint", ctypes.c_uint8 * 32),
    ]


class _CgTrainRolloutCatalog(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("deck_count", ctypes.c_uint32),
        ("card_vocab_size", ctypes.c_uint32),
        ("supporter_count", ctypes.c_uint32),
        ("posterior_cache_capacity", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("entry_counts", ctypes.POINTER(ctypes.c_int16)),
        ("exact_log_priors", ctypes.POINTER(ctypes.c_double)),
        ("log_combinations", ctypes.POINTER(ctypes.c_double)),
        ("log_factorials", ctypes.POINTER(ctypes.c_double)),
        ("unknown_card_probabilities", ctypes.POINTER(ctypes.c_double)),
        (
            "unknown_log_card_probabilities",
            ctypes.POINTER(ctypes.c_double),
        ),
        ("unknown_log_prior", ctypes.c_double),
        ("supporter_card_ids", ctypes.POINTER(ctypes.c_int32)),
    ]


class _CgTrainRolloutShape(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("state_token_width", ctypes.c_uint32),
        ("state_attachment_width", ctypes.c_uint32),
        ("option_width", ctypes.c_uint32),
        ("deck_width", ctypes.c_uint32),
        ("belief_row_count", ctypes.c_uint32),
        ("belief_width", ctypes.c_uint32),
    ]


class _CgTrainRolloutOutput(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("row_capacity", ctypes.c_uint32),
        ("state_token_width", ctypes.c_uint32),
        ("state_attachment_width", ctypes.c_uint32),
        ("option_width", ctypes.c_uint32),
        ("deck_width", ctypes.c_uint32),
        ("belief_row_capacity", ctypes.c_uint32),
        ("belief_width", ctypes.c_uint32),
        ("state_card_ids", ctypes.POINTER(ctypes.c_int64)),
        ("state_areas", ctypes.POINTER(ctypes.c_int64)),
        ("state_owner_roles", ctypes.POINTER(ctypes.c_int64)),
        ("state_token_kinds", ctypes.POINTER(ctypes.c_int64)),
        ("state_scalars", ctypes.POINTER(ctypes.c_float)),
        ("state_last_attack_ids", ctypes.POINTER(ctypes.c_int64)),
        ("state_padding_mask", ctypes.POINTER(ctypes.c_uint8)),
        (
            "state_attachment_card_ids",
            ctypes.POINTER(ctypes.c_uint16),
        ),
        (
            "state_attachment_parent_indices",
            ctypes.POINTER(ctypes.c_uint16),
        ),
        ("state_attachment_kinds", ctypes.POINTER(ctypes.c_uint8)),
        ("state_entity_slots", ctypes.POINTER(ctypes.c_uint8)),
        ("state_sequence_lengths", ctypes.POINTER(ctypes.c_uint32)),
        ("option_types", ctypes.POINTER(ctypes.c_int64)),
        ("option_contexts", ctypes.POINTER(ctypes.c_int64)),
        ("option_entity_slots", ctypes.POINTER(ctypes.c_int64)),
        ("option_entity_slot_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("option_attack_ids", ctypes.POINTER(ctypes.c_int64)),
        ("option_card_ids", ctypes.POINTER(ctypes.c_int64)),
        ("option_scalars", ctypes.POINTER(ctypes.c_float)),
        (
            "option_dynamic_effect_features",
            ctypes.POINTER(ctypes.c_float),
        ),
        (
            "option_dynamic_effect_masks",
            ctypes.POINTER(ctypes.c_uint8),
        ),
        ("option_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("option_min_counts", ctypes.POINTER(ctypes.c_int64)),
        ("option_max_counts", ctypes.POINTER(ctypes.c_int64)),
        ("option_lengths", ctypes.POINTER(ctypes.c_uint32)),
        ("option_maximum_counts", ctypes.POINTER(ctypes.c_uint32)),
        ("deck_card_ids", ctypes.POINTER(ctypes.c_int64)),
        ("deck_counts", ctypes.POINTER(ctypes.c_float)),
        ("deck_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("belief_card_ids", ctypes.POINTER(ctypes.c_int64)),
        ("belief_expected_counts", ctypes.POINTER(ctypes.c_float)),
        ("belief_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("belief_scalars", ctypes.POINTER(ctypes.c_float)),
        ("belief_row_indices", ctypes.POINTER(ctypes.c_int64)),
    ]


@dataclass(frozen=True, slots=True)
class NativeRolloutShape:
    """Exact output dimensions for one selected native source."""

    row_count: int
    state_token_width: int
    state_attachment_width: int
    option_width: int
    deck_width: int
    belief_row_count: int
    belief_width: int


class NativeRolloutEncoder:
    """Stateful native public tracker and direct selected-slot encoder."""

    def __init__(
        self,
        *,
        slot_capacity: int,
        library: Any,
        catalog: PublicDeckCatalog,
        input_contract_fingerprint: str,
        supporter_card_ids: Sequence[int] | None = None,
        posterior_cache_capacity: int = 32_768,
    ) -> None:
        """Copy immutable catalog data and create one native slot tracker."""
        if slot_capacity <= 0:
            raise ValueError("native rollout slot capacity must be positive")
        if posterior_cache_capacity <= 0:
            raise ValueError("native rollout posterior cache must be positive")
        _bind_rollout_library(library)
        _validate_rollout_abi(library)
        native_arrays = catalog.native_arrays()
        supporters = np.ascontiguousarray(
            tuple(
                int(card_id)
                for card_id in (
                    default_supporter_card_ids()
                    if supporter_card_ids is None
                    else supporter_card_ids
                )
            ),
            dtype=np.int32,
        )
        native_catalog = _CgTrainRolloutCatalog(
            struct_size=ctypes.sizeof(_CgTrainRolloutCatalog),
            deck_count=native_arrays.entry_counts.shape[0],
            card_vocab_size=native_arrays.entry_counts.shape[1] - 1,
            supporter_count=supporters.size,
            posterior_cache_capacity=posterior_cache_capacity,
            reserved0=0,
            reserved1=0,
            reserved2=0,
            entry_counts=_array_pointer(
                native_arrays.entry_counts,
                ctypes.c_int16,
            ),
            exact_log_priors=_array_pointer(
                native_arrays.exact_log_priors,
                ctypes.c_double,
            ),
            log_combinations=_array_pointer(
                native_arrays.log_combinations,
                ctypes.c_double,
            ),
            log_factorials=_array_pointer(
                native_arrays.log_factorials,
                ctypes.c_double,
            ),
            unknown_card_probabilities=_array_pointer(
                native_arrays.unknown_card_probabilities,
                ctypes.c_double,
            ),
            unknown_log_card_probabilities=_array_pointer(
                native_arrays.unknown_log_card_probabilities,
                ctypes.c_double,
            ),
            unknown_log_prior=native_arrays.unknown_log_prior,
            supporter_card_ids=_array_pointer(
                supporters,
                ctypes.c_int32,
            ),
        )
        pointer = library.CgTrainRolloutCreate(
            slot_capacity,
            ctypes.byref(native_catalog),
        )
        if not pointer:
            raise NativeRolloutLoadError(
                f"native rollout creation failed: {_last_error(library)}"
            )
        self.library = library
        self.slot_capacity = int(slot_capacity)
        self.catalog_fingerprint = catalog.fingerprint
        self.input_contract_fingerprint = input_contract_fingerprint
        self.model_encoding_fingerprint = (
            NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT
        )
        self._pointer: int | None = int(pointer)
        self._deck_signatures: dict[tuple[int, int], str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Destroy the native tracker exactly once."""
        pointer = self._pointer
        if pointer is None:
            return
        self._pointer = None
        self.library.CgTrainRolloutDestroy(pointer)

    def consume_reset(
        self,
        batch: NativeTrainingBatchView,
        decks: npt.ArrayLike,
    ) -> None:
        """Transactionally reset selected slots and consume their first logs."""
        deck_rows = np.ascontiguousarray(decks, dtype=np.int32)
        expected = (batch.batch_size, 2, 60)
        if deck_rows.shape != expected:
            raise ValueError(f"native rollout reset decks must have shape {expected}")
        staged: dict[tuple[int, int], str] = {}
        for row in range(batch.batch_size):
            if int(batch.status[row]) not in (1, 2):
                continue
            slot = int(batch.slots[row])
            for perspective in range(2):
                staged[(slot, perspective)] = canonicalize_deck(
                    deck_rows[row, perspective]
                ).signature
        source = _raw_source(batch)
        result = int(
            self.library.CgTrainRolloutConsumeReset(
                self._require_pointer(),
                batch.batch_size,
                _array_pointer(batch.slots, ctypes.c_uint32),
                _array_pointer(deck_rows, ctypes.c_int32),
                ctypes.byref(source._native),
            )
        )
        self._raise_call_error(result, operation="consume reset")
        self._deck_signatures.update(staged)

    def consume_step(self, batch: NativeTrainingBatchView) -> None:
        """Transactionally consume current public state and log deltas."""
        source = _raw_source(batch)
        result = int(
            self.library.CgTrainRolloutConsumeStep(
                self._require_pointer(),
                batch.batch_size,
                _array_pointer(batch.slots, ctypes.c_uint32),
                ctypes.byref(source._native),
            )
        )
        self._raise_call_error(result, operation="consume step")

    def clear_slots(self, slots: npt.ArrayLike) -> None:
        """Release terminal slot history and immutable identities."""
        slot_rows = _aligned_uint32(slots, name="slots")
        result = int(
            self.library.CgTrainRolloutClearSlots(
                self._require_pointer(),
                slot_rows.size,
                _array_pointer(slot_rows, ctypes.c_uint32),
            )
        )
        self._raise_call_error(result, operation="clear slots")
        for slot in slot_rows:
            self._deck_signatures.pop((int(slot), 0), None)
            self._deck_signatures.pop((int(slot), 1), None)

    def plan_rows(
        self,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
    ) -> NativeRolloutShape:
        """Return exact model tensor dimensions for a direct selection."""
        slot_rows, perspective_rows = _selection(slots, perspectives)
        shape = _CgTrainRolloutShape(struct_size=ctypes.sizeof(_CgTrainRolloutShape))
        result = int(
            self.library.CgTrainRolloutPlanRows(
                self._require_pointer(),
                slot_rows.size,
                _array_pointer(slot_rows, ctypes.c_uint32),
                _array_pointer(perspective_rows, ctypes.c_int32),
                ctypes.byref(shape),
            )
        )
        self._raise_call_error(result, operation="plan rows")
        return NativeRolloutShape(
            row_count=int(shape.row_count),
            state_token_width=int(shape.state_token_width),
            state_attachment_width=int(shape.state_attachment_width),
            option_width=int(shape.option_width),
            deck_width=int(shape.deck_width),
            belief_row_count=int(shape.belief_row_count),
            belief_width=int(shape.belief_width),
        )

    def encode_rows(
        self,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
        *,
        row_offset: int,
        belief_row_offset: int,
        output: _CgTrainRolloutOutput,
    ) -> None:
        """Write selected slots directly into a shared caller-owned buffer."""
        slot_rows, perspective_rows = _selection(slots, perspectives)
        result = int(
            self.library.CgTrainRolloutEncodeRows(
                self._require_pointer(),
                slot_rows.size,
                _array_pointer(slot_rows, ctypes.c_uint32),
                _array_pointer(perspective_rows, ctypes.c_int32),
                row_offset,
                belief_row_offset,
                ctypes.byref(output),
            )
        )
        self._raise_call_error(result, operation="encode rows")

    def deck_signatures(
        self,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
    ) -> tuple[str, ...]:
        """Return immutable exact-deck identities in caller row order."""
        slot_rows, perspective_rows = _selection(slots, perspectives)
        try:
            return tuple(
                self._deck_signatures[(int(slot), int(perspective))]
                for slot, perspective in zip(
                    slot_rows,
                    perspective_rows,
                    strict=True,
                )
            )
        except KeyError as error:
            raise NativeRolloutCallError(
                "native rollout deck identity requested before reset"
            ) from error

    def known_opponent_batch(
        self,
        slots: npt.ArrayLike,
        perspectives: npt.ArrayLike,
    ) -> NativeKnownOpponentBatch:
        """Emit sorted learner-only known public evidence."""
        slot_rows, perspective_rows = _selection(slots, perspectives)
        value_count = ctypes.c_uint32()
        result = int(
            self.library.CgTrainRolloutPlanKnown(
                self._require_pointer(),
                slot_rows.size,
                _array_pointer(slot_rows, ctypes.c_uint32),
                _array_pointer(perspective_rows, ctypes.c_int32),
                ctypes.byref(value_count),
            )
        )
        self._raise_call_error(result, operation="plan known")
        offsets = np.empty(slot_rows.size + 1, dtype=np.uint32)
        card_ids = np.empty(value_count.value, dtype=np.int32)
        counts = np.empty(value_count.value, dtype=np.int32)
        result = int(
            self.library.CgTrainRolloutWriteKnown(
                self._require_pointer(),
                slot_rows.size,
                _array_pointer(slot_rows, ctypes.c_uint32),
                _array_pointer(perspective_rows, ctypes.c_int32),
                value_count.value,
                _array_pointer(offsets, ctypes.c_uint32),
                _array_pointer(card_ids, ctypes.c_int32),
                _array_pointer(counts, ctypes.c_int32),
            )
        )
        self._raise_call_error(result, operation="write known")
        return NativeKnownOpponentBatch(
            offsets=offsets,
            card_ids=card_ids,
            counts=counts,
        )

    def _require_pointer(self) -> int:
        if self._pointer is None:
            raise NativeRolloutCallError("native rollout encoder is closed")
        return self._pointer

    def _raise_call_error(self, code: int, *, operation: str) -> None:
        if code == _OK:
            return
        error_type = (
            NativeRolloutCapacityError
            if code == _INSUFFICIENT_CAPACITY
            else NativeRolloutCallError
        )
        raise error_type(
            f"native rollout {operation} failed with code={code}: "
            f"{_last_error(self.library)}"
        )


def _bind_rollout_library(library: Any) -> None:
    with _BIND_LOCK:
        descriptor = library.CgTrainRolloutGetAbiDescriptor
        descriptor.argtypes = []
        descriptor.restype = ctypes.POINTER(_CgTrainRolloutAbiDescriptor)
        library.CgTrainRolloutLastError.argtypes = []
        library.CgTrainRolloutLastError.restype = ctypes.c_char_p
        library.CgTrainRolloutCreate.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_CgTrainRolloutCatalog),
        ]
        library.CgTrainRolloutCreate.restype = ctypes.c_void_p
        library.CgTrainRolloutDestroy.argtypes = [ctypes.c_void_p]
        library.CgTrainRolloutDestroy.restype = None
        library.CgTrainRolloutConsumeReset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(_CgTrainOutput),
        ]
        library.CgTrainRolloutConsumeReset.restype = ctypes.c_int32
        library.CgTrainRolloutConsumeStep.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_CgTrainOutput),
        ]
        library.CgTrainRolloutConsumeStep.restype = ctypes.c_int32
        library.CgTrainRolloutClearSlots.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.CgTrainRolloutClearSlots.restype = ctypes.c_int32
        library.CgTrainRolloutPlanRows.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(_CgTrainRolloutShape),
        ]
        library.CgTrainRolloutPlanRows.restype = ctypes.c_int32
        library.CgTrainRolloutEncodeRows.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_CgTrainRolloutOutput),
        ]
        library.CgTrainRolloutEncodeRows.restype = ctypes.c_int32
        library.CgTrainRolloutPlanKnown.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.CgTrainRolloutPlanKnown.restype = ctypes.c_int32
        library.CgTrainRolloutWriteKnown.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        library.CgTrainRolloutWriteKnown.restype = ctypes.c_int32


def _validate_rollout_abi(library: Any) -> None:
    pointer = library.CgTrainRolloutGetAbiDescriptor()
    if not pointer:
        raise NativeRolloutLoadError("native rollout ABI descriptor is null")
    descriptor = pointer.contents
    expected = (
        descriptor.magic == _ABI_MAGIC
        and descriptor.abi_version == _ABI_VERSION
        and descriptor.descriptor_size == ctypes.sizeof(_CgTrainRolloutAbiDescriptor)
        and descriptor.catalog_size == ctypes.sizeof(_CgTrainRolloutCatalog)
        and descriptor.shape_size == ctypes.sizeof(_CgTrainRolloutShape)
        and descriptor.output_size == ctypes.sizeof(_CgTrainRolloutOutput)
        and descriptor.token_scalar_size == _TOKEN_SCALAR_SIZE
        and descriptor.option_scalar_size == _OPTION_SCALAR_SIZE
        and descriptor.dynamic_effect_size == _DYNAMIC_EFFECT_SIZE
        and descriptor.maximum_entity_slots == _MAXIMUM_ENTITY_SLOTS
        and descriptor.belief_scalar_size == _BELIEF_SCALAR_SIZE
        and descriptor.history_size == _HISTORY_SIZE
        and descriptor.deck_flow_size == _DECK_FLOW_SIZE
        and descriptor.features & _REQUIRED_FEATURES == _REQUIRED_FEATURES
        and bytes(descriptor.model_encoding_fingerprint)
        == _MODEL_ENCODING_FINGERPRINT
    )
    if not expected:
        raise NativeRolloutLoadError("native rollout ABI descriptor is incompatible")


def _raw_source(batch: NativeTrainingBatchView) -> NativeTrainingOutputBuffer:
    owner = batch._owner
    if (
        batch.status.ctypes.data != owner.status.ctypes.data
        or batch.option_offsets.ctypes.data != owner.option_offsets.ctypes.data
        or batch.visible_card_offsets.ctypes.data
        != owner.visible_card_offsets.ctypes.data
        or batch.attachment_offsets.ctypes.data != owner.attachment_offsets.ctypes.data
        or batch.log_offsets.ctypes.data != owner.log_offsets.ctypes.data
    ):
        raise ValueError("native rollout consume requires an unselected raw arena view")
    return owner


def _selection(
    slots: npt.ArrayLike,
    perspectives: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    slot_rows = _aligned_uint32(slots, name="slots")
    perspective_rows = np.ascontiguousarray(perspectives, dtype=np.int32)
    if perspective_rows.shape != slot_rows.shape:
        raise ValueError("native rollout selection vectors must align")
    return slot_rows, perspective_rows


def _aligned_uint32(
    values: npt.ArrayLike,
    *,
    name: str,
) -> npt.NDArray[np.uint32]:
    result = np.ascontiguousarray(values, dtype=np.uint32)
    if result.ndim != 1 or result.size <= 0:
        raise ValueError(f"native rollout {name} must be a non-empty vector")
    return result


def _array_pointer(
    values: np.ndarray,
    ctype: Any,
) -> Any:
    return values.ctypes.data_as(ctypes.POINTER(ctype))


def _last_error(library: Any) -> str:
    raw = library.CgTrainRolloutLastError()
    return "unknown error" if raw is None else raw.decode("utf-8", "replace")


__all__ = [
    "NATIVE_ROLLOUT_MODEL_ENCODING_FINGERPRINT",
    "NativeRolloutCallError",
    "NativeRolloutCapacityError",
    "NativeRolloutEncoder",
    "NativeRolloutLoadError",
    "NativeRolloutShape",
]
