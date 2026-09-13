"""Zero-JSON ctypes binding for the native batched training arena."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt

_ABI_MAGIC = 0x31544743
_ABI_VERSION = 4
NATIVE_TRAINING_ABI_VERSION = _ABI_VERSION
_REQUIRED_FEATURES = (
    ((1 << 10) - 1)
    | (1 << 11)
    | (1 << 12)
    | (1 << 13)
    | (1 << 14)
    | (1 << 15)
    | (1 << 16)
)
_OK = 0
_INSUFFICIENT_CAPACITY = -2
_BIND_LOCK = threading.Lock()

Int32Array = npt.NDArray[np.int32]
Uint32Array = npt.NDArray[np.uint32]


class NativeTrainingLoadError(RuntimeError):
    """Raised when the training-only native library is unavailable or stale."""


class NativeTrainingCallError(RuntimeError):
    """Raised when a whole native arena call cannot be accepted."""


class NativeTrainingCapacityError(NativeTrainingCallError):
    """Raised without state mutation when caller-owned buffers are too small."""


class _CgTrainAbiDescriptor(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("descriptor_size", ctypes.c_uint32),
        ("output_size", ctypes.c_uint32),
        ("deck_size", ctypes.c_uint32),
        ("players_per_game", ctypes.c_uint32),
        ("option_param_count", ctypes.c_uint32),
        ("option_type_count", ctypes.c_uint32),
        ("max_lane_capacity", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
    ]


class _CgTrainOutput(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("slot_capacity", ctypes.c_uint32),
        ("option_capacity", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("status", ctypes.POINTER(ctypes.c_int32)),
        ("error", ctypes.POINTER(ctypes.c_int32)),
        ("select_player", ctypes.POINTER(ctypes.c_int32)),
        ("select_type", ctypes.POINTER(ctypes.c_int32)),
        ("select_context", ctypes.POINTER(ctypes.c_int32)),
        ("select_min", ctypes.POINTER(ctypes.c_int32)),
        ("select_max", ctypes.POINTER(ctypes.c_int32)),
        ("result", ctypes.POINTER(ctypes.c_int32)),
        ("turn", ctypes.POINTER(ctypes.c_int32)),
        ("option_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("option_type", ctypes.POINTER(ctypes.c_int32)),
        ("option_p0", ctypes.POINTER(ctypes.c_int32)),
        ("option_p1", ctypes.POINTER(ctypes.c_int32)),
        ("option_p2", ctypes.POINTER(ctypes.c_int32)),
        ("option_p3", ctypes.POINTER(ctypes.c_int32)),
        ("option_p4", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_capacity", ctypes.c_uint32),
        ("attachment_capacity", ctypes.c_uint32),
        ("reserved_v2_0", ctypes.c_uint32),
        ("reserved_v2_1", ctypes.c_uint32),
        ("turn_action_count", ctypes.POINTER(ctypes.c_int32)),
        ("first_player", ctypes.POINTER(ctypes.c_int32)),
        ("turn_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("remain_damage_counter", ctypes.POINTER(ctypes.c_int32)),
        ("remain_energy_cost", ctypes.POINTER(ctypes.c_int32)),
        ("player0_deck_count", ctypes.POINTER(ctypes.c_int32)),
        ("player1_deck_count", ctypes.POINTER(ctypes.c_int32)),
        ("player0_hand_count", ctypes.POINTER(ctypes.c_int32)),
        ("player1_hand_count", ctypes.POINTER(ctypes.c_int32)),
        ("player0_prize_count", ctypes.POINTER(ctypes.c_int32)),
        ("player1_prize_count", ctypes.POINTER(ctypes.c_int32)),
        ("player0_bench_max", ctypes.POINTER(ctypes.c_int32)),
        ("player1_bench_max", ctypes.POINTER(ctypes.c_int32)),
        ("player0_status_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("player1_status_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("looking_mode", ctypes.POINTER(ctypes.c_int32)),
        ("select_deck_visible", ctypes.POINTER(ctypes.c_int32)),
        ("context_card_row", ctypes.POINTER(ctypes.c_uint32)),
        ("effect_card_row", ctypes.POINTER(ctypes.c_uint32)),
        ("visible_card_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("visible_card_owner", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_area", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_area_index", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_id", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_serial", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_hp", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_max_hp", ctypes.POINTER(ctypes.c_int32)),
        ("visible_card_appear_this_turn", ctypes.POINTER(ctypes.c_int32)),
        ("attachment_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("attachment_parent", ctypes.POINTER(ctypes.c_uint32)),
        ("attachment_kind", ctypes.POINTER(ctypes.c_int32)),
        ("attachment_card_id", ctypes.POINTER(ctypes.c_int32)),
        ("attachment_card_serial", ctypes.POINTER(ctypes.c_int32)),
        ("attachment_energy_type", ctypes.POINTER(ctypes.c_int32)),
        ("attachment_energy_units", ctypes.POINTER(ctypes.c_int32)),
        ("log_capacity", ctypes.c_uint32),
        ("reserved_v3", ctypes.c_uint32),
        ("log_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("log_type", ctypes.POINTER(ctypes.c_int32)),
        ("log_param_count", ctypes.POINTER(ctypes.c_uint32)),
        ("log_p0", ctypes.POINTER(ctypes.c_int32)),
        ("log_p1", ctypes.POINTER(ctypes.c_int32)),
        ("log_p2", ctypes.POINTER(ctypes.c_int32)),
        ("log_p3", ctypes.POINTER(ctypes.c_int32)),
        ("log_p4", ctypes.POINTER(ctypes.c_int32)),
        ("log_p5", ctypes.POINTER(ctypes.c_int32)),
        ("log_p6", ctypes.POINTER(ctypes.c_int32)),
        ("selection_advance_count", ctypes.POINTER(ctypes.c_uint32)),
    ]


@dataclass(frozen=True)
class NativeTrainingAbi:
    """Validated immutable dimensions exported by the native library."""

    deck_size: int
    players_per_game: int
    option_type_count: int
    maximum_lane_capacity: int
    features: int


@dataclass(frozen=True)
class NativeTrainingBatchView:
    """Borrowed SoA slices valid until their output buffer is reused."""

    slots: Uint32Array
    status: Int32Array
    error: Int32Array
    select_player: Int32Array
    select_type: Int32Array
    select_context: Int32Array
    select_min: Int32Array
    select_max: Int32Array
    result: Int32Array
    turn: Int32Array
    option_offsets: Uint32Array
    option_type: Int32Array
    option_params: tuple[
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
    ]
    turn_action_count: Int32Array
    first_player: Int32Array
    turn_flags: Uint32Array
    remain_damage_counter: Int32Array
    remain_energy_cost: Int32Array
    player_deck_counts: tuple[Int32Array, Int32Array]
    player_hand_counts: tuple[Int32Array, Int32Array]
    player_prize_counts: tuple[Int32Array, Int32Array]
    player_bench_max: tuple[Int32Array, Int32Array]
    player_status_flags: tuple[Uint32Array, Uint32Array]
    looking_mode: Int32Array
    select_deck_visible: Int32Array
    context_card_row: Uint32Array
    effect_card_row: Uint32Array
    visible_card_offsets: Uint32Array
    visible_card_owner: Int32Array
    visible_card_area: Int32Array
    visible_card_area_index: Int32Array
    visible_card_id: Int32Array
    visible_card_serial: Int32Array
    visible_card_hp: Int32Array
    visible_card_max_hp: Int32Array
    visible_card_appear_this_turn: Int32Array
    attachment_offsets: Uint32Array
    attachment_parent: Uint32Array
    attachment_kind: Int32Array
    attachment_card_id: Int32Array
    attachment_card_serial: Int32Array
    attachment_energy_type: Int32Array
    attachment_energy_units: Int32Array
    log_offsets: Uint32Array
    log_type: Int32Array
    log_param_count: Uint32Array
    log_params: tuple[
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
        Int32Array,
    ]
    selection_advance_count: Uint32Array
    _owner: NativeTrainingOutputBuffer

    @property
    def batch_size(self) -> int:
        """Return the number of aligned arena slots."""
        return int(self.status.shape[0])

    @property
    def option_count(self) -> int:
        """Return the number of flattened engine-legal options."""
        return int(self.option_type.shape[0])

    @property
    def visible_card_count(self) -> int:
        """Return the number of public visible-card rows."""
        return int(self.visible_card_owner.shape[0])

    @property
    def attachment_count(self) -> int:
        """Return the number of public attachment rows."""
        return int(self.attachment_parent.shape[0])

    @property
    def log_count(self) -> int:
        """Return the number of perspective-projected public log rows."""
        return int(self.log_type.shape[0])


class NativeTrainingOutputBuffer:
    """Reusable caller-owned SoA storage for one native batch call."""

    def __init__(
        self,
        *,
        slot_capacity: int,
        option_capacity: int,
        visible_card_capacity: int | None = None,
        attachment_capacity: int | None = None,
        log_capacity: int | None = None,
    ) -> None:
        """Allocate aligned NumPy columns once outside the rollout hot path."""
        if slot_capacity <= 0 or option_capacity <= 0:
            raise ValueError("native output capacities must be positive")
        resolved_visible_capacity = (
            slot_capacity * 256
            if visible_card_capacity is None
            else int(visible_card_capacity)
        )
        resolved_attachment_capacity = (
            slot_capacity * 512
            if attachment_capacity is None
            else int(attachment_capacity)
        )
        resolved_log_capacity = (
            slot_capacity * 256 if log_capacity is None else int(log_capacity)
        )
        if (
            resolved_visible_capacity <= 0
            or resolved_attachment_capacity <= 0
            or resolved_log_capacity <= 0
        ):
            raise ValueError("native public-state capacities must be positive")
        self.slot_capacity = int(slot_capacity)
        self.option_capacity = int(option_capacity)
        self.visible_card_capacity = resolved_visible_capacity
        self.attachment_capacity = resolved_attachment_capacity
        self.log_capacity = resolved_log_capacity
        self.status = np.empty(slot_capacity, dtype=np.int32)
        self.error = np.empty(slot_capacity, dtype=np.int32)
        self.select_player = np.empty(slot_capacity, dtype=np.int32)
        self.select_type = np.empty(slot_capacity, dtype=np.int32)
        self.select_context = np.empty(slot_capacity, dtype=np.int32)
        self.select_min = np.empty(slot_capacity, dtype=np.int32)
        self.select_max = np.empty(slot_capacity, dtype=np.int32)
        self.result = np.empty(slot_capacity, dtype=np.int32)
        self.turn = np.empty(slot_capacity, dtype=np.int32)
        self.option_offsets = np.empty(slot_capacity + 1, dtype=np.uint32)
        self.option_type = np.empty(option_capacity, dtype=np.int32)
        self.option_params = tuple(
            np.empty(option_capacity, dtype=np.int32) for _ in range(5)
        )
        self.turn_action_count = np.empty(slot_capacity, dtype=np.int32)
        self.first_player = np.empty(slot_capacity, dtype=np.int32)
        self.turn_flags = np.empty(slot_capacity, dtype=np.uint32)
        self.remain_damage_counter = np.empty(slot_capacity, dtype=np.int32)
        self.remain_energy_cost = np.empty(slot_capacity, dtype=np.int32)
        self.player_deck_counts = tuple(
            np.empty(slot_capacity, dtype=np.int32) for _ in range(2)
        )
        self.player_hand_counts = tuple(
            np.empty(slot_capacity, dtype=np.int32) for _ in range(2)
        )
        self.player_prize_counts = tuple(
            np.empty(slot_capacity, dtype=np.int32) for _ in range(2)
        )
        self.player_bench_max = tuple(
            np.empty(slot_capacity, dtype=np.int32) for _ in range(2)
        )
        self.player_status_flags = tuple(
            np.empty(slot_capacity, dtype=np.uint32) for _ in range(2)
        )
        self.looking_mode = np.empty(slot_capacity, dtype=np.int32)
        self.select_deck_visible = np.empty(slot_capacity, dtype=np.int32)
        self.context_card_row = np.empty(slot_capacity, dtype=np.uint32)
        self.effect_card_row = np.empty(slot_capacity, dtype=np.uint32)
        self.visible_card_offsets = np.empty(slot_capacity + 1, dtype=np.uint32)
        self.visible_card_owner = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_area = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_area_index = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_id = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_serial = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_hp = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_max_hp = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.visible_card_appear_this_turn = np.empty(
            resolved_visible_capacity,
            dtype=np.int32,
        )
        self.attachment_offsets = np.empty(slot_capacity + 1, dtype=np.uint32)
        self.attachment_parent = np.empty(
            resolved_attachment_capacity,
            dtype=np.uint32,
        )
        self.attachment_kind = np.empty(
            resolved_attachment_capacity,
            dtype=np.int32,
        )
        self.attachment_card_id = np.empty(
            resolved_attachment_capacity,
            dtype=np.int32,
        )
        self.attachment_card_serial = np.empty(
            resolved_attachment_capacity,
            dtype=np.int32,
        )
        self.attachment_energy_type = np.empty(
            resolved_attachment_capacity,
            dtype=np.int32,
        )
        self.attachment_energy_units = np.empty(
            resolved_attachment_capacity,
            dtype=np.int32,
        )
        self.log_offsets = np.empty(slot_capacity + 1, dtype=np.uint32)
        self.log_type = np.empty(resolved_log_capacity, dtype=np.int32)
        self.log_param_count = np.empty(
            resolved_log_capacity,
            dtype=np.uint32,
        )
        self.log_params = tuple(
            np.empty(resolved_log_capacity, dtype=np.int32) for _ in range(7)
        )
        self.selection_advance_count = np.empty(
            slot_capacity,
            dtype=np.uint32,
        )
        self._native = _CgTrainOutput(
            struct_size=ctypes.sizeof(_CgTrainOutput),
            slot_capacity=slot_capacity,
            option_capacity=option_capacity,
            reserved=0,
            status=_int32_pointer(self.status),
            error=_int32_pointer(self.error),
            select_player=_int32_pointer(self.select_player),
            select_type=_int32_pointer(self.select_type),
            select_context=_int32_pointer(self.select_context),
            select_min=_int32_pointer(self.select_min),
            select_max=_int32_pointer(self.select_max),
            result=_int32_pointer(self.result),
            turn=_int32_pointer(self.turn),
            option_offsets=_uint32_pointer(self.option_offsets),
            option_type=_int32_pointer(self.option_type),
            option_p0=_int32_pointer(self.option_params[0]),
            option_p1=_int32_pointer(self.option_params[1]),
            option_p2=_int32_pointer(self.option_params[2]),
            option_p3=_int32_pointer(self.option_params[3]),
            option_p4=_int32_pointer(self.option_params[4]),
            visible_card_capacity=resolved_visible_capacity,
            attachment_capacity=resolved_attachment_capacity,
            reserved_v2_0=0,
            reserved_v2_1=0,
            turn_action_count=_int32_pointer(self.turn_action_count),
            first_player=_int32_pointer(self.first_player),
            turn_flags=_uint32_pointer(self.turn_flags),
            remain_damage_counter=_int32_pointer(self.remain_damage_counter),
            remain_energy_cost=_int32_pointer(self.remain_energy_cost),
            player0_deck_count=_int32_pointer(self.player_deck_counts[0]),
            player1_deck_count=_int32_pointer(self.player_deck_counts[1]),
            player0_hand_count=_int32_pointer(self.player_hand_counts[0]),
            player1_hand_count=_int32_pointer(self.player_hand_counts[1]),
            player0_prize_count=_int32_pointer(self.player_prize_counts[0]),
            player1_prize_count=_int32_pointer(self.player_prize_counts[1]),
            player0_bench_max=_int32_pointer(self.player_bench_max[0]),
            player1_bench_max=_int32_pointer(self.player_bench_max[1]),
            player0_status_flags=_uint32_pointer(self.player_status_flags[0]),
            player1_status_flags=_uint32_pointer(self.player_status_flags[1]),
            looking_mode=_int32_pointer(self.looking_mode),
            select_deck_visible=_int32_pointer(self.select_deck_visible),
            context_card_row=_uint32_pointer(self.context_card_row),
            effect_card_row=_uint32_pointer(self.effect_card_row),
            visible_card_offsets=_uint32_pointer(self.visible_card_offsets),
            visible_card_owner=_int32_pointer(self.visible_card_owner),
            visible_card_area=_int32_pointer(self.visible_card_area),
            visible_card_area_index=_int32_pointer(self.visible_card_area_index),
            visible_card_id=_int32_pointer(self.visible_card_id),
            visible_card_serial=_int32_pointer(self.visible_card_serial),
            visible_card_hp=_int32_pointer(self.visible_card_hp),
            visible_card_max_hp=_int32_pointer(self.visible_card_max_hp),
            visible_card_appear_this_turn=_int32_pointer(
                self.visible_card_appear_this_turn
            ),
            attachment_offsets=_uint32_pointer(self.attachment_offsets),
            attachment_parent=_uint32_pointer(self.attachment_parent),
            attachment_kind=_int32_pointer(self.attachment_kind),
            attachment_card_id=_int32_pointer(self.attachment_card_id),
            attachment_card_serial=_int32_pointer(self.attachment_card_serial),
            attachment_energy_type=_int32_pointer(self.attachment_energy_type),
            attachment_energy_units=_int32_pointer(self.attachment_energy_units),
            log_capacity=resolved_log_capacity,
            reserved_v3=0,
            log_offsets=_uint32_pointer(self.log_offsets),
            log_type=_int32_pointer(self.log_type),
            log_param_count=_uint32_pointer(self.log_param_count),
            log_p0=_int32_pointer(self.log_params[0]),
            log_p1=_int32_pointer(self.log_params[1]),
            log_p2=_int32_pointer(self.log_params[2]),
            log_p3=_int32_pointer(self.log_params[3]),
            log_p4=_int32_pointer(self.log_params[4]),
            log_p5=_int32_pointer(self.log_params[5]),
            log_p6=_int32_pointer(self.log_params[6]),
            selection_advance_count=_uint32_pointer(self.selection_advance_count),
        )

    def view(self, slots: Uint32Array) -> NativeTrainingBatchView:
        """Return used slices after one successful native call."""
        rows = int(slots.shape[0])
        if rows > self.slot_capacity:
            raise ValueError("slot view exceeds its output buffer")
        option_count = int(self.option_offsets[rows])
        if option_count > self.option_capacity:
            raise RuntimeError("native option count exceeds validated capacity")
        visible_card_count = int(self.visible_card_offsets[rows])
        attachment_count = int(self.attachment_offsets[rows])
        log_count = int(self.log_offsets[rows])
        if visible_card_count > self.visible_card_capacity:
            raise RuntimeError("native visible-card count exceeds validated capacity")
        if attachment_count > self.attachment_capacity:
            raise RuntimeError("native attachment count exceeds validated capacity")
        if log_count > self.log_capacity:
            raise RuntimeError("native log count exceeds validated capacity")
        return NativeTrainingBatchView(
            slots=slots,
            status=self.status[:rows],
            error=self.error[:rows],
            select_player=self.select_player[:rows],
            select_type=self.select_type[:rows],
            select_context=self.select_context[:rows],
            select_min=self.select_min[:rows],
            select_max=self.select_max[:rows],
            result=self.result[:rows],
            turn=self.turn[:rows],
            option_offsets=self.option_offsets[: rows + 1],
            option_type=self.option_type[:option_count],
            option_params=(
                self.option_params[0][:option_count],
                self.option_params[1][:option_count],
                self.option_params[2][:option_count],
                self.option_params[3][:option_count],
                self.option_params[4][:option_count],
            ),
            turn_action_count=self.turn_action_count[:rows],
            first_player=self.first_player[:rows],
            turn_flags=self.turn_flags[:rows],
            remain_damage_counter=self.remain_damage_counter[:rows],
            remain_energy_cost=self.remain_energy_cost[:rows],
            player_deck_counts=(
                self.player_deck_counts[0][:rows],
                self.player_deck_counts[1][:rows],
            ),
            player_hand_counts=(
                self.player_hand_counts[0][:rows],
                self.player_hand_counts[1][:rows],
            ),
            player_prize_counts=(
                self.player_prize_counts[0][:rows],
                self.player_prize_counts[1][:rows],
            ),
            player_bench_max=(
                self.player_bench_max[0][:rows],
                self.player_bench_max[1][:rows],
            ),
            player_status_flags=(
                self.player_status_flags[0][:rows],
                self.player_status_flags[1][:rows],
            ),
            looking_mode=self.looking_mode[:rows],
            select_deck_visible=self.select_deck_visible[:rows],
            context_card_row=self.context_card_row[:rows],
            effect_card_row=self.effect_card_row[:rows],
            visible_card_offsets=self.visible_card_offsets[: rows + 1],
            visible_card_owner=self.visible_card_owner[:visible_card_count],
            visible_card_area=self.visible_card_area[:visible_card_count],
            visible_card_area_index=self.visible_card_area_index[:visible_card_count],
            visible_card_id=self.visible_card_id[:visible_card_count],
            visible_card_serial=self.visible_card_serial[:visible_card_count],
            visible_card_hp=self.visible_card_hp[:visible_card_count],
            visible_card_max_hp=self.visible_card_max_hp[:visible_card_count],
            visible_card_appear_this_turn=self.visible_card_appear_this_turn[
                :visible_card_count
            ],
            attachment_offsets=self.attachment_offsets[: rows + 1],
            attachment_parent=self.attachment_parent[:attachment_count],
            attachment_kind=self.attachment_kind[:attachment_count],
            attachment_card_id=self.attachment_card_id[:attachment_count],
            attachment_card_serial=self.attachment_card_serial[:attachment_count],
            attachment_energy_type=self.attachment_energy_type[:attachment_count],
            attachment_energy_units=self.attachment_energy_units[:attachment_count],
            log_offsets=self.log_offsets[: rows + 1],
            log_type=self.log_type[:log_count],
            log_param_count=self.log_param_count[:log_count],
            log_params=(
                self.log_params[0][:log_count],
                self.log_params[1][:log_count],
                self.log_params[2][:log_count],
                self.log_params[3][:log_count],
                self.log_params[4][:log_count],
                self.log_params[5][:log_count],
                self.log_params[6][:log_count],
            ),
            selection_advance_count=self.selection_advance_count[:rows],
            _owner=self,
        )


class NativeTrainingLane:
    """Opaque fixed-capacity batch of live source-engine games."""

    def __init__(
        self,
        capacity: int,
        *,
        library_path: Path | str | None = None,
        worker_count: int | None = None,
    ) -> None:
        """Load the public-state ABI and allocate a stable native slot arena."""
        self.library, self.library_path = _load_library(library_path)
        _bind_library(self.library)
        self.abi = _read_abi(self.library)
        if capacity <= 0 or capacity > self.abi.maximum_lane_capacity:
            raise ValueError("native lane capacity is outside the ABI limit")
        if worker_count is not None and worker_count <= 0:
            raise ValueError("native lane worker_count must be positive")
        pointer = (
            self.library.CgTrainCreate(capacity)
            if worker_count is None
            else self.library.CgTrainCreateWithWorkers(capacity, worker_count)
        )
        if not pointer:
            raise NativeTrainingCallError(
                f"native lane creation failed: {_last_error(self.library)}"
            )
        self._pointer: int | None = int(pointer)
        self.capacity = int(capacity)
        self.worker_count = int(self.library.CgTrainWorkerCount(pointer))
        if self.worker_count <= 0:
            self.close()
            raise NativeTrainingCallError(
                "native lane returned an invalid worker count"
            )

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
        """Destroy every live engine state exactly once."""
        pointer = self._pointer
        if pointer is None:
            return
        self._pointer = None
        self.library.CgTrainDestroy(pointer)

    def clear_slots(self, slots: npt.ArrayLike) -> None:
        """Destroy retired engine states while keeping the lane reusable."""
        slot_rows = self._slots(slots)
        result = int(
            self.library.CgTrainClearSlots(
                self._require_pointer(),
                slot_rows.size,
                _uint32_pointer(slot_rows),
            )
        )
        self._raise_call_error(result, operation="clear slots")

    def reset(
        self,
        decks: npt.ArrayLike,
        seeds: npt.ArrayLike,
        *,
        output: NativeTrainingOutputBuffer,
        slots: npt.ArrayLike | None = None,
    ) -> NativeTrainingBatchView:
        """Reset selected slots from `[batch, 2, 60]` decks and exact seeds."""
        deck_rows = np.ascontiguousarray(decks, dtype=np.int32)
        expected_tail = (self.abi.players_per_game, self.abi.deck_size)
        if deck_rows.ndim != 3 or tuple(deck_rows.shape[1:]) != expected_tail:
            raise ValueError(f"native decks must have shape [batch, {expected_tail}]")
        batch_size = int(deck_rows.shape[0])
        slot_rows = self._slots(slots, batch_size=batch_size)
        seed_rows = np.ascontiguousarray(seeds, dtype=np.uint32)
        if seed_rows.shape != (batch_size,):
            raise ValueError("native seeds must have shape [batch]")
        self._require_output(output, batch_size=batch_size)
        result = int(
            self.library.CgTrainReset(
                self._require_pointer(),
                batch_size,
                _uint32_pointer(slot_rows),
                _int32_pointer(deck_rows),
                _uint32_pointer(seed_rows),
                ctypes.byref(output._native),
            )
        )
        self._raise_call_error(result, operation="reset")
        return output.view(slot_rows)

    def step(
        self,
        slots: npt.ArrayLike,
        action_offsets: npt.ArrayLike,
        action_choices: npt.ArrayLike,
        *,
        output: NativeTrainingOutputBuffer,
    ) -> NativeTrainingBatchView:
        """Advance a slot batch using complete selections encoded as CSR."""
        slot_rows = self._slots(slots)
        batch_size = int(slot_rows.shape[0])
        offsets = np.ascontiguousarray(action_offsets, dtype=np.uint32)
        choices = np.ascontiguousarray(action_choices, dtype=np.int32)
        if choices.ndim != 1:
            raise ValueError("native action choices must be one-dimensional")
        if (
            offsets.shape != (batch_size + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != choices.shape[0]
            or bool(np.any(offsets[1:] < offsets[:-1]))
        ):
            raise ValueError("native action offsets are not valid CSR")
        self._require_output(output, batch_size=batch_size)
        result = int(
            self.library.CgTrainStep(
                self._require_pointer(),
                batch_size,
                _uint32_pointer(slot_rows),
                _uint32_pointer(offsets),
                _int32_pointer(choices),
                int(choices.shape[0]),
                ctypes.byref(output._native),
            )
        )
        self._raise_call_error(result, operation="step")
        return output.view(slot_rows)

    def export_public_state_tokens(
        self,
        slots: npt.ArrayLike,
    ) -> tuple[bytes, ...]:
        """Return privacy-erased Search API roots without mutating the lane."""
        return self._export_csr_bytes(
            slots,
            function=self.library.CgTrainExportPublicStateTokens,
            initial_bytes_per_row=32_768,
            operation="export state tokens",
        )

    def export_public_observations(
        self,
        slots: npt.ArrayLike,
    ) -> tuple[bytes, ...]:
        """Return exact cached public observations from the latest lane output."""
        return self._export_csr_bytes(
            slots,
            function=self.library.CgTrainExportPublicObservations,
            initial_bytes_per_row=32_768,
            operation="export public observations",
        )

    def _export_csr_bytes(
        self,
        slots: npt.ArrayLike,
        *,
        function: Any,
        initial_bytes_per_row: int,
        operation: str,
    ) -> tuple[bytes, ...]:
        """Read one immutable variable-width byte row per selected slot."""
        slot_rows = self._slots(slots)
        offsets = np.empty(slot_rows.size + 1, dtype=np.uint32)
        initial_capacity = max(
            1,
            int(slot_rows.size) * initial_bytes_per_row,
        )
        data = ctypes.create_string_buffer(initial_capacity)
        result = int(
            function(
                self._require_pointer(),
                slot_rows.size,
                _uint32_pointer(slot_rows),
                initial_capacity,
                _uint32_pointer(offsets),
                data,
            )
        )
        if result == _INSUFFICIENT_CAPACITY:
            required = int(offsets[-1])
            if required <= initial_capacity:
                self._raise_call_error(result, operation=operation)
            data = ctypes.create_string_buffer(required)
            result = int(
                function(
                    self._require_pointer(),
                    slot_rows.size,
                    _uint32_pointer(slot_rows),
                    required,
                    _uint32_pointer(offsets),
                    data,
                )
            )
        self._raise_call_error(result, operation=operation)
        return tuple(
            ctypes.string_at(
                ctypes.addressof(data) + int(offsets[row]),
                int(offsets[row + 1]) - int(offsets[row]),
            )
            for row in range(slot_rows.size)
        )

    def _slots(
        self,
        values: npt.ArrayLike | None,
        *,
        batch_size: int | None = None,
    ) -> Uint32Array:
        if values is None:
            if batch_size is None:
                raise ValueError("native slots are required")
            result = np.arange(batch_size, dtype=np.uint32)
        else:
            result = np.ascontiguousarray(values, dtype=np.uint32)
        if result.ndim != 1 or (batch_size is not None and result.size != batch_size):
            raise ValueError("native slots must have shape [batch]")
        if result.size == 0 or result.size > self.capacity:
            raise ValueError("native slot batch is empty or exceeds lane capacity")
        if (
            bool(np.any(result >= self.capacity))
            or np.unique(result).size != result.size
        ):
            raise ValueError("native slots must be unique and in range")
        return result

    def _require_output(
        self,
        output: NativeTrainingOutputBuffer,
        *,
        batch_size: int,
    ) -> None:
        if output.slot_capacity < batch_size:
            raise NativeTrainingCapacityError(
                "native output slot capacity is too small"
            )

    def _require_pointer(self) -> int:
        if self._pointer is None:
            raise NativeTrainingCallError("native training lane is closed")
        return self._pointer

    def _raise_call_error(self, code: int, *, operation: str) -> None:
        if code == _OK:
            return
        error_type = (
            NativeTrainingCapacityError
            if code == _INSUFFICIENT_CAPACITY
            else NativeTrainingCallError
        )
        raise error_type(
            f"native training {operation} failed with code={code}: "
            f"{_last_error(self.library)}"
        )


def _bind_library(library: Any) -> None:
    with _BIND_LOCK:
        descriptor = library.CgTrainGetAbiDescriptor
        descriptor.argtypes = []
        descriptor.restype = ctypes.POINTER(_CgTrainAbiDescriptor)
        library.CgTrainLastError.argtypes = []
        library.CgTrainLastError.restype = ctypes.c_char_p
        library.CgTrainInitialize.argtypes = []
        library.CgTrainInitialize.restype = ctypes.c_int32
        library.CgTrainCreate.argtypes = [ctypes.c_uint32]
        library.CgTrainCreate.restype = ctypes.c_void_p
        library.CgTrainCreateWithWorkers.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.CgTrainCreateWithWorkers.restype = ctypes.c_void_p
        library.CgTrainDestroy.argtypes = [ctypes.c_void_p]
        library.CgTrainDestroy.restype = None
        library.CgTrainCapacity.argtypes = [ctypes.c_void_p]
        library.CgTrainCapacity.restype = ctypes.c_uint32
        library.CgTrainWorkerCount.argtypes = [ctypes.c_void_p]
        library.CgTrainWorkerCount.restype = ctypes.c_uint32
        library.CgTrainClearSlots.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.CgTrainClearSlots.restype = ctypes.c_int32
        library.CgTrainReset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_CgTrainOutput),
        ]
        library.CgTrainReset.restype = ctypes.c_int32
        library.CgTrainStep.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.POINTER(_CgTrainOutput),
        ]
        library.CgTrainStep.restype = ctypes.c_int32
        library.CgTrainExportPublicStateTokens.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_char),
        ]
        library.CgTrainExportPublicStateTokens.restype = ctypes.c_int32
        library.CgTrainExportPublicObservations.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_char),
        ]
        library.CgTrainExportPublicObservations.restype = ctypes.c_int32


def _read_abi(library: Any) -> NativeTrainingAbi:
    pointer = library.CgTrainGetAbiDescriptor()
    if not pointer:
        raise NativeTrainingLoadError("native training ABI descriptor is null")
    raw = pointer.contents
    if (
        raw.magic != _ABI_MAGIC
        or raw.abi_version != _ABI_VERSION
        or raw.descriptor_size != ctypes.sizeof(_CgTrainAbiDescriptor)
        or raw.output_size != ctypes.sizeof(_CgTrainOutput)
        or raw.option_param_count != 5
        or raw.deck_size <= 0
        or raw.players_per_game != 2
        or raw.features & _REQUIRED_FEATURES != _REQUIRED_FEATURES
    ):
        raise NativeTrainingLoadError("native training ABI is incompatible")
    return NativeTrainingAbi(
        deck_size=int(raw.deck_size),
        players_per_game=int(raw.players_per_game),
        option_type_count=int(raw.option_type_count),
        maximum_lane_capacity=int(raw.max_lane_capacity),
        features=int(raw.features),
    )


def resolve_native_training_library(
    library_path: Path | str | None = None,
) -> tuple[Path, NativeTrainingAbi]:
    """Resolve one concrete training engine artifact and validate its ABI."""
    library, resolved = _load_library(library_path)
    _bind_library(library)
    return resolved, _read_abi(library)


def _load_library(library_path: Path | str | None) -> tuple[Any, Path]:
    candidates: list[Path] = []
    if library_path is not None:
        candidates.append(Path(library_path))
    else:
        environment_path = os.environ.get("PTCG_RL_CG_TRAIN_LIB")
        if environment_path:
            candidates.append(Path(environment_path))
        repository_root = Path(__file__).resolve().parents[3]
        candidates.append(
            repository_root / "src" / "native" / "cg_train" / "libcg_train.so"
        )
    attempts: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if not resolved.is_file():
            attempts.append(str(resolved))
            continue
        try:
            return (ctypes.CDLL(str(resolved)), resolved)
        except OSError as error:
            attempts.append(f"{resolved}: {error}")
    raise NativeTrainingLoadError(
        "native training library is not loadable; tried " + ", ".join(attempts)
    )


def _last_error(library: Any) -> str:
    raw = library.CgTrainLastError()
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else ""


def _int32_pointer(
    values: Int32Array,
) -> Any:
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _uint32_pointer(
    values: Uint32Array,
) -> Any:
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


__all__ = [
    "NativeTrainingAbi",
    "NativeTrainingBatchView",
    "NativeTrainingCallError",
    "NativeTrainingCapacityError",
    "NativeTrainingLane",
    "NativeTrainingLoadError",
    "NativeTrainingOutputBuffer",
    "resolve_native_training_library",
]
