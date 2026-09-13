"""ctypes binding for the native aggregate-quota lease scheduler."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_NO_FROZEN_ARTIFACT = (1 << 32) - 1


class _NativeQuotaRow(ctypes.Structure):
    _fields_ = [
        ("game_count", ctypes.c_uint64),
        ("frozen_artifact_slot", ctypes.c_uint32),
        ("cohort_artifact_slot", ctypes.c_uint32),
    ]


class NativeQuotaScheduler:
    """Own one native quota cursor and issue artifact-coherent row recipes."""

    def __init__(
        self,
        game_counts: Sequence[int],
        frozen_artifact_ids: Sequence[str | None],
        *,
        cohort_artifact_ids: Sequence[str | None] | None = None,
        library_path: Path | None = None,
    ) -> None:
        """Create a scheduler without materializing any per-game assignment.

        Cohort identities constrain which rows can share a shard without making
        current-policy rows consume that frozen artifact.  When omitted, the
        frozen identity is also the scheduling cohort for compatibility.
        """
        self._pointer: int | None = None
        counts = tuple(int(value) for value in game_counts)
        artifacts = tuple(frozen_artifact_ids)
        cohorts = (
            artifacts if cohort_artifact_ids is None else tuple(cohort_artifact_ids)
        )
        if not counts or len(counts) != len(artifacts) or len(counts) != len(cohorts):
            raise ValueError("quota scheduler rows must be non-empty and aligned")
        if any(value <= 0 for value in counts):
            raise ValueError("quota scheduler counts must be positive")
        unique_artifacts = tuple(
            sorted({item for item in artifacts if item is not None})
        )
        slot_ids = tuple(
            sorted(
                {item for item in artifacts if item is not None}
                | {item for item in cohorts if item is not None}
            )
        )
        if len(slot_ids) > 20:
            raise ValueError("quota scheduler supports at most 20 frozen artifacts")
        self.artifact_ids = unique_artifacts
        self._slot_ids = slot_ids
        slots = {artifact_id: index for index, artifact_id in enumerate(slot_ids)}
        rows = (_NativeQuotaRow * len(counts))(
            *(
                _NativeQuotaRow(
                    game_count=count,
                    frozen_artifact_slot=(
                        _NO_FROZEN_ARTIFACT
                        if artifact_id is None
                        else slots[artifact_id]
                    ),
                    cohort_artifact_slot=(
                        _NO_FROZEN_ARTIFACT if cohort_id is None else slots[cohort_id]
                    ),
                )
                for count, artifact_id, cohort_id in zip(
                    counts,
                    artifacts,
                    cohorts,
                    strict=True,
                )
            )
        )
        self._library = _load_library(library_path)
        self._configure_abi(self._library)
        pointer = self._library.CgTrainQuotaSchedulerCreate(
            rows,
            ctypes.c_uint32(len(counts)),
            ctypes.c_uint32(max(1, len(slot_ids))),
        )
        if not pointer:
            raise RuntimeError(self._last_error())
        self._pointer = int(pointer)

    @property
    def remaining(self) -> int:
        """Return the number of unissued games."""
        return int(
            self._library.CgTrainQuotaSchedulerRemaining(self._require_pointer())
        )

    def can_take(
        self,
        game_count: int,
        *,
        frozen_artifact_limit: int,
        required_artifact_id: str | None = None,
    ) -> bool:
        """Return whether one full coherent recipe is currently available."""
        return bool(
            self._library.CgTrainQuotaSchedulerCanTake(
                self._require_pointer(),
                ctypes.c_uint32(game_count),
                ctypes.c_uint32(self._effective_artifact_limit(frozen_artifact_limit)),
                ctypes.c_uint32(self._artifact_slot(required_artifact_id)),
            )
        )

    def take(
        self,
        game_count: int,
        *,
        frozen_artifact_limit: int,
        required_artifact_id: str | None = None,
    ) -> tuple[int, ...]:
        """Atomically consume one recipe and return its aggregate row indices."""
        if game_count <= 0:
            raise ValueError("quota recipe size must be positive")
        output = (ctypes.c_uint32 * game_count)()
        output_count = ctypes.c_uint32()
        result = int(
            self._library.CgTrainQuotaSchedulerTake(
                self._require_pointer(),
                ctypes.c_uint32(game_count),
                ctypes.c_uint32(self._effective_artifact_limit(frozen_artifact_limit)),
                ctypes.c_uint32(self._artifact_slot(required_artifact_id)),
                output,
                ctypes.c_uint32(game_count),
                ctypes.byref(output_count),
            )
        )
        if result != 0:
            raise RuntimeError(self._last_error())
        if output_count.value != game_count:
            raise RuntimeError("native quota scheduler returned a partial recipe")
        return tuple(int(output[index]) for index in range(output_count.value))

    def close(self) -> None:
        """Release the native scheduler deterministically."""
        if self._pointer is not None:
            self._library.CgTrainQuotaSchedulerDestroy(self._pointer)
            self._pointer = None

    def __enter__(self) -> NativeQuotaScheduler:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _artifact_slot(self, artifact_id: str | None) -> int:
        if artifact_id is None:
            return _NO_FROZEN_ARTIFACT
        try:
            return self._slot_ids.index(artifact_id)
        except ValueError as exc:
            raise ValueError("required quota artifact is absent") from exc

    def _effective_artifact_limit(self, requested: int) -> int:
        if requested <= 0:
            raise ValueError("frozen artifact limit must be positive")
        return min(requested, max(1, len(self._slot_ids)))

    def _require_pointer(self) -> int:
        if self._pointer is None:
            raise RuntimeError("native quota scheduler is closed")
        return self._pointer

    def _last_error(self) -> str:
        raw = self._library.CgTrainQuotaSchedulerLastError()
        return "native quota scheduler failed" if raw is None else raw.decode()

    @staticmethod
    def _configure_abi(library: Any) -> None:
        library.CgTrainQuotaSchedulerLastError.restype = ctypes.c_char_p
        library.CgTrainQuotaSchedulerCreate.argtypes = [
            ctypes.POINTER(_NativeQuotaRow),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.CgTrainQuotaSchedulerCreate.restype = ctypes.c_void_p
        library.CgTrainQuotaSchedulerDestroy.argtypes = [ctypes.c_void_p]
        library.CgTrainQuotaSchedulerRemaining.argtypes = [ctypes.c_void_p]
        library.CgTrainQuotaSchedulerRemaining.restype = ctypes.c_uint64
        library.CgTrainQuotaSchedulerCanTake.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.CgTrainQuotaSchedulerCanTake.restype = ctypes.c_int32
        library.CgTrainQuotaSchedulerTake.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.CgTrainQuotaSchedulerTake.restype = ctypes.c_int32


def _load_library(path: Path | None) -> Any:
    resolved = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "native"
        / "cg_train"
        / "libcg_train.so"
        if path is None
        else path
    )
    if not resolved.is_file():
        raise RuntimeError(f"native quota scheduler library is absent: {resolved}")
    return ctypes.CDLL(str(resolved))


__all__ = ["NativeQuotaScheduler"]
