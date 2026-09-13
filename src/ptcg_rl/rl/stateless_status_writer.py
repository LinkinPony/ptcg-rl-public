"""Latest-only background publication for stateless training observability."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.stateless_curriculum_codec import encode_compact_mapping

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StatelessStatusSnapshot:
    """Detached payloads for one latest-status publication."""

    learner: dict[str, Any]
    allocation: dict[str, Any] | None
    curriculum_detail: dict[str, Any] | None = None


class NonBlockingStatelessStatusWriter:
    """Coalesce latest-only status updates without blocking the learner."""

    def __init__(self, output_dir: Path) -> None:
        self._learner_path = output_dir / "learner_status.json"
        self._allocation_path = output_dir / "control" / "opponent_allocation.json"
        self._control_dir = output_dir / "control"
        self._condition = threading.Condition()
        self._pending: StatelessStatusSnapshot | None = None
        self._closed = False
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="stateless-status-writer",
            daemon=True,
        )
        self._thread.start()

    def publish(self, snapshot: StatelessStatusSnapshot) -> None:
        """Replace an unstarted snapshot because both artifacts are latest-only."""
        with self._condition:
            if self._closed:
                raise RuntimeError("stateless status writer is closed")
            self._pending = snapshot
            self._condition.notify()

    @property
    def last_error(self) -> str | None:
        """Return the most recent non-fatal observability failure."""
        with self._condition:
            return self._last_error

    def close(self) -> None:
        """Flush the newest accepted snapshot and stop the writer."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._closed
                )
                snapshot = self._pending
                self._pending = None
                if snapshot is None and self._closed:
                    return
            if snapshot is None:
                continue
            try:
                if snapshot.curriculum_detail is not None:
                    curriculum_path = _publish_compact_field(
                        snapshot.learner,
                        field_path=("curriculum", "detail_artifact"),
                        detail=snapshot.curriculum_detail,
                        directory=self._control_dir / "curriculum_observability",
                        pointer_base=self._learner_path.parent,
                    )
                allocation = snapshot.allocation
                if allocation is not None:
                    report = allocation.get("report")
                    if not isinstance(report, dict):
                        raise ValueError("opponent allocation report must be a mapping")
                    allocation_path = _publish_compact_field(
                        allocation,
                        field_path=("report",),
                        detail=report,
                        directory=self._control_dir / "opponent_allocations",
                        pointer_base=self._allocation_path.parent,
                    )
                atomic_write_bytes(
                    self._learner_path,
                    json_payload(snapshot.learner),
                    overwrite=True,
                )
                if snapshot.curriculum_detail is not None:
                    _prune_compact_details(curriculum_path.parent)
                if allocation is not None:
                    atomic_write_bytes(
                        self._allocation_path,
                        json_payload(allocation),
                        overwrite=True,
                    )
                    _prune_compact_details(allocation_path.parent)
                with self._condition:
                    self._last_error = None
            except Exception as error:  # noqa: BLE001 - telemetry is non-critical.
                rendered = f"{type(error).__name__}: {error}"
                with self._condition:
                    self._last_error = rendered
                _LOGGER.exception("stateless status publication failed")


def _publish_compact_field(
    envelope: dict[str, Any],
    *,
    field_path: tuple[str, ...],
    detail: dict[str, Any],
    directory: Path,
    pointer_base: Path,
) -> Path:
    """Publish immutable detail first, then replace it with a verified pointer."""
    payload = encode_compact_mapping(detail)
    sha256 = hashlib.sha256(payload).hexdigest()
    path = directory / f"detail-{sha256}.msgpack.zlib"
    if path.exists():
        if path.stat().st_size != len(payload) or path.read_bytes() != payload:
            raise FileExistsError(f"compact status detail differs from retry: {path}")
    else:
        atomic_write_bytes(path, payload, overwrite=False)
    pointer: dict[str, Any] = {
        "format": "compact-mapping-msgpack-zlib-v1",
        "path": str(path.relative_to(pointer_base)),
        "size_bytes": len(payload),
        "sha256": sha256,
    }
    target = envelope
    for key in field_path[:-1]:
        nested = target.get(key)
        if not isinstance(nested, dict):
            raise ValueError("compact status pointer parent must be a mapping")
        target = nested
    target[field_path[-1]] = pointer
    return path


def _prune_compact_details(directory: Path, *, keep_last: int = 2) -> None:
    """Bound superseded latest-only details after their pointer is durable."""
    try:
        paths = sorted(
            directory.glob("detail-*.msgpack.zlib"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths[keep_last:]:
            path.unlink(missing_ok=True)
    except OSError:
        _LOGGER.exception("compact status detail pruning failed")


__all__ = [
    "NonBlockingStatelessStatusWriter",
    "StatelessStatusSnapshot",
]
