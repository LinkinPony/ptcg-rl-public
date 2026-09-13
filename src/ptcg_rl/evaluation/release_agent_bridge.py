"""Stdlib-only bridge for one isolated packaged submission episode.

This file is executed directly with ``python -I``.  Keep it free of project and
third-party imports so the child can import only the selected release archive.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import struct
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_MAX_TRACEBACK_CHARS = 32_768


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--protocol-fd", type=int, required=True)
    return parser.parse_args()


def _read_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("release-agent protocol closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(channel: socket.socket) -> dict[str, Any]:
    size = _HEADER.unpack(_read_exact(channel, _HEADER.size))[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"release-agent message is too large: {size}")
    value = json.loads(_read_exact(channel, size))
    if not isinstance(value, dict):
        raise TypeError("release-agent request must be a JSON object")
    return value


def _write_message(channel: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise ValueError("release-agent response exceeds protocol limit")
    channel.sendall(_HEADER.pack(len(encoded)) + encoded)


def _runtime_status(main_module: ModuleType) -> dict[str, Any] | None:
    runtime = getattr(main_module, "_AGENT", None)
    status_method = getattr(runtime, "runtime_status", None)
    if not callable(status_method):
        return None
    value = status_method()
    return dict(value) if isinstance(value, dict) else None


def _last_telemetry(main_module: ModuleType) -> dict[str, Any] | None:
    runtime = getattr(main_module, "_AGENT", None)
    telemetry_method = getattr(runtime, "last_act_telemetry", None)
    if not callable(telemetry_method):
        return None
    value = telemetry_method()
    return dict(value) if isinstance(value, dict) else None


def _close_runtime(main_module: ModuleType | None) -> None:
    if main_module is None:
        return
    runtime = getattr(main_module, "_AGENT", None)
    close_method = getattr(runtime, "close", None)
    if callable(close_method):
        close_method()


def _import_agent(agent_root: Path, work_dir: Path) -> ModuleType:
    agent_root = agent_root.resolve(strict=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir.resolve(strict=True))
    sys.path.insert(0, str(agent_root))
    main_module = importlib.import_module("main")
    sys.path = [item for item in sys.path if Path(item or ".").resolve() != agent_root]
    return main_module


def _module_path(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    return str(Path(path).resolve()) if path is not None else None


def _policy_device(main_module: ModuleType) -> str | None:
    runtime = getattr(main_module, "_AGENT", None)
    policy = getattr(runtime, "_policy", None)
    device = getattr(policy, "device", None)
    if device is None:
        # Older fixed-deck runtimes keep the resolved torch.device private.
        device = getattr(policy, "_device", None)
    if callable(device):
        try:
            return str(device())
        except Exception:  # noqa: BLE001 - status must not break inference.
            return None
    return str(device) if device is not None else None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "numpy", "pyarrow", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _execution_status(main_module: ModuleType) -> dict[str, Any]:
    affinity = (
        sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    torch_module = sys.modules.get("torch")
    get_num_threads = getattr(torch_module, "get_num_threads", None)
    torch_threads = int(get_num_threads()) if callable(get_num_threads) else None
    return {
        "affinity": affinity,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "torch_num_threads": torch_threads,
        "policy_device": _policy_device(main_module),
        "python": platform.python_version(),
        "packages": _package_versions(),
    }


def _serve(channel: socket.socket, main_module: ModuleType) -> None:
    agent = getattr(main_module, "agent", None)
    if not callable(agent):
        raise TypeError("packaged main.py does not expose callable agent()")
    _write_message(
        channel,
        {
            "type": "ready",
            "main_path": _module_path("main"),
            "package_path": _module_path("ptcg_rl"),
            "runtime_status": _runtime_status(main_module),
            "execution_status": _execution_status(main_module),
        },
    )
    while True:
        request = _read_message(channel)
        request_type = request.get("type")
        if request_type == "close":
            _write_message(channel, {"type": "closed"})
            return
        if request_type != "act":
            raise ValueError(f"unsupported release-agent request: {request_type!r}")
        try:
            action = agent(request.get("observation"), request.get("configuration"))
            _write_message(
                channel,
                {
                    "type": "action",
                    "action": action,
                    "runtime_status": _runtime_status(main_module),
                    "telemetry": _last_telemetry(main_module),
                    "execution_status": _execution_status(main_module),
                },
            )
        except Exception as exc:  # noqa: BLE001 - return package failure verbatim.
            _write_message(
                channel,
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc()[-_MAX_TRACEBACK_CHARS:],
                },
            )


def main() -> None:
    """Import one packaged submission and serve exactly one episode."""
    args = _parse_args()
    channel = socket.socket(fileno=args.protocol_fd)
    main_module: ModuleType | None = None
    try:
        try:
            main_module = _import_agent(args.agent_root, args.work_dir)
            _serve(channel, main_module)
        except Exception as exc:  # noqa: BLE001 - preserve startup/protocol evidence.
            _write_message(
                channel,
                {
                    "type": "fatal",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc()[-_MAX_TRACEBACK_CHARS:],
                },
            )
            raise
    finally:
        try:
            _close_runtime(main_module)
        finally:
            channel.close()


if __name__ == "__main__":
    main()
