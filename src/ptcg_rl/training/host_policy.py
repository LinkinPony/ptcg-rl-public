"""Hardware policy checks for reusable RL training runs."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

_NVIDIA_SMI_TIMEOUT_SECONDS = 10.0


def require_cuda_training_host(
    accelerator_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Require a visible NVIDIA accelerator before starting an RL training run.

    Args:
        accelerator_names: Optional detected accelerator names. Supplying names
            makes the policy independently testable; production callers should
            omit this argument and use the local ``nvidia-smi`` inventory.

    Returns:
        The normalized visible accelerator names.

    Raises:
        RuntimeError: No NVIDIA accelerator is visible on the current host.
    """
    names = tuple(
        name.strip()
        for name in (
            accelerator_names
            if accelerator_names is not None
            else _query_nvidia_accelerator_names()
        )
        if name.strip()
    )
    if names:
        return names
    detected = ", ".join(names) if names else "none"
    raise RuntimeError(
        "RL training requires an NVIDIA accelerator; visible accelerators: "
        f"{detected}. Dry-run, monitoring, and configuration inspection remain "
        "available on other hosts."
    )


def _query_nvidia_accelerator_names() -> tuple[str, ...]:
    """Return names reported by the local NVIDIA driver."""
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode != 0:
        return ()
    return tuple(completed.stdout.splitlines())
