"""Install one verified release-evaluation runtime on SSH worker hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ptcg_rl.evaluation.distributed_hosts import load_ssh_targets
from ptcg_rl.evaluation.search_identity import file_sha256

_LOADED_IMAGE_PATTERN = re.compile(r"Loaded image(?: ID)?: (\S+)")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--hosts-env", type=Path, default=Path(".env"))
    parser.add_argument("--hosts-env-key", default="SSH_HOSTS")
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=Path("src/tools/distributed_eval/container_python.sh"),
    )
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path(".cache/ptcg-rl-eval"),
    )
    parser.add_argument("--connect-timeout-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Provision all configured hosts concurrently and emit public evidence."""
    args = _parse_args()
    archive = args.archive.resolve(strict=True)
    wrapper = args.wrapper.resolve(strict=True)
    expected_sha256 = str(args.archive_sha256)
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("--archive-sha256 must be 64 lowercase hex characters")
    if file_sha256(archive) != expected_sha256:
        raise ValueError("local runtime archive does not match --archive-sha256")
    remote_root = Path(args.remote_root)
    if remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("--remote-root must be relative to remote HOME")
    targets = load_ssh_targets(
        args.hosts_env,
        key=str(args.hosts_env_key),
    )
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = [
            executor.submit(
                _provision_host,
                target,
                archive=archive,
                archive_sha256=expected_sha256,
                image_reference=str(args.image_reference),
                wrapper=wrapper,
                remote_root=remote_root,
                connect_timeout_seconds=int(args.connect_timeout_seconds),
            )
            for target in targets
        ]
        results = [future.result() for future in futures]
    print(json.dumps({"hosts": results}, indent=2, sort_keys=True))


def _provision_host(
    target: str,
    *,
    archive: Path,
    archive_sha256: str,
    image_reference: str,
    wrapper: Path,
    remote_root: Path,
    connect_timeout_seconds: int,
) -> dict[str, Any]:
    home = Path(
        _run_capture(
            _ssh_command(
                target,
                "python3 -c 'import pathlib; print(pathlib.Path.home())'",
                timeout=connect_timeout_seconds,
            )
        ).strip()
    )
    if not home.is_absolute():
        raise ValueError(f"remote HOME is not absolute for {target}")
    evaluation_root = home / remote_root
    image_dir = evaluation_root / "images"
    remote_archive = image_dir / f"{archive_sha256}.tar.gz"
    remote_wrapper = home / ".local" / "libexec" / "ptcg-rl-eval-python"
    _run_checked(
        _ssh_command(
            target,
            "mkdir -p "
            f"{shlex.quote(str(image_dir))} "
            f"{shlex.quote(str(remote_wrapper.parent))}",
            timeout=connect_timeout_seconds,
        )
    )
    _run_checked(("rsync", "-az", str(archive), f"{target}:{remote_archive}"))
    _run_checked(("rsync", "-az", str(wrapper), f"{target}:{remote_wrapper}"))
    observed_sha256 = _run_capture(
        _ssh_command(
            target,
            f"sha256sum {shlex.quote(str(remote_archive))}",
            timeout=connect_timeout_seconds,
        )
    ).split()[0]
    if observed_sha256 != archive_sha256:
        raise ValueError(f"remote runtime archive hash mismatch for {target}")
    load_output = _run_capture(
        _ssh_command(
            target,
            f"docker load --input {shlex.quote(str(remote_archive))}",
            timeout=connect_timeout_seconds,
        ),
        timeout=1200.0,
    )
    matches = _LOADED_IMAGE_PATTERN.findall(load_output)
    if not matches:
        raise RuntimeError(f"docker load returned no image identity for {target}")
    loaded_reference = matches[-1]
    remote_command = (
        f"chmod 0755 {shlex.quote(str(remote_wrapper))} && "
        f"docker tag {shlex.quote(loaded_reference)} "
        f"{shlex.quote(image_reference)} && "
        f"docker image inspect {shlex.quote(image_reference)} "
        "--format '{{.Id}}'"
    )
    local_image_id = _run_capture(
        _ssh_command(
            target,
            remote_command,
            timeout=connect_timeout_seconds,
        )
    ).strip()
    return {
        "host_id": _public_host_id(target),
        "archive_sha256": observed_sha256,
        "image_reference": image_reference,
        "local_image_id": local_image_id,
        "wrapper_path": str(remote_wrapper),
    }


def _public_host_id(target: str) -> str:
    return hashlib.sha256(target.encode("utf-8")).hexdigest()[:12]


def _ssh_command(target: str, command: str, *, timeout: int) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        target,
        command,
    )


def _run_checked(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, timeout=1200.0)


def _run_capture(command: tuple[str, ...], *, timeout: float = 120.0) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


if __name__ == "__main__":
    main()
