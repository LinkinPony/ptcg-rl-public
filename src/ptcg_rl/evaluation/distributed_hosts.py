"""SSH host discovery and resource allocation for distributed evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.evaluation.search_identity import fingerprint_payload

_SSH_TARGET_PATTERN = re.compile(
    r"(?:(?P<user>[A-Za-z_][A-Za-z0-9_.-]*)@)?"
    r"(?P<host>[A-Za-z0-9][A-Za-z0-9_.-]*)"
)
_DOCKER_IMAGE_REFERENCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_MODULES = ("torch", "numpy", "pyarrow", "pydantic")


class GPUResourceProbe(BaseModel):
    """One GPU visible to a remote login session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int


class PythonRuntimeProbe(BaseModel):
    """Versioned Python environment usable by an evaluation worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: str
    python_version: str
    implementation: str
    libc: tuple[str, str]
    packages: dict[str, str | None]
    torch_cuda_version: str | None
    torch_cuda_available: bool
    runtime_environment_id: str | None = None
    container_image_id: str | None = None
    runtime_fingerprint: str = ""

    @model_validator(mode="after")
    def derive_fingerprint(self) -> PythonRuntimeProbe:
        payload = {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "libc": self.libc,
            "packages": self.packages,
            "torch_cuda_version": self.torch_cuda_version,
            "runtime_environment_id": self.runtime_environment_id,
        }
        fingerprint = fingerprint_payload(payload)
        if self.runtime_fingerprint and self.runtime_fingerprint != fingerprint:
            raise ValueError("Python runtime fingerprint does not match its payload")
        if not self.runtime_fingerprint:
            object.__setattr__(self, "runtime_fingerprint", fingerprint)
        return self

    @property
    def compatible(self) -> bool:
        """Whether all packages required by the host worker are installed."""
        return all(self.packages.get(name) is not None for name in _REQUIRED_MODULES)


class SSHHostProbe(BaseModel):
    """Read-only CPU, GPU, storage, and runtime evidence for one SSH target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    host_label: str
    hostname: str
    home: Path
    affinity: tuple[int, ...]
    cpu_model: str | None
    memory_total_bytes: int
    memory_available_bytes: int
    home_free_bytes: int
    load_average: tuple[float, float, float]
    gpus: tuple[GPUResourceProbe, ...] = ()
    gpu_probe_error: str | None = None
    docker_available: bool = False
    container_archive_sha256: str | None = None
    runtimes: tuple[PythonRuntimeProbe, ...] = ()
    selected_runtime: PythonRuntimeProbe | None = None


class DistributedHostConfig(BaseModel):
    """Dynamic cluster policy; host identities remain in an ignored env file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hosts_env_path: Path = Path(".env")
    hosts_env_key: str = "SSH_HOSTS"
    python_candidates: tuple[str, ...] = (
        "$HOME/miniconda3/envs/rl/bin/python",
        "$HOME/miniconda3/envs/dl/bin/python",
        "python3",
    )
    runtime_backend: Literal["host", "docker"] = "host"
    container_python_wrapper: str = (
        "$HOME/.local/libexec/ptcg-rl-eval-python"
    )
    container_image: str | None = None
    container_archive_sha256: str | None = None
    requested_resource: Literal["cpu", "cuda"] = "cpu"
    require_homogeneous_runtime: bool = True
    require_all_hosts: bool = False
    remote_root: Path = Path(".cache/ptcg-rl-eval")
    reserve_cpus_per_host: int = 2
    max_workers_per_host: int = 8
    cpu_threads_per_worker: int = 1
    workers_per_gpu: int = 1
    minimum_memory_per_worker_bytes: int = 2 * 1024**3
    minimum_home_free_bytes: int = 5 * 1024**3
    connect_timeout_seconds: int = 10
    poll_interval_seconds: float = 10.0

    @field_validator(
        "reserve_cpus_per_host",
        "max_workers_per_host",
        "cpu_threads_per_worker",
        "workers_per_gpu",
        "minimum_memory_per_worker_bytes",
        "minimum_home_free_bytes",
        "connect_timeout_seconds",
    )
    @classmethod
    def nonnegative_resources(cls, value: int) -> int:
        if value < 0:
            raise ValueError("distributed host resource values cannot be negative")
        return value

    @field_validator("max_workers_per_host", "cpu_threads_per_worker")
    @classmethod
    def positive_worker_values(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("distributed worker counts must be positive")
        return value

    @field_validator("poll_interval_seconds")
    @classmethod
    def positive_poll_interval(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")
        return value

    @field_validator("remote_root")
    @classmethod
    def relative_remote_root(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("remote_root must be a safe path relative to remote HOME")
        return value

    @model_validator(mode="after")
    def valid_runtime_backend(self) -> DistributedHostConfig:
        if self.runtime_backend == "docker":
            if self.container_image is None:
                raise ValueError("docker runtime requires container_image")
            if (
                _DOCKER_IMAGE_REFERENCE_PATTERN.fullmatch(self.container_image)
                is None
            ):
                raise ValueError(
                    "container_image must be a safe local Docker image reference"
                )
            if (
                self.container_archive_sha256 is None
                or _SHA256_PATTERN.fullmatch(self.container_archive_sha256) is None
            ):
                raise ValueError(
                    "docker runtime requires the verified image archive SHA256"
                )
        elif (
            self.container_image is not None
            or self.container_archive_sha256 is not None
        ):
            raise ValueError(
                "container image identity is only valid for the docker runtime"
            )
        return self


class HostAllocation(BaseModel):
    """One validated worker allocation and its immutable runtime identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    host_label: str
    hostname: str
    home: Path
    python_executable: str
    runtime_fingerprint: str
    workers: int
    cpu_threads_per_worker: int
    cpu_affinity: tuple[int, ...]
    gpu_indices: tuple[int, ...] = ()
    requested_resource: Literal["cpu", "cuda"]
    runtime_backend: Literal["host", "docker"] = "host"
    container_image: str | None = None
    container_archive_sha256: str | None = None


def load_ssh_targets(path: Path, *, key: str = "SSH_HOSTS") -> tuple[str, ...]:
    """Parse validated ``user@host`` targets without evaluating an env file."""
    value: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        if name.strip() == key:
            value = raw_value.strip()
            break
    if value is None:
        raise ValueError(f"{key} is missing from {path}")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(decoded, list):
        raise ValueError(f"{key} must be a JSON list or comma-separated targets")
    targets = tuple(str(item).strip() for item in decoded)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError(f"{key} must contain distinct SSH targets")
    for target in targets:
        if _SSH_TARGET_PATTERN.fullmatch(target) is None:
            raise ValueError(f"unsafe SSH target in {key}: {target!r}")
    return targets


def probe_ssh_hosts(
    targets: tuple[str, ...],
    config: DistributedHostConfig,
) -> tuple[SSHHostProbe, ...]:
    """Probe all hosts concurrently without changing remote state."""
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = [executor.submit(_probe_ssh_host, target, config) for target in targets]
        return tuple(future.result() for future in futures)


def probe_ssh_hosts_resilient(
    targets: tuple[str, ...],
    config: DistributedHostConfig,
) -> tuple[tuple[SSHHostProbe, ...], dict[str, str]]:
    """Probe hosts concurrently while preserving explicit connection failures."""
    probes: list[SSHHostProbe] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = {
            target: executor.submit(_probe_ssh_host, target, config)
            for target in targets
        }
        for target, future in futures.items():
            try:
                probes.append(future.result())
            except Exception as exc:  # A single unavailable host may be optional.
                failures[target] = f"{type(exc).__name__}: {exc}"
    return tuple(probes), failures


def allocate_hosts(
    probes: tuple[SSHHostProbe, ...],
    config: DistributedHostConfig,
) -> tuple[tuple[HostAllocation, ...], dict[str, str]]:
    """Select compatible hosts and assign bounded CPU/GPU worker counts."""
    usable: list[SSHHostProbe] = []
    skipped: dict[str, str] = {}
    for probe in probes:
        runtime = probe.selected_runtime
        if runtime is None or not runtime.compatible:
            skipped[probe.target] = "missing_required_python_runtime"
            continue
        if (
            config.runtime_backend == "docker"
            and probe.container_archive_sha256
            != config.container_archive_sha256
        ):
            skipped[probe.target] = "container_archive_identity_mismatch"
            continue
        if probe.home_free_bytes < config.minimum_home_free_bytes:
            skipped[probe.target] = "insufficient_home_storage"
            continue
        if config.requested_resource == "cuda" and not probe.gpus:
            skipped[probe.target] = "cuda_unavailable"
            continue
        usable.append(probe)
    if config.require_all_hosts and skipped:
        raise RuntimeError(f"required evaluation hosts are unavailable: {skipped}")
    if not usable:
        raise RuntimeError(f"no compatible evaluation hosts are available: {skipped}")

    cohorts: dict[str, list[SSHHostProbe]] = defaultdict(list)
    for probe in usable:
        assert probe.selected_runtime is not None
        cohorts[probe.selected_runtime.runtime_fingerprint].append(probe)
    if config.require_homogeneous_runtime and len(cohorts) != 1:
        summary = {fingerprint: [item.target for item in values] for fingerprint, values in cohorts.items()}
        raise RuntimeError(
            "distributed formal evaluation requires one Python runtime fingerprint; "
            f"found {summary}"
        )

    allocations: list[HostAllocation] = []
    for probe in usable:
        runtime = probe.selected_runtime
        assert runtime is not None
        available_cpu_count = max(
            0,
            len(probe.affinity) - config.reserve_cpus_per_host,
        )
        cpu_capacity = available_cpu_count // config.cpu_threads_per_worker
        memory_capacity = (
            probe.memory_available_bytes // config.minimum_memory_per_worker_bytes
            if config.minimum_memory_per_worker_bytes
            else config.max_workers_per_host
        )
        resource_capacity = (
            len(probe.gpus) * config.workers_per_gpu
            if config.requested_resource == "cuda"
            else config.max_workers_per_host
        )
        workers = min(
            config.max_workers_per_host,
            cpu_capacity,
            memory_capacity,
            resource_capacity,
        )
        if workers <= 0:
            skipped[probe.target] = "zero_worker_capacity"
            continue
        cpu_count = workers * config.cpu_threads_per_worker
        cpu_affinity = probe.affinity[
            config.reserve_cpus_per_host : config.reserve_cpus_per_host + cpu_count
        ]
        gpu_indices = (
            tuple(gpu.index for gpu in probe.gpus)
            if config.requested_resource == "cuda"
            else ()
        )
        allocations.append(
            HostAllocation(
                target=probe.target,
                host_label=probe.host_label,
                hostname=probe.hostname,
                home=probe.home,
                python_executable=runtime.executable,
                runtime_fingerprint=runtime.runtime_fingerprint,
                workers=workers,
                cpu_threads_per_worker=config.cpu_threads_per_worker,
                cpu_affinity=cpu_affinity,
                gpu_indices=gpu_indices,
                requested_resource=config.requested_resource,
                runtime_backend=config.runtime_backend,
                container_image=config.container_image,
                container_archive_sha256=config.container_archive_sha256,
            )
        )
    if not allocations:
        raise RuntimeError(f"no evaluation host has worker capacity: {skipped}")
    return tuple(allocations), skipped


def host_probe_fingerprint(probes: tuple[SSHHostProbe, ...]) -> str:
    """Fingerprint a sorted probe snapshot for campaign evidence."""
    payload = [
        probe.model_dump(mode="json")
        for probe in sorted(probes, key=lambda item: item.target)
    ]
    return fingerprint_payload({"hosts": payload})


def _probe_ssh_host(
    target: str,
    config: DistributedHostConfig,
) -> SSHHostProbe:
    candidates = (
        (config.container_python_wrapper,)
        if config.runtime_backend == "docker"
        else config.python_candidates
    )
    script = _remote_probe_script(
        candidates,
        runtime_backend=config.runtime_backend,
        container_image=config.container_image,
        container_archive_sha256=config.container_archive_sha256,
        remote_root=config.remote_root,
    )
    command = f"python3 -c {shlex.quote(script)}"
    completed = subprocess.run(
        _ssh_command(target, command, timeout=config.connect_timeout_seconds),
        check=False,
        capture_output=True,
        text=True,
        timeout=max(30, config.connect_timeout_seconds * 4),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip()[-1000:]
        raise RuntimeError(f"SSH resource probe failed for {target}: {message}")
    raw = json.loads(completed.stdout)
    if not isinstance(raw, dict):
        raise ValueError(f"SSH resource probe returned invalid JSON for {target}")
    runtimes = tuple(PythonRuntimeProbe.model_validate(item) for item in raw["runtimes"])
    selected_runtime = next((item for item in runtimes if item.compatible), None)
    raw.update(
        {
            "target": target,
            "host_label": _host_label(target, str(raw["hostname"])),
            "runtimes": runtimes,
            "selected_runtime": selected_runtime,
        }
    )
    return SSHHostProbe.model_validate(raw)


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


def _host_label(target: str, hostname: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", hostname).strip("-") or "host"
    suffix = hashlib.sha256(target.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{suffix}"


def _remote_probe_script(
    python_candidates: tuple[str, ...],
    *,
    runtime_backend: Literal["host", "docker"],
    container_image: str | None,
    container_archive_sha256: str | None,
    remote_root: Path,
) -> str:
    candidates_json = json.dumps(python_candidates)
    required_json = json.dumps(_REQUIRED_MODULES)
    child_script = f'''import importlib.metadata
import json
import os
import platform
import sys
packages = {{}}
for name in json.loads({required_json!r}):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
cuda_version = None
cuda_available = False
if packages.get("torch") is not None:
    import torch
    cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
    cuda_available = bool(torch.cuda.is_available())
print(json.dumps({{
    "executable": sys.executable,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "libc": list(platform.libc_ver()),
    "packages": packages,
    "torch_cuda_version": cuda_version,
    "torch_cuda_available": cuda_available,
    "runtime_environment_id": os.environ.get("PTCG_EVAL_RUNTIME_ID"),
    "container_image_id": os.environ.get("PTCG_EVAL_LOCAL_IMAGE_ID"),
}}))'''
    return f'''import importlib.metadata
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

candidates = json.loads({candidates_json!r})
required = json.loads({required_json!r})
runtime_backend = {runtime_backend!r}
container_image = {container_image!r}
expected_container_archive_sha256 = {container_archive_sha256!r}
remote_root = pathlib.Path.home() / {str(remote_root)!r}

def memory_value(name):
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.lower().startswith(name.lower() + ":"):
            return int(line.split()[1]) * 1024
    return 0

def cpu_model():
    for line in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None

child = {child_script!r}

runtimes = []
seen = set()
for raw in candidates:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    executable = shutil.which(expanded) if "/" not in expanded else expanded
    if not executable or executable in seen or not os.path.exists(executable):
        continue
    seen.add(executable)
    child_env = dict(os.environ)
    if runtime_backend == "docker":
        child_env.update({{
            "PTCG_EVAL_IMAGE": container_image,
            "PTCG_EVAL_REMOTE_ROOT": str(remote_root),
        }})
    run = subprocess.run(
        [executable, "-c", child],
        text=True,
        capture_output=True,
        env=child_env,
    )
    if run.returncode == 0:
        runtime = json.loads(run.stdout)
        runtime["executable"] = executable
        runtimes.append(runtime)

gpus = []
gpu_probe_error = None
if shutil.which("nvidia-smi"):
    run = subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True)
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                gpus.append({{
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }})
    else:
        gpu_probe_error = "driver_unavailable"

home = pathlib.Path.home()
usage = shutil.disk_usage(home)
affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 0))
docker = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 if shutil.which("docker") else False
observed_container_archive_sha256 = None
if expected_container_archive_sha256 is not None:
    archive_path = (
        remote_root
        / "images"
        / (expected_container_archive_sha256 + ".tar.gz")
    )
    if archive_path.is_file():
        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                digest.update(chunk)
        observed_container_archive_sha256 = digest.hexdigest()
print(json.dumps({{
    "hostname": platform.node(),
    "home": str(home),
    "affinity": affinity,
    "cpu_model": cpu_model(),
    "memory_total_bytes": memory_value("MemTotal"),
    "memory_available_bytes": memory_value("MemAvailable"),
    "home_free_bytes": usage.free,
    "load_average": list(os.getloadavg()),
    "gpus": gpus,
    "gpu_probe_error": gpu_probe_error,
    "docker_available": docker,
    "container_archive_sha256": observed_container_archive_sha256,
    "runtimes": runtimes,
}}))'''


__all__ = [
    "DistributedHostConfig",
    "GPUResourceProbe",
    "HostAllocation",
    "PythonRuntimeProbe",
    "SSHHostProbe",
    "allocate_hosts",
    "host_probe_fingerprint",
    "load_ssh_targets",
    "probe_ssh_hosts",
    "probe_ssh_hosts_resilient",
]
