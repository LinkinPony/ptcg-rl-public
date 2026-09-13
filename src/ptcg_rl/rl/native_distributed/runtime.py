"""Worker/coordinator identity construction and CUDA/native preflight."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zmq

from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.engine.native_training import (
    NATIVE_TRAINING_ABI_VERSION,
    resolve_native_training_library,
)
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactProducer
from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.rl.native_distributed.contracts import (
    NativeCollectionCapacityTier,
    NativeCollectionWorkerManifest,
    NativeRolloutWorkerIdentity,
)
from ptcg_rl.rl.stateless_training_config import (
    SimpleStatelessTrainingConfig,
    StatelessNativeWorkerProfileConfig,
)
from ptcg_rl.training.source_identity import resolve_training_source_identity

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_DOMAIN = b"ptcg-rl/native-distributed-worker-runtime/v1\x00"


def source_snapshot_fingerprint(repo_root: Path = _REPO_ROOT) -> str:
    """Bind formal workers to executable content, independent of Git HEAD."""
    return resolve_training_source_identity(repo_root).training_source_fingerprint


def build_coordinator_worker_contract(
    config: SimpleStatelessTrainingConfig,
) -> NativeRolloutWorkerIdentity:
    """Build the host-independent compatibility contract expected from workers."""
    from ptcg_rl.rl.stateless_training import (
        resolve_simple_stateless_training_resources,
    )

    resources = resolve_simple_stateless_training_resources(config)
    native_path, _native_abi = resolve_native_training_library()
    native_fingerprint = _file_sha256(native_path)
    fact_contract, _fact_binary = _engine_fact_fingerprints(resources.model_config)
    source_identity = resolve_training_source_identity(_REPO_ROOT)
    return NativeRolloutWorkerIdentity(
        worker_id="coordinator-contract",
        session_id="coordinator-contract",
        runtime_fingerprint="0" * 64,
        source_git_commit=source_identity.source_git_commit,
        source_snapshot_fingerprint=(
            source_identity.training_source_fingerprint
        ),
        native_library_fingerprint=native_fingerprint,
        native_abi_version=NATIVE_TRAINING_ABI_VERSION,
        engine_fact_contract_fingerprint=fact_contract,
        feature_schema_fingerprint=(
            resources.policy_identity.fragment_static_contract_fingerprint
        ),
        card_catalog_fingerprint=(resources.catalog_manifest.card_catalog_fingerprint),
        static_features_fingerprint=_file_sha256(
            _resolve_repo_path(resources.model_config.card_encoder.feature_table_path)
        ),
        exact_registry_fingerprint=str(resources.model_config.resolved_registry_sha256),
        scripted_opponents_fingerprint=resources.scripted_manifest_fingerprint,
        historical_opponents_fingerprint=resources.pinned_manifest_fingerprint,
        model_config_fingerprint=(resources.policy_identity.model_config_fingerprint),
        resolved_config_fingerprint=(
            resources.policy_identity.resolved_config_fingerprint
        ),
    )


def build_worker_manifest(
    config: SimpleStatelessTrainingConfig,
    *,
    worker_id: str,
    session_id: str,
    worker_profile: str,
) -> NativeCollectionWorkerManifest:
    """Run worker preflight and construct its immutable runtime manifest."""
    from ptcg_rl.rl.stateless_training import (
        resolve_simple_stateless_training_resources,
    )

    try:
        profile = config.native_distributed.worker_profiles[worker_profile]
    except KeyError as exc:
        raise KeyError(
            f"native distributed worker profile is unknown: {worker_profile}"
        ) from exc
    gpu = _validate_cuda_profile(profile)
    resources = resolve_simple_stateless_training_resources(config)
    native_path, _native_abi = resolve_native_training_library()
    native_fingerprint = _file_sha256(native_path)
    fact_contract, fact_binary = _engine_fact_fingerprints(resources.model_config)
    source_identity = resolve_training_source_identity(_REPO_ROOT)
    runtime_fingerprint = _runtime_fingerprint(
        profile,
        worker_profile=worker_profile,
        gpu=gpu,
        native_library_fingerprint=native_fingerprint,
        engine_fact_contract_fingerprint=fact_contract,
        engine_fact_binary_fingerprint=fact_binary,
    )
    identity = NativeRolloutWorkerIdentity(
        worker_id=worker_id,
        session_id=session_id,
        runtime_fingerprint=runtime_fingerprint,
        source_git_commit=source_identity.source_git_commit,
        source_snapshot_fingerprint=(
            source_identity.training_source_fingerprint
        ),
        native_library_fingerprint=native_fingerprint,
        native_abi_version=NATIVE_TRAINING_ABI_VERSION,
        engine_fact_contract_fingerprint=fact_contract,
        feature_schema_fingerprint=(
            resources.policy_identity.fragment_static_contract_fingerprint
        ),
        card_catalog_fingerprint=(resources.catalog_manifest.card_catalog_fingerprint),
        static_features_fingerprint=_file_sha256(
            _resolve_repo_path(resources.model_config.card_encoder.feature_table_path)
        ),
        exact_registry_fingerprint=str(resources.model_config.resolved_registry_sha256),
        scripted_opponents_fingerprint=resources.scripted_manifest_fingerprint,
        historical_opponents_fingerprint=resources.pinned_manifest_fingerprint,
        model_config_fingerprint=(resources.policy_identity.model_config_fingerprint),
        resolved_config_fingerprint=(
            resources.policy_identity.resolved_config_fingerprint
        ),
    )
    return NativeCollectionWorkerManifest(
        identity=identity,
        worker_profile=worker_profile,
        cuda_device_name=str(gpu["name"]),
        cuda_device_uuid=str(gpu["uuid"]),
        cuda_total_memory_bytes=cast(int, gpu["total_memory"]),
        cuda_compute_capability=str(gpu["compute_capability"]),
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda),
        capacity_tiers=tuple(
            NativeCollectionCapacityTier.model_validate(item.model_dump(mode="python"))
            for item in profile.capacity_tiers
        ),
    )


def _validate_cuda_profile(
    profile: StatelessNativeWorkerProfileConfig,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("native collection worker requires CUDA")
    device_index = profile.cuda_device_index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError("native collection worker CUDA device is absent")
    properties = torch.cuda.get_device_properties(device_index)
    if properties.name != profile.expected_cuda_device_name:
        raise RuntimeError(
            "native collection worker GPU differs from its runtime profile: "
            f"expected={profile.expected_cuda_device_name!r}, "
            f"actual={properties.name!r}"
        )
    if int(properties.total_memory) < profile.minimum_cuda_memory_bytes:
        raise RuntimeError("native collection worker GPU memory is below profile")
    if profile.require_bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("native collection worker GPU lacks BF16 support")
    return {
        "name": properties.name,
        "uuid": _cuda_uuid(device_index),
        "total_memory": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def _runtime_fingerprint(
    profile: StatelessNativeWorkerProfileConfig,
    *,
    worker_profile: str,
    gpu: dict[str, object],
    native_library_fingerprint: str,
    engine_fact_contract_fingerprint: str | None,
    engine_fact_binary_fingerprint: str | None,
) -> str:
    payload = {
        "worker_profile": worker_profile,
        "profile": profile.model_dump(mode="json"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "pyzmq": zmq.__version__,
        "gpu": gpu,
        "native_library_fingerprint": native_library_fingerprint,
        "engine_fact_contract_fingerprint": engine_fact_contract_fingerprint,
        "engine_fact_binary_fingerprint": engine_fact_binary_fingerprint,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(_RUNTIME_DOMAIN + encoded).hexdigest()


def _engine_fact_fingerprints(
    model_config: SimpleStatelessModelConfig,
) -> tuple[str | None, str | None]:
    """Resolve portable fact semantics and the host-specific binary identity."""
    sequence = model_config.sequence
    if sequence is None or not sequence.engine_facts.enabled:
        return None, None
    producer = ProspectiveEngineFactProducer(
        sampler=BeliefSampler(config=sequence.engine_facts.sampler),
        config=sequence.engine_facts,
    )
    return producer.contract_fingerprint, producer.fingerprint


def _cuda_uuid(device_index: int) -> str:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise RuntimeError("native collection worker could not resolve GPU UUID")
    return value


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "build_coordinator_worker_contract",
    "build_worker_manifest",
    "source_snapshot_fingerprint",
]
