"""Orchestrate immutable deck-conditioned policy verification."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks import (
    DeckBatch,
    canonicalize_deck,
    private_deck_profile,
    private_registry_fingerprint,
)
from ptcg_rl.evaluation.deck_conditioning_audit_models import (
    DeckConditioningAuditArchitecture,
    DeckConditioningAuditConfig,
)
from ptcg_rl.evaluation.deck_conditioning_audit_probes import (
    benchmark_model,
    migration_audit,
    mixed_eager_audit,
    model_inputs,
    parameter_summary,
    synchronize,
)
from ptcg_rl.evaluation.deck_conditioning_audit_runtime import (
    export_and_benchmark_runtime,
)
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic
from ptcg_rl.model import (
    AgentNetworkConfig,
    DeckConditioningConfig,
    build_agent_policy_value_net,
    load_agent_policy_value_state_dict,
)


def run_deck_conditioning_audit(
    config: DeckConditioningAuditConfig,
) -> dict[str, Any]:
    """Run the preregistered compatibility and local performance audit."""
    source_path = _required_file(config.source_checkpoint, "source checkpoint")
    source_sha256 = file_sha256(source_path)
    if source_sha256 != config.source_checkpoint_sha256:
        raise ValueError(
            "source checkpoint SHA256 mismatch: "
            f"expected {config.source_checkpoint_sha256}, got {source_sha256}"
        )
    output_dir = records.repo_path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable audit output already exists: {output_dir}")
    device = _resolve_device(config.device, require_cuda=config.require_cuda)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    source_checkpoint = torch.load(source_path, map_location="cpu")
    source_config = _checkpoint_model_config(source_checkpoint)
    if source_config.deck_conditioning is not None:
        raise ValueError("migration audit source must be a legacy checkpoint")
    target_config, deck_rows, registry_sha256 = _target_config(
        source_config,
        config,
    )
    if registry_sha256 != config.resolved_registry_sha256:
        raise ValueError(
            "resolved private registry SHA256 mismatch: "
            f"expected {config.resolved_registry_sha256}, got {registry_sha256}"
        )
    output_dir.mkdir(parents=True)
    states, options = model_inputs(batch_size=len(deck_rows), device=device)
    decks = DeckBatch.from_card_ids(deck_rows, device=device)
    actions = tuple((0, 1) for _ in deck_rows)

    source_state = _checkpoint_state_dict(source_checkpoint)
    legacy = build_agent_policy_value_net(source_config).to(device).eval()
    load_agent_policy_value_state_dict(
        legacy,
        source_state,
        source_config=source_config,
    )
    candidate = build_agent_policy_value_net(target_config).to(device).eval()
    load_agent_policy_value_state_dict(
        candidate,
        source_state,
        source_config=source_config,
    )
    synchronize(device)
    migration = migration_audit(
        legacy,
        candidate,
        states=states,
        options=options,
        decks=decks,
        actions=actions,
        atol=config.migration_atol,
        rtol=config.migration_rtol,
    )
    eager = mixed_eager_audit(
        candidate,
        states=states,
        options=options,
        decks=decks,
        atol=config.mixed_eager_atol,
        rtol=config.mixed_eager_rtol,
    )
    performance = {
        "legacy": benchmark_model(
            legacy,
            states=states,
            options=options,
            decks=None,
            device=device,
            warmup=config.benchmark_warmup_iterations,
            iterations=config.benchmark_iterations,
        ),
        "conditioned_mixed": benchmark_model(
            candidate,
            states=states,
            options=options,
            decks=decks,
            device=device,
            warmup=config.benchmark_warmup_iterations,
            iterations=config.benchmark_iterations,
        ),
    }
    parameters = parameter_summary(legacy, candidate)

    raw_target = output_dir / "migrated_v1_full_fp32.pt"
    candidate.cpu()
    legacy.cpu()
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in candidate.state_dict().items()
            },
            "model_config": target_config.model_dump(mode="json"),
            "migration_source": {
                "path": records.display_path(source_path),
                "sha256": file_sha256(source_path),
            },
        },
        raw_target,
    )
    del legacy, candidate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    release_deck = _required_file(
        config.selected_release_deck_path,
        "selected release deck",
    )
    runtime_artifact = export_and_benchmark_runtime(
        raw_target,
        release_deck=release_deck,
        output_dir=output_dir,
        iterations=config.runtime_iterations,
    )
    result = _result(
        config=config,
        source_path=source_path,
        source_config=source_config,
        target_config=target_config,
        registry_sha256=registry_sha256,
        device=device,
        migration=migration,
        eager=eager,
        parameters=parameters,
        performance=performance,
        runtime_artifact=runtime_artifact,
    )
    write_identity_atomic(output_dir / "summary.json", result)
    (output_dir / "report.md").write_text(
        _markdown_report(result),
        encoding="utf-8",
    )
    return result


def _target_config(
    source: AgentNetworkConfig,
    config: DeckConditioningAuditConfig,
) -> tuple[AgentNetworkConfig, tuple[tuple[int, ...], ...], str]:
    profiles = tuple(
        sorted(
            (
                private_deck_profile(
                    canonicalize_deck(
                        records.read_deck(_required_file(path, "private deck"))
                    )
                )
                for path in config.private_deck_paths
            ),
            key=lambda profile: profile.module_key,
        )
    )
    registry_sha256 = private_registry_fingerprint(profiles)
    architecture = config.architecture
    conditioning = DeckConditioningConfig(
        resolved_registry_sha256=registry_sha256,
        encoder_hidden_dim=architecture.encoder_hidden_dim,
        adapter_layer_indices=architecture.adapter_layer_indices,
        adapter_bottleneck_dim=architecture.adapter_bottleneck_dim,
        policy_bottleneck_dim=architecture.policy_bottleneck_dim,
        value_bottleneck_dim=architecture.value_bottleneck_dim,
        adapter_dropout=architecture.adapter_dropout,
        private_profiles=profiles,
    )
    target = source.model_copy(update={"deck_conditioning": conditioning})
    generic = canonicalize_deck(
        records.read_deck(_required_file(config.generic_deck_path, "generic deck"))
    )
    if generic.signature in conditioning.profile_by_signature:
        raise ValueError("generic audit deck is registered as a private profile")
    rows = tuple(profile.canonical_card_ids for profile in profiles) + (
        generic.card_ids,
    )
    return target, rows, registry_sha256


def _result(
    *,
    config: DeckConditioningAuditConfig,
    source_path: Path,
    source_config: AgentNetworkConfig,
    target_config: AgentNetworkConfig,
    registry_sha256: str,
    device: torch.device,
    migration: Mapping[str, Any],
    eager: Mapping[str, Any],
    parameters: Mapping[str, Any],
    performance: Mapping[str, Any],
    runtime_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "DECK-CONDITIONING-VERIFICATION-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_checkpoint": records.display_path(source_path),
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_model_config": source_config.model_dump(mode="json"),
        "target_model_config": target_config.model_dump(mode="json"),
        "registry_sha256": registry_sha256,
        "private_profile_count": len(config.private_deck_paths),
        "device": _device_summary(device),
        "git": _git_identity(),
        "tolerances": {
            "migration_atol": config.migration_atol,
            "migration_rtol": config.migration_rtol,
            "mixed_eager_atol": config.mixed_eager_atol,
            "mixed_eager_rtol": config.mixed_eager_rtol,
        },
        "migration": dict(migration),
        "mixed_eager": dict(eager),
        "parameters": dict(parameters),
        "performance": dict(performance),
        "runtime_artifact": dict(runtime_artifact),
        "compile_or_graph_required": False,
        "compile_or_graph_reason": (
            "formal profiles keep compile_model and compile_evaluate_actions false; "
            "graph parity remains covered only for the optional graph path"
        ),
    }


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    raise ValueError("source checkpoint has no AgentNetworkConfig")


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return cast(Mapping[str, Tensor], value)
    return cast(Mapping[str, Tensor], checkpoint)


def _resolve_device(raw: str, *, require_cuda: bool) -> torch.device:
    normalized = raw.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("this audit preregisters an eager CUDA requirement")
    return device


def _device_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"type": "cpu", "cuda_available": torch.cuda.is_available()}
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _required_file(path: Path, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _git_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def _markdown_report(result: Mapping[str, Any]) -> str:
    migration = cast(Mapping[str, Any], result["migration"])
    eager = cast(Mapping[str, Any], result["mixed_eager"])
    performance = cast(Mapping[str, Any], result["performance"])
    legacy = cast(Mapping[str, Any], performance["legacy"])
    conditioned = cast(Mapping[str, Any], performance["conditioned_mixed"])
    runtime_artifact = cast(Mapping[str, Any], result["runtime_artifact"])
    cpu = cast(Mapping[str, Any], runtime_artifact["cpu_benchmark"])
    device = cast(Mapping[str, Any], result["device"])
    return "\n".join(
        (
            "# Deck-conditioned policy verification",
            "",
            f"- Source: `{result['source_checkpoint']}`",
            f"- Source SHA256: `{result['source_checkpoint_sha256']}`",
            f"- Registry SHA256: `{result['registry_sha256']}`",
            f"- Device: `{device.get('name', 'cpu')}`",
            "",
            "## Correctness",
            "",
            f"- Migration max abs: `{migration['max_abs']}`; mask/greedy "
            f"disagreements: `{migration['mask_disagreements']}/"
            f"{migration['greedy_disagreements']}`.",
            f"- Mixed eager max abs: `{eager['max_abs']}`; mask/greedy "
            f"disagreements: `{eager['mask_disagreements']}/"
            f"{eager['greedy_disagreements']}`.",
            "",
            "## Performance diagnostics",
            "",
            f"- Legacy CUDA mean: `{legacy['mean_ms']:.3f} ms`.",
            f"- Conditioned mixed CUDA mean: `{conditioned['mean_ms']:.3f} ms`.",
            f"- CPU load/prewarm: `{cpu['load_seconds']:.3f}s / "
            f"{cpu['prewarm_seconds']:.3f}s`.",
            f"- Final runtime checkpoint SHA256: `{runtime_artifact['runtime_sha256']}`.",
            "",
            "Throughput and conversion differences are diagnostics, not promotion gates.",
            "",
        )
    )


__all__ = [
    "DeckConditioningAuditArchitecture",
    "DeckConditioningAuditConfig",
    "run_deck_conditioning_audit",
]
