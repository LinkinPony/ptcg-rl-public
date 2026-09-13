"""Diagnose shared-gradient conflicts between exact stateless deck routes.

This tool never constructs or steps an optimizer. It consumes immutable
checkpoint files and already-published compact fragment parts, then writes only
to its explicit output directory. CUDA execution is bounded by an explicit
per-process allocator fraction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore[import-untyped]
from torch import Tensor

from ptcg_rl.data.kaggle_deck.records import read_deck
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.model.simple_stateless import SimpleStatelessPolicyValueNet
from ptcg_rl.rl.stateless_array_replay import (
    StatelessArrayOptimizerWindow,
    prepare_stateless_array_optimizer_window,
)
from ptcg_rl.rl.stateless_checkpoint import (
    StatelessPolicyIdentity,
    load_stateless_policy_checkpoint,
)
from ptcg_rl.rl.stateless_fragment_io import (
    CompactFragmentPart,
    load_compact_fragment_part,
)
from ptcg_rl.rl.stateless_gradient_diagnostics import (
    DeckGradientObservation,
    accumulate_stateless_deck_gradient,
    build_stateless_gradient_layout,
    cosine_matrix,
    cosine_to_other_mean,
    grouped_gradient_grams,
    sample_stateless_deck_decisions,
    sample_stateless_matched_cells,
    vector_cosine,
)
from ptcg_rl.rl.stateless_ppo import SimpleStatelessPpoConfig

_REPORT_FORMAT = "simple_stateless_deck_gradient_diagnostic_v1"
_COMPONENT_NAMES = ("actor", "critic", "belief")
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactEvidence:
    """Immutable inputs bound to one diagnostic run."""

    checkpoint_version: int
    checkpoint_sha256: str
    checkpoint_model_fingerprint: str
    learner_state_sha256: str
    pair_manifest_sha256: str
    ppo_config_fingerprint: str
    resolved_config_fingerprint: str
    fragment_snapshot_fingerprint: str
    fragment_parts: int
    fragment_bytes: int
    fragment_static_contract_fingerprint: str


@dataclass(frozen=True)
class FragmentEvidence:
    """Loaded fragment snapshot and its aggregate identity."""

    parts: tuple[CompactFragmentPart, ...]
    fingerprint: str
    total_bytes: int
    behavior_versions: tuple[int, ...]
    behavior_fingerprints: tuple[str, ...]


def main() -> None:
    """Run a bounded read-only diagnostic and publish compact evidence."""
    arguments = _arguments()
    started = time.perf_counter()
    torch.set_num_threads(arguments.torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device(
        "cuda:0" if arguments.device == "cuda" else arguments.device
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(
            arguments.cuda_memory_fraction,
            device=device,
        )
        torch.cuda.manual_seed_all(arguments.seed)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_stateless_policy_checkpoint(arguments.checkpoint)
    ppo_config, learner_sha, pair_sha = _load_bound_ppo_config(
        learner_state=arguments.learner_state,
        pair_manifest=arguments.pair_manifest,
        checkpoint_sha=loaded.artifact.policy_sha256,
        checkpoint_model_fingerprint=loaded.artifact.policy_model_fingerprint,
        checkpoint_version=loaded.artifact.version,
        checkpoint_identity=loaded.identity,
    )
    fragments = _load_fragment_snapshot(arguments.fragment_root)
    window = prepare_stateless_array_optimizer_window(
        fragments.parts,
        current_policy_version=loaded.artifact.version,
        maximum_version_age=ppo_config.maximum_version_age,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        normalize_epsilon=ppo_config.normalize_epsilon,
    )
    _validate_snapshot_binding(
        window=window,
        fragments=fragments,
        checkpoint_version=loaded.artifact.version,
        checkpoint_model_fingerprint=(
            loaded.artifact.policy_model_fingerprint
        ),
        checkpoint_identity=loaded.identity,
    )
    labels, profile_sha = _deck_labels(
        arguments.deck_registry_profile,
        expected_digests=loaded.identity.active_exact_deck_digests,
    )
    model = SimpleStatelessPolicyValueNet(
        loaded.model_config_value,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(loaded.model_state, strict=True)
    model.float().to(device).train()
    checkpoint_version = loaded.artifact.version
    checkpoint_sha = loaded.artifact.policy_sha256
    model_fingerprint = loaded.artifact.policy_model_fingerprint
    resolved_config_fingerprint = loaded.identity.resolved_config_fingerprint
    static_contract = loaded.identity.fragment_static_contract_fingerprint
    del loaded
    gc.collect()

    layout = build_stateless_gradient_layout(model)
    matched_plan = None
    if arguments.sampling_mode == "matched":
        matched_plan = sample_stateless_matched_cells(
            window,
            replicates=arguments.replicates,
            seed=arguments.seed,
        )
        samples = matched_plan.samples
        sampled_decisions_per_deck = matched_plan.decisions_per_deck
    else:
        samples = sample_stateless_deck_decisions(
            window,
            decisions_per_deck=arguments.decisions_per_deck,
            replicates=arguments.replicates,
            seed=arguments.seed,
        )
        sampled_decisions_per_deck = arguments.decisions_per_deck
    deck_digests = tuple(sorted(samples))
    evidence = ArtifactEvidence(
        checkpoint_version=checkpoint_version,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_model_fingerprint=model_fingerprint,
        learner_state_sha256=learner_sha,
        pair_manifest_sha256=pair_sha,
        ppo_config_fingerprint=ppo_config.fingerprint,
        resolved_config_fingerprint=resolved_config_fingerprint,
        fragment_snapshot_fingerprint=fragments.fingerprint,
        fragment_parts=len(fragments.parts),
        fragment_bytes=fragments.total_bytes,
        fragment_static_contract_fingerprint=static_contract,
    )
    pair_rows: list[dict[str, Any]] = []
    deck_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    print(
        "validated "
        f"v{checkpoint_version} parts={len(fragments.parts)} "
        f"fragments={window.fragments_retained} "
        f"decisions={window.decision_count} decks={len(deck_digests)}",
        flush=True,
    )
    for replicate in range(arguments.replicates):
        total_gradients = torch.empty(
            (len(deck_digests), layout.shared_numel),
            dtype=torch.float32,
        )
        component_gradients = {
            name: torch.empty(
                (len(deck_digests), layout.late_trunk_numel),
                dtype=torch.float32,
            )
            for name in _COMPONENT_NAMES
        }
        observations: list[DeckGradientObservation] = []
        replicate_sample_rows: list[dict[str, Any]] = []
        for deck_index, digest in enumerate(deck_digests):
            indices = samples[digest][replicate]
            observation = accumulate_stateless_deck_gradient(
                model=model,
                window=window,
                indices=tuple(int(value) for value in indices),
                config=ppo_config,
                layout=layout,
                microbatch_decisions=arguments.microbatch_decisions,
                total_destination=total_gradients[deck_index],
                component_destinations={
                    name: values[deck_index]
                    for name, values in component_gradients.items()
                },
            )
            observations.append(observation)
            replicate_sample_rows.extend(
                _sample_manifest_rows(
                    window=window,
                    indices=indices,
                    digest=digest,
                    label=labels[digest],
                    replicate=replicate,
                )
            )
            print(
                f"replicate={replicate + 1}/{arguments.replicates} "
                f"deck={deck_index + 1:02d}/{len(deck_digests)} "
                f"{labels[digest]} loss={observation.loss:.6g} "
                f"tokens={observation.active_tokens}",
                flush=True,
            )

        total_grams = grouped_gradient_grams(
            total_gradients,
            layout.shared_group_slices,
        )
        component_grams = {
            name: grouped_gradient_grams(
                values,
                {"late_trunk": slice(0, layout.late_trunk_numel)},
            )["late_trunk"]
            for name, values in component_gradients.items()
        }
        pair_rows.extend(
            _pair_rows(
                deck_digests=deck_digests,
                labels=labels,
                replicate=replicate,
                total_grams=total_grams,
                component_grams=component_grams,
            )
        )
        deck_rows.extend(
            _deck_rows(
                window=window,
                deck_digests=deck_digests,
                labels=labels,
                replicate=replicate,
                observations=observations,
                sample_rows=replicate_sample_rows,
                total_grams=total_grams,
                component_grams=component_grams,
                component_gradients=component_gradients,
            )
        )
        sample_rows.extend(replicate_sample_rows)
        del total_gradients, component_gradients, total_grams, component_grams
        gc.collect()

    pair_aggregate = _aggregate_pair_rows(pair_rows)
    deck_aggregate = _aggregate_deck_rows(deck_rows)
    group_summary = _group_summary(pair_aggregate)
    elapsed = time.perf_counter() - started
    summary = {
        "format": _REPORT_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "execution": {
            "device": str(device),
            "torch_threads": arguments.torch_threads,
            "cuda_memory_fraction": (
                arguments.cuda_memory_fraction
                if device.type == "cuda"
                else None
            ),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "elapsed_seconds": elapsed,
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "gradient_clipping_applied": False,
        },
        "artifacts": asdict(evidence),
        "deck_registry_profile": {
            "path": str(arguments.deck_registry_profile.resolve()),
            "sha256": profile_sha,
        },
        "window": {
            "fragments_seen": window.fragments_seen,
            "fragments_retained": window.fragments_retained,
            "fragments_stale": window.fragments_stale,
            "decisions": window.decision_count,
            "tokens": window.token_count,
            "advantage_mean": window.advantage_mean,
            "advantage_std": window.advantage_std,
            "behavior_versions": list(fragments.behavior_versions),
            "behavior_fingerprints": list(fragments.behavior_fingerprints),
        },
        "sampling": {
            "mode": arguments.sampling_mode,
            "seed": arguments.seed,
            "replicates": arguments.replicates,
            "decisions_per_deck": sampled_decisions_per_deck,
            "microbatch_decisions": arguments.microbatch_decisions,
            "fragment_disjoint_across_replicates": True,
            "global_window_advantage_normalization": True,
            "matched_cell_fingerprint": (
                matched_plan.fingerprint if matched_plan is not None else None
            ),
            "matched_cell_quotas": (
                [
                    {
                        "seat": seat,
                        "opponent_deck_digest": opponent,
                        "opponent_artifact_fingerprint": artifact,
                        "decisions_per_replicate": quota,
                    }
                    for seat, opponent, artifact, quota in matched_plan.cell_quotas
                ]
                if matched_plan is not None
                else None
            ),
        },
        "parameters": {
            "shared": layout.shared_numel,
            "private": layout.private_numel,
            "late_trunk_probe": layout.late_trunk_numel,
            "total": layout.shared_numel + layout.private_numel,
        },
        "deck_summary": deck_aggregate,
        "group_summary": group_summary,
        "top_conflicts": _top_pairs(pair_aggregate, reverse=False),
        "top_alignments": _top_pairs(pair_aggregate, reverse=True),
        "limitations": (
            "Operational rollout mixtures can confound deck identity with "
            "opponent and seat distributions.",
            (
                "The diagnostic uses the learner's H200 BF16 autocast path, "
                "but floating-point reduction order is not bitwise replayed."
                if device.type == "cuda"
                else "CPU FP32 replay is not bit-identical to H200 BF16 learner math."
            ),
            "Raw gradient cosine omits AdamW moment preconditioning, weight "
            "decay, aggregate clipping, and sequential logical-batch updates.",
            "Two fragment-disjoint samples measure local stability, not a "
            "population confidence interval.",
        ),
    }
    _write_parquet(output_dir / "pairwise.parquet", pair_rows)
    _write_parquet(output_dir / "pairwise_aggregate.parquet", pair_aggregate)
    _write_parquet(output_dir / "deck_summary.parquet", deck_rows)
    _write_parquet(output_dir / "sample_manifest.parquet", sample_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_text(
        output_dir / "report.md",
        _markdown_report(
            summary=summary,
            pair_aggregate=pair_aggregate,
            deck_aggregate=deck_aggregate,
        ),
    )
    print(f"wrote diagnostic artifacts to {output_dir}", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--learner-state", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--deck-registry-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decisions-per-deck", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--microbatch-decisions", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.04)
    parser.add_argument(
        "--sampling-mode",
        choices=("operational", "matched"),
        default="operational",
    )
    arguments = parser.parse_args()
    for name in (
        "decisions_per_deck",
        "replicates",
        "microbatch_decisions",
        "torch_threads",
    ):
        if int(getattr(arguments, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < arguments.cuda_memory_fraction <= 0.25:
        parser.error("--cuda-memory-fraction must be in (0, 0.25]")
    return arguments


def _load_bound_ppo_config(
    *,
    learner_state: Path,
    pair_manifest: Path,
    checkpoint_sha: str,
    checkpoint_model_fingerprint: str,
    checkpoint_version: int,
    checkpoint_identity: StatelessPolicyIdentity,
) -> tuple[SimpleStatelessPpoConfig, str, str]:
    manifest_bytes = pair_manifest.read_bytes()
    manifest = _mapping(json.loads(manifest_bytes), "pair manifest")
    if (
        manifest.get("format") != "exact_policy_learner_pair_v1"
        or int(manifest["version"]) != checkpoint_version
    ):
        raise ValueError("pair manifest version differs from checkpoint")
    policy_record = _mapping(manifest["policy"], "policy record")
    learner_record = _mapping(manifest["training_state"], "learner record")
    if (
        str(policy_record["sha256"]) != checkpoint_sha
        or str(policy_record["model_fingerprint"])
        != checkpoint_model_fingerprint
        or str(learner_record["policy_sha256"]) != checkpoint_sha
    ):
        raise ValueError("pair manifest is not bound to the checkpoint")
    if learner_state.stat().st_size != int(learner_record["size_bytes"]):
        raise ValueError("learner-state size differs from pair manifest")
    learner_sha = _file_sha256(learner_state)
    if learner_sha != str(learner_record["sha256"]):
        raise ValueError("learner-state SHA-256 differs from pair manifest")
    payload = _mapping(
        torch.load(learner_state, map_location="cpu", weights_only=False),
        "learner payload",
    )
    if (
        payload.get("format") != "simple_stateless_learner_state_v1"
        or int(payload["version"]) != checkpoint_version
        or str(payload["policy_sha256"]) != checkpoint_sha
        or str(payload["policy_model_fingerprint"])
        != checkpoint_model_fingerprint
        or StatelessPolicyIdentity.model_validate(payload["identity"])
        != checkpoint_identity
    ):
        raise ValueError("learner-state payload is not bound to the checkpoint")
    config = SimpleStatelessPpoConfig.model_validate(payload["ppo_config"])
    del payload
    gc.collect()
    return config, learner_sha, hashlib.sha256(manifest_bytes).hexdigest()


def _load_fragment_snapshot(root: Path) -> FragmentEvidence:
    paths = tuple(sorted(root.rglob("*.npz")))
    if not paths:
        raise ValueError("fragment snapshot contains no NPZ parts")
    aggregate = hashlib.sha256()
    parts: list[CompactFragmentPart] = []
    total_bytes = 0
    behavior_versions: set[int] = set()
    behavior_fingerprints: set[str] = set()
    for path in paths:
        size = path.stat().st_size
        sha = _file_sha256(path)
        relative = path.relative_to(root)
        aggregate.update(str(relative).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii"))
        aggregate.update(b"\n")
        part = load_compact_fragment_part(path)
        parts.append(part)
        total_bytes += size
        behavior_versions.update(
            int(value) for value in part.arrays["behavior_policy_versions"]
        )
        behavior_fingerprints.update(
            str(value) for value in part.arrays["behavior_policy_fingerprints"]
        )
    return FragmentEvidence(
        parts=tuple(parts),
        fingerprint=aggregate.hexdigest(),
        total_bytes=total_bytes,
        behavior_versions=tuple(sorted(behavior_versions)),
        behavior_fingerprints=tuple(sorted(behavior_fingerprints)),
    )


def _validate_snapshot_binding(
    *,
    window: StatelessArrayOptimizerWindow,
    fragments: FragmentEvidence,
    checkpoint_version: int,
    checkpoint_model_fingerprint: str,
    checkpoint_identity: StatelessPolicyIdentity,
) -> None:
    if fragments.behavior_versions != (checkpoint_version,):
        raise ValueError("diagnostic snapshot is not exactly on-policy")
    if fragments.behavior_fingerprints != (checkpoint_model_fingerprint,):
        raise ValueError("fragment behavior fingerprint differs from checkpoint")
    if np.any(window.behavior_version_ages != 0):
        raise ValueError("diagnostic window contains stale behavior")
    observed_decks = tuple(
        sorted({str(value) for value in window.deck_digests})
    )
    if observed_decks != checkpoint_identity.active_exact_deck_digests:
        raise ValueError("diagnostic window does not cover the exact active roster")
    if (
        window.static_contract_fingerprint
        != checkpoint_identity.fragment_static_contract_fingerprint
    ):
        raise ValueError("fragment snapshot contract differs from checkpoint")


def _deck_labels(
    profile: Path,
    *,
    expected_digests: Sequence[str],
) -> tuple[dict[str, str], str]:
    payload = yaml.safe_load(profile.read_text(encoding="utf-8"))
    root = _mapping(payload, "deck registry profile")
    registry = _mapping(root["private_deck_registry"], "private deck registry")
    raw_decks = registry["decks"]
    if not isinstance(raw_decks, list):
        raise TypeError("private deck registry decks must be a list")
    labels: dict[str, str] = {}
    for raw_deck in raw_decks:
        deck = _mapping(raw_deck, "private deck")
        path = Path(str(deck["path"]))
        resolved = path if path.is_absolute() else _REPO_ROOT / path
        digest = canonicalize_deck(read_deck(resolved)).deck_digest
        labels[digest] = str(deck["label"])
    if set(labels) != set(expected_digests):
        raise ValueError("deck registry profile differs from checkpoint routes")
    return labels, _file_sha256(profile)


def _sample_manifest_rows(
    *,
    window: StatelessArrayOptimizerWindow,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    digest: str,
    label: str,
    replicate: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_index in indices:
        index = int(raw_index)
        fragment_index = int(window.decision_fragment_indices[index])
        part_index = int(window.retained_fragment_part_indices[fragment_index])
        part_row = int(window.retained_fragment_rows[fragment_index])
        source = window.source_arrays[part_index]
        rows.append(
            {
                "replicate": replicate,
                "deck_digest": digest,
                "deck_label": label,
                "decision_index": index,
                "fragment_id": str(window.fragment_ids[fragment_index]),
                "game_id": str(source["game_ids"][part_row]),
                "seat": int(source["seats"][part_row]),
                "opponent_deck_digest": str(
                    source["opponent_deck_digests"][part_row]
                ),
                "opponent_artifact_fingerprint": str(
                    source["opponent_artifact_fingerprints"][part_row]
                ),
                "normalized_advantage": float(
                    window.normalized_advantages[index]
                ),
                "raw_advantage": float(window.raw_advantages[index]),
                "belief_target_valid": bool(
                    window.belief_target_valid[index]
                ),
            }
        )
    return rows


def _pair_rows(
    *,
    deck_digests: Sequence[str],
    labels: Mapping[str, str],
    replicate: int,
    total_grams: Mapping[str, Tensor],
    component_grams: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices: list[tuple[str, str, Tensor]] = [
        ("total", group, gram) for group, gram in total_grams.items()
    ]
    matrices.extend(
        (objective, "late_trunk", gram)
        for objective, gram in component_grams.items()
    )
    for objective, group, gram in matrices:
        cosine = cosine_matrix(gram)
        norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
        for left in range(len(deck_digests)):
            for right in range(left + 1, len(deck_digests)):
                rows.append(
                    {
                        "replicate": replicate,
                        "objective": objective,
                        "parameter_group": group,
                        "deck_a_digest": deck_digests[left],
                        "deck_a_label": labels[deck_digests[left]],
                        "deck_b_digest": deck_digests[right],
                        "deck_b_label": labels[deck_digests[right]],
                        "dot": float(gram[left, right]),
                        "cosine": float(cosine[left, right]),
                        "norm_a": float(norms[left]),
                        "norm_b": float(norms[right]),
                        "negative_dot": bool(gram[left, right] < 0.0),
                    }
                )
    return rows


def _deck_rows(
    *,
    window: StatelessArrayOptimizerWindow,
    deck_digests: Sequence[str],
    labels: Mapping[str, str],
    replicate: int,
    observations: Sequence[DeckGradientObservation],
    sample_rows: Sequence[Mapping[str, Any]],
    total_grams: Mapping[str, Tensor],
    component_grams: Mapping[str, Tensor],
    component_gradients: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    samples_by_deck: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample_row in sample_rows:
        samples_by_deck[str(sample_row["deck_digest"])].append(sample_row)
    deck_values = np.asarray(window.deck_digests, dtype=np.str_)
    total_cosine = cosine_matrix(total_grams["all_shared"])
    alignments = cosine_to_other_mean(total_grams["all_shared"])
    component_cosines = {
        name: cosine_matrix(gram) for name, gram in component_grams.items()
    }
    rows: list[dict[str, Any]] = []
    for index, digest in enumerate(deck_digests):
        full_indices = np.flatnonzero(deck_values == digest)
        selected = samples_by_deck[digest]
        peer_mask = [peer for peer in range(len(deck_digests)) if peer != index]
        group_norms = {
            group: math.sqrt(max(float(gram[index, index]), 0.0))
            for group, gram in total_grams.items()
        }
        deck_row: dict[str, Any] = {
            "replicate": replicate,
            "deck_digest": digest,
            "deck_label": labels[digest],
            "window_decisions": int(full_indices.size),
            "window_fragments": int(
                np.unique(window.decision_fragment_indices[full_indices]).size
            ),
            "sample_decisions": len(selected),
            "sample_seat0": sum(int(value["seat"]) == 0 for value in selected),
            "sample_seat1": sum(int(value["seat"]) == 1 for value in selected),
            "sample_opponent_decks": len(
                {str(value["opponent_deck_digest"]) for value in selected}
            ),
            "sample_opponent_artifacts": len(
                {
                    str(value["opponent_artifact_fingerprint"])
                    for value in selected
                }
            ),
            "sample_advantage_mean": statistics.fmean(
                float(value["normalized_advantage"]) for value in selected
            ),
            "sample_advantage_std": statistics.pstdev(
                float(value["normalized_advantage"]) for value in selected
            ),
            "window_advantage_mean": float(
                np.mean(window.normalized_advantages[full_indices])
            ),
            "window_advantage_std": float(
                np.std(window.normalized_advantages[full_indices])
            ),
            "total_gradient_norm": group_norms["all_shared"],
            "total_mean_peer_cosine": statistics.fmean(
                float(total_cosine[index, peer]) for peer in peer_mask
            ),
            "total_negative_peer_count": sum(
                float(total_cosine[index, peer]) < 0.0 for peer in peer_mask
            ),
            "total_cosine_to_other_mean": alignments[index],
            "actor_mean_peer_cosine": statistics.fmean(
                float(component_cosines["actor"][index, peer])
                for peer in peer_mask
            ),
            "critic_mean_peer_cosine": statistics.fmean(
                float(component_cosines["critic"][index, peer])
                for peer in peer_mask
            ),
            "belief_mean_peer_cosine": statistics.fmean(
                float(component_cosines["belief"][index, peer])
                for peer in peer_mask
            ),
            "actor_gradient_norm_late": math.sqrt(
                max(float(component_grams["actor"][index, index]), 0.0)
            ),
            "critic_gradient_norm_late": math.sqrt(
                max(float(component_grams["critic"][index, index]), 0.0)
            ),
            "belief_gradient_norm_late": math.sqrt(
                max(float(component_grams["belief"][index, index]), 0.0)
            ),
            "actor_critic_cosine_late": vector_cosine(
                component_gradients["actor"][index],
                component_gradients["critic"][index],
            ),
            "group_norms_json": json.dumps(group_norms, sort_keys=True),
        }
        deck_row.update(asdict(observations[index]))
        rows.append(deck_row)
    return rows


def _aggregate_pair_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[
        tuple[str, str, str, str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["objective"]),
            str(row["parameter_group"]),
            str(row["deck_a_digest"]),
            str(row["deck_a_label"]),
            str(row["deck_b_digest"]),
            str(row["deck_b_label"]),
        )
        grouped[key].append(row)
    aggregated: list[dict[str, Any]] = []
    for key, values in grouped.items():
        cosines = [float(value["cosine"]) for value in values]
        dots = [float(value["dot"]) for value in values]
        aggregated.append(
            {
                "objective": key[0],
                "parameter_group": key[1],
                "deck_a_digest": key[2],
                "deck_a_label": key[3],
                "deck_b_digest": key[4],
                "deck_b_label": key[5],
                "replicates": len(values),
                "mean_cosine": statistics.fmean(cosines),
                "min_cosine": min(cosines),
                "max_cosine": max(cosines),
                "mean_dot": statistics.fmean(dots),
                "negative_fraction": sum(value < 0.0 for value in dots)
                / len(dots),
                "stable_negative": max(dots) < 0.0,
                "stable_positive": min(dots) > 0.0,
            }
        )
    return aggregated


def _aggregate_deck_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["deck_digest"])].append(row)
    passthrough = (
        "deck_digest",
        "deck_label",
        "window_decisions",
        "window_fragments",
    )
    numeric = (
        "sample_advantage_mean",
        "total_gradient_norm",
        "total_mean_peer_cosine",
        "total_negative_peer_count",
        "total_cosine_to_other_mean",
        "actor_mean_peer_cosine",
        "critic_mean_peer_cosine",
        "belief_mean_peer_cosine",
        "actor_gradient_norm_late",
        "critic_gradient_norm_late",
        "belief_gradient_norm_late",
        "actor_critic_cosine_late",
        "private_policy_norm",
        "private_value_norm",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "belief_loss",
        "ratio_mean",
        "approximate_kl",
        "clip_fraction",
    )
    result: list[dict[str, Any]] = []
    for digest in sorted(grouped):
        values = grouped[digest]
        row = {name: values[0][name] for name in passthrough}
        row.update(
            {
                name: statistics.fmean(float(value[name]) for value in values)
                for name in numeric
            }
        )
        result.append(row)
    return result


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["objective"]), str(row["parameter_group"]))].append(row)
    return [
        {
            "objective": objective,
            "parameter_group": group,
            "pairs": len(values),
            "mean_pair_cosine": statistics.fmean(
                float(value["mean_cosine"]) for value in values
            ),
            "stable_negative_pairs": sum(
                bool(value["stable_negative"]) for value in values
            ),
            "any_negative_pairs": sum(
                float(value["negative_fraction"]) > 0.0 for value in values
            ),
        }
        for (objective, group), values in sorted(grouped.items())
    ]


def _top_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    reverse: bool,
    limit: int = 12,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["objective"] == "total"
        and row["parameter_group"] == "all_shared"
    ]
    ordered = sorted(
        candidates,
        key=lambda row: float(row["mean_cosine"]),
        reverse=reverse,
    )
    return [dict(row) for row in ordered[:limit]]


def _markdown_report(
    *,
    summary: Mapping[str, Any],
    pair_aggregate: Sequence[Mapping[str, Any]],
    deck_aggregate: Sequence[Mapping[str, Any]],
) -> str:
    artifacts = _mapping(summary["artifacts"], "summary artifacts")
    window = _mapping(summary["window"], "summary window")
    sampling = _mapping(summary["sampling"], "summary sampling")
    execution = _mapping(summary["execution"], "summary execution")
    lines = [
        "# Simple-stateless 22-deck gradient diagnostic",
        "",
        "## Evidence and method",
        "",
        f"- Checkpoint: v{artifacts['checkpoint_version']} / "
        f"`{str(artifacts['checkpoint_sha256'])[:16]}…`",
        f"- Snapshot: `{str(artifacts['fragment_snapshot_fingerprint'])[:16]}…`; "
        f"{window['fragments_retained']} fragments, {window['decisions']} decisions",
        f"- Sampling: {sampling['replicates']} fragment-disjoint replicates × "
        f"{sampling['decisions_per_deck']} decisions per deck "
        f"({sampling['mode']})",
        "- Objective: exact current deck-macro PPO loss at the frozen checkpoint; "
        "each sampled deck is reweighted to unit task mass.",
        "- Conflict scope: shared parameters only. Route-private residuals are "
        "reported as norms and excluded from pairwise cosine.",
        "",
        "## Per-deck summary",
        "",
        "| Deck | window decisions | shared norm | mean peer cos | negative peers | "
        "cos to other mean | actor↔critic late |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_decks = sorted(
        deck_aggregate,
        key=lambda row: float(row["total_cosine_to_other_mean"]),
    )
    for row in ordered_decks:
        lines.append(
            f"| {row['deck_label']} | {row['window_decisions']} | "
            f"{float(row['total_gradient_norm']):.4g} | "
            f"{float(row['total_mean_peer_cosine']):+.3f} | "
            f"{float(row['total_negative_peer_count']):.1f} | "
            f"{float(row['total_cosine_to_other_mean']):+.3f} | "
            f"{float(row['actor_critic_cosine_late']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest stable shared conflicts",
            "",
            "| Deck A | Deck B | mean cosine | replicate range |",
            "|---|---|---:|---:|",
        ]
    )
    conflicts = [
        row
        for row in pair_aggregate
        if row["objective"] == "total"
        and row["parameter_group"] == "all_shared"
        and bool(row["stable_negative"])
    ]
    for row in sorted(conflicts, key=lambda value: float(value["mean_cosine"]))[:20]:
        lines.append(
            f"| {row['deck_a_label']} | {row['deck_b_label']} | "
            f"{float(row['mean_cosine']):+.3f} | "
            f"[{float(row['min_cosine']):+.3f}, "
            f"{float(row['max_cosine']):+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A negative dot product has the local first-order meaning that one "
            "deck's descent direction raises the other deck's sampled loss.",
            (
                "- Opponent artifact, opponent deck, and seat counts are matched "
                "across decks in this report."
                if sampling["mode"] == "matched"
                else "- This is an operational snapshot. Different opponent "
                "mixtures or seat support can appear as deck conflict."
            ),
            (
                "- The H200 BF16 learner autocast path was used; AdamW moment "
                "preconditioning is omitted."
                if str(execution["device"]).startswith("cuda")
                else "- CPU FP32 gradients omit H200 BF16 rounding and AdamW "
                "moment preconditioning."
            ),
            "- No optimizer, clipping, or parameter mutation ran.",
            "",
        ]
    )
    return "\n".join(lines)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
