"""Deterministic routed retention audit for full-model supervised policies."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.evaluation.simple_stateless_retention_sample import (
    RetentionSampleRef as _SampleRef,
)
from ptcg_rl.evaluation.simple_stateless_retention_sample import (
    sample_retention_rows,
)
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    materialize_simple_stateless_checkpoint_model,
    resolve_simple_pretraining_routes,
)
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.policy_inputs import collate_simple_stateless_actor_rows
from ptcg_rl.rl.stateless_checkpoint import load_stateless_policy_checkpoint
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    SupervisedPolicyArtifactManifest,
    load_supervised_policy_artifact,
)
from ptcg_rl.training.simple_stateless_pretrain_data import (
    LEGACY_PRETRAINING_SHARD_SCHEMA,
    ReplayPretrainingExample,
    file_sha256,
    iter_pretraining_rows,
    load_pretraining_manifest,
    load_pretraining_part,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DOMAIN = b"ptcg-rl/simple-stateless-retention-config/v1\x00"


class SimpleStatelessRetentionAuditConfig(BaseModel):
    """Immutable inputs for one routed retention comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_checkpoint_path: Path
    supervised_manifest_paths: tuple[Path, ...]
    dataset_manifest_path: Path
    sample_rows: int = Field(default=8192, ge=1, le=100_000)
    seed: int = Field(default=20260727, ge=0)
    output_dir: Path
    device: str = "cuda"
    batch_size: int = Field(default=256, ge=1, le=4096)

    @field_validator("supervised_manifest_paths")
    @classmethod
    def nonempty_unique_manifests(
        cls,
        value: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        """Reject ambiguous duplicate candidate inputs."""
        normalized = tuple(path.resolve() for path in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("supervised manifests must be nonempty and unique")
        return normalized

    @field_validator(
        "baseline_checkpoint_path",
        "dataset_manifest_path",
        "output_dir",
    )
    @classmethod
    def resolved_path(cls, value: Path) -> Path:
        """Bind audit paths before hashing and execution."""
        return value.resolve()

    @field_validator("device")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        """Reject an empty torch device."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("retention audit device cannot be empty")
        return normalized

    @model_validator(mode="after")
    def bounded_model_count(self) -> Self:
        """Keep simultaneous routed models within an auditable bound."""
        if len(self.supervised_manifest_paths) > 16:
            raise ValueError("retention audit supports at most 16 candidates")
        return self


@dataclass(frozen=True)
class _SelectedExample:
    sample_index: int
    reference: _SampleRef
    example: ReplayPretrainingExample


@dataclass
class _MetricAccumulator:
    rows: int = 0
    nll_sum: float = 0.0
    teacher_matches: int = 0
    greedy_flips: int = 0
    episode_nll_sum: defaultdict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    episode_rows: defaultdict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def add(
        self,
        *,
        episode_id: int,
        teacher_nll: float,
        teacher_match: bool,
        greedy_flip: bool,
    ) -> None:
        """Accumulate one sampled decision without retaining model tensors."""
        self.rows += 1
        self.nll_sum += teacher_nll
        self.teacher_matches += int(teacher_match)
        self.greedy_flips += int(greedy_flip)
        self.episode_nll_sum[episode_id] += teacher_nll
        self.episode_rows[episode_id] += 1

    def metrics(self) -> dict[str, int | float | None]:
        """Return decision and episode-equal policy diagnostics."""
        if self.rows == 0:
            return {
                "rows": 0,
                "episodes": 0,
                "episode_normalized_teacher_nll": None,
                "decision_teacher_nll": None,
                "greedy_teacher_agreement": None,
                "greedy_flip_from_baseline": None,
            }
        episode_nll = sum(
            total / self.episode_rows[episode_id]
            for episode_id, total in self.episode_nll_sum.items()
        ) / len(self.episode_nll_sum)
        return {
            "rows": self.rows,
            "episodes": len(self.episode_nll_sum),
            "episode_normalized_teacher_nll": episode_nll,
            "decision_teacher_nll": self.nll_sum / self.rows,
            "greedy_teacher_agreement": self.teacher_matches / self.rows,
            "greedy_flip_from_baseline": self.greedy_flips / self.rows,
        }


@dataclass(frozen=True)
class _AuditModel:
    role: Literal["baseline", "supervised"]
    model_fingerprint: str
    artifact_sha256: str
    source_path: Path
    model: SimpleStatelessPolicyValueNet
    manifest: SupervisedPolicyArtifactManifest | None = None


def run(config: SimpleStatelessRetentionAuditConfig) -> dict[str, Any]:
    """Sample a V1 corpus once and compare routed policies on identical rows."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA retention audit requested but CUDA is unavailable")
    _configure_torch_runtime(device)
    baseline = load_stateless_policy_checkpoint(config.baseline_checkpoint_path)
    dataset = load_pretraining_manifest(config.dataset_manifest_path)
    if dataset.format != LEGACY_PRETRAINING_SHARD_SCHEMA:
        raise ValueError("retention audit requires the immutable V1 compact corpus")
    if (
        dataset.public_catalog_fingerprint
        != baseline.identity.public_deck_catalog_fingerprint
        or dataset.input_contract_fingerprint
        != baseline.identity.input_contract_fingerprint
    ):
        raise ValueError("retention corpus and baseline input identities differ")
    exact_registry = baseline.model_config_value.resolved_registry_sha256
    if exact_registry is None:
        raise ValueError("baseline checkpoint has no resolved exact registry")

    models = [
        _baseline_model(
            baseline.model_config_value,
            baseline.model_state,
            model_fingerprint=baseline.artifact.policy_model_fingerprint,
            artifact_sha256=baseline.artifact.policy_sha256,
            source_path=config.baseline_checkpoint_path,
            device=device,
        )
    ]
    manifest_sha256s: dict[str, str] = {}
    for path in config.supervised_manifest_paths:
        manifest, state = load_supervised_policy_artifact(
            path,
            expected_model_config=baseline.model_config_value,
            expected_exact_registry_fingerprint=exact_registry,
            expected_public_catalog_fingerprint=(
                baseline.identity.public_deck_catalog_fingerprint
            ),
            expected_input_contract_fingerprint=(
                baseline.identity.input_contract_fingerprint
            ),
        )
        _validate_candidate_source(
            manifest,
            baseline_policy_sha256=baseline.artifact.policy_sha256,
            baseline_model_fingerprint=(baseline.artifact.policy_model_fingerprint),
        )
        manifest_sha256s[str(path)] = file_sha256(path)
        models.append(
            _supervised_model(
                manifest,
                state,
                model_config=baseline.model_config_value,
                source_path=path,
                device=device,
            )
        )
    fingerprints = tuple(model.model_fingerprint for model in models)
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("retention audit models have duplicate state fingerprints")

    references, sampling = sample_retention_rows(
        dataset_manifest_path=config.dataset_manifest_path,
        sample_rows=config.sample_rows,
        seed=config.seed,
        dataset=dataset,
        exact_routes=baseline.model_config_value.exact_routes,
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "rows.parquet"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    for path in (parquet_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"retention audit output already exists: {path}")

    aggregates, stratum_aggregates = _evaluate_to_parquet(
        config,
        dataset=dataset,
        references=references,
        models=tuple(models),
        parquet_path=parquet_path,
    )
    baseline_metrics = aggregates[models[0].model_fingerprint].metrics()
    model_summaries = []
    for audit_model in models:
        metrics = aggregates[audit_model.model_fingerprint].metrics()
        baseline_nll = baseline_metrics["episode_normalized_teacher_nll"]
        model_nll = metrics["episode_normalized_teacher_nll"]
        model_summaries.append(
            {
                "role": audit_model.role,
                "model_fingerprint": audit_model.model_fingerprint,
                "artifact_sha256": audit_model.artifact_sha256,
                "source_path": str(audit_model.source_path),
                "supervised_manifest_sha256": (
                    None
                    if audit_model.manifest is None
                    else manifest_sha256s[str(audit_model.source_path)]
                ),
                "selection": (
                    None
                    if audit_model.manifest is None
                    or audit_model.manifest.selection is None
                    else audit_model.manifest.selection.model_dump(mode="json")
                ),
                "metrics": metrics,
                "episode_normalized_teacher_nll_delta": (
                    None
                    if baseline_nll is None or model_nll is None
                    else float(model_nll) - float(baseline_nll)
                ),
                "strata": {
                    stratum: stratum_aggregates[
                        (audit_model.model_fingerprint, stratum)
                    ].metrics()
                    for stratum in sampling["strata"]
                },
            }
        )
    summary = {
        "format": "simple-stateless-full-model-retention-audit-v1",
        "config_fingerprint": _config_fingerprint(config),
        "inputs": {
            "baseline_checkpoint_path": str(config.baseline_checkpoint_path),
            "baseline_checkpoint_sha256": baseline.artifact.policy_sha256,
            "baseline_model_fingerprint": (baseline.artifact.policy_model_fingerprint),
            "dataset_manifest_path": str(config.dataset_manifest_path),
            "dataset_manifest_sha256": file_sha256(config.dataset_manifest_path),
            "dataset_fingerprint": dataset.fingerprint,
            "public_catalog_fingerprint": dataset.public_catalog_fingerprint,
            "input_contract_fingerprint": dataset.input_contract_fingerprint,
            "exact_registry_fingerprint": exact_registry,
            "device": str(device),
        },
        "sampling": sampling,
        "models": model_summaries,
        "rows_parquet": {
            "path": str(parquet_path),
            "rows": len(references) * len(models),
            "size_bytes": parquet_path.stat().st_size,
            "sha256": file_sha256(parquet_path),
        },
    }
    atomic_write_bytes(
        summary_path,
        json_payload(summary),
        overwrite=False,
    )
    atomic_write_bytes(
        report_path,
        _markdown_report(summary).encode("utf-8"),
        overwrite=False,
    )
    return summary


def _baseline_model(
    model_config: Any,
    state: Mapping[str, torch.Tensor],
    *,
    model_fingerprint: str,
    artifact_sha256: str,
    source_path: Path,
    device: torch.device,
) -> _AuditModel:
    model = _loaded_model(model_config, state, device=device)
    return _AuditModel(
        role="baseline",
        model_fingerprint=model_fingerprint,
        artifact_sha256=artifact_sha256,
        source_path=source_path,
        model=model,
    )


def _supervised_model(
    manifest: SupervisedPolicyArtifactManifest,
    state: Mapping[str, torch.Tensor],
    *,
    model_config: Any,
    source_path: Path,
    device: torch.device,
) -> _AuditModel:
    return _AuditModel(
        role="supervised",
        model_fingerprint=manifest.model_state_fingerprint,
        artifact_sha256=manifest.policy_sha256,
        source_path=source_path,
        model=_loaded_model(model_config, state, device=device),
        manifest=manifest,
    )


def _loaded_model(
    model_config: Any,
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> SimpleStatelessPolicyValueNet:
    model = materialize_simple_stateless_checkpoint_model(
        model_config,
        state,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def _configure_torch_runtime(device: torch.device) -> None:
    """Avoid oversized CPU pools around a CUDA-only sequential audit."""
    if device.type != "cuda":
        return
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _validate_candidate_source(
    manifest: SupervisedPolicyArtifactManifest,
    *,
    baseline_policy_sha256: str,
    baseline_model_fingerprint: str,
) -> None:
    scope = manifest.trainable_scope
    if scope is None or scope.mode != "full_model":
        raise ValueError("retention audit accepts only full-model artifacts")
    if manifest.selection is None:
        raise ValueError("full-model retention candidate has no validation selection")
    initialization = manifest.initialization
    if (
        initialization.mode != "rl_pair"
        or initialization.source_policy_sha256 != baseline_policy_sha256
        or initialization.source_policy_model_fingerprint != baseline_model_fingerprint
    ):
        raise ValueError("supervised candidate was not initialized from baseline")


def _iter_sample_batches(
    config: SimpleStatelessRetentionAuditConfig,
    *,
    dataset: Any,
    references: Sequence[_SampleRef],
) -> Iterator[tuple[_SelectedExample, ...]]:
    indexed = tuple(enumerate(references))
    by_part: defaultdict[int, list[tuple[int, _SampleRef]]] = defaultdict(list)
    for sample_index, reference in indexed:
        by_part[reference.part_index].append((sample_index, reference))
    parts_dir = config.dataset_manifest_path.parent / "parts"
    pending: list[_SelectedExample] = []
    for part_index in sorted(by_part):
        record = dataset.parts[part_index]
        part = load_pretraining_part(parts_dir / record.filename)
        entries = by_part[part_index]
        examples = iter_pretraining_rows(
            part,
            [reference.row_index for _sample_index, reference in entries],
            catalog_fingerprint=dataset.public_catalog_fingerprint,
            input_contract_fingerprint=dataset.input_contract_fingerprint,
        )
        for (sample_index, reference), example in zip(
            entries,
            examples,
            strict=True,
        ):
            if example.episode_id != reference.episode_id:
                raise ValueError("sampled retention row changed episode identity")
            pending.append(
                _SelectedExample(
                    sample_index=sample_index,
                    reference=reference,
                    example=example,
                )
            )
            if len(pending) >= config.batch_size:
                yield tuple(pending)
                pending.clear()
    if pending:
        yield tuple(pending)


def _evaluate_to_parquet(
    config: SimpleStatelessRetentionAuditConfig,
    *,
    dataset: Any,
    references: Sequence[_SampleRef],
    models: tuple[_AuditModel, ...],
    parquet_path: Path,
) -> tuple[
    dict[str, _MetricAccumulator],
    defaultdict[tuple[str, str], _MetricAccumulator],
]:
    schema = pa.schema(
        (
            ("sample_index", pa.int64()),
            ("episode_id", pa.int64()),
            ("part_filename", pa.string()),
            ("part_row", pa.int32()),
            ("stratum", pa.string()),
            ("model_role", pa.string()),
            ("model_fingerprint", pa.string()),
            ("teacher_action_nll", pa.float64()),
            ("greedy_matches_teacher", pa.bool_()),
            ("greedy_flip_from_baseline", pa.bool_()),
        )
    )
    temporary_dir = _REPO_ROOT / "tmp" / f"retention_audit_{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    temporary_path = temporary_dir / "rows.parquet"
    aggregates = {model.model_fingerprint: _MetricAccumulator() for model in models}
    strata: defaultdict[tuple[str, str], _MetricAccumulator] = defaultdict(
        _MetricAccumulator
    )
    try:
        with pq.ParquetWriter(
            temporary_path,
            schema,
            compression="zstd",
        ) as writer:
            for batch in _iter_sample_batches(
                config,
                dataset=dataset,
                references=references,
            ):
                evaluations = [
                    _evaluate_batch(
                        model.model, batch, device=torch.device(config.device)
                    )
                    for model in models
                ]
                baseline_greedy = evaluations[0][1]
                rows: list[dict[str, Any]] = []
                for audit_model, (nlls, greedy) in zip(
                    models,
                    evaluations,
                    strict=True,
                ):
                    for selected, teacher_nll, predicted, baseline_predicted in zip(
                        batch,
                        nlls,
                        greedy,
                        baseline_greedy,
                        strict=True,
                    ):
                        teacher_match = predicted == selected.example.action
                        greedy_flip = predicted != baseline_predicted
                        accumulator = aggregates[audit_model.model_fingerprint]
                        accumulator.add(
                            episode_id=selected.reference.episode_id,
                            teacher_nll=teacher_nll,
                            teacher_match=teacher_match,
                            greedy_flip=greedy_flip,
                        )
                        strata[
                            (
                                audit_model.model_fingerprint,
                                selected.reference.stratum,
                            )
                        ].add(
                            episode_id=selected.reference.episode_id,
                            teacher_nll=teacher_nll,
                            teacher_match=teacher_match,
                            greedy_flip=greedy_flip,
                        )
                        rows.append(
                            {
                                "sample_index": selected.sample_index,
                                "episode_id": selected.reference.episode_id,
                                "part_filename": dataset.parts[
                                    selected.reference.part_index
                                ].filename,
                                "part_row": selected.reference.row_index,
                                "stratum": selected.reference.stratum,
                                "model_role": audit_model.role,
                                "model_fingerprint": (audit_model.model_fingerprint),
                                "teacher_action_nll": teacher_nll,
                                "greedy_matches_teacher": teacher_match,
                                "greedy_flip_from_baseline": greedy_flip,
                            }
                        )
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        os.replace(temporary_path, parquet_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_dir.rmdir()
    return aggregates, strata


def _evaluate_batch(
    model: SimpleStatelessPolicyValueNet,
    batch: tuple[_SelectedExample, ...],
    *,
    device: torch.device,
) -> tuple[list[float], tuple[tuple[int, ...], ...]]:
    examples = tuple(selected.example for selected in batch)
    inputs = collate_simple_stateless_actor_rows(
        tuple(example.actor_row for example in examples),
        device=device,
        deduplicate_belief=True,
    )
    routes = resolve_simple_pretraining_routes(
        inputs.deck_signatures,
        model.config,
        device=device,
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        state = model.encode_observation_state(
            state=inputs.states,
            unique_deck_card_ids=inputs.unique_deck_card_ids,
            deck_counts=inputs.deck_counts,
            deck_valid_mask=inputs.deck_valid_mask,
            belief_summary=inputs.belief_summary,
            route_plan=routes,
            allow_unrouted_rows=True,
        )
        options = model.encode_legal_options(
            state,
            inputs.options,
            route_plan=routes,
            allow_unrouted_rows=True,
        )
        teacher = model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            options,
            inputs.options,
            tuple(example.action for example in examples),
            route_plan=routes,
            evaluate_prefix_values=False,
        )
        greedy = model.heads.greedy_decode(
            state.policy,
            state.opponent_belief,
            options,
            inputs.options,
            route_plan=routes,
        )
    nlls = (-teacher.action_logprobs.float()).cpu().tolist()
    if any(not math.isfinite(float(value)) for value in nlls):
        raise FloatingPointError("retention audit produced non-finite policy NLL")
    return [float(value) for value in nlls], greedy


def _config_fingerprint(config: SimpleStatelessRetentionAuditConfig) -> str:
    return hashlib.sha256(
        _CONFIG_DOMAIN
        + json.dumps(
            config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _markdown_report(summary: Mapping[str, Any]) -> str:
    sampling = summary["sampling"]
    lines = [
        "# Full-model retention audit",
        "",
        f"- Sample fingerprint: `{sampling['sample_fingerprint']}`",
        f"- Selected rows: {sampling['selected_rows']}",
        f"- Dataset fingerprint: `{summary['inputs']['dataset_fingerprint']}`",
        "",
        "## Stratum coverage",
        "",
        "| Stratum | Available | Sampled |",
        "|---|---:|---:|",
    ]
    for stratum, counts in sampling["strata"].items():
        lines.append(
            f"| `{stratum}` | {counts['available_rows']} | {counts['sampled_rows']} |"
        )
    lines.extend(
        (
            "",
            "## Model comparison",
            "",
            "| Role | Model | Episode NLL | Δ NLL | Greedy flip |",
            "|---|---|---:|---:|---:|",
        )
    )
    for model in summary["models"]:
        metrics = model["metrics"]
        delta = model["episode_normalized_teacher_nll_delta"]
        lines.append(
            f"| {model['role']} | `{model['model_fingerprint'][:16]}` | "
            f"{_format_metric(metrics['episode_normalized_teacher_nll'])} | "
            f"{_format_metric(delta)} | "
            f"{_format_metric(metrics['greedy_flip_from_baseline'])} |"
        )
    lines.extend(
        (
            "",
            "Teacher NLL is averaged within sampled episodes and then equally "
            "across episodes. Greedy flip compares the candidate action sequence "
            "with the routed baseline on the identical public state.",
            "",
        )
    )
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("retention report metric is not numeric")
    return f"{value:.6f}"


__all__ = [
    "SimpleStatelessRetentionAuditConfig",
    "run",
]
