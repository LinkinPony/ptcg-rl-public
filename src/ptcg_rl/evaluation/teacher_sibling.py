"""Public-teacher afterstate sibling generation and paired critic audit."""

from __future__ import annotations

import time
from typing import Any

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks import parse_canonical_signature
from ptcg_rl.evaluation.search_identity import file_sha256, write_identity_atomic
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingConfig
from ptcg_rl.evaluation.teacher_sibling_engine import (
    evaluate_teacher_root,
    step_action,
)
from ptcg_rl.evaluation.teacher_sibling_io import TeacherSiblingWriter
from ptcg_rl.evaluation.teacher_sibling_metrics import TeacherSiblingMetrics
from ptcg_rl.evaluation.teacher_sibling_report import (
    require_absent_output,
    teacher_sibling_campaign_identity,
    verified_checkpoint,
    write_teacher_sibling_report,
)
from ptcg_rl.evaluation.teacher_sibling_support import (
    campaign_fingerprint,
    sample_teacher_validation_roots,
)
from ptcg_rl.training.bc_dataset import observation_from_step_row

_ARTIFACT_NAMES = (
    "roots.parquet",
    "worlds.parquet",
    "evaluations.parquet",
    "pairs.parquet",
)


def run_teacher_sibling_evaluation(
    config: TeacherSiblingConfig,
) -> dict[str, Any]:
    """Generate replayable engine siblings and compare BC value rankings."""
    started_at = time.perf_counter()
    output_dir = records.repo_path(config.output_dir)
    require_absent_output(output_dir)
    initial_path = verified_checkpoint(
        config.initial_checkpoint.path,
        config.initial_checkpoint.expected_sha256,
    )
    trained_path = verified_checkpoint(
        config.trained_checkpoint.path,
        config.trained_checkpoint.expected_sha256,
    )
    sampled_roots, sampling = sample_teacher_validation_roots(
        config.data,
        reservoir_size=config.reservoir_size,
        seed=config.seed,
    )
    identity = teacher_sibling_campaign_identity(
        config,
        sampling=sampling,
        initial_path=initial_path,
        trained_path=trained_path,
    )
    campaign_fp = campaign_fingerprint(identity)

    own_deck = parse_canonical_signature(
        config.data.expected_deck_signature
    ).card_ids
    initial_policy = CheckpointPolicy(
        initial_path,
        device=config.device,
        own_deck=own_deck,
    )
    trained_policy = CheckpointPolicy(
        trained_path,
        device=config.device,
        own_deck=own_deck,
    )
    initial_policy.configure_inference_cache(enabled=True)
    trained_policy.configure_inference_cache(enabled=True)
    initial_policy.prewarm()
    trained_policy.prewarm()
    belief_producer = OpponentBeliefFeatureProducer.from_config(config.belief)
    sampler = BeliefSampler(config=config.sampler)
    metrics = TeacherSiblingMetrics()
    skipped = {"teacher_illegal": 0, "fewer_than_two_candidates": 0}
    processed_roots = 0
    world_rows_total = 0
    evaluation_rows_total = 0
    pair_rows_total = 0
    with TeacherSiblingWriter(
        output_dir,
        compression=config.compression,
    ) as writer:
        for sampled in sampled_roots:
            if processed_roots >= config.max_roots:
                break
            root_group = evaluate_teacher_root(
                sampled,
                config=config,
                campaign_fp=campaign_fp,
                initial_policy=initial_policy,
                trained_policy=trained_policy,
                belief_producer=belief_producer,
                sampler=sampler,
                metrics=metrics,
            )
            if root_group is None:
                observation = observation_from_step_row(sampled.row)
                select = observation.get("select")
                teacher_action = step_action(sampled.row.get("action"))
                if select is None or not is_legal_action(select, teacher_action):
                    skipped["teacher_illegal"] += 1
                else:
                    skipped["fewer_than_two_candidates"] += 1
                continue
            root_row, world_rows, evaluation_rows, pair_rows = root_group
            writer.write_root_group(
                root=root_row,
                worlds=world_rows,
                evaluations=evaluation_rows,
                pairs=pair_rows,
            )
            processed_roots += 1
            world_rows_total += len(world_rows)
            evaluation_rows_total += len(evaluation_rows)
            pair_rows_total += len(pair_rows)

    elapsed_seconds = time.perf_counter() - started_at
    metric_summary = metrics.summary(config.references)
    artifact_paths = {name: output_dir / name for name in _ARTIFACT_NAMES}
    summary = {
        "protocol": "PUBLIC-TEACHER-ENGINE-SIBLING-v1",
        "experiment_id": config.experiment_id,
        "campaign_fp": campaign_fp,
        "behavior_kind": config.behavior_kind,
        "ppo_ratio_eligible": False,
        "identity": identity,
        "sampling": sampling,
        "processed_roots": processed_roots,
        "skipped_roots": skipped,
        "world_rows": world_rows_total,
        "evaluation_rows": evaluation_rows_total,
        "pair_rows": pair_rows_total,
        "elapsed_seconds": elapsed_seconds,
        "roots_per_second": (
            processed_roots / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        **metric_summary,
        "artifacts": {
            name: {
                "path": records.display_path(path),
                "sha256": file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "config": config.model_dump(mode="json"),
    }
    write_identity_atomic(output_dir / "summary.json", summary)
    write_identity_atomic(
        output_dir / "resolved_config.json",
        config.model_dump(mode="json"),
    )
    write_teacher_sibling_report(output_dir / "report.md", summary)
    return summary


__all__ = ["run_teacher_sibling_evaluation"]
