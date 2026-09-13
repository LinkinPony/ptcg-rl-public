"""H200 candidate proposal for bounded CPU native reanalysis jobs."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal, cast

import torch

from ptcg_rl.model import AgentPolicyValueNet
from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
    NativeReanalysisJob,
    NativeReanalysisJobBatch,
    ReanalysisRoot,
    native_producer_fingerprint,
)
from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    AmortizedPolicyIterationConfig,
)
from ptcg_rl.rl.amortized_policy_iteration.improvement_actor import (
    action_spaces_from_roots,
    provisional_proposal,
    reweight_proposals,
    sample_behavior_candidates,
)
from ptcg_rl.rl.amortized_policy_iteration.tensor_batch import (
    collate_information_sets,
    ordered_rows,
)

AutocastMode = Literal["bf16", "off"]
_JOB_SEED_DOMAIN = b"ptcg-rl/api-native-job-seed/v1\x00"


@dataclass(frozen=True, slots=True)
class ReanalysisProposalBatch:
    """Proposed native jobs and immutable admission diagnostics."""

    jobs: tuple[NativeReanalysisJob, ...]
    wire_batches: tuple[NativeReanalysisJobBatch, ...]
    counters: dict[str, int]


def build_native_reanalysis_jobs(
    model: AgentPolicyValueNet,
    roots: Sequence[ReanalysisRoot],
    *,
    current_policy_version: int,
    config: AmortizedPolicyIterationConfig,
    device: torch.device | str,
    seed: int,
    autocast: AutocastMode,
) -> ReanalysisProposalBatch:
    """Evaluate a frozen current policy and attach exact complete-action density."""
    if not model.config.action_value.enabled:
        raise ValueError("reanalysis proposal requires the complete-action Q head")
    counters: Counter[str] = Counter()
    eligible = []
    seen_roots: set[str] = set()
    for root in roots:
        counters["roots_seen"] += 1
        if root.student.root_id in seen_roots:
            counters["duplicate_root"] += 1
            continue
        seen_roots.add(root.student.root_id)
        policy_lag = current_policy_version - root.student.policy_version
        if policy_lag < 0:
            counters["policy_version_mismatch"] += 1
            counters["policy_version_future"] += 1
            continue
        if policy_lag > config.learner.max_root_policy_lag:
            counters["policy_version_mismatch"] += 1
            counters["policy_version_too_old"] += 1
            continue
        eligible.append(root)
    if not eligible:
        return ReanalysisProposalBatch(
            jobs=(),
            wire_batches=(),
            counters=dict(counters),
        )

    by_temperature: dict[float, list[ReanalysisRoot]] = defaultdict(list)
    for root in eligible:
        by_temperature[root.student.sampling_temperature].append(root)
    jobs: list[NativeReanalysisJob] = []
    rng = random.Random(seed)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for temperature, group in by_temperature.items():
            for start in range(0, len(group), config.learner.proposal_batch_size):
                chunk = tuple(group[start : start + config.learner.proposal_batch_size])
                jobs.extend(
                    _propose_chunk(
                        model,
                        chunk,
                        current_policy_version=current_policy_version,
                        config=config,
                        device=device,
                        temperature=temperature,
                        rng=rng,
                        seed=seed,
                        autocast=autocast,
                        generator=generator,
                    )
                )
    finally:
        model.train(was_training)
    wire_batches = batch_native_reanalysis_jobs(
        jobs,
        roots_per_batch=config.native.roots_per_job,
    )
    counters["jobs_proposed"] += len(jobs)
    counters["job_batches_proposed"] += len(wire_batches)
    return ReanalysisProposalBatch(
        jobs=tuple(jobs),
        wire_batches=wire_batches,
        counters=dict(counters),
    )


def batch_native_reanalysis_jobs(
    jobs: Sequence[NativeReanalysisJob],
    *,
    roots_per_batch: int,
) -> tuple[NativeReanalysisJobBatch, ...]:
    """Pack complete root jobs into queue envelopes without changing a job."""
    if roots_per_batch <= 0:
        raise ValueError("roots_per_batch must be positive")
    return tuple(
        NativeReanalysisJobBatch(tuple(jobs[start : start + roots_per_batch]))
        for start in range(0, len(jobs), roots_per_batch)
    )


def _propose_chunk(
    model: AgentPolicyValueNet,
    roots: tuple[ReanalysisRoot, ...],
    *,
    current_policy_version: int,
    config: AmortizedPolicyIterationConfig,
    device: torch.device | str,
    temperature: float,
    rng: random.Random,
    seed: int,
    autocast: AutocastMode,
    generator: torch.Generator,
) -> tuple[NativeReanalysisJob, ...]:
    students = tuple(root.student for root in roots)
    spaces = action_spaces_from_roots(students)
    batch = collate_information_sets(
        states=tuple(root.state for root in students),
        options=tuple(root.options for root in students),
        min_counts=tuple(root.min_count for root in students),
        max_counts=tuple(root.max_count for root in students),
        decks=tuple(root.deck for root in students),
        device=device,
    )
    with torch.inference_mode(), _autocast_context(model, autocast=autocast):
        conditioned = model.encode_conditioned_state(batch.states, batch.decks)
        context = model.policy_context_from_conditioned(conditioned, batch.options)
        behavior_candidates = sample_behavior_candidates(
            model,
            context,
            batch.options,
            batch.decks,
            spaces=spaces,
            config=config.candidate_proposal,
            temperature=temperature,
            generator=generator,
        )
        provisional = tuple(
            provisional_proposal(
                space,
                behavior_candidates=behavior_candidates[row],
                config=config.candidate_proposal,
                rng=rng,
            )
            for row, space in enumerate(spaces)
        )
        action_groups = tuple(
            tuple(candidate.action for candidate in proposal.candidates)
            for proposal in provisional
        )
        evaluation = model.evaluate_action_values_from_context(
            context,
            batch.options,
            action_groups,
            ordered_rows=ordered_rows(batch.options),
            decks=batch.decks,
            policy_temperature=temperature,
            validate_candidate_actions=False,
        )
        if evaluation.action_logprobs is None:
            raise RuntimeError("reanalysis proposal omitted exact policy density")
        proposals = reweight_proposals(
            provisional,
            spaces,
            old_probabilities=evaluation.action_logprobs.float().exp(),
            candidate_counts=evaluation.candidate_counts,
            config=config.candidate_proposal,
        )
    contract = native_producer_fingerprint(config)
    return tuple(
        NativeReanalysisJob(
            root=root,
            proposal=proposal,
            proposal_policy_version=current_policy_version,
            producer_contract_fingerprint=contract,
            stochastic_seed=_job_seed(seed, root.student.root_id),
        )
        for root, proposal in zip(roots, proposals, strict=True)
    )


def _job_seed(seed: int, root_id: str) -> int:
    digest = hashlib.sha256()
    digest.update(_JOB_SEED_DOMAIN)
    digest.update(seed.to_bytes(16, "big", signed=True))
    digest.update(bytes.fromhex(root_id))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _autocast_context(
    model: AgentPolicyValueNet,
    *,
    autocast: AutocastMode,
) -> AbstractContextManager[None]:
    if autocast == "off" or next(model.parameters()).device.type != "cuda":
        return nullcontext()
    return cast(
        AbstractContextManager[None],
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    )


__all__ = [
    "ReanalysisProposalBatch",
    "batch_native_reanalysis_jobs",
    "build_native_reanalysis_jobs",
]
