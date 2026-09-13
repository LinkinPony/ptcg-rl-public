"""H200 inference wrapper for fixed-share real improvement actors."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor

from ptcg_rl.actions.encoding import (
    EncodedOptionArrayFeatures,
    EncodedOptionInput,
)
from ptcg_rl.actions.selection import is_unordered_set_selection
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.model.action_value import _wdl_expected_score_unchecked
from ptcg_rl.model.network import (
    AgentPolicyValueNet,
    PolicyEvaluationContext,
)
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.rl.amortized_policy_iteration.candidate_proposal import (
    BehaviorCandidate,
    CandidateProposal,
    CompleteActionSpace,
    build_candidate_proposal,
    reweight_candidate_proposal,
    structural_action_probability,
)
from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    CandidateProposalConfig,
    CmpoConfig,
)
from ptcg_rl.rl.amortized_policy_iteration.improvement_policy import (
    _cmpo_distribution_unchecked,
    sample_improvement_distribution,
)

if TYPE_CHECKING:
    from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
        StudentReanalysisRoot,
    )

AutocastMode = Literal["bf16", "off"]


class ImprovementRolloutPolicy:
    """Sample bounded CMPO actions from the persistent Q head on the H200."""

    behavior_kind = "improvement"

    def __init__(
        self,
        model: AgentPolicyValueNet,
        *,
        publication_source: object,
        proposal_config: CandidateProposalConfig,
        cmpo_config: CmpoConfig,
        seed: int,
        autocast: AutocastMode,
        generator: torch.Generator | None = None,
    ) -> None:
        """Bind one shared mutable inference model and fixed proposal contract."""
        if not model.config.action_value.enabled:
            raise ValueError("improvement behavior requires an action-value head")
        self._model = model
        self._publication_source = publication_source
        self._proposal_config = proposal_config
        self._cmpo_config = cmpo_config
        self._rng = random.Random(seed)
        self._autocast = autocast
        self._generator = generator

    @property
    def policy_version(self) -> int:
        """Mirror the ordinary candidate policy publication."""
        return int(getattr(self._publication_source, "policy_version", 0))

    @property
    def model_fingerprint(self) -> str:
        """Mirror the exact shared candidate-model fingerprint."""
        return str(getattr(self._publication_source, "model_fingerprint", ""))

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        """Return real improved actions and their true categorical log-probability."""
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("improvement behavior temperature must be positive")
        with torch.inference_mode(), self._autocast_context():
            conditioned = self._model.encode_conditioned_state(states, decks)
            context = self._model.policy_context_from_conditioned(
                conditioned,
                options,
            )
            spaces = action_spaces(options)
            behavior_candidates = sample_behavior_candidates(
                self._model,
                context,
                options,
                decks,
                spaces=spaces,
                config=self._proposal_config,
                temperature=temperature,
                generator=self._generator,
            )
            proposals = tuple(
                provisional_proposal(
                    spaces[row],
                    behavior_candidates=behavior_candidates[row],
                    config=self._proposal_config,
                    rng=self._rng,
                )
                for row in range(len(spaces))
            )
            action_groups = tuple(
                tuple(candidate.action for candidate in proposal.candidates)
                for proposal in proposals
            )
            ordered_rows = torch.tensor(
                [space.ordered for space in spaces],
                dtype=torch.bool,
                device=options.valid_options.device,
            )
            evaluation = self._model.evaluate_action_values_from_context(
                context,
                options,
                action_groups,
                ordered_rows=ordered_rows,
                decks=decks,
                policy_temperature=temperature,
                validate_candidate_actions=False,
            )
            action_logprobs = evaluation.action_logprobs
            if action_logprobs is None:
                raise RuntimeError("action-value evaluation omitted policy density")
            old_probabilities = action_logprobs.float().exp()
            proposal_probabilities = cmpo_proposal_probabilities(
                proposals,
                spaces,
                old_probabilities=old_probabilities,
                candidate_counts=evaluation.candidate_counts,
                config=self._proposal_config,
            )
            state_probabilities = torch.softmax(
                evaluation.information_set_state_logits.float(),
                dim=-1,
            )
            improved = _cmpo_distribution_unchecked(
                evaluation.expected_scores.float(),
                _wdl_expected_score_unchecked(state_probabilities),
                old_probabilities,
                proposal_probabilities,
                candidate_counts=evaluation.candidate_counts,
                exhaustive_rows=tuple(proposal.exhaustive for proposal in proposals),
                config=self._cmpo_config,
            )
            sample = sample_improvement_distribution(
                improved,
                generator=self._generator,
            )
            values = self._model.root_values_from_conditioned(conditioned)
            flattened_actions = tuple(
                action for group in action_groups for action in group
            )
            actions = tuple(
                flattened_actions[int(index)] for index in sample.flat_indices.tolist()
            )
        return (
            actions,
            sample.log_probabilities.to(dtype=torch.float32),
            values,
        )

    def _autocast_context(self) -> AbstractContextManager[None]:
        if self._autocast == "off" or _module_device(self._model).type != "cuda":
            return nullcontext()
        return cast(
            AbstractContextManager[None],
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        )


def action_spaces(options: OptionBatch) -> tuple[CompleteActionSpace, ...]:
    """Materialize complete-action metadata from a collated tensor batch."""
    batch_size = int(options.valid_options.shape[0])
    if batch_size == 0:
        return ()
    option_counts = options.valid_options.to(dtype=torch.long).sum(dim=1)
    if int(options.contexts.shape[1]) == 0:
        first_contexts = torch.full_like(option_counts, -1)
    else:
        first_contexts = options.contexts[:, 0].to(dtype=torch.long)
    metadata = (
        torch.stack(
            (
                option_counts,
                options.min_counts.to(dtype=torch.long),
                options.max_counts.to(dtype=torch.long),
                first_contexts,
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )
    return tuple(
        _complete_action_space(
            option_count=option_count,
            min_count=min_count,
            max_count=max_count,
            context=first_context,
        )
        for option_count, min_count, max_count, first_context in metadata
    )


def action_spaces_from_roots(
    roots: Sequence[StudentReanalysisRoot],
) -> tuple[CompleteActionSpace, ...]:
    """Build action spaces from CPU root metadata without synchronizing CUDA."""
    return tuple(
        _complete_action_space(
            option_count=len(root.options),
            min_count=root.min_count,
            max_count=root.max_count,
            context=_first_option_context(root.options),
        )
        for root in roots
    )


def _first_option_context(options: EncodedOptionInput) -> int:
    if not len(options):
        return -1
    if isinstance(options, EncodedOptionArrayFeatures):
        return int(options.contexts[0])
    return int(options[0].context)


def _complete_action_space(
    *,
    option_count: int,
    min_count: int,
    max_count: int,
    context: int,
) -> CompleteActionSpace:
    normalized_context = -1 if option_count <= 0 else int(context)
    unordered = is_unordered_set_selection(
        context=normalized_context,
        min_count=min_count,
        max_count=max_count,
    )
    return CompleteActionSpace(
        option_count=option_count,
        min_count=min_count,
        max_count=max_count,
        ordered=bool(max_count > 1 and not unordered),
    )


def sample_behavior_candidates(
    model: AgentPolicyValueNet,
    context: PolicyEvaluationContext,
    options: OptionBatch,
    decks: DeckBatch,
    *,
    spaces: tuple[CompleteActionSpace, ...],
    config: CandidateProposalConfig,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[tuple[BehaviorCandidate, ...], ...]:
    """Draw all non-exhaustive behavior candidates in one GPU batch."""
    if len(spaces) != int(options.valid_options.shape[0]):
        raise ValueError("action spaces must align with option rows")
    candidate_counts = tuple(
        0
        if space.legal_action_count <= config.exhaustive_action_cap
        else config.max_candidates
        for space in spaces
    )
    max_select_steps = max(
        (
            space.max_count
            for space, candidate_count in zip(
                spaces,
                candidate_counts,
                strict=True,
            )
            if candidate_count > 0
        ),
        default=0,
    )
    samples = model.sample_action_candidates_from_context(
        context,
        options,
        candidate_counts,
        max_select_steps=max_select_steps,
        decks=decks,
        temperature=temperature,
        generator=generator,
    )
    if not any(samples.candidate_counts):
        return tuple(() for _ in samples.candidate_counts)
    probabilities = samples.action_logprobs.float().exp().detach().cpu().tolist()
    result = []
    start = 0
    for actions, count in zip(
        samples.actions,
        samples.candidate_counts,
        strict=True,
    ):
        stop = start + count
        result.append(
            tuple(
                BehaviorCandidate(action=action, probability=float(probability))
                for action, probability in zip(
                    actions,
                    probabilities[start:stop],
                    strict=True,
                )
            )
        )
        start = stop
    return tuple(result)


def provisional_proposal(
    space: CompleteActionSpace,
    *,
    behavior_candidates: Sequence[BehaviorCandidate],
    config: CandidateProposalConfig,
    rng: random.Random,
) -> CandidateProposal:
    candidates = []
    known: dict[tuple[int, ...], float] = {}
    for candidate in behavior_candidates:
        candidates.append(candidate)
        previous = known.setdefault(candidate.action, candidate.probability)
        if not math.isclose(
            previous,
            candidate.probability,
            rel_tol=1.0e-5,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError("frozen policy assigned inconsistent action probability")
    return build_candidate_proposal(
        space,
        behavior_candidates=candidates,
        behavior_probability=lambda action: known.get(action, 0.0),
        config=config,
        rng=rng,
    )


def reweight_proposals(
    proposals: tuple[CandidateProposal, ...],
    spaces: tuple[CompleteActionSpace, ...],
    *,
    old_probabilities: Tensor,
    candidate_counts: tuple[int, ...],
    config: CandidateProposalConfig,
) -> tuple[CandidateProposal, ...]:
    if len(proposals) != len(spaces) or len(proposals) != len(candidate_counts):
        raise ValueError("proposal groups are misaligned")
    result = []
    start = 0
    cpu_probabilities = old_probabilities.detach().to(device="cpu").tolist()
    for proposal, space, count in zip(
        proposals,
        spaces,
        candidate_counts,
        strict=True,
    ):
        stop = start + count
        result.append(
            reweight_candidate_proposal(
                proposal,
                space,
                behavior_probabilities=tuple(
                    float(value) for value in cpu_probabilities[start:stop]
                ),
                config=config,
            )
        )
        start = stop
    return tuple(result)


def cmpo_proposal_probabilities(
    proposals: tuple[CandidateProposal, ...],
    spaces: tuple[CompleteActionSpace, ...],
    *,
    old_probabilities: Tensor,
    candidate_counts: tuple[int, ...],
    config: CandidateProposalConfig,
) -> Tensor:
    """Compute exact CMPO proposal densities without a GPU-to-CPU round trip."""
    if len(proposals) != len(spaces) or len(proposals) != len(candidate_counts):
        raise ValueError("proposal groups are misaligned")
    if sum(candidate_counts) != int(old_probabilities.numel()):
        raise ValueError("candidate counts must cover old policy probabilities")

    behavior_coefficients = []
    fixed_components = []
    for proposal, space, count in zip(
        proposals,
        spaces,
        candidate_counts,
        strict=True,
    ):
        if len(proposal.candidates) != count:
            raise ValueError("candidate counts must align with proposals")
        for candidate in proposal.candidates:
            if proposal.exhaustive:
                behavior_coefficients.append(0.0)
                fixed_components.append(1.0 / float(proposal.legal_action_count))
            else:
                behavior_coefficients.append(config.behavior_weight)
                fixed_components.append(
                    config.structural_weight
                    * structural_action_probability(space, candidate.action)
                    + config.exploration_weight / float(proposal.legal_action_count)
                )
    coefficients = old_probabilities.new_tensor(behavior_coefficients)
    fixed = old_probabilities.new_tensor(fixed_components)
    return coefficients * old_probabilities + fixed


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


__all__ = [
    "ImprovementRolloutPolicy",
    "action_spaces",
    "action_spaces_from_roots",
    "cmpo_proposal_probabilities",
    "provisional_proposal",
    "reweight_proposals",
    "sample_behavior_candidates",
]
