"""Read-only gradient diagnostics for exact-deck stateless PPO tasks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    normalized_sparse_belief_row_losses,
    resolve_simple_exact_routes,
)
from ptcg_rl.model.tensor_validation import require_tensor_condition
from ptcg_rl.rl.stateless_array_collation import (
    collate_stateless_array_microbatch,
)
from ptcg_rl.rl.stateless_array_replay import StatelessArrayOptimizerWindow
from ptcg_rl.rl.stateless_ppo import (
    SimpleStatelessPpoConfig,
    StatelessPpoLossInputs,
    stateless_deck_macro_ppo_loss,
)

_TRUNK_LAYER_PATTERN = re.compile(r"^backbone\.trunk\.layers\.(\d+)\.")
_V2_STAGE_PATTERN = re.compile(
    r"^backbone\.v2_adapters\.stages\.after_layer_(\d+)\."
)
_SHARED_GROUP_ORDER = (
    "input_tokens",
    "input_card",
    "input_deck",
    "input_belief",
    "input_state",
    "trunk_00_07",
    "trunk_08_15",
    "trunk_16_23",
    "trunk_24_33",
    "trunk_output",
    "policy_options",
    "policy_decode",
    "value_shared",
    "belief_head",
)
_COMPONENT_NAMES = ("actor", "critic", "belief")


@dataclass(frozen=True)
class ParameterSpan:
    """One parameter and its flat coordinates in a diagnostic vector."""

    name: str
    parameter: nn.Parameter
    start: int
    stop: int


@dataclass(frozen=True)
class StatelessGradientLayout:
    """Shared, late-trunk, and route-private parameter partitions."""

    shared_spans: tuple[ParameterSpan, ...]
    shared_group_slices: Mapping[str, slice]
    late_trunk_spans: tuple[ParameterSpan, ...]
    private_policy: Mapping[str, tuple[nn.Parameter, ...]]
    private_value: Mapping[str, tuple[nn.Parameter, ...]]
    shared_numel: int
    late_trunk_numel: int
    private_numel: int


@dataclass(frozen=True)
class DeckGradientObservation:
    """Scalar observations accumulated while materializing one deck gradient."""

    decisions: int
    active_tokens: int
    valid_belief_rows: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    belief_loss: float
    ratio_mean: float
    approximate_kl: float
    clip_fraction: float
    private_policy_norm: float
    private_value_norm: float


@dataclass(frozen=True)
class MatchedCellSamplePlan:
    """Deck samples with identical opponent-artifact, deck, and seat support."""

    samples: Mapping[str, tuple[npt.NDArray[np.int64], ...]]
    cell_quotas: tuple[tuple[int, str, str, int], ...]
    decisions_per_deck: int
    fingerprint: str


def build_stateless_gradient_layout(
    model: SimpleStatelessPolicyValueNet,
) -> StatelessGradientLayout:
    """Partition the model without padding disjoint private routes with zeros."""
    named_parameters = dict(model.named_parameters())
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        group: [] for group in _SHARED_GROUP_ORDER
    }
    private_policy_lists: dict[str, list[nn.Parameter]] = {
        route.deck_digest: [] for route in model.config.exact_routes
    }
    private_value_lists: dict[str, list[nn.Parameter]] = {
        route.deck_digest: [] for route in model.config.exact_routes
    }
    private_names: set[str] = set()
    routes_by_key = {
        route.module_key: route for route in model.config.exact_routes
    }
    for bank_name, bank in model.route_private_banks():
        if set(bank) != set(routes_by_key):
            raise ValueError(
                f"private route bank differs from exact registry: {bank_name}"
            )
        for module_key, route in routes_by_key.items():
            module = bank[module_key]
            for local_name, parameter in module.named_parameters():
                full_name = f"{bank_name}.{module_key}.{local_name}"
                if named_parameters.get(full_name) is not parameter:
                    raise ValueError(
                        f"private route parameter is absent from model: {full_name}"
                    )
                component = _private_parameter_component(bank_name, local_name)
                destination = (
                    private_value_lists
                    if component == "value"
                    else private_policy_lists
                )
                destination[route.deck_digest].append(parameter)
                private_names.add(full_name)
    if any(not parameters for parameters in private_policy_lists.values()):
        raise ValueError("an exact route has no private policy parameters")
    if any(not parameters for parameters in private_value_lists.values()):
        raise ValueError("an exact route has no private value parameters")
    private_policy = {
        digest: tuple(parameters)
        for digest, parameters in private_policy_lists.items()
    }
    private_value = {
        digest: tuple(parameters)
        for digest, parameters in private_value_lists.items()
    }

    for name, parameter in named_parameters.items():
        if name in private_names:
            continue
        grouped[_shared_group(name)].append((name, parameter))

    shared_spans: list[ParameterSpan] = []
    group_slices: dict[str, slice] = {}
    offset = 0
    for group in _SHARED_GROUP_ORDER:
        start = offset
        for name, parameter in grouped[group]:
            stop = offset + parameter.numel()
            shared_spans.append(
                ParameterSpan(
                    name=name,
                    parameter=parameter,
                    start=offset,
                    stop=stop,
                )
            )
            offset = stop
        group_slices[group] = slice(start, offset)

    late_spans: list[ParameterSpan] = []
    late_offset = 0
    for name, parameter in named_parameters.items():
        match = _TRUNK_LAYER_PATTERN.match(name)
        if match is None or int(match.group(1)) < model.config.num_layers - 2:
            continue
        stop = late_offset + parameter.numel()
        late_spans.append(
            ParameterSpan(
                name=name,
                parameter=parameter,
                start=late_offset,
                stop=stop,
            )
        )
        late_offset = stop

    private_numel = sum(
        parameter.numel()
        for name, parameter in named_parameters.items()
        if name in private_names
    )
    if offset + private_numel != sum(
        parameter.numel() for parameter in model.parameters()
    ):
        raise ValueError("gradient parameter partition does not cover the model")
    if not late_spans:
        raise ValueError("late-trunk diagnostic partition is empty")
    return StatelessGradientLayout(
        shared_spans=tuple(shared_spans),
        shared_group_slices=group_slices,
        late_trunk_spans=tuple(late_spans),
        private_policy=private_policy,
        private_value=private_value,
        shared_numel=offset,
        late_trunk_numel=late_offset,
        private_numel=private_numel,
    )


def sample_stateless_deck_decisions(
    window: StatelessArrayOptimizerWindow,
    *,
    decisions_per_deck: int,
    replicates: int,
    seed: int,
) -> Mapping[str, tuple[npt.NDArray[np.int64], ...]]:
    """Select advantage-stratified, fragment-disjoint deck samples."""
    if decisions_per_deck <= 0 or replicates <= 0:
        raise ValueError("diagnostic sample sizes must be positive")
    samples: dict[str, tuple[npt.NDArray[np.int64], ...]] = {}
    deck_values = np.asarray(window.deck_digests, dtype=np.str_)
    fragment_values = np.asarray(window.decision_fragment_indices, dtype=np.int64)
    advantages = np.asarray(window.normalized_advantages, dtype=np.float64)
    for digest in sorted({str(value) for value in deck_values}):
        deck_indices = np.flatnonzero(deck_values == digest).astype(
            np.int64,
            copy=False,
        )
        fragment_count = np.unique(fragment_values[deck_indices]).size
        required = decisions_per_deck * replicates
        if fragment_count < required:
            raise ValueError(
                f"deck {digest} has {fragment_count} fragments, fewer than "
                f"the required disjoint sample count {required}"
            )
        digest_seed = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest_seed[:8], "big"))
        ranked = deck_indices[
            np.argsort(advantages[deck_indices], kind="stable")
        ]
        strata = tuple(
            np.asarray(values, dtype=np.int64)
            for values in np.array_split(ranked, 4)
        )
        used_fragments: set[int] = set()
        deck_samples: list[npt.NDArray[np.int64]] = []
        for _replicate in range(replicates):
            selected: list[int] = []
            quotas = _stratum_quotas(decisions_per_deck, len(strata))
            for stratum, quota in zip(strata, quotas, strict=True):
                if quota == 0:
                    continue
                candidates = rng.permutation(stratum)
                selected_in_stratum = 0
                for raw_index in candidates:
                    index = int(raw_index)
                    fragment = int(fragment_values[index])
                    if fragment in used_fragments:
                        continue
                    selected.append(index)
                    used_fragments.add(fragment)
                    selected_in_stratum += 1
                    if selected_in_stratum >= quota:
                        break
            if len(selected) < decisions_per_deck:
                candidates = rng.permutation(deck_indices)
                for raw_index in candidates:
                    index = int(raw_index)
                    fragment = int(fragment_values[index])
                    if fragment in used_fragments:
                        continue
                    selected.append(index)
                    used_fragments.add(fragment)
                    if len(selected) == decisions_per_deck:
                        break
            if len(selected) != decisions_per_deck:
                raise RuntimeError(f"failed to sample deck {digest}")
            deck_samples.append(np.asarray(sorted(selected), dtype=np.int64))
        samples[digest] = tuple(deck_samples)
    return samples


def sample_stateless_matched_cells(
    window: StatelessArrayOptimizerWindow,
    *,
    replicates: int,
    seed: int,
) -> MatchedCellSamplePlan:
    """Select disjoint samples with exactly matched opponent/seat cell counts."""
    if replicates <= 0:
        raise ValueError("diagnostic replicate count must be positive")
    fragment_cells: list[tuple[int, str, str]] = []
    for fragment_index in range(window.fragments_retained):
        part_index = int(
            window.retained_fragment_part_indices[fragment_index]
        )
        part_row = int(window.retained_fragment_rows[fragment_index])
        source = window.source_arrays[part_index]
        fragment_cells.append(
            (
                int(source["seats"][part_row]),
                str(source["opponent_deck_digests"][part_row]),
                str(source["opponent_artifact_fingerprints"][part_row]),
            )
        )
    fragment_values = np.asarray(window.decision_fragment_indices, dtype=np.int64)
    deck_values = np.asarray(window.deck_digests, dtype=np.str_)
    deck_digests = tuple(sorted({str(value) for value in deck_values}))
    indices_by_deck_cell: dict[
        str, dict[tuple[int, str, str], list[int]]
    ] = {}
    fragment_counts: dict[str, dict[tuple[int, str, str], int]] = {}
    for digest in deck_digests:
        grouped: dict[tuple[int, str, str], list[int]] = {}
        deck_indices = np.flatnonzero(deck_values == digest)
        for raw_index in deck_indices:
            index = int(raw_index)
            cell = fragment_cells[int(fragment_values[index])]
            grouped.setdefault(cell, []).append(index)
        indices_by_deck_cell[digest] = grouped
        fragment_counts[digest] = {
            cell: len(
                {
                    int(fragment_values[index])
                    for index in indices
                }
            )
            for cell, indices in grouped.items()
        }
    common_cells = set(indices_by_deck_cell[deck_digests[0]])
    for digest in deck_digests[1:]:
        common_cells.intersection_update(indices_by_deck_cell[digest])
    cell_quotas = tuple(
        (
            cell[0],
            cell[1],
            cell[2],
            min(fragment_counts[digest][cell] for digest in deck_digests)
            // replicates,
        )
        for cell in sorted(common_cells)
        if min(fragment_counts[digest][cell] for digest in deck_digests)
        // replicates
        > 0
    )
    decisions_per_deck = sum(cell[3] for cell in cell_quotas)
    if decisions_per_deck <= 0:
        raise ValueError("snapshot has no repeatable matched opponent/seat support")

    samples: dict[str, tuple[npt.NDArray[np.int64], ...]] = {}
    for digest in deck_digests:
        digest_seed = hashlib.sha256(f"{seed}:{digest}:matched".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest_seed[:8], "big"))
        used_fragments: set[int] = set()
        deck_replicates: list[npt.NDArray[np.int64]] = []
        for _replicate in range(replicates):
            selected: list[int] = []
            for seat, opponent_digest, artifact_fingerprint, quota in cell_quotas:
                cell = (seat, opponent_digest, artifact_fingerprint)
                candidates = rng.permutation(indices_by_deck_cell[digest][cell])
                chosen = 0
                for raw_index in candidates:
                    index = int(raw_index)
                    fragment = int(fragment_values[index])
                    if fragment in used_fragments:
                        continue
                    selected.append(index)
                    used_fragments.add(fragment)
                    chosen += 1
                    if chosen == quota:
                        break
                if chosen != quota:
                    raise RuntimeError(
                        f"failed to fill matched cell for deck {digest}"
                    )
            if len(selected) != decisions_per_deck:
                raise RuntimeError(f"failed to sample matched deck {digest}")
            deck_replicates.append(
                np.asarray(sorted(selected), dtype=np.int64)
            )
        samples[digest] = tuple(deck_replicates)
    fingerprint_payload = json.dumps(
        cell_quotas,
        separators=(",", ":"),
    ).encode()
    return MatchedCellSamplePlan(
        samples=samples,
        cell_quotas=cell_quotas,
        decisions_per_deck=decisions_per_deck,
        fingerprint=hashlib.sha256(
            b"ptcg-rl/matched-gradient-cells/v1\x00" + fingerprint_payload
        ).hexdigest(),
    )


def accumulate_stateless_deck_gradient(
    *,
    model: SimpleStatelessPolicyValueNet,
    window: StatelessArrayOptimizerWindow,
    indices: Sequence[int],
    config: SimpleStatelessPpoConfig,
    layout: StatelessGradientLayout,
    microbatch_decisions: int,
    total_destination: Tensor,
    component_destinations: Mapping[str, Tensor],
) -> DeckGradientObservation:
    """Accumulate one normalized deck task gradient without an optimizer step."""
    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("deck diagnostic indices must be non-empty and unique")
    if microbatch_decisions <= 0:
        raise ValueError("diagnostic microbatch size must be positive")
    digests = {str(window.deck_digests[index]) for index in selected}
    if len(digests) != 1:
        raise ValueError("one diagnostic gradient must contain exactly one deck")
    digest = next(iter(digests))
    if total_destination.shape != (layout.shared_numel,):
        raise ValueError("shared gradient destination has the wrong shape")
    if set(component_destinations) != set(_COMPONENT_NAMES):
        raise ValueError("component gradient destinations are incomplete")
    if any(
        destination.shape != (layout.late_trunk_numel,)
        for destination in component_destinations.values()
    ):
        raise ValueError("component gradient destination has the wrong shape")

    model.zero_grad(set_to_none=True)
    total_destination.zero_()
    for destination in component_destinations.values():
        destination.zero_()
    device = next(model.parameters()).device
    component_accumulators = {
        name: torch.zeros(
            layout.late_trunk_numel,
            dtype=torch.float32,
            device=device,
        )
        for name in _COMPONENT_NAMES
    }
    decision_weight = 1.0 / float(len(selected))
    valid_belief_rows = int(window.belief_target_valid[list(selected)].sum())
    belief_weight = (
        0.0 if valid_belief_rows == 0 else 1.0 / float(valid_belief_rows)
    )
    scalar_sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "belief_loss": 0.0,
        "ratio_sum": 0.0,
        "approximate_kl_sum": 0.0,
        "clipped_token_count": 0.0,
        "active_token_count": 0.0,
    }
    probe_parameters = tuple(span.parameter for span in layout.late_trunk_spans)
    for start in range(0, len(selected), microbatch_decisions):
        chunk = selected[start : start + microbatch_decisions]
        inputs = evaluate_stateless_array_microbatch(
            model=model,
            window=window,
            decision_indices=chunk,
            behavior_temperature=config.behavior_temperature,
        )
        inputs = replace(
            inputs,
            decision_macro_weights=torch.full_like(
                inputs.decision_macro_weights,
                decision_weight,
            ),
            belief_macro_weights=(
                inputs.belief_valid_mask.to(
                    dtype=inputs.belief_macro_weights.dtype
                )
                * belief_weight
            ),
        )
        loss = stateless_deck_macro_ppo_loss(inputs, config)
        objectives = {
            "actor": (
                loss.policy_loss
                + config.entropy_coefficient * loss.entropy_loss
            ),
            "critic": config.value_coefficient * loss.value_loss,
            "belief": loss.belief_loss,
        }
        for name in _COMPONENT_NAMES:
            gradients = torch.autograd.grad(
                objectives[name],
                probe_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            _accumulate_flat_gradients(
                gradients,
                layout.late_trunk_spans,
                component_accumulators[name],
            )
        loss.loss.backward()  # type: ignore[no-untyped-call]
        for name in scalar_sums:
            scalar_sums[name] += float(getattr(loss, name).detach())

    _copy_parameter_gradients(layout.shared_spans, total_destination)
    for name in _COMPONENT_NAMES:
        component_destinations[name].copy_(
            component_accumulators[name].to(
                device=component_destinations[name].device
            )
        )
    active_tokens = int(scalar_sums["active_token_count"])
    ratio_mean = scalar_sums["ratio_sum"] / max(active_tokens, 1)
    approximate_kl = scalar_sums["approximate_kl_sum"] / max(active_tokens, 1)
    clip_fraction = scalar_sums["clipped_token_count"] / max(active_tokens, 1)
    return DeckGradientObservation(
        decisions=len(selected),
        active_tokens=active_tokens,
        valid_belief_rows=valid_belief_rows,
        loss=scalar_sums["loss"],
        policy_loss=scalar_sums["policy_loss"],
        value_loss=scalar_sums["value_loss"],
        entropy_loss=scalar_sums["entropy_loss"],
        belief_loss=scalar_sums["belief_loss"],
        ratio_mean=ratio_mean,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        private_policy_norm=_parameter_gradient_norm(
            layout.private_policy[digest]
        ),
        private_value_norm=_parameter_gradient_norm(layout.private_value[digest]),
    )


def evaluate_stateless_array_microbatch(
    *,
    model: SimpleStatelessPolicyValueNet,
    window: StatelessArrayOptimizerWindow,
    decision_indices: Sequence[int],
    behavior_temperature: float,
) -> StatelessPpoLossInputs:
    """Reproduce the learner's pure compact-column forward evaluation."""
    device = next(model.parameters()).device
    card_vocab_size = model.backbone.input_encoder.card_encoder.num_card_ids
    batch = collate_stateless_array_microbatch(
        window,
        decision_indices,
        card_vocab_size=card_vocab_size,
        device=device,
    )
    routes = resolve_simple_exact_routes(
        batch.deck_signatures,
        model.config,
        device=device,
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        state = model.encode_observation_state(
            state=batch.states,
            unique_deck_card_ids=batch.unique_deck_card_ids,
            deck_counts=batch.deck_counts,
            deck_valid_mask=batch.deck_valid_mask,
            belief_summary=batch.belief_summary,
            route_plan=routes,
        )
        option_embeddings = model.encode_legal_options(
            state,
            batch.options,
            route_plan=routes,
        )
        evaluation = model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            batch.actions,
            route_plan=routes,
            temperature=behavior_temperature,
        )
        root_values = model.heads.root_value(
            state.value,
            state.opponent_belief,
            route_plan=routes,
        )
        belief_logits = model.belief_logits(state)
    belief_row_losses, belief_valid = normalized_sparse_belief_row_losses(
        belief_logits,
        batch.sparse_belief_targets,
    )
    require_tensor_condition(
        belief_valid.eq(batch.expected_belief_valid).all(),
        "diagnostic belief target validity changed",
    )
    require_tensor_condition(
        evaluation.token_mask.eq(batch.token_mask).all(),
        "diagnostic token mask differs from behavior",
    )
    return StatelessPpoLossInputs(
        current_token_logprobs=evaluation.token_logprobs,
        current_token_entropies=evaluation.token_entropies,
        current_prefix_values=evaluation.prefix_values,
        token_mask=evaluation.token_mask,
        old_token_logprobs=batch.old_token_logprobs,
        old_prefix_values=batch.old_prefix_values,
        token_advantages=batch.token_advantages,
        token_returns=batch.token_returns,
        current_root_values=root_values,
        old_root_values=batch.old_root_values,
        root_returns=batch.root_returns,
        decision_macro_weights=batch.decision_macro_weights,
        belief_row_losses=belief_row_losses,
        belief_valid_mask=belief_valid,
        belief_macro_weights=batch.belief_macro_weights,
    )


def grouped_gradient_grams(
    gradients: Tensor,
    group_slices: Mapping[str, slice],
    *,
    chunk_elements: int = 1_000_000,
) -> Mapping[str, Tensor]:
    """Calculate stable FP64 Gram matrices from compact FP32 gradient rows."""
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradient matrix must contain at least two rows")
    if chunk_elements <= 0:
        raise ValueError("Gram chunk size must be positive")
    result: dict[str, Tensor] = {}
    total = torch.zeros(
        (gradients.shape[0], gradients.shape[0]),
        dtype=torch.float64,
    )
    for group, coordinates in group_slices.items():
        start = int(coordinates.start or 0)
        stop = int(coordinates.stop or gradients.shape[1])
        gram = torch.zeros_like(total)
        for chunk_start in range(start, stop, chunk_elements):
            block = gradients[
                :,
                chunk_start : min(chunk_start + chunk_elements, stop),
            ]
            gram.add_(torch.mm(block, block.transpose(0, 1)).double())
        result[group] = gram
        total.add_(gram)
    result["all_shared"] = total
    return result


def cosine_matrix(gram: Tensor) -> Tensor:
    """Convert a symmetric Gram matrix to cosine similarities."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("cosine conversion requires a square Gram matrix")
    norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
    denominator = torch.outer(norms, norms)
    cosine = torch.zeros_like(gram)
    valid = denominator > 0.0
    cosine[valid] = gram[valid] / denominator[valid]
    return cosine.clamp(min=-1.0, max=1.0)


def cosine_to_other_mean(gram: Tensor) -> tuple[float, ...]:
    """Measure each task against the equally weighted mean of all other tasks."""
    task_count = int(gram.shape[0])
    if gram.shape != (task_count, task_count) or task_count < 2:
        raise ValueError("other-task alignment requires a square multi-task Gram")
    values: list[float] = []
    total_sum = float(gram.sum())
    row_sums = gram.sum(dim=1)
    for index in range(task_count):
        own_norm = math.sqrt(max(float(gram[index, index]), 0.0))
        dot_other = (
            float(row_sums[index]) - float(gram[index, index])
        ) / float(task_count - 1)
        other_square = (
            total_sum
            - 2.0 * float(row_sums[index])
            + float(gram[index, index])
        ) / float((task_count - 1) ** 2)
        denominator = own_norm * math.sqrt(max(other_square, 0.0))
        values.append(0.0 if denominator == 0.0 else dot_other / denominator)
    return tuple(values)


def vector_cosine(left: Tensor, right: Tensor) -> float:
    """Return a bounded cosine for two flat component gradients."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("vector cosine requires equal flat vectors")
    denominator = float(torch.linalg.vector_norm(left)) * float(
        torch.linalg.vector_norm(right)
    )
    if denominator == 0.0:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(torch.dot(left, right)) / denominator),
    )


def _shared_group(name: str) -> str:
    if name.startswith("backbone.input_encoder.card_encoder."):
        return "input_card"
    if name.startswith("backbone.input_encoder.deck_encoder."):
        return "input_deck"
    if (
        name.startswith("backbone.input_encoder.belief_summary_encoder.")
        or name.startswith("backbone.input_encoder.belief_public_projection.")
    ):
        return "input_belief"
    if name.startswith("backbone.input_encoder.raw_state_encoder."):
        return "input_state"
    if name.startswith("backbone.input_encoder."):
        return "input_tokens"
    match = _TRUNK_LAYER_PATTERN.match(name)
    if match is not None:
        layer = int(match.group(1))
        if layer <= 7:
            return "trunk_00_07"
        if layer <= 15:
            return "trunk_08_15"
        if layer <= 23:
            return "trunk_16_23"
        return "trunk_24_33"
    if name.startswith("backbone.trunk."):
        return "trunk_output"
    v2_stage = _V2_STAGE_PATTERN.match(name)
    if v2_stage is not None:
        completed_layers = int(v2_stage.group(1))
        if completed_layers <= 8:
            return "trunk_08_15"
        if completed_layers <= 17:
            return "trunk_16_23"
        if completed_layers < 34:
            return "trunk_24_33"
        return "trunk_output"
    if name.startswith("belief_head."):
        return "belief_head"
    if name.startswith("heads.option_encoder."):
        return "policy_options"
    if (
        name.startswith("heads.shared_root_value.")
        or name.startswith("heads.prefix_value.")
        or name.startswith("heads.value_belief_projection.")
    ):
        return "value_shared"
    if name.startswith("heads."):
        return "policy_decode"
    raise ValueError(f"unclassified shared parameter: {name}")


def _private_parameter_component(
    bank_name: str,
    local_name: str,
) -> str:
    """Classify one exact-route parameter as actor-side or critic-side."""
    if bank_name in {
        "heads.policy_residuals",
        "heads.option_residuals",
    }:
        return "policy"
    if bank_name == "heads.value_residuals":
        return "value"
    if bank_name == "backbone.v2_adapters.prompts":
        return "value" if local_name == "value" else "policy"
    if ".exact_capsules" in bank_name:
        return "value" if local_name.startswith("value_") else "policy"
    raise ValueError(f"unclassified private route bank: {bank_name}")


def _stratum_quotas(total: int, strata: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, strata)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(strata)
    )


@torch.no_grad()
def _copy_parameter_gradients(
    spans: Sequence[ParameterSpan],
    destination: Tensor,
) -> None:
    for span in spans:
        gradient = span.parameter.grad
        target = destination[span.start : span.stop]
        if gradient is None:
            target.zero_()
        else:
            target.copy_(
                gradient.detach().reshape(-1).float().to(
                    device=destination.device
                )
            )


@torch.no_grad()
def _accumulate_flat_gradients(
    gradients: Sequence[Tensor | None],
    spans: Sequence[ParameterSpan],
    destination: Tensor,
) -> None:
    for gradient, span in zip(gradients, spans, strict=True):
        if gradient is not None:
            destination[span.start : span.stop].add_(
                gradient.detach().reshape(-1).float()
            )


def _parameter_gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    square_sum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            square_sum += float(
                torch.sum(parameter.grad.detach().float().square())
            )
    return math.sqrt(square_sum)


__all__ = [
    "DeckGradientObservation",
    "MatchedCellSamplePlan",
    "StatelessGradientLayout",
    "accumulate_stateless_deck_gradient",
    "build_stateless_gradient_layout",
    "cosine_matrix",
    "cosine_to_other_mean",
    "evaluate_stateless_array_microbatch",
    "grouped_gradient_grams",
    "sample_stateless_deck_decisions",
    "sample_stateless_matched_cells",
    "vector_cosine",
]
