"""Fail-closed parameter ownership and artifact audits for replay pretraining."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.decks.registry import DeckExpertRoute
from ptcg_rl.model.simple_stateless import (
    SimpleExactRoutePlan,
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
    resolve_simple_pretraining_routes,
    uses_exact_v2_topology,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    PrivateParameterBankTrainingRecord,
    PrivateResidualTrainingRecord,
    TrainableParameterScopeAudit,
)
from ptcg_rl.training.simple_stateless_pretrain_config import (
    ReplayPretrainingTrainableScopeConfig,
)
from ptcg_rl.training.simple_stateless_pretrain_data import (
    ReplayPretrainingDatasetManifest,
    load_pretraining_part,
)


@dataclass(frozen=True)
class TrainableParameterScope:
    """Resolved model tensors owned by one supervised optimizer."""

    mode: Literal["full_model", "exact_actor_private_v2"]
    target_deck_digest: str | None
    target_expert_id: str | None
    target_module_key: str | None
    parameter_names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    parameter_elements: int
    frozen_state_names: tuple[str, ...]


def resolve_trainable_parameter_scope(
    model: SimpleStatelessPolicyValueNet,
    config: ReplayPretrainingTrainableScopeConfig,
) -> TrainableParameterScope:
    """Freeze the model, then enable only the validated optimizer whitelist."""
    named_parameters = dict(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("pretraining model has no parameters")
    target: DeckExpertRoute | None = None
    if config.mode == "full_model":
        selected_names = tuple(sorted(named_parameters))
    else:
        if not uses_exact_v2_topology(model.config):
            raise ValueError("exact actor-private scope requires V2 exact routes")
        target = target_route(model.config, config.target_deck_digest)
        route_banks = private_bank_parameter_names(model)[target.module_key]
        required_actor_banks = {
            "heads.policy_residuals",
            "heads.option_residuals",
            "backbone.v2_adapters.prompts",
            *(
                bank_name
                for bank_name in route_banks
                if bank_name.startswith("backbone.v2_adapters.stages.")
            ),
        }
        missing = required_actor_banks.difference(route_banks)
        if missing:
            raise RuntimeError(
                f"v2 actor-private scope is missing physical banks: {sorted(missing)}"
            )
        selected_by_bank = {
            bank_name: tuple(
                name for name in names if is_actor_private_trainable_parameter(name)
            )
            for bank_name, names in route_banks.items()
        }
        empty_required = tuple(
            sorted(
                bank_name
                for bank_name in required_actor_banks
                if not selected_by_bank[bank_name]
            )
        )
        if empty_required:
            raise RuntimeError(
                f"v2 actor-private bank matched no trainable tensors: {empty_required}"
            )
        selected_names = tuple(
            sorted(name for names in selected_by_bank.values() for name in names)
        )
    if not selected_names or len(selected_names) != len(set(selected_names)):
        raise RuntimeError("pretraining parameter whitelist is empty or ambiguous")
    missing_parameters = tuple(
        name for name in selected_names if name not in named_parameters
    )
    if missing_parameters:
        raise RuntimeError(
            f"pretraining whitelist references non-parameters: {missing_parameters}"
        )
    selected = frozenset(selected_names)
    for name, parameter in named_parameters.items():
        parameter.requires_grad_(name in selected)
    enabled = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if enabled != selected_names:
        raise RuntimeError("model trainable parameters differ from the whitelist")
    frozen_state_names = tuple(
        name for name in sorted(model.state_dict()) if name not in selected
    )
    parameters = tuple(named_parameters[name] for name in selected_names)
    return TrainableParameterScope(
        mode=config.mode,
        target_deck_digest=None if target is None else target.deck_digest,
        target_expert_id=None if target is None else target.expert_id,
        target_module_key=None if target is None else target.module_key,
        parameter_names=selected_names,
        parameters=parameters,
        parameter_elements=sum(parameter.numel() for parameter in parameters),
        frozen_state_names=frozen_state_names,
    )


def target_route(
    model_config: SimpleStatelessModelConfig,
    target_deck_digest: str | None,
) -> DeckExpertRoute:
    """Resolve exactly one configured deck identity from the active registry."""
    if target_deck_digest is None:
        raise ValueError("exact actor-private scope is missing its target deck")
    matches = tuple(
        route
        for route in model_config.exact_routes
        if route.deck_digest == target_deck_digest
    )
    if len(matches) != 1:
        raise ValueError(
            "target deck digest does not resolve exactly once in the active registry"
        )
    return matches[0]


def validate_target_dataset(
    dataset: ReplayPretrainingDatasetManifest,
    *,
    dataset_dir: Path,
    model: SimpleStatelessPolicyValueNet,
    trainable_scope: TrainableParameterScope,
) -> None:
    """Scan compact shards before the first update and reject foreign own decks."""
    if trainable_scope.target_deck_digest is None:
        return
    if dataset.target_deck_digest != trainable_scope.target_deck_digest:
        raise ValueError("exact actor-private dataset target identity changed")
    route = target_route(model.config, trainable_scope.target_deck_digest)
    expected = np.asarray(route.canonical_card_ids, dtype=np.int64)
    for record in dataset.parts:
        part = load_pretraining_part(dataset_dir / "parts" / record.filename)
        own_decks = np.asarray(part.arrays["own_decks"], dtype=np.int64)
        matching = np.all(own_decks == expected, axis=1)
        if not bool(np.all(matching)):
            foreign_rows = int(np.count_nonzero(~matching))
            raise ValueError(
                "exact actor-private dataset contains foreign own-deck rows: "
                f"part={record.filename}, rows={foreign_rows}"
            )


def resolve_pretraining_batch_routes(
    deck_signatures: Sequence[str],
    model_config: SimpleStatelessModelConfig,
    *,
    device: torch.device,
    trainable_scope: TrainableParameterScope,
) -> SimpleExactRoutePlan:
    """Use exact online routing for target BC and legacy offline routing otherwise."""
    if trainable_scope.target_deck_digest is None:
        return resolve_simple_pretraining_routes(
            deck_signatures,
            model_config,
            device=device,
        )
    route = target_route(model_config, trainable_scope.target_deck_digest)
    if any(signature != route.signature for signature in deck_signatures):
        raise ValueError("exact actor-private batch contains a foreign deck")
    plan = resolve_simple_exact_routes(
        deck_signatures,
        model_config,
        device=device,
    )
    if plan.module_keys != (route.module_key,):
        raise RuntimeError("exact actor-private route differs from configured target")
    return plan


def private_parameter_state(
    model: SimpleStatelessPolicyValueNet,
) -> dict[str, Tensor]:
    """Snapshot every route-private physical bank without the shared model."""
    names = {
        name
        for route_banks in private_bank_parameter_names(model).values()
        for bank_names in route_banks.values()
        for name in bank_names
    }
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in sorted(names)}


def private_training_records(
    model: SimpleStatelessPolicyValueNet,
    *,
    initial_private_state: Mapping[str, Tensor],
    route_examples: Counter[str],
    trainable_parameter_names: frozenset[str],
) -> tuple[PrivateResidualTrainingRecord, ...]:
    """Report movement for every physical private bank in every exact route."""
    current = model.state_dict()
    bank_names = private_bank_parameter_names(model)
    private_names = {
        name
        for route_banks in bank_names.values()
        for names in route_banks.values()
        for name in names
    }
    unclassified = tuple(
        sorted(
            name
            for name in private_names
            if is_policy_private_parameter(name) == is_value_private_parameter(name)
        )
    )
    if unclassified:
        raise RuntimeError(
            f"private parameter policy/value ownership is ambiguous: {unclassified}"
        )
    records: list[PrivateResidualTrainingRecord] = []
    for route in model.config.exact_routes:
        route_banks = bank_names[route.module_key]
        records.append(
            PrivateResidualTrainingRecord(
                deck_digest=route.deck_digest,
                expert_id=route.expert_id,
                matched_examples=route_examples[route.deck_digest],
                policy_parameter_delta_l2=parameter_delta_l2(
                    current,
                    initial_private_state,
                    names=tuple(
                        name
                        for names in route_banks.values()
                        for name in names
                        if is_policy_private_parameter(name)
                    ),
                ),
                value_parameter_delta_l2=parameter_delta_l2(
                    current,
                    initial_private_state,
                    names=tuple(
                        name
                        for names in route_banks.values()
                        for name in names
                        if is_value_private_parameter(name)
                    ),
                ),
                parameter_banks=tuple(
                    _private_bank_training_record(
                        bank_name,
                        names=names,
                        current=current,
                        initial=initial_private_state,
                        trainable_parameter_names=trainable_parameter_names,
                    )
                    for bank_name, names in route_banks.items()
                ),
            )
        )
    return tuple(records)


def private_bank_parameter_names(
    model: SimpleStatelessPolicyValueNet,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Inventory every physical route-private state tensor by bank and expert."""
    route_module_keys = tuple(route.module_key for route in model.config.exact_routes)
    inventory: dict[str, dict[str, tuple[str, ...]]] = {
        module_key: {} for module_key in route_module_keys
    }
    model_state_names = frozenset(model.state_dict())
    for bank_name, bank in model.route_private_banks():
        if tuple(sorted(bank)) != tuple(sorted(route_module_keys)):
            raise RuntimeError(
                f"private bank {bank_name} differs from the exact registry"
            )
        for module_key in route_module_keys:
            module = bank[module_key]
            names = tuple(
                sorted(
                    f"{bank_name}.{module_key}.{local_name}"
                    for local_name in module.state_dict()
                )
            )
            if not names or any(name not in model_state_names for name in names):
                raise RuntimeError(
                    f"private bank {bank_name} has incomplete model state"
                )
            inventory[module_key][bank_name] = names
    if route_module_keys and any(not banks for banks in inventory.values()):
        raise RuntimeError("exact route has no physical private parameter bank")
    return inventory


def is_actor_private_trainable_parameter(name: str) -> bool:
    """Select only the target V2 prompt, residual, and option actor paths."""
    if name.startswith(("heads.policy_residuals.", "heads.option_residuals.")):
        return True
    if name.startswith("backbone.v2_adapters.prompts."):
        return name.endswith(".policy_and_scratch")
    if ".exact_capsules." in name:
        return ".policy_residual." in name
    return False


def is_policy_private_parameter(name: str) -> bool:
    """Classify every route-private actor tensor for aggregate reporting."""
    return is_actor_private_trainable_parameter(name) or (
        ".exact_capsules." in name and name.endswith(".policy_coefficients")
    )


def is_value_private_parameter(name: str) -> bool:
    """Classify every route-private critic tensor for aggregate reporting."""
    if name.startswith("heads.value_residuals."):
        return True
    if name.startswith("backbone.v2_adapters.prompts."):
        return name.endswith(".value")
    if ".exact_capsules." in name:
        return ".value_residual." in name or name.endswith(".value_coefficients")
    return False


def parameter_delta_l2(
    current: Mapping[str, Tensor],
    initial: Mapping[str, Tensor],
    *,
    names: Sequence[str],
) -> float:
    """Return the joint L2 movement over one named parameter subset."""
    if not names:
        raise RuntimeError("private parameter selection matched no tensors")
    total = 0.0
    for name in names:
        before = initial[name]
        after = current[name].detach().cpu()
        total += float(torch.square(after.float() - before.float()).sum())
    return math.sqrt(total)


def state_subset_fingerprint(
    state: Mapping[str, Tensor],
    names: Sequence[str],
) -> str:
    """Hash one named state subset, including a stable empty-set identity."""
    selected = {name: state[name] for name in names}
    if selected:
        return canonical_model_state_fingerprint(selected)
    return hashlib.sha256(
        b"ptcg-rl/simple-stateless-pretraining-empty-state/v1\x00"
    ).hexdigest()


def assert_frozen_state_unchanged(
    model: SimpleStatelessPolicyValueNet,
    *,
    trainable_scope: TrainableParameterScope,
    initial_fingerprint: str,
) -> None:
    """Reject resumed or final state whose optimizer-external tensors moved."""
    current = state_subset_fingerprint(
        model.state_dict(),
        trainable_scope.frozen_state_names,
    )
    if current != initial_fingerprint:
        raise RuntimeError("frozen model state changed during pretraining")


def trainable_scope_audit(
    model: SimpleStatelessPolicyValueNet,
    *,
    trainable_scope: TrainableParameterScope,
    frozen_initial_fingerprint: str,
) -> TrainableParameterScopeAudit:
    """Materialize the immutable optimizer whitelist and frozen-state proof."""
    frozen_final_fingerprint = state_subset_fingerprint(
        model.state_dict(),
        trainable_scope.frozen_state_names,
    )
    return TrainableParameterScopeAudit(
        mode=trainable_scope.mode,
        target_deck_digest=trainable_scope.target_deck_digest,
        target_expert_id=trainable_scope.target_expert_id,
        trainable_parameter_names=trainable_scope.parameter_names,
        trainable_tensor_count=len(trainable_scope.parameter_names),
        trainable_parameter_elements=trainable_scope.parameter_elements,
        frozen_tensor_count=len(trainable_scope.frozen_state_names),
        frozen_initial_fingerprint=frozen_initial_fingerprint,
        frozen_final_fingerprint=frozen_final_fingerprint,
        frozen_bitwise_identical=(
            frozen_initial_fingerprint == frozen_final_fingerprint
        ),
    )


def _private_bank_training_record(
    bank_name: str,
    *,
    names: Sequence[str],
    current: Mapping[str, Tensor],
    initial: Mapping[str, Tensor],
    trainable_parameter_names: frozenset[str],
) -> PrivateParameterBankTrainingRecord:
    """Build exact per-bank movement evidence for one physical expert."""
    initial_bank = {name: initial[name] for name in names}
    final_bank = {name: current[name].detach().cpu() for name in names}
    changed = sum(
        not _tensors_bitwise_equal(initial_bank[name], final_bank[name])
        for name in names
    )
    trainable_names = tuple(name for name in names if name in trainable_parameter_names)
    return PrivateParameterBankTrainingRecord(
        bank_name=bank_name,
        tensor_count=len(names),
        parameter_elements=sum(initial[name].numel() for name in names),
        trainable_tensor_count=len(trainable_names),
        trainable_parameter_elements=sum(
            initial[name].numel() for name in trainable_names
        ),
        changed_tensor_count=changed,
        parameter_delta_l2=parameter_delta_l2(
            current,
            initial,
            names=names,
        ),
        initial_fingerprint=canonical_model_state_fingerprint(initial_bank),
        final_fingerprint=canonical_model_state_fingerprint(final_bank),
    )


def _tensors_bitwise_equal(first: Tensor, second: Tensor) -> bool:
    """Compare tensor storage bytes, preserving signed zero and NaN payloads."""
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    return bool(
        torch.equal(
            first.detach().contiguous().view(torch.uint8).cpu(),
            second.detach().contiguous().view(torch.uint8).cpu(),
        )
    )


__all__ = [
    "TrainableParameterScope",
    "assert_frozen_state_unchanged",
    "private_parameter_state",
    "private_training_records",
    "resolve_pretraining_batch_routes",
    "resolve_trainable_parameter_scope",
    "state_subset_fingerprint",
    "trainable_scope_audit",
    "validate_target_dataset",
]
