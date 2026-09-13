"""Fail-closed exact-roster transitions for the simple-stateless policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.decks.identity import deck_module_key
from ptcg_rl.decks.registry import DeckExpertRoute
from ptcg_rl.model.simple_stateless import (
    ExactCompositionalCapsule,
    ExactPolicyQueryResidual,
    ExactScalarValueResidual,
    ExactSemanticPrompt,
    ExactWdlValueResidual,
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
)
from ptcg_rl.rl.stateless_deck_balance import (
    DeckSeatHistory,
    DeckTargetShare,
    StatelessDeckBalanceConfig,
    StatelessDeckBalanceState,
    StatelessDynamicDeckBalanceConfig,
    StatelessWeightedDeckBalanceConfig,
)
from ptcg_rl.rl.stateless_private_optimizer import (
    STATELESS_OPTIMIZER_GROUP_ROLE_KEY,
    STATELESS_PRIVATE_GROUP_ROLE,
    STATELESS_SHARED_GROUP_ROLE,
)
from ptcg_rl.rl.stateless_training_config import (
    StatelessExactStrategyInitialization,
    StatelessFamilyStrategyInitialization,
    StatelessFamilyStrategyReassignment,
    StatelessRegistryTransitionConfig,
)

_ROUTE_FIELDS = {
    "exact_routes",
    "family_routes",
    "resolved_family_registry_sha256",
    "resolved_registry_sha256",
}


@dataclass(frozen=True)
class StatelessRegistryTransitionPlan:
    """Validated source-to-target exact strategy lifecycle."""

    source_registry_sha256: str
    target_registry_sha256: str
    continued_routes: tuple[DeckExpertRoute, ...]
    added_routes: tuple[DeckExpertRoute, ...]
    retired_routes: tuple[DeckExpertRoute, ...]
    initializations: tuple[StatelessExactStrategyInitialization, ...]
    source_family_registry_sha256: str | None
    target_family_registry_sha256: str | None
    target_family_ids: tuple[str, ...]
    added_family_ids: tuple[str, ...]
    family_initializations: tuple[StatelessFamilyStrategyInitialization, ...]
    family_reassignments: tuple[StatelessFamilyStrategyReassignment, ...]
    retired_family_ids: tuple[str, ...]

    @property
    def summary(self) -> dict[str, object]:
        """Return path-free transition evidence for the durable source record."""
        added_decks = {route.deck_digest for route in self.added_routes}
        retired_decks = {route.deck_digest for route in self.retired_routes}
        return {
            "source_registry_sha256": self.source_registry_sha256,
            "target_registry_sha256": self.target_registry_sha256,
            "continued_expert_ids": [
                route.expert_id for route in self.continued_routes
            ],
            "initialized": [
                {
                    "deck_digest": item.target_deck_digest,
                    "expert_id": item.target_expert_id,
                    "mode": item.mode,
                    **(
                        {"source_expert_id": item.source_expert_id}
                        if item.source_expert_id is not None
                        else {}
                    ),
                }
                for item in self.initializations
            ],
            "retired_expert_ids": [route.expert_id for route in self.retired_routes],
            "source_family_registry_sha256": (self.source_family_registry_sha256),
            "target_family_registry_sha256": (self.target_family_registry_sha256),
            "initialized_families": [
                {
                    "family_id": item.target_family_id,
                    "mode": item.mode,
                    **(
                        {"source_family_id": item.source_family_id}
                        if item.source_family_id is not None
                        else {}
                    ),
                }
                for item in self.family_initializations
            ],
            "reassigned_families": [
                {
                    "deck_digest": item.deck_digest,
                    "source_family_id": item.source_family_id,
                    "target_family_id": item.target_family_id,
                }
                for item in self.family_reassignments
            ],
            "retired_family_ids": list(self.retired_family_ids),
            "reinitialized_deck_digests": sorted(added_decks & retired_decks),
        }


def build_stateless_registry_transition_plan(
    source: SimpleStatelessModelConfig,
    target: SimpleStatelessModelConfig,
    declaration: StatelessRegistryTransitionConfig,
) -> StatelessRegistryTransitionPlan:
    """Validate an explicit exact-roster edit without implicit route reuse."""
    if (
        source.resolved_registry_sha256 is None
        or target.resolved_registry_sha256 is None
    ):
        raise ValueError("stateless registry transition requires routed registries")
    if declaration.source_registry_sha256 != source.resolved_registry_sha256:
        raise ValueError("stateless transition source registry is not authorized")
    if _model_topology_without_routes(source) != _model_topology_without_routes(target):
        raise ValueError("stateless registry transition changed shared model topology")

    source_by_deck = {route.deck_digest: route for route in source.exact_routes}
    target_by_deck = {route.deck_digest: route for route in target.exact_routes}
    shared_decks = set(source_by_deck) & set(target_by_deck)
    continued = tuple(
        source_by_deck[digest]
        for digest in sorted(shared_decks)
        if source_by_deck[digest] == target_by_deck[digest]
    )
    reinitialized_decks = {
        digest
        for digest in shared_decks
        if source_by_deck[digest] != target_by_deck[digest]
    }
    added_decks = (set(target_by_deck) - set(source_by_deck)) | reinitialized_decks
    retired_decks = (set(source_by_deck) - set(target_by_deck)) | reinitialized_decks
    added = tuple(target_by_deck[digest] for digest in sorted(added_decks))
    retired = tuple(source_by_deck[digest] for digest in sorted(retired_decks))
    declared_additions = {
        (item.target_deck_digest, item.target_expert_id)
        for item in declaration.initializations
    }
    actual_additions = {(route.deck_digest, route.expert_id) for route in added}
    if declared_additions != actual_additions:
        raise ValueError(
            "stateless target-only routes differ from initialization declarations"
        )
    declared_retirements = set(declaration.retired_expert_ids)
    actual_retirements = {route.expert_id for route in retired}
    if declared_retirements != actual_retirements:
        raise ValueError(
            "stateless source-only routes differ from retirement declarations"
        )
    source_experts = {route.expert_id for route in source.exact_routes}
    added_experts = {route.expert_id for route in added}
    if source_experts & added_experts:
        raise ValueError("stateless target-only routes must use new expert lineages")
    target_experts = {route.expert_id for route in target.exact_routes}
    if target_experts & declared_retirements:
        raise ValueError("retired stateless expert remains in the target registry")
    continued_experts = {route.expert_id for route in continued}
    invalid_clone_sources = {
        item.source_expert_id
        for item in declaration.initializations
        if item.mode == "clone"
        and item.source_expert_id is not None
        and item.source_expert_id not in continued_experts
    }
    if invalid_clone_sources:
        raise ValueError(
            "stateless exact clone source must remain in the target registry: "
            f"{sorted(invalid_clone_sources)}"
        )
    (
        source_family_registry_sha256,
        target_family_registry_sha256,
        target_family_ids,
        added_family_ids,
        family_reassignments,
        retired_family_ids,
    ) = _validate_family_registry_transition(
        source,
        target,
        declaration=declaration,
    )
    if not added and not retired and not family_reassignments:
        raise ValueError("stateless registry declaration does not change the roster")

    return StatelessRegistryTransitionPlan(
        source_registry_sha256=source.resolved_registry_sha256,
        target_registry_sha256=target.resolved_registry_sha256,
        continued_routes=continued,
        added_routes=added,
        retired_routes=retired,
        initializations=tuple(
            sorted(
                declaration.initializations,
                key=lambda item: item.target_expert_id,
            )
        ),
        source_family_registry_sha256=source_family_registry_sha256,
        target_family_registry_sha256=target_family_registry_sha256,
        target_family_ids=target_family_ids,
        added_family_ids=added_family_ids,
        family_initializations=tuple(
            sorted(
                declaration.family_initializations,
                key=lambda item: item.target_family_id,
            )
        ),
        family_reassignments=family_reassignments,
        retired_family_ids=retired_family_ids,
    )


def migrate_stateless_registry_model(
    *,
    target: SimpleStatelessPolicyValueNet,
    source_state: Mapping[str, Tensor],
    plan: StatelessRegistryTransitionPlan,
) -> dict[str, int]:
    """Copy retained tensors and initialize each declared private lineage."""
    target_state = target.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    route_private_banks = _validated_route_private_banks(target, plan=plan)
    family_private_banks = _validated_family_private_banks(target, plan=plan)
    expected_missing = _route_private_state_keys(
        route_private_banks,
        routes=plan.added_routes,
    )
    added_family_state = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.added_family_ids
        ),
        parameters_only=False,
    )
    expected_missing |= added_family_state
    unexpected_missing = (target_keys - source_keys) - expected_missing
    expected_extra = _route_private_state_keys(
        route_private_banks,
        routes=plan.retired_routes,
    )
    retired_family_state = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.retired_family_ids
        ),
        parameters_only=False,
    )
    expected_extra |= retired_family_state
    unexpected_extra = (source_keys - target_keys) - expected_extra
    if unexpected_missing or unexpected_extra:
        raise ValueError(
            "stateless registry model inventory mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"extra={sorted(unexpected_extra)}"
        )
    incompatible = target.load_state_dict(dict(source_state), strict=False)
    if set(incompatible.missing_keys) != expected_missing:
        raise ValueError("stateless registry migration loaded unexpected missing keys")
    if set(incompatible.unexpected_keys) != expected_extra:
        raise ValueError("stateless registry migration loaded unexpected source keys")
    initialized_cloned_tensors = _initialize_added_route_private_modules(
        route_private_banks,
        plan=plan,
    )
    _initialize_added_family_private_tails(target, plan=plan)
    return {
        "copied_tensors": len(source_keys - expected_extra),
        "initialized_tensors": len(expected_missing),
        "initialized_cloned_tensors": initialized_cloned_tensors,
        "initialized_family_tensors": len(added_family_state),
        "retired_tensors": len(expected_extra),
        "retired_family_tensors": len(retired_family_state),
    }


def transplant_stateless_registry_optimizer(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: SimpleStatelessPolicyValueNet,
    source_model_config: SimpleStatelessModelConfig,
    source_optimizer_state: Mapping[str, Any],
    plan: StatelessRegistryTransitionPlan,
) -> dict[str, int]:
    """Transplant Adam state by stable parameter name and reset new routes."""
    with torch.random.fork_rng(devices=[]):
        source_model = SimpleStatelessPolicyValueNet(
            source_model_config,
            load_static_features=False,
            initialize=False,
        )
    target_name_groups = _optimizer_parameter_name_groups(optimizer, target_model)
    target_names = tuple(name for group in target_name_groups for name in group)
    route_private_banks = _validated_route_private_banks(target_model, plan=plan)
    family_private_banks = _validated_family_private_banks(
        target_model,
        plan=plan,
    )
    expected_added = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.added_routes,
    )
    expected_added |= _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.added_family_ids
        ),
        parameters_only=True,
    )
    expected_retired = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.retired_routes,
    )
    expected_retired |= _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.retired_family_ids
        ),
        parameters_only=True,
    )
    source_name_set = (set(target_names) - expected_added) | expected_retired
    source_names = tuple(
        name for name, _ in source_model.named_parameters() if name in source_name_set
    )
    if set(source_names) != source_name_set:
        raise ValueError("stateless source optimizer names cannot be reconstructed")
    if len(target_name_groups) == 1:
        return _transplant_stateless_optimizer_state(
            optimizer=optimizer,
            target_model=target_model,
            source_parameter_names=source_names,
            source_optimizer_state=source_optimizer_state,
            plan=plan,
        )
    if len(target_name_groups) == 2:
        source_name_groups = _hybrid_parameter_name_groups(source_model)
        if {name for group in source_name_groups for name in group} != source_name_set:
            raise ValueError(
                "stateless hybrid source optimizer names cannot be reconstructed"
            )
        if target_name_groups != _hybrid_parameter_name_groups(target_model):
            raise ValueError(
                "stateless hybrid target optimizer groups changed ownership"
            )
        return _transplant_stateless_hybrid_optimizer_state(
            optimizer=optimizer,
            target_model=target_model,
            source_parameter_name_groups=source_name_groups,
            source_optimizer_state=source_optimizer_state,
            plan=plan,
        )
    raise ValueError("stateless registry transition has unsupported optimizer groups")


def _transplant_stateless_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: nn.Module,
    source_parameter_names: tuple[str, ...],
    source_optimizer_state: Mapping[str, Any],
    plan: StatelessRegistryTransitionPlan,
) -> dict[str, int]:
    """Apply a name-bound one-group Adam transplant after inventory validation."""
    source_names = source_parameter_names
    target_parameters = dict(target_model.named_parameters())
    target_names = _optimizer_parameter_names(optimizer, target_model)
    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    target_state = optimizer.state_dict()
    target_groups = target_state.get("param_groups")
    if (
        not isinstance(source_groups, list)
        or not isinstance(source_states, Mapping)
        or not isinstance(target_groups, list)
        or len(source_groups) != 1
        or len(target_groups) != 1
    ):
        raise ValueError("stateless registry transition requires one optimizer group")
    source_ids = source_groups[0].get("params")
    target_ids = target_groups[0].get("params")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != len(source_names)
        or not isinstance(target_ids, list)
        or len(target_ids) != len(target_names)
    ):
        raise ValueError("stateless optimizer parameter inventory is malformed")
    source_by_name: dict[str, Mapping[str, Any]] = {}
    for parameter_id, name in zip(source_ids, source_names, strict=True):
        parameter_state = source_states.get(parameter_id, {})
        if not isinstance(parameter_state, Mapping):
            raise ValueError(f"stateless optimizer state is malformed: {name}")
        source_by_name[name] = cast(Mapping[str, Any], parameter_state)
    if set(source_states) - set(source_ids):
        raise ValueError("stateless optimizer has unreferenced parameter state")

    route_private_banks = _validated_route_private_banks(target_model, plan=plan)
    family_private_banks = _validated_family_private_banks(
        target_model,
        plan=plan,
    )
    expected_added = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.added_routes,
    )
    added_family_parameters = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.added_family_ids
        ),
        parameters_only=True,
    )
    expected_added |= added_family_parameters
    expected_retired = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.retired_routes,
    )
    retired_family_parameters = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.retired_family_ids
        ),
        parameters_only=True,
    )
    expected_retired |= retired_family_parameters
    if set(target_names) - set(source_names) != expected_added:
        raise ValueError("stateless optimizer target parameters changed implicitly")
    if set(source_names) - set(target_names) != expected_retired:
        raise ValueError("stateless optimizer source parameters retired implicitly")

    transplanted: dict[int, Any] = {}
    copied = 0
    preserved_without_state = 0
    reset = 0
    for parameter_id, name in zip(target_ids, target_names, strict=True):
        if name in expected_added:
            reset += 1
            continue
        state = source_by_name[name]
        _validate_adam_state(state, target_parameters[name], name=name)
        if state:
            transplanted[int(parameter_id)] = {
                key: value.detach().clone() if isinstance(value, Tensor) else value
                for key, value in state.items()
            }
            copied += 1
        else:
            preserved_without_state += 1
    target_group = dict(source_groups[0])
    target_group["params"] = target_ids
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": [target_group],
        }
    )
    return {
        "copied_parameters": copied,
        "preserved_without_state": preserved_without_state,
        "reset_parameters": reset,
        "initialized_family_parameters": len(added_family_parameters),
        "retired_parameters": len(expected_retired),
        "retired_family_parameters": len(retired_family_parameters),
    }


def _transplant_stateless_hybrid_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: SimpleStatelessPolicyValueNet,
    source_parameter_name_groups: tuple[tuple[str, ...], tuple[str, ...]],
    source_optimizer_state: Mapping[str, Any],
    plan: StatelessRegistryTransitionPlan,
) -> dict[str, int]:
    """Preserve hybrid Adam state by group and reset only new private tails."""
    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != 2
        or not isinstance(source_states, Mapping)
        or not isinstance(target_groups, list)
        or len(target_groups) != 2
        or not isinstance(target_states, Mapping)
        or target_states
    ):
        raise ValueError(
            "stateless hybrid registry transition requires two optimizer groups"
        )
    _require_hybrid_optimizer_group_roles(source_groups, label="source")
    _require_hybrid_optimizer_group_roles(target_groups, label="target")
    target_name_groups = _optimizer_parameter_name_groups(optimizer, target_model)
    if len(target_name_groups) != 2:
        raise ValueError("stateless hybrid optimizer group inventory is malformed")

    source_id_groups: list[list[int]] = []
    target_id_groups: list[list[int]] = []
    for index, (source_names, target_names) in enumerate(
        zip(source_parameter_name_groups, target_name_groups, strict=True)
    ):
        source_ids = source_groups[index].get("params")
        target_ids = target_groups[index].get("params")
        if (
            not isinstance(source_ids, list)
            or len(source_ids) != len(source_names)
            or len(set(source_ids)) != len(source_ids)
            or not isinstance(target_ids, list)
            or len(target_ids) != len(target_names)
            or len(set(target_ids)) != len(target_ids)
        ):
            raise ValueError("stateless hybrid optimizer inventory is malformed")
        source_id_groups.append(source_ids)
        target_id_groups.append(target_ids)
    all_source_ids = {
        parameter_id for group in source_id_groups for parameter_id in group
    }
    all_target_ids = {
        parameter_id for group in target_id_groups for parameter_id in group
    }
    if (
        sum(len(group) for group in source_id_groups) != len(all_source_ids)
        or sum(len(group) for group in target_id_groups) != len(all_target_ids)
        or set(source_states) - all_source_ids
    ):
        raise ValueError("stateless hybrid optimizer parameter IDs overlap")

    route_private_banks = _validated_route_private_banks(target_model, plan=plan)
    family_private_banks = _validated_family_private_banks(
        target_model,
        plan=plan,
    )
    expected_added = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.added_routes,
    )
    added_family_parameters = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.added_family_ids
        ),
        parameters_only=True,
    )
    expected_added |= added_family_parameters
    expected_retired = _route_private_parameter_keys(
        route_private_banks,
        routes=plan.retired_routes,
    )
    retired_family_parameters = _bank_state_keys(
        family_private_banks,
        module_keys=tuple(
            deck_module_key(family_id) for family_id in plan.retired_family_ids
        ),
        parameters_only=True,
    )
    expected_retired |= retired_family_parameters
    source_private, source_shared = map(set, source_parameter_name_groups)
    target_private, target_shared = map(set, target_name_groups)
    if target_private - source_private != expected_added:
        raise ValueError("stateless hybrid private parameters changed implicitly")
    if source_private - target_private != expected_retired:
        raise ValueError("stateless hybrid private parameters retired implicitly")
    if source_shared != target_shared:
        raise ValueError("stateless hybrid shared parameters changed")

    source_by_name: dict[str, Mapping[str, Any]] = {}
    for source_ids, source_names in zip(
        source_id_groups,
        source_parameter_name_groups,
        strict=True,
    ):
        for parameter_id, name in zip(source_ids, source_names, strict=True):
            raw_state = source_states.get(parameter_id, {})
            if not isinstance(raw_state, Mapping):
                raise ValueError(f"stateless optimizer state is malformed: {name}")
            source_by_name[name] = cast(Mapping[str, Any], raw_state)

    target_parameters = dict(target_model.named_parameters())
    transplanted: dict[int, Any] = {}
    copied = 0
    preserved_without_state = 0
    reset = 0
    for target_ids, target_names in zip(
        target_id_groups,
        target_name_groups,
        strict=True,
    ):
        for parameter_id, name in zip(target_ids, target_names, strict=True):
            if name in expected_added:
                reset += 1
                continue
            raw_state = source_by_name[name]
            _validate_adam_state(raw_state, target_parameters[name], name=name)
            if raw_state:
                transplanted[int(parameter_id)] = {
                    key: value.detach().clone() if isinstance(value, Tensor) else value
                    for key, value in raw_state.items()
                }
                copied += 1
            else:
                preserved_without_state += 1

    loaded_groups = [dict(group) for group in source_groups]
    for loaded, target_ids in zip(loaded_groups, target_id_groups, strict=True):
        loaded["params"] = target_ids
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": loaded_groups,
        }
    )
    _require_hybrid_optimizer_group_roles(optimizer.param_groups, label="loaded")
    return {
        "copied_parameters": copied,
        "preserved_without_state": preserved_without_state,
        "reset_parameters": reset,
        "initialized_family_parameters": len(added_family_parameters),
        "retired_parameters": len(expected_retired),
        "retired_family_parameters": len(retired_family_parameters),
        "optimizer_groups": 2,
    }


def _hybrid_parameter_name_groups(
    model: SimpleStatelessPolicyValueNet,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve model-order private/shared names without changing grad flags."""
    private_ids: set[int] = set()
    private_banks = (*model.route_private_banks(), *model.family_private_banks())
    for _bank_name, bank in private_banks:
        for parameter in bank.parameters():
            parameter_id = id(parameter)
            if parameter_id in private_ids:
                raise ValueError("stateless private parameter ownership overlaps")
            private_ids.add(parameter_id)
    named_parameters = tuple(model.named_parameters())
    model_ids = {id(parameter) for _name, parameter in named_parameters}
    if not private_ids or private_ids - model_ids:
        raise ValueError("stateless hybrid private inventory is malformed")
    private_names = tuple(
        name for name, parameter in named_parameters if id(parameter) in private_ids
    )
    shared_names = tuple(
        name for name, parameter in named_parameters if id(parameter) not in private_ids
    )
    if not private_names or not shared_names:
        raise ValueError("stateless hybrid optimizer groups cannot be empty")
    return private_names, shared_names


def _require_hybrid_optimizer_group_roles(
    groups: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    roles = tuple(group.get(STATELESS_OPTIMIZER_GROUP_ROLE_KEY) for group in groups)
    if roles != (STATELESS_PRIVATE_GROUP_ROLE, STATELESS_SHARED_GROUP_ROLE):
        raise ValueError(f"stateless hybrid {label} optimizer roles changed")


def _optimizer_parameter_name_groups(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> tuple[tuple[str, ...], ...]:
    """Resolve every optimizer group's parameters to stable model names."""
    names_by_parameter = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    seen: set[int] = set()
    groups: list[tuple[str, ...]] = []
    for group in optimizer.param_groups:
        raw_parameters = group.get("params")
        if not isinstance(raw_parameters, list):
            raise ValueError("stateless optimizer parameter group is malformed")
        names: list[str] = []
        for parameter in raw_parameters:
            parameter_id = id(parameter)
            if parameter_id in seen or parameter_id not in names_by_parameter:
                raise ValueError("stateless optimizer parameter ownership is ambiguous")
            seen.add(parameter_id)
            names.append(names_by_parameter[parameter_id])
        groups.append(tuple(names))
    return tuple(groups)


def _optimizer_parameter_names(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> tuple[str, ...]:
    """Resolve one optimizer group's parameters to stable model names."""
    groups = _optimizer_parameter_name_groups(optimizer, model)
    if len(groups) != 1:
        raise ValueError("stateless registry transition requires one optimizer group")
    return groups[0]


def _validate_family_registry_transition(
    source: SimpleStatelessModelConfig,
    target: SimpleStatelessModelConfig,
    *,
    declaration: StatelessRegistryTransitionConfig,
) -> tuple[
    str | None,
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[StatelessFamilyStrategyReassignment, ...],
    tuple[str, ...],
]:
    """Authorize explicit family additions, route moves, and retirements."""
    source_routes = source.family_routes
    target_routes = target.family_routes
    if not source_routes and not target_routes:
        if (
            declaration.source_family_registry_sha256 is not None
            or declaration.retired_family_ids
        ):
            raise ValueError(
                "stateless non-family registry transition declared family lifecycle"
            )
        if declaration.family_initializations or declaration.family_reassignments:
            raise ValueError(
                "stateless non-family registry transition changed families"
            )
        return None, None, (), (), (), ()
    if not source_routes or not target_routes:
        raise ValueError("stateless registry transition changed family topology")

    source_registry = source.resolved_family_registry_sha256
    target_registry = target.resolved_family_registry_sha256
    if source_registry is None or target_registry is None:
        raise ValueError("stateless family transition requires routed registries")
    source_by_deck = {route.deck_digest: route for route in source_routes}
    target_by_deck = {route.deck_digest: route for route in target_routes}
    shared_decks = set(source_by_deck) & set(target_by_deck)
    actual_reassignments = {
        (
            digest,
            source_by_deck[digest].family_id,
            target_by_deck[digest].family_id,
        )
        for digest in shared_decks
        if source_by_deck[digest].family_id != target_by_deck[digest].family_id
    }
    declared_reassignments = {
        (item.deck_digest, item.source_family_id, item.target_family_id)
        for item in declaration.family_reassignments
    }
    if declared_reassignments != actual_reassignments:
        raise ValueError(
            "stateless family route moves differ from reassignment declarations"
        )

    source_family_ids = {route.family_id for route in source_routes}
    target_family_ids = {route.family_id for route in target_routes}
    added_family_ids = tuple(sorted(target_family_ids - source_family_ids))
    declared_family_additions = {
        item.target_family_id for item in declaration.family_initializations
    }
    if declared_family_additions != set(added_family_ids):
        raise ValueError(
            "stateless added families differ from initialization declarations"
        )
    continued_family_ids = source_family_ids & target_family_ids
    invalid_clone_sources = {
        item.source_family_id
        for item in declaration.family_initializations
        if item.mode == "clone"
        and item.source_family_id is not None
        and item.source_family_id not in continued_family_ids
    }
    if invalid_clone_sources:
        raise ValueError(
            "stateless family clone source must remain in the target registry: "
            f"{sorted(invalid_clone_sources)}"
        )
    retired_family_ids = tuple(sorted(source_family_ids - target_family_ids))
    family_registry_changed = source_registry != target_registry
    if family_registry_changed:
        if declaration.source_family_registry_sha256 != source_registry:
            raise ValueError("stateless source family registry is not authorized")
        if set(declaration.retired_family_ids) != set(retired_family_ids):
            raise ValueError(
                "stateless retired families differ from retirement declarations"
            )
    elif (
        declaration.source_family_registry_sha256 is not None
        or declaration.family_initializations
        or declaration.family_reassignments
        or declaration.retired_family_ids
    ):
        raise ValueError("stateless family transition declaration is unnecessary")
    return (
        source_registry,
        target_registry,
        tuple(sorted(target_family_ids)),
        added_family_ids,
        tuple(
            sorted(
                declaration.family_reassignments,
                key=lambda item: item.deck_digest,
            )
        ),
        retired_family_ids,
    )


def migrate_stateless_deck_balance_state(
    source: StatelessDeckBalanceState,
    *,
    source_config: (
        StatelessDeckBalanceConfig
        | StatelessDynamicDeckBalanceConfig
        | StatelessWeightedDeckBalanceConfig
    ),
    target_config: (
        StatelessDeckBalanceConfig
        | StatelessDynamicDeckBalanceConfig
        | StatelessWeightedDeckBalanceConfig
    ),
    plan: StatelessRegistryTransitionPlan,
) -> StatelessDeckBalanceState:
    """Preserve settled balance history and add fresh cells for new decks."""
    if source.config_fingerprint != source_config.fingerprint:
        raise ValueError("stateless deck-balance source fingerprint mismatch")
    if source.inflight_assignments:
        raise ValueError("stateless deck-balance transition has in-flight assignments")
    roster_field = (
        "target_deck_shares"
        if isinstance(source_config, StatelessWeightedDeckBalanceConfig)
        else "active_deck_digests"
    )
    source_settings = source_config.model_dump(
        mode="python",
        exclude={roster_field},
    )
    target_settings = target_config.model_dump(
        mode="python",
        exclude={roster_field},
    )
    if (
        type(source_config) is not type(target_config)
        or source_settings != target_settings
    ):
        raise ValueError("stateless deck-balance transition changed scheduler settings")
    source_decks = set(source_config.active_deck_digests)
    target_decks = set(target_config.active_deck_digests)
    added_decks = {
        route.deck_digest
        for route in plan.added_routes
        if route.deck_digest not in source_decks
    }
    retired_decks = {
        route.deck_digest
        for route in plan.retired_routes
        if route.deck_digest not in target_decks
    }
    if target_decks - source_decks != added_decks:
        raise ValueError("stateless deck-balance additions differ from registry plan")
    if source_decks - target_decks != retired_decks:
        raise ValueError("stateless deck-balance retirements differ from registry plan")
    retained_history = tuple(
        item for item in source.history if item.deck_digest in target_decks
    )
    new_history = tuple(
        DeckSeatHistory(
            deck_digest=deck_digest,
            seat=seat,
            last_finish_cursor=source.decision_cursor,
        )
        for deck_digest in target_config.active_deck_digests
        if deck_digest in added_decks
        for seat in (0, 1)
    )
    retained_events = tuple(
        event for event in source.rolling_events if event.deck_digest in target_decks
    )
    retained_terminal_events = tuple(
        event
        for event in source.terminal_score_events
        if event.deck_digest in target_decks
    )
    migrated = source.model_copy(
        update={
            "config_fingerprint": target_config.fingerprint,
            "rolling_events": retained_events,
            "terminal_score_events": retained_terminal_events,
            "history": retained_history + new_history,
        }
    )
    expected_cells = {
        (deck, seat) for deck in target_config.active_deck_digests for seat in (0, 1)
    }
    actual_cells = {(item.deck_digest, item.seat) for item in migrated.history}
    if actual_cells != expected_cells:
        raise ValueError("stateless migrated deck-balance history is incomplete")
    return migrated


def source_weighted_balance_for_registry_transition(
    target: StatelessWeightedDeckBalanceConfig,
    *,
    source_deck_digests: tuple[str, ...],
    source_target_deck_shares: tuple[DeckTargetShare, ...] = (),
    expected_source_config_fingerprint: str | None = None,
) -> StatelessWeightedDeckBalanceConfig:
    """Reconstruct or verify the prior weighted deck-balance configuration."""
    source_decks = set(source_deck_digests)
    has_explicit_shares = bool(source_target_deck_shares)
    has_expected_fingerprint = expected_source_config_fingerprint is not None
    if has_explicit_shares != has_expected_fingerprint:
        raise ValueError(
            "weighted registry source balance requires both explicit shares and "
            "its fingerprint"
        )
    if has_explicit_shares:
        source = StatelessWeightedDeckBalanceConfig.model_validate(
            {
                **target.model_dump(mode="python"),
                "target_deck_shares": source_target_deck_shares,
            }
        )
        if set(source.active_deck_digests) != source_decks:
            raise ValueError(
                "weighted registry source balance differs from the source roster"
            )
        if source.fingerprint != expected_source_config_fingerprint:
            raise ValueError("weighted registry source balance fingerprint mismatch")
        return source
    target_decks = set(target.active_deck_digests)
    added = target_decks - source_decks
    retired = source_decks - target_decks
    if (
        not added
        or len(source_decks) != len(target_decks)
        or len(added) != len(retired)
    ):
        raise ValueError(
            "weighted registry migration requires an equal-size non-empty roster edit"
        )
    target_shares = {
        item.deck_digest: item.target_share for item in target.target_deck_shares
    }
    added_shares = {target_shares[deck_digest] for deck_digest in added}
    if len(added_shares) != 1:
        raise ValueError(
            "weighted registry additions must share one auxiliary allocation"
        )
    auxiliary_share = next(iter(added_shares))
    source_shares = tuple(
        DeckTargetShare(
            deck_digest=deck_digest,
            target_share=target_shares.get(deck_digest, auxiliary_share),
        )
        for deck_digest in sorted(source_decks)
    )
    return target.model_copy(update={"target_deck_shares": source_shares})


def _model_topology_without_routes(
    config: SimpleStatelessModelConfig,
) -> dict[str, object]:
    return config.model_dump(mode="python", exclude=_ROUTE_FIELDS)


def _validated_route_private_banks(
    model: nn.Module,
    *,
    plan: StatelessRegistryTransitionPlan,
) -> tuple[tuple[str, nn.ModuleDict], ...]:
    """Validate the model-declared route-keyed parameter banks."""
    enumerate_banks = getattr(model, "route_private_banks", None)
    if not callable(enumerate_banks):
        raise ValueError(
            "stateless registry model does not enumerate route-private banks"
        )
    bank_items = enumerate_banks()
    if not isinstance(bank_items, tuple) or not bank_items:
        raise ValueError("stateless registry model has no route-private banks")

    target_module_keys = {
        route.module_key for route in (*plan.continued_routes, *plan.added_routes)
    }
    registered_modules = dict(model.named_modules())
    banks: list[tuple[str, nn.ModuleDict]] = []
    for item in bank_items:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], nn.ModuleDict)
        ):
            raise ValueError("stateless route-private bank declaration is malformed")
        bank_name = item[0]
        bank = item[1]
        if (
            not bank_name
            or any(name == bank_name for name, _bank in banks)
            or registered_modules.get(bank_name) is not bank
        ):
            raise ValueError(
                "stateless route-private bank is not uniquely registered: "
                f"{bank_name!r}"
            )
        if set(bank) != target_module_keys:
            raise ValueError(
                "stateless route-private bank has incomplete target routes: "
                f"{bank_name}"
            )
        banks.append((bank_name, bank))
    return tuple(banks)


def _validated_family_private_banks(
    model: nn.Module,
    *,
    plan: StatelessRegistryTransitionPlan,
) -> tuple[tuple[str, nn.ModuleDict], ...]:
    """Validate separately declared family-keyed parameter banks."""
    if plan.target_family_registry_sha256 is None:
        return ()
    enumerate_banks = getattr(model, "family_private_banks", None)
    if not callable(enumerate_banks):
        raise ValueError(
            "stateless family registry model does not enumerate family banks"
        )
    bank_items = enumerate_banks()
    if not isinstance(bank_items, tuple) or not bank_items:
        raise ValueError("stateless family registry model has no family banks")

    target_module_keys = {
        deck_module_key(family_id) for family_id in plan.target_family_ids
    }
    registered_modules = dict(model.named_modules())
    banks: list[tuple[str, nn.ModuleDict]] = []
    for item in bank_items:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], nn.ModuleDict)
        ):
            raise ValueError("stateless family bank declaration is malformed")
        bank_name = item[0]
        bank = item[1]
        if (
            not bank_name
            or any(name == bank_name for name, _bank in banks)
            or registered_modules.get(bank_name) is not bank
        ):
            raise ValueError(
                f"stateless family bank is not uniquely registered: {bank_name!r}"
            )
        if set(bank) != target_module_keys:
            raise ValueError(
                f"stateless family bank has incomplete target families: {bank_name}"
            )
        banks.append((bank_name, bank))
    return tuple(banks)


def _initialize_added_family_private_tails(
    model: nn.Module,
    *,
    plan: StatelessRegistryTransitionPlan,
) -> None:
    """Initialize every declared new family from its bound source path."""
    if not plan.added_family_ids:
        return
    for bank_name, _bank in _validated_family_private_banks(model, plan=plan):
        owner_name, separator, _leaf_name = bank_name.rpartition(".")
        if not separator:
            raise ValueError(
                f"stateless family bank has no registered owner: {bank_name}"
            )
        owner = model.get_submodule(owner_name)
        for declaration in plan.family_initializations:
            target_key = deck_module_key(declaration.target_family_id)
            if declaration.mode == "clone_generic":
                initialize_generic = getattr(
                    owner,
                    "initialize_tail_from_generic",
                    None,
                )
                if not callable(initialize_generic):
                    raise ValueError(
                        "stateless family bank does not expose generic-clone "
                        f"initialization: {bank_name}"
                    )
                initialize_generic(target_key)
                continue
            initialize_family = getattr(
                owner,
                "initialize_tail_from_family",
                None,
            )
            if not callable(initialize_family):
                raise ValueError(
                    "stateless family bank does not expose learned-family clone "
                    f"initialization: {bank_name}"
                )
            source_family_id = declaration.source_family_id
            if source_family_id is None:
                raise RuntimeError("validated family clone has no source lineage")
            initialize_family(
                target_key,
                source_module_key=deck_module_key(source_family_id),
            )


def _initialize_added_route_private_modules(
    banks: tuple[tuple[str, nn.ModuleDict], ...],
    *,
    plan: StatelessRegistryTransitionPlan,
) -> int:
    """Initialize each new exact route from zero or a retained donor."""
    added_by_expert = {route.expert_id: route for route in plan.added_routes}
    continued_by_expert = {
        route.expert_id: route for route in plan.continued_routes
    }
    cloned_tensors = 0
    for declaration in plan.initializations:
        route = added_by_expert[declaration.target_expert_id]
        if declaration.mode == "clone":
            source_expert_id = declaration.source_expert_id
            if source_expert_id is None:
                raise RuntimeError("validated exact clone has no source lineage")
            source_route = continued_by_expert[source_expert_id]
            for _bank_name, bank in banks:
                source_state = bank[source_route.module_key].state_dict()
                bank[route.module_key].load_state_dict(source_state, strict=True)
                cloned_tensors += len(source_state)
            continue
        for bank_name, bank in banks:
            module = bank[route.module_key]
            zero_parameters = _route_private_zero_parameters(module)
            if not zero_parameters:
                raise ValueError(
                    "stateless route-private module has no zero-init contract: "
                    f"{bank_name}.{route.module_key}"
                )
            for relative_name, parameter in zero_parameters:
                name = f"{bank_name}.{route.module_key}.{relative_name}"
                if parameter.is_meta:
                    raise ValueError(
                        "cannot validate stateless route-private initialization "
                        f"on meta device: {name}"
                    )
                if torch.count_nonzero(parameter).item() != 0:
                    raise ValueError(
                        "new stateless exact route private initialization "
                        f"is not zero: {name}"
                    )
    return cloned_tensors


def _route_private_zero_parameters(
    module: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Enumerate direct output gates and neutral route-private mixture logits."""
    if isinstance(module, (ExactScalarValueResidual, ExactWdlValueResidual)):
        return tuple(
            (f"output.{name}", parameter)
            for name, parameter in module.output.named_parameters()
        )
    if isinstance(module, ExactPolicyQueryResidual):
        return tuple(
            (f"up.{name}", parameter)
            for name, parameter in module.up.named_parameters()
        )
    if isinstance(module, ExactSemanticPrompt):
        return tuple(module.named_parameters())
    if isinstance(module, ExactCompositionalCapsule):
        return tuple(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if name.endswith(
                (
                    "_coefficients",
                    ".up.weight",
                    ".up.bias",
                )
            )
        )
    raise ValueError(
        f"unsupported stateless route-private module type: {type(module).__qualname__}"
    )


def _route_private_state_keys(
    banks: tuple[tuple[str, nn.ModuleDict], ...],
    *,
    routes: tuple[DeckExpertRoute, ...],
) -> set[str]:
    """Build the exact state inventory for declared routes from target modules."""
    return _bank_state_keys(
        banks,
        module_keys=tuple(route.module_key for route in routes),
        parameters_only=False,
    )


def _route_private_parameter_keys(
    banks: tuple[tuple[str, nn.ModuleDict], ...],
    *,
    routes: tuple[DeckExpertRoute, ...],
) -> set[str]:
    """Build the exact optimizer inventory for declared routes."""
    return _bank_state_keys(
        banks,
        module_keys=tuple(route.module_key for route in routes),
        parameters_only=True,
    )


def _bank_state_keys(
    banks: tuple[tuple[str, nn.ModuleDict], ...],
    *,
    module_keys: tuple[str, ...],
    parameters_only: bool,
) -> set[str]:
    """Build an exact state inventory for uniform module-keyed banks."""
    return {
        f"{bank_name}.{module_key}.{suffix}"
        for bank_name, bank in banks
        for suffix in _uniform_route_suffixes(
            bank_name,
            bank,
            parameters_only=parameters_only,
        )
        for module_key in module_keys
    }


def _uniform_route_suffixes(
    bank_name: str,
    bank: nn.ModuleDict,
    *,
    parameters_only: bool,
) -> tuple[str, ...]:
    """Require one exact per-route module inventory throughout a bank."""
    inventories = {
        tuple(
            sorted(
                name
                for name, _value in (
                    module.named_parameters()
                    if parameters_only
                    else module.state_dict().items()
                )
            )
        )
        for module in bank.values()
    }
    if len(inventories) != 1:
        raise ValueError(
            "stateless route-private bank modules have inconsistent inventory: "
            f"{bank_name}"
        )
    suffixes = next(iter(inventories), ())
    if not suffixes:
        inventory_kind = "parameter" if parameters_only else "state"
        raise ValueError(
            f"stateless route-private bank has empty {inventory_kind} inventory: "
            f"{bank_name}"
        )
    return suffixes


def _validate_adam_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
    *,
    name: str,
) -> None:
    if not state:
        return
    required = {"step", "exp_avg", "exp_avg_sq"}
    if required - set(state):
        raise ValueError(f"stateless optimizer state is incomplete: {name}")
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is not None and (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != parameter.shape
        ):
            raise ValueError(
                f"stateless optimizer moment is incompatible: {name}:{key}"
            )
    step = state["step"]
    if isinstance(step, Tensor) and step.numel() != 1:
        raise ValueError(f"stateless optimizer step is malformed: {name}")


__all__ = [
    "StatelessRegistryTransitionPlan",
    "build_stateless_registry_transition_plan",
    "migrate_stateless_deck_balance_state",
    "migrate_stateless_registry_model",
    "transplant_stateless_registry_optimizer",
]
