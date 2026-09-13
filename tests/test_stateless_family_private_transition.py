"""One-to-many v2 upper-layer cloning into family-private v3."""

from __future__ import annotations

import torch
from torch import nn

from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.decks.registry import (
    deck_expert_registry_fingerprint,
    deck_expert_route,
    deck_family_registry_fingerprint,
    deck_family_route,
)
from ptcg_rl.model.sequence.config import GeneralistSequenceConfig
from ptcg_rl.model.simple_stateless.config import SimpleStatelessModelConfig
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.stateless_family_private_transition import (
    StatelessFamilyPrivateRouteInitialization,
    StatelessFamilyPrivateTransitionDeclaration,
    migrate_family_private_model,
)
from ptcg_rl.rl.stateless_topology_transition import StatelessTopologyRouteClone


class _ToyTrunk(nn.Module):
    """Minimal state-dict-compatible Transformer owner."""

    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(layers))
        self.output_norm = nn.Linear(1, 1)


class _ToyFamilyTail(nn.Module):
    """Minimal cloned/appended family state owner."""

    def __init__(self) -> None:
        super().__init__()
        self.cloned_layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(5))
        self.appended_layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(3))


class _ToySource(nn.Module):
    """State names matching the relevant v2 policy paths."""

    def __init__(self, exact_key: str) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.trunk = _ToyTrunk(20)
        self.heads = nn.Module()
        self.heads.policy_residuals = nn.ModuleDict(
            {exact_key: nn.Linear(1, 1)}
        )


class _ToyTarget(nn.Module):
    """State names matching the v3 shared, family, and exact owners."""

    def __init__(self, exact_key: str, family_key: str) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.trunk = _ToyTrunk(15)
        self.backbone.family_private = nn.Module()
        self.backbone.family_private.generic_upper = nn.ModuleList(
            nn.Linear(1, 1) for _ in range(5)
        )
        self.backbone.family_private.tails = nn.ModuleDict(
            {family_key: _ToyFamilyTail()}
        )
        self.heads = nn.Module()
        self.heads.policy_residuals = nn.ModuleDict(
            {exact_key: nn.Linear(1, 1)}
        )


class _ToyZeroResidual(nn.Module):
    """Minimal exact residual with the production zero-gate naming contract."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(1, 1)
        self.up = nn.Linear(1, 1)


class _ToyTargetWithAddition(_ToyTarget):
    """Target topology containing one cloned and one newly initialized leaf."""

    def __init__(
        self,
        cloned_exact_key: str,
        initialized_exact_key: str,
        family_key: str,
    ) -> None:
        super().__init__(cloned_exact_key, family_key)
        residuals = self.heads.policy_residuals
        assert isinstance(residuals, nn.ModuleDict)
        residuals[initialized_exact_key] = _ToyZeroResidual()


def test_family_private_migration_clones_upper_and_remaps_exact_leaf() -> None:
    """Every inherited tensor is exact-copied and only appended state is fresh."""
    deck = canonicalize_deck((7,) * 60)
    source_route = deck_expert_route(deck, expert_id="a" * 64)
    target_route = deck_expert_route(deck, expert_id="b" * 64)
    family_route = deck_family_route(deck, family_id="c" * 64)
    source_config = SimpleStatelessModelConfig(
        architecture="generalist_sequence_v2",
        num_layers=20,
        exact_routes=(source_route,),
        resolved_registry_sha256=deck_expert_registry_fingerprint((source_route,)),
        sequence=GeneralistSequenceConfig(),
    )
    target_config = SimpleStatelessModelConfig(
        architecture="generalist_sequence_v3",
        num_layers=23,
        exact_routes=(target_route,),
        resolved_registry_sha256=deck_expert_registry_fingerprint((target_route,)),
        family_routes=(family_route,),
        resolved_family_registry_sha256=deck_family_registry_fingerprint(
            (family_route,)
        ),
        sequence=GeneralistSequenceConfig(),
    )
    source = _ToySource(source_route.module_key)
    target = _ToyTarget(target_route.module_key, family_route.module_key)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
        for name, parameter in target.named_parameters():
            if ".appended_layers." in name:
                parameter.zero_()
    source_fingerprint = canonical_model_state_fingerprint(source)
    declaration = StatelessFamilyPrivateTransitionDeclaration(
        source_registry_sha256=source_config.resolved_registry_sha256 or "",
        target_registry_sha256=target_config.resolved_registry_sha256 or "",
        target_family_registry_sha256=(
            target_config.resolved_family_registry_sha256 or ""
        ),
        source_pair_manifest_sha256="d" * 64,
        source_policy_sha256="e" * 64,
        source_learner_state_sha256="f" * 64,
        source_model_config_fingerprint=model_config_fingerprint(source_config),
        source_model_state_fingerprint=source_fingerprint,
        target_model_config_fingerprint=model_config_fingerprint(target_config),
        route_clones=(
            StatelessTopologyRouteClone(
                deck_digest=deck.deck_digest,
                source_expert_id=source_route.expert_id,
                target_expert_id=target_route.expert_id,
            ),
        ),
    )
    inert_names = tuple(
        sorted(
            name
            for name in target.state_dict()
            if ".appended_layers." in name
        )
    )

    result = migrate_family_private_model(
        source_config=source_config,
        source_state=source.state_dict(),
        target_model=target,
        target_config=target_config,
        declaration=declaration,
        inert_output_tensor_names=inert_names,
    )

    state = result.state_dict
    for index in range(5):
        source_prefix = f"backbone.trunk.layers.{15 + index}"
        generic_prefix = f"backbone.family_private.generic_upper.{index}"
        family_prefix = (
            f"backbone.family_private.tails.{family_route.module_key}."
            f"cloned_layers.{index}"
        )
        for suffix in ("weight", "bias"):
            torch.testing.assert_close(
                state[f"{generic_prefix}.{suffix}"],
                source.state_dict()[f"{source_prefix}.{suffix}"],
            )
            torch.testing.assert_close(
                state[f"{family_prefix}.{suffix}"],
                source.state_dict()[f"{source_prefix}.{suffix}"],
            )
    torch.testing.assert_close(
        state[f"heads.policy_residuals.{target_route.module_key}.weight"],
        source.state_dict()[
            f"heads.policy_residuals.{source_route.module_key}.weight"
        ],
    )
    assert all(torch.count_nonzero(state[name]).item() == 0 for name in inert_names)
    assert result.audit.source_state_fingerprint == source_fingerprint
    assert result.audit.initialized_tensor_names == inert_names


def test_family_private_migration_explicitly_zero_initializes_new_exact_leaf() -> None:
    """A target-only deck gets a fresh exact leaf while sharing a family tail."""
    source_deck = canonicalize_deck((7,) * 60)
    added_deck = canonicalize_deck((8,) * 60)
    source_route = deck_expert_route(source_deck, expert_id="a" * 64)
    cloned_route = deck_expert_route(source_deck, expert_id="b" * 64)
    added_route = deck_expert_route(added_deck, expert_id="c" * 64)
    target_routes = tuple(
        sorted((cloned_route, added_route), key=lambda route: route.deck_digest)
    )
    family_routes = tuple(
        sorted(
            (
                deck_family_route(source_deck, family_id="d" * 64),
                deck_family_route(added_deck, family_id="d" * 64),
            ),
            key=lambda route: route.deck_digest,
        )
    )
    source_config = SimpleStatelessModelConfig(
        architecture="generalist_sequence_v2",
        num_layers=20,
        exact_routes=(source_route,),
        resolved_registry_sha256=deck_expert_registry_fingerprint((source_route,)),
        sequence=GeneralistSequenceConfig(),
    )
    target_config = SimpleStatelessModelConfig(
        architecture="generalist_sequence_v3",
        num_layers=23,
        exact_routes=target_routes,
        resolved_registry_sha256=deck_expert_registry_fingerprint(target_routes),
        family_routes=family_routes,
        resolved_family_registry_sha256=deck_family_registry_fingerprint(
            family_routes
        ),
        sequence=GeneralistSequenceConfig(),
    )
    source = _ToySource(source_route.module_key)
    target = _ToyTargetWithAddition(
        cloned_route.module_key,
        added_route.module_key,
        family_routes[0].module_key,
    )
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
        for name, parameter in target.named_parameters():
            if ".appended_layers." in name or (
                f".{added_route.module_key}.up." in name
            ):
                parameter.zero_()
    source_fingerprint = canonical_model_state_fingerprint(source)
    declaration = StatelessFamilyPrivateTransitionDeclaration(
        source_registry_sha256=source_config.resolved_registry_sha256 or "",
        target_registry_sha256=target_config.resolved_registry_sha256 or "",
        target_family_registry_sha256=(
            target_config.resolved_family_registry_sha256 or ""
        ),
        source_pair_manifest_sha256="e" * 64,
        source_policy_sha256="f" * 64,
        source_learner_state_sha256="1" * 64,
        source_model_config_fingerprint=model_config_fingerprint(source_config),
        source_model_state_fingerprint=source_fingerprint,
        target_model_config_fingerprint=model_config_fingerprint(target_config),
        route_clones=(
            StatelessTopologyRouteClone(
                deck_digest=source_deck.deck_digest,
                source_expert_id=source_route.expert_id,
                target_expert_id=cloned_route.expert_id,
            ),
        ),
        route_initializations=(
            StatelessFamilyPrivateRouteInitialization(
                target_deck_digest=added_deck.deck_digest,
                target_expert_id=added_route.expert_id,
            ),
        ),
    )
    family_inert_names = tuple(
        sorted(
            name
            for name in target.state_dict()
            if ".appended_layers." in name
        )
    )

    result = migrate_family_private_model(
        source_config=source_config,
        source_state=source.state_dict(),
        target_model=target,
        target_config=target_config,
        declaration=declaration,
        inert_output_tensor_names=family_inert_names,
    )

    added_prefix = f"heads.policy_residuals.{added_route.module_key}."
    assert result.audit.initialized_exact_target_tensors == 4
    assert result.audit.initialized_appended_target_tensors == 6
    assert all(
        torch.count_nonzero(value).item() == 0
        for name, value in result.state_dict.items()
        if name.startswith(added_prefix) and ".up." in name
    )
