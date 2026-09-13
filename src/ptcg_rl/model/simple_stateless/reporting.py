"""Generated topology and parameter ownership reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from torch import nn

from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_V2_ARCHITECTURE,
    GENERALIST_SEQUENCE_V3_ARCHITECTURE,
)
from ptcg_rl.model.simple_stateless.config import SimpleStatelessModelConfig
from ptcg_rl.model.simple_stateless.family_private import (
    FAMILY_PRIVATE_APPENDED_LAYERS,
    FAMILY_PRIVATE_CLONED_LAYERS,
    FAMILY_PRIVATE_SHARED_LAYERS,
    FamilyPrivateStrategyBank,
)
from ptcg_rl.model.simple_stateless.layers import PackedTransformerTrunk


def simple_stateless_parameter_report(
    model: nn.Module,
    config: SimpleStatelessModelConfig,
) -> dict[str, Any]:
    """Report real registered parameters, grouped without double counting."""
    try:
        trunk = model.get_submodule("trunk")
    except AttributeError:
        trunk = model.get_submodule("backbone.trunk")
    if not isinstance(trunk, PackedTransformerTrunk):
        raise ValueError("model does not contain a packed Transformer trunk")
    components: defaultdict[str, int] = defaultdict(int)
    seen: set[int] = set()
    duplicate_parameter_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen:
            duplicate_parameter_names.append(name)
            continue
        seen.add(identity)
        components[name.split(".", maxsplit=1)[0]] += parameter.numel()

    state_names = tuple(model.state_dict().keys())
    card_parameter_names = tuple(
        name
        for name in state_names
        if ".card_encoder." in f".{name}." and not name.endswith("static_features")
    )
    card_owner_prefixes = sorted(
        {
            name.split("card_encoder.", maxsplit=1)[0] + "card_encoder"
            for name in card_parameter_names
        }
    )
    topology: dict[str, Any] = {
        "shared_transformer_blocks": len(trunk.layers),
        "attention_heads": config.attention_heads,
        "feedforward_dim": config.feedforward_dim,
        "scratch_tokens": config.scratch_tokens,
    }
    try:
        family_private = model.get_submodule("family_private")
    except AttributeError:
        try:
            family_private = model.get_submodule("backbone.family_private")
        except AttributeError:
            family_private = None
    if isinstance(family_private, FamilyPrivateStrategyBank):
        topology.update(
            {
                "family_private_families": len(family_private.tails),
                "family_private_cloned_blocks": FAMILY_PRIVATE_CLONED_LAYERS,
                "family_private_appended_blocks": FAMILY_PRIVATE_APPENDED_LAYERS,
                "family_private_has_generic_upper": (
                    family_private.generic_upper is not None
                ),
            }
        )
    return {
        "architecture": config.architecture,
        "resolved_model_config": config.model_dump(mode="json"),
        "topology": topology,
        "tensor_elements": {
            "total_unique": sum(components.values()),
            "by_component": dict(sorted(components.items())),
        },
        "parameter_ownership": {
            "duplicate_parameter_names": sorted(duplicate_parameter_names),
            "card_encoder_owners": card_owner_prefixes,
        },
    }


def validate_simple_stateless_parameter_report(report: Mapping[str, Any]) -> None:
    """Fail closed when generated ownership/topology evidence violates identity."""
    topology = report.get("topology")
    ownership = report.get("parameter_ownership")
    if not isinstance(topology, Mapping) or not isinstance(ownership, Mapping):
        raise ValueError("simple stateless parameter report is incomplete")
    resolved = report.get("resolved_model_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("parameter report has no resolved model configuration")
    architecture = resolved.get("architecture")
    if not isinstance(architecture, str):
        raise ValueError("parameter report architecture is invalid")
    expected_layers = {
        GENERALIST_SEQUENCE_V2_ARCHITECTURE: 20,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE: FAMILY_PRIVATE_SHARED_LAYERS,
    }.get(architecture, 34)
    if topology.get("shared_transformer_blocks") != expected_layers:
        raise ValueError(
            f"parameter report does not contain {expected_layers} shared blocks"
        )
    if topology.get("scratch_tokens") != 8:
        raise ValueError("parameter report does not contain 8 scratch tokens")
    if architecture == GENERALIST_SEQUENCE_V3_ARCHITECTURE and (
        topology.get("family_private_cloned_blocks")
        != FAMILY_PRIVATE_CLONED_LAYERS
        or topology.get("family_private_appended_blocks")
        != FAMILY_PRIVATE_APPENDED_LAYERS
        or not isinstance(topology.get("family_private_families"), int)
        or int(topology["family_private_families"]) <= 0
    ):
        raise ValueError("parameter report has an invalid family-private tail")
    if ownership.get("duplicate_parameter_names"):
        raise ValueError("model registers duplicate parameter owners")
    card_owners = ownership.get("card_encoder_owners")
    if (
        not isinstance(card_owners, list)
        or len(card_owners) != 1
        or not str(card_owners[0]).endswith("input_encoder.card_encoder")
    ):
        raise ValueError("model must register exactly one CardEncoder owner")
