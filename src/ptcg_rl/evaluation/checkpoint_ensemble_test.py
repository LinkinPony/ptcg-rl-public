"""Behavior tests for parameter-space checkpoint ensembles."""

from __future__ import annotations

import torch

from ptcg_rl.evaluation.checkpoint_ensemble import (
    is_routed_parameter,
    merge_model_states,
)


def test_merge_scopes_keep_unselected_anchor_tensors() -> None:
    anchor = {
        "backbone.trunk.weight": torch.tensor([2.0]),
        "backbone.family_private.tails.deck_x.weight": torch.tensor([4.0]),
        "heads.policy_residuals.deck_x.weight": torch.tensor([6.0]),
    }
    older = {name: tensor - 2.0 for name, tensor in anchor.items()}

    shared, selected = merge_model_states(
        {"anchor": anchor, "older": older},
        weights={"anchor": 0.5, "older": 0.5},
        scope="shared",
        anchor_name="anchor",
    )

    assert selected == ("backbone.trunk.weight",)
    assert torch.equal(shared["backbone.trunk.weight"], torch.tensor([1.0]))
    assert torch.equal(
        shared["backbone.family_private.tails.deck_x.weight"], torch.tensor([4.0])
    )
    assert torch.equal(
        shared["heads.policy_residuals.deck_x.weight"], torch.tensor([6.0])
    )


def test_routed_parameter_classification_covers_exact_and_family_capacity() -> None:
    assert is_routed_parameter("backbone.family_private.tails.deck_x.weight")
    assert is_routed_parameter(
        "backbone.v2_adapters.stages.layer.exact_capsules.deck_x.weight"
    )
    assert is_routed_parameter("backbone.v2_adapters.prompts.deck_x")
    assert is_routed_parameter("heads.option_residuals.deck_x.weight")
    assert not is_routed_parameter("backbone.trunk.layers.0.weight")
