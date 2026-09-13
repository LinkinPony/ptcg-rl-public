"""One-time architecture-v2 LoRA to dense-private weight migration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
)
from ptcg_rl.rl.dense_private_transition_support import (
    POLICY_TARGETS,
    TARGET_PRIVATE_PREFIXES,
    TRANSFORMER_TARGET_BY_SUFFIX,
    TransitionContext,
    build_transition_context,
    lower_lora_inventory,
    merged_lora_weight,
    reject_invalid_target_state,
    validate_source_lora_inventory,
)

_OPTION_SET_COMPONENTS = frozenset(
    {
        "option_projection",
        "self_norm",
        "self_attention",
        "cross_norm",
        "state_projection",
        "cross_attention",
        "ffn_norm",
        "ffn",
        "output_projection",
    }
)
_DENSE_RESIDUAL_SUFFIXES = frozenset(
    {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
        "3.weight",
        "3.bias",
        "5.weight",
        "5.bias",
    }
)
_PRIVATE_RESIDUAL_SUFFIXES = frozenset(
    {"down.weight", "down.bias", "up.weight", "up.bias"}
)
_PRIVATE_SCALAR_SUFFIXES = frozenset(
    {"hidden.weight", "hidden.bias", "output.weight", "output.bias"}
)
_COUNT_POLICY_TARGETS = (
    "count_state_projection",
    "count_feature_projection",
    "count_output_projection",
)


@dataclass
class _MigrationRecorder:
    """Accumulate portable migration counts and per-key actions."""

    actions: dict[str, dict[str, str | None]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(
        self,
        target_key: str,
        *,
        action: str,
        source_key: str | None,
        numel: int,
    ) -> None:
        """Record one target tensor exactly once."""
        if target_key in self.actions:
            raise ValueError(f"migration target handled twice: {target_key}")
        self.actions[target_key] = {
            "action": action,
            "source_key": source_key,
        }
        self.counts[f"{action}_tensors"] += 1
        self.counts[f"{action}_numel"] += numel


def migrate_lora_v2_to_dense_private(
    target_model: AgentPolicyValueNet,
    source_state_dict: Mapping[str, Any],
    *,
    source_config: AgentNetworkConfig,
    expert_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Migrate a routed v2 checkpoint into a fresh dense-private v3 model.

    ``expert_sources`` maps each target expert ID (or ``deck_`` module key) to
    its v2 donor. Shared tensors retain exact state-dict names. Upper
    Transformer and policy LoRA weights are materialized as ordinary dense
    weights; legacy dense residuals are moved under each strategy. New
    option-set/count/value residual branches retain their output-inert target
    initialization.

    Lower LoRA cannot be represented below the new shared/private split. The
    returned bound manifest fingerprints every such factor pair, records its
    delta norm, and explicitly marks that boundary as non-equivalent.
    """
    target_state = target_model.state_dict()
    context = build_transition_context(
        target_model=target_model,
        source_config=source_config,
        target_to_source_experts=expert_sources,
    )
    reject_invalid_target_state(target_state)
    validate_source_lora_inventory(source_state_dict, context=context)

    migrated = dict(target_state)
    recorder = _MigrationRecorder()
    _copy_shared_state(
        migrated,
        target_state=target_state,
        source_state=source_state_dict,
        recorder=recorder,
    )
    for target_module, source_module in sorted(context.target_to_source.items()):
        _migrate_state_strategy(
            migrated,
            target_state=target_state,
            source_state=source_state_dict,
            target_module=target_module,
            source_module=source_module,
            context=context,
            recorder=recorder,
        )
        _migrate_policy_strategy(
            migrated,
            target_state=target_state,
            source_state=source_state_dict,
            target_module=target_module,
            source_module=source_module,
            context=context,
            recorder=recorder,
        )
        for kind in ("root", "prefix"):
            _migrate_value_strategy(
                migrated,
                target_state=target_state,
                source_state=source_state_dict,
                target_module=target_module,
                source_module=source_module,
                kind=kind,
                recorder=recorder,
            )
    unhandled = sorted(set(target_state) - set(recorder.actions))
    if unhandled:
        raise ValueError(f"dense-private target state was not handled: {unhandled}")

    lower_inventory = lower_lora_inventory(
        source_state=source_state_dict,
        context=context,
    )
    target_model.load_state_dict(migrated, strict=True)
    summary: dict[str, Any] = {
        "schema": "dense-private-strategy-weight-transition-v1",
        "source_architecture_version": DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
        "target_architecture_version": (
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ),
        "shared_transformer_layers": context.shared_layer_count,
        "private_transformer_layers": (
            context.transformer_layer_count - context.shared_layer_count
        ),
        "target_to_source_module_keys": dict(sorted(context.target_to_source.items())),
        "source_lora": {
            "rank": context.rank,
            "alpha": context.alpha,
            "scaling": context.scaling,
        },
        "lower_lora_handling": (
            "retired_without_merge" if lower_inventory else "none"
        ),
        "non_equivalent_boundary": bool(lower_inventory),
        "lower_lora_delta_count": len(lower_inventory),
        "lower_lora_factor_numel": sum(
            int(item["factor_numel"]) for item in lower_inventory
        ),
        "lower_lora_delta_numel": sum(
            int(item["delta_numel"]) for item in lower_inventory
        ),
        "lower_lora_deltas": lower_inventory,
        "parameter_actions_sha256": fingerprint_payload(recorder.actions),
        **dict(sorted(recorder.counts.items())),
    }
    summary["manifest_sha256"] = fingerprint_payload(summary)
    return summary


def migrate_dense_private_strategy_weights(
    target_model: AgentPolicyValueNet,
    source_state_dict: Mapping[str, Any],
    source_config: AgentNetworkConfig,
    target_to_source_experts: Mapping[str, str],
) -> dict[str, Any]:
    """Compatibility spelling for direct callers of the migration helper."""
    return migrate_lora_v2_to_dense_private(
        target_model,
        source_state_dict,
        source_config=source_config,
        expert_sources=target_to_source_experts,
    )


def _copy_shared_state(
    migrated: dict[str, Tensor],
    *,
    target_state: Mapping[str, Tensor],
    source_state: Mapping[str, Any],
    recorder: _MigrationRecorder,
) -> None:
    """Copy every v3 non-private tensor by its unchanged exact key."""
    for target_key, target_value in target_state.items():
        if target_key.startswith(TARGET_PRIVATE_PREFIXES):
            continue
        source_value = _source_tensor(source_state, target_key)
        migrated[target_key] = _copy_tensor(source_value, target_value, name=target_key)
        recorder.record(
            target_key,
            action="copied_shared",
            source_key=target_key,
            numel=int(target_value.numel()),
        )


def _migrate_state_strategy(
    migrated: dict[str, Tensor],
    *,
    target_state: Mapping[str, Tensor],
    source_state: Mapping[str, Any],
    target_module: str,
    source_module: str,
    context: TransitionContext,
    recorder: _MigrationRecorder,
) -> None:
    """Clone and merge one private upper Transformer stack."""
    target_prefix = f"state_encoder.private_strategy_stacks.{target_module}."
    expected: set[str] = set()
    for source_layer in range(
        context.shared_layer_count,
        context.transformer_layer_count,
    ):
        relative_layer = source_layer - context.shared_layer_count
        source_prefix = f"state_encoder.transformer.layers.{source_layer}."
        source_keys = _subtree_keys(source_state, source_prefix)
        if not source_keys:
            raise ValueError(f"source Transformer layer is absent: {source_prefix}")
        for source_key in source_keys:
            suffix = source_key.removeprefix(source_prefix)
            target_key = f"{target_prefix}layers.{relative_layer}.layer.{suffix}"
            expected.add(target_key)
            target_value = _target_tensor(target_state, target_key)
            lora_target = TRANSFORMER_TARGET_BY_SUFFIX.get(suffix)
            if lora_target is None:
                migrated[target_key] = _copy_tensor(
                    _source_tensor(source_state, source_key),
                    target_value,
                    name=target_key,
                )
                action = "copied_private"
            else:
                spec = context.specs_by_target[
                    f"transformer.{source_layer}.{lora_target}"
                ]
                migrated[target_key] = merged_lora_weight(
                    source_state,
                    spec=spec,
                    module_key=source_module,
                    rank=context.rank,
                    scaling=context.scaling,
                    target=target_value,
                )
                action = "merged_transformer_lora"
            recorder.record(
                target_key,
                action=action,
                source_key=source_key,
                numel=int(target_value.numel()),
            )

    for source_layer in context.source_conditioning.adapter_layer_indices:
        relative_layer = source_layer - context.shared_layer_count
        source_prefix = (
            f"state_encoder.private_adapters.layer_{source_layer:02d}."
            f"{source_module}."
        )
        for source_key in _strict_subtree_keys(
            source_state,
            prefix=source_prefix,
            expected_suffixes=_PRIVATE_RESIDUAL_SUFFIXES,
        ):
            suffix = source_key.removeprefix(source_prefix)
            target_key = f"{target_prefix}layers.{relative_layer}.residual.{suffix}"
            expected.add(target_key)
            _copy_private_key(
                migrated,
                target_state=target_state,
                source_state=source_state,
                target_key=target_key,
                source_key=source_key,
                recorder=recorder,
            )

    norm_prefix = "state_encoder.layer_norm."
    norm_keys = _subtree_keys(source_state, norm_prefix)
    if not norm_keys:
        raise ValueError("source state output norm is absent")
    for source_key in norm_keys:
        suffix = source_key.removeprefix(norm_prefix)
        target_key = f"{target_prefix}output_norm.{suffix}"
        expected.add(target_key)
        _copy_private_key(
            migrated,
            target_state=target_state,
            source_state=source_state,
            target_key=target_key,
            source_key=source_key,
            recorder=recorder,
        )
    _require_exact_target_subtree(target_state, prefix=target_prefix, expected=expected)


def _migrate_policy_strategy(
    migrated: dict[str, Tensor],
    *,
    target_state: Mapping[str, Tensor],
    source_state: Mapping[str, Any],
    target_module: str,
    source_module: str,
    context: TransitionContext,
    recorder: _MigrationRecorder,
) -> None:
    """Clone and merge one private policy/count strategy."""
    target_prefix = f"policy_head.private_strategies.{target_module}."
    expected: set[str] = set()
    for policy_target in POLICY_TARGETS:
        spec = context.specs_by_target[f"policy.{policy_target}"]
        source_prefix = spec.base_weight_key.removesuffix("weight")
        source_keys = _subtree_keys(source_state, source_prefix)
        if not source_keys:
            raise ValueError(f"source policy Linear is absent: {source_prefix}")
        for source_key in source_keys:
            suffix = source_key.removeprefix(source_prefix)
            target_key = f"{target_prefix}linears.{policy_target}.{suffix}"
            expected.add(target_key)
            target_value = _target_tensor(target_state, target_key)
            if suffix == "weight":
                migrated[target_key] = merged_lora_weight(
                    source_state,
                    spec=spec,
                    module_key=source_module,
                    rank=context.rank,
                    scaling=context.scaling,
                    target=target_value,
                )
                action = "merged_policy_lora"
            else:
                migrated[target_key] = _copy_tensor(
                    _source_tensor(source_state, source_key),
                    target_value,
                    name=target_key,
                )
                action = "copied_private"
            recorder.record(
                target_key,
                action=action,
                source_key=source_key,
                numel=int(target_value.numel()),
            )

    for policy_target in _COUNT_POLICY_TARGETS:
        source_prefix = f"policy_head.{policy_target}."
        source_keys = _subtree_keys(source_state, source_prefix)
        if not source_keys:
            raise ValueError(f"source count Linear is absent: {source_prefix}")
        for source_key in source_keys:
            suffix = source_key.removeprefix(source_prefix)
            target_key = f"{target_prefix}linears.{policy_target}.{suffix}"
            expected.add(target_key)
            _copy_private_key(
                migrated,
                target_state=target_state,
                source_state=source_state,
                target_key=target_key,
                source_key=source_key,
                recorder=recorder,
            )

    stop_key = f"{target_prefix}stop_embedding"
    expected.add(stop_key)
    _copy_private_key(
        migrated,
        target_state=target_state,
        source_state=source_state,
        target_key=stop_key,
        source_key="policy_head.stop_embedding",
        recorder=recorder,
    )
    adapter_prefix = f"private_policy_adapters.{source_module}."
    for source_key in _strict_subtree_keys(
        source_state,
        prefix=adapter_prefix,
        expected_suffixes=_PRIVATE_RESIDUAL_SUFFIXES,
    ):
        suffix = source_key.removeprefix(adapter_prefix)
        target_key = f"{target_prefix}global_adapter.{suffix}"
        expected.add(target_key)
        _copy_private_key(
            migrated,
            target_state=target_state,
            source_state=source_state,
            target_key=target_key,
            source_key=source_key,
            recorder=recorder,
        )

    option_prefix = f"{target_prefix}option_set."
    option_keys = _subtree_keys(target_state, option_prefix)
    option_components = {
        key.removeprefix(option_prefix).split(".", maxsplit=1)[0]
        for key in option_keys
    }
    if option_components != _OPTION_SET_COMPONENTS:
        raise ValueError(
            "option-set inventory mismatch: "
            f"expected={sorted(_OPTION_SET_COMPONENTS)}, "
            f"actual={sorted(option_components)}"
        )
    _require_zero_output(target_state, prefix=f"{option_prefix}output_projection.")
    for target_key in option_keys:
        expected.add(target_key)
        _retain_initialized_key(
            target_state,
            target_key=target_key,
            recorder=recorder,
            action="initialized_option_set",
        )

    count_prefix = f"{target_prefix}count_set_projection."
    count_keys = _subtree_keys(target_state, count_prefix)
    _require_zero_output(target_state, prefix=count_prefix)
    for target_key in count_keys:
        expected.add(target_key)
        _retain_initialized_key(
            target_state,
            target_key=target_key,
            recorder=recorder,
            action="initialized_count_set_projection",
        )
    _require_exact_target_subtree(target_state, prefix=target_prefix, expected=expected)


def _migrate_value_strategy(
    migrated: dict[str, Tensor],
    *,
    target_state: Mapping[str, Tensor],
    source_state: Mapping[str, Any],
    target_module: str,
    source_module: str,
    kind: str,
    recorder: _MigrationRecorder,
) -> None:
    """Move one root or prefix value path into a private dense head."""
    if kind == "root":
        target_bank = "dense_private_root_value_heads"
        base_bank = "value_head"
        legacy_bank = "private_root_value_heads"
    elif kind == "prefix":
        target_bank = "dense_private_prefix_value_heads"
        base_bank = "prefix_value_delta_head"
        legacy_bank = "private_prefix_value_heads"
    else:
        raise ValueError(f"unknown value strategy kind: {kind}")
    target_prefix = f"{target_bank}.{target_module}."
    expected: set[str] = set()
    for target_name, source_name in (("base_hidden", "0"), ("base_output", "2")):
        source_prefix = f"{base_bank}.{source_name}."
        source_keys = _subtree_keys(source_state, source_prefix)
        if not source_keys:
            raise ValueError(f"source value Linear is absent: {source_prefix}")
        for source_key in source_keys:
            suffix = source_key.removeprefix(source_prefix)
            target_key = f"{target_prefix}{target_name}.{suffix}"
            expected.add(target_key)
            _copy_private_key(
                migrated,
                target_state=target_state,
                source_state=source_state,
                target_key=target_key,
                source_key=source_key,
                recorder=recorder,
            )

    legacy_prefix = f"{legacy_bank}.{source_module}."
    for source_key in _strict_subtree_keys(
        source_state,
        prefix=legacy_prefix,
        expected_suffixes=_PRIVATE_SCALAR_SUFFIXES,
    ):
        suffix = source_key.removeprefix(legacy_prefix)
        target_key = f"{target_prefix}legacy_residual.{suffix}"
        expected.add(target_key)
        _copy_private_key(
            migrated,
            target_state=target_state,
            source_state=source_state,
            target_key=target_key,
            source_key=source_key,
            recorder=recorder,
        )

    dense_prefix = f"{target_prefix}dense_residual."
    dense_keys = _subtree_keys(target_state, dense_prefix)
    dense_suffixes = {key.removeprefix(dense_prefix) for key in dense_keys}
    if dense_suffixes != _DENSE_RESIDUAL_SUFFIXES:
        raise ValueError(
            "dense value residual inventory mismatch: "
            f"expected={sorted(_DENSE_RESIDUAL_SUFFIXES)}, "
            f"actual={sorted(dense_suffixes)}"
        )
    _require_zero_output(target_state, prefix=f"{dense_prefix}5.")
    for target_key in dense_keys:
        expected.add(target_key)
        _retain_initialized_key(
            target_state,
            target_key=target_key,
            recorder=recorder,
            action="initialized_dense_value_residual",
        )
    _require_exact_target_subtree(target_state, prefix=target_prefix, expected=expected)


def _copy_private_key(
    migrated: dict[str, Tensor],
    *,
    target_state: Mapping[str, Tensor],
    source_state: Mapping[str, Any],
    target_key: str,
    source_key: str,
    recorder: _MigrationRecorder,
) -> None:
    target_value = _target_tensor(target_state, target_key)
    migrated[target_key] = _copy_tensor(
        _source_tensor(source_state, source_key),
        target_value,
        name=target_key,
    )
    recorder.record(
        target_key,
        action="copied_private",
        source_key=source_key,
        numel=int(target_value.numel()),
    )


def _retain_initialized_key(
    target_state: Mapping[str, Tensor],
    *,
    target_key: str,
    recorder: _MigrationRecorder,
    action: str,
) -> None:
    target_value = _target_tensor(target_state, target_key)
    if target_value.is_floating_point() and not torch.isfinite(target_value).all():
        raise ValueError(f"initialized target tensor is non-finite: {target_key}")
    recorder.record(
        target_key,
        action=action,
        source_key=None,
        numel=int(target_value.numel()),
    )


def _copy_tensor(source: Tensor, target: Tensor, *, name: str) -> Tensor:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(f"transition tensor shape mismatch: {name}")
    if source.dtype != target.dtype:
        raise ValueError(f"transition tensor dtype mismatch: {name}")
    if source.is_floating_point() and not torch.isfinite(source).all():
        raise ValueError(f"transition source tensor is non-finite: {name}")
    return source.detach().to(device=target.device).clone()


def _source_tensor(source_state: Mapping[str, Any], key: str) -> Tensor:
    value = source_state.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"transition source tensor is missing: {key}")
    return value


def _target_tensor(target_state: Mapping[str, Tensor], key: str) -> Tensor:
    value = target_state.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"transition target tensor is missing: {key}")
    return value


def _subtree_keys(state: Mapping[str, Any], prefix: str) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in state if str(key).startswith(prefix)))


def _strict_subtree_keys(
    state: Mapping[str, Any],
    *,
    prefix: str,
    expected_suffixes: frozenset[str],
) -> tuple[str, ...]:
    keys = _subtree_keys(state, prefix)
    actual_suffixes = {key.removeprefix(prefix) for key in keys}
    if actual_suffixes != expected_suffixes:
        raise ValueError(
            f"legacy residual inventory mismatch for {prefix}: "
            f"expected={sorted(expected_suffixes)}, actual={sorted(actual_suffixes)}"
        )
    return keys


def _require_exact_target_subtree(
    target_state: Mapping[str, Tensor],
    *,
    prefix: str,
    expected: set[str],
) -> None:
    actual = set(_subtree_keys(target_state, prefix))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"private target inventory mismatch for {prefix}: "
            f"missing={missing}, extra={extra}"
        )


def _require_zero_output(
    target_state: Mapping[str, Tensor],
    *,
    prefix: str,
) -> None:
    keys = _subtree_keys(target_state, prefix)
    if not keys:
        raise ValueError(f"zero-initialized output is absent: {prefix}")
    for key in keys:
        value = _target_tensor(target_state, key)
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"new output projection is invalid: {key}")
        if torch.count_nonzero(value).item() != 0:
            raise ValueError(f"new output projection is not zero-initialized: {key}")


__all__ = [
    "migrate_dense_private_strategy_weights",
    "migrate_lora_v2_to_dense_private",
]
