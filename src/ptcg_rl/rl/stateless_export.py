"""Fixed-deck FP16 deployment export for the clean stateless model."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from ptcg_rl.decks.registry import (
    deck_expert_registry_fingerprint,
    deck_family_registry_fingerprint,
)
from ptcg_rl.model.simple_stateless.config import (
    SimpleStatelessModelConfig,
    uses_exact_v2_topology,
    uses_family_private_topology,
    uses_temporal_prefusion,
)
from ptcg_rl.model.simple_stateless.network import SimpleStatelessPolicyValueNet
from ptcg_rl.model.simple_stateless.routing import resolve_simple_exact_routes
from ptcg_rl.model.simple_stateless.v2 import (
    GENERALIST_SEQUENCE_V2_CAPSULE_STAGES,
    SIMPLE_STATELESS_V2_CAPSULE_STAGES,
)
from ptcg_rl.rl.checkpoint_pair_io import publish_torch_file
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.policy_inputs import (
    SimpleStatelessActorRow,
    collate_simple_stateless_actor_rows,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PLANNING_BUDGET_BYTES = 197_000_000


class ExportComponentRecord(BaseModel):
    """Actual serialized tensor inventory for one deployment component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str
    tensors: int = Field(gt=0)
    tensor_elements: int = Field(gt=0)
    storage_bytes: int = Field(gt=0)


class FixedDeckExportReport(BaseModel):
    """Generated evidence for one fixed-route deployment checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["simple-stateless-fixed-deck-export-v1"] = (
        "simple-stateless-fixed-deck-export-v1"
    )
    source_policy_sha256: str
    source_model_fingerprint: str
    target_deck_digest: str
    target_expert_id: str
    target_family_id: str | None = None
    source_model_config_fingerprint: str
    fixed_model_config_fingerprint: str
    fixed_registry_fingerprint: str
    fixed_family_registry_fingerprint: str | None = None
    storage_dtype: Literal["float16"] = "float16"
    source_exact_routes: int = Field(gt=0)
    source_strategy_families: int = Field(default=0, ge=0)
    exported_exact_routes: Literal[1] = 1
    removed_tensors: int = Field(ge=0)
    removed_tensor_elements: int = Field(ge=0)
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int = Field(gt=0)
    planning_budget_bytes: int = _PLANNING_BUDGET_BYTES
    planning_budget_ratio: float = Field(gt=0.0)
    components: tuple[ExportComponentRecord, ...]

    @field_validator(
        "source_policy_sha256",
        "source_model_fingerprint",
        "target_deck_digest",
        "target_expert_id",
        "source_model_config_fingerprint",
        "fixed_model_config_fingerprint",
        "fixed_registry_fingerprint",
        "checkpoint_sha256",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable input and output identities."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("export fingerprints must be lowercase SHA-256")
        return normalized

    @field_validator(
        "target_family_id",
        "fixed_family_registry_fingerprint",
    )
    @classmethod
    def valid_optional_fingerprint(cls, value: str | None) -> str | None:
        """Validate optional v3 family identities without changing old reports."""
        if value is None:
            return None
        return cls.valid_fingerprint(value)


class FixedDeckParityReport(BaseModel):
    """Measured source/fixed-route equivalence on public actor rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int = Field(gt=0)
    actions_equal: bool
    maximum_policy_logit_error: float = Field(ge=0.0)
    maximum_root_value_error: float = Field(ge=0.0)
    maximum_belief_latent_error: float = Field(ge=0.0)
    absolute_tolerance: float = Field(gt=0.0)


def fixed_deck_model_config(
    source: SimpleStatelessModelConfig,
    *,
    target_deck_digest: str,
) -> SimpleStatelessModelConfig:
    """Return the same topology with exactly one immutable route."""
    routes = tuple(
        route
        for route in source.exact_routes
        if route.deck_digest == target_deck_digest
    )
    if len(routes) != 1:
        raise ValueError("target exact deck must resolve to exactly one source route")
    registry = deck_expert_registry_fingerprint(routes)
    family_routes = tuple(
        route
        for route in source.family_routes
        if route.deck_digest == target_deck_digest
    )
    if uses_family_private_topology(source) and len(family_routes) != 1:
        raise ValueError("target exact deck must resolve to one strategy family")
    payload = source.model_dump(mode="python")
    payload.update(
        {
            "exact_routes": routes,
            "resolved_registry_sha256": registry,
            "family_routes": family_routes,
            "resolved_family_registry_sha256": (
                deck_family_registry_fingerprint(family_routes)
                if family_routes
                else None
            ),
            "export_mode": "fixed",
        }
    )
    return SimpleStatelessModelConfig.model_validate(payload)


def prune_fixed_deck_state(
    state: Mapping[str, Tensor],
    *,
    source_config: SimpleStatelessModelConfig,
    fixed_config: SimpleStatelessModelConfig,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Remove only non-target exact residual tensors from a full policy state."""
    if fixed_config.export_mode != "fixed" or len(fixed_config.exact_routes) != 1:
        raise ValueError("pruning requires a one-route fixed deployment config")
    if not state:
        raise ValueError("deployment export requires a non-empty model state")
    target_key = fixed_config.exact_routes[0].module_key
    source_keys = {route.module_key for route in source_config.exact_routes}
    if target_key not in source_keys:
        raise ValueError("fixed route does not belong to the source registry")
    target_family_key: str | None = None
    source_family_keys: set[str] = set()
    if uses_family_private_topology(source_config):
        if len(fixed_config.family_routes) != 1:
            raise ValueError("fixed family-private config requires one family route")
        target_family_key = fixed_config.family_routes[0].module_key
        source_family_keys = {
            route.module_key for route in source_config.family_routes
        }
        observed_family_keys = {
            family_key
            for name in state
            if (family_key := _family_route_key(name)) is not None
        }
        if observed_family_keys != source_family_keys:
            raise ValueError(
                "source model state does not contain every declared family tail"
            )
    route_banks = _expected_private_route_banks(source_config)
    source_bank_keys: dict[str, set[str]] = {
        bank: set() for bank in route_banks
    }
    for name in state:
        reference = _private_route_reference(name)
        if reference is None:
            continue
        bank, route_key = reference
        if bank not in source_bank_keys:
            raise ValueError(f"unrecognized exact-route bank in model state: {bank}")
        source_bank_keys[bank].add(route_key)
    incomplete_banks = {
        bank: sorted(keys)
        for bank, keys in source_bank_keys.items()
        if keys != source_keys
    }
    if incomplete_banks:
        raise ValueError(
            "source model state does not contain every exact route in every "
            f"private bank: {incomplete_banks}"
        )
    removed: list[str] = []
    retained: dict[str, Tensor] = {}
    for name, tensor in state.items():
        private_key = _private_route_key(name)
        if private_key is not None and private_key != target_key:
            removed.append(name)
            continue
        family_key = _family_route_key(name)
        if family_key is not None and family_key != target_family_key:
            removed.append(name)
            continue
        if name.startswith("backbone.family_private.generic_upper."):
            removed.append(name)
            continue
        retained[name] = tensor
    expected_removed_keys = source_keys - {target_key}
    observed_removed_keys = {
        key
        for name in removed
        if (key := _private_route_key(name)) is not None
    }
    if observed_removed_keys != expected_removed_keys:
        raise ValueError("fixed export did not prune every non-target exact route")
    retained_bank_keys: dict[str, set[str]] = {
        bank: set() for bank in route_banks
    }
    for name in retained:
        reference = _private_route_reference(name)
        if reference is not None:
            bank, route_key = reference
            retained_bank_keys[bank].add(route_key)
    if any(keys != {target_key} for keys in retained_bank_keys.values()):
        raise ValueError("fixed export must retain only the target in every route bank")
    if target_family_key is not None:
        retained_family_keys = {
            family_key
            for name in retained
            if (family_key := _family_route_key(name)) is not None
        }
        if retained_family_keys != {target_family_key}:
            raise ValueError("fixed export must retain only the target family tail")
        if any(
            name.startswith("backbone.family_private.generic_upper.")
            for name in retained
        ):
            raise ValueError("fixed export retained the offline generic upper path")
    return retained, tuple(sorted(removed))


def fp16_storage_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Materialize a CPU FP16 storage view while preserving integer buffers."""
    return {
        name: (
            tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
            if tensor.is_floating_point()
            else tensor.detach().to(device="cpu", copy=True).contiguous()
        )
        for name, tensor in state.items()
    }


def export_fixed_deck_checkpoint(
    path: Path,
    *,
    source_state: Mapping[str, Tensor],
    source_config: SimpleStatelessModelConfig,
    source_policy_sha256: str,
    source_model_fingerprint: str,
    target_deck_digest: str,
    public_deck_catalog_fingerprint: str,
    asset_manifest_fingerprints: Mapping[str, str],
) -> FixedDeckExportReport:
    """Prune non-target routes and publish the actual FP16 deployment payload."""
    _require_sha256(source_policy_sha256, "source policy")
    _require_sha256(source_model_fingerprint, "source model")
    _require_sha256(public_deck_catalog_fingerprint, "public deck catalog")
    for name, fingerprint in asset_manifest_fingerprints.items():
        if not name.strip():
            raise ValueError("deployment asset manifest names must be non-empty")
        _require_sha256(fingerprint, f"deployment asset {name}")
    actual_source_fingerprint = canonical_model_state_fingerprint(source_state)
    if actual_source_fingerprint != source_model_fingerprint:
        raise ValueError("source model state differs from its declared identity")
    fixed_config = fixed_deck_model_config(
        source_config,
        target_deck_digest=target_deck_digest,
    )
    pruned, removed = prune_fixed_deck_state(
        source_state,
        source_config=source_config,
        fixed_config=fixed_config,
    )
    storage = fp16_storage_state(pruned)
    payload = {
        "format": "simple_stateless_fixed_deck_policy_v1",
        "source_policy_sha256": source_policy_sha256,
        "source_model_fingerprint": source_model_fingerprint,
        "target_deck_digest": target_deck_digest,
        "target_expert_id": fixed_config.exact_routes[0].expert_id,
        "target_family_id": (
            fixed_config.family_routes[0].family_id
            if fixed_config.family_routes
            else None
        ),
        "model_config": fixed_config.model_dump(mode="json"),
        "model_config_fingerprint": model_config_fingerprint(fixed_config),
        "public_deck_catalog_fingerprint": public_deck_catalog_fingerprint,
        "asset_manifest_fingerprints": dict(sorted(asset_manifest_fingerprints.items())),
        "storage_dtype": "float16",
        "model_state": storage,
        "model_fingerprint": canonical_model_state_fingerprint(storage),
    }
    size_bytes, sha256 = publish_torch_file(path.resolve(), payload)
    removed_elements = sum(source_state[name].numel() for name in removed)
    return FixedDeckExportReport(
        source_policy_sha256=source_policy_sha256,
        source_model_fingerprint=source_model_fingerprint,
        target_deck_digest=target_deck_digest,
        target_expert_id=fixed_config.exact_routes[0].expert_id,
        target_family_id=(
            fixed_config.family_routes[0].family_id
            if fixed_config.family_routes
            else None
        ),
        source_model_config_fingerprint=model_config_fingerprint(source_config),
        fixed_model_config_fingerprint=model_config_fingerprint(fixed_config),
        fixed_registry_fingerprint=(
            fixed_config.resolved_registry_sha256
            if fixed_config.resolved_registry_sha256 is not None
            else ""
        ),
        fixed_family_registry_fingerprint=(
            fixed_config.resolved_family_registry_sha256
        ),
        source_exact_routes=len(source_config.exact_routes),
        source_strategy_families=len(
            {route.family_id for route in source_config.family_routes}
        ),
        removed_tensors=len(removed),
        removed_tensor_elements=removed_elements,
        checkpoint_path=path.resolve(),
        checkpoint_sha256=sha256,
        checkpoint_size_bytes=size_bytes,
        planning_budget_ratio=size_bytes / float(_PLANNING_BUDGET_BYTES),
        components=_component_inventory(storage),
    )


def write_fixed_deck_export_report(
    path: Path,
    report: FixedDeckExportReport,
) -> None:
    """Persist human-inspectable evidence beside an immutable checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_fixed_deck_parity(
    source_model: SimpleStatelessPolicyValueNet,
    fixed_model: SimpleStatelessPolicyValueNet,
    rows: Sequence[SimpleStatelessActorRow],
    *,
    device: torch.device | str,
    absolute_tolerance: float,
) -> FixedDeckParityReport:
    """Compare first-step logits, decoded actions, value, and belief latent."""
    if not rows:
        raise ValueError("fixed-deck parity requires at least one public actor row")
    if absolute_tolerance <= 0.0 or not math.isfinite(absolute_tolerance):
        raise ValueError("fixed-deck parity tolerance must be finite and positive")
    target_digest = fixed_model.config.exact_routes[0].deck_digest
    if any(row.own_deck.deck_digest != target_digest for row in rows):
        raise ValueError("parity rows must all use the fixed target deck")
    target_device = torch.device(device)
    source = _deployment_outputs(source_model, rows, device=target_device)
    fixed = _deployment_outputs(fixed_model, rows, device=target_device)
    policy_error = _maximum_error(source["logits"], fixed["logits"])
    value_error = _maximum_error(source["root_values"], fixed["root_values"])
    belief_error = _maximum_error(source["belief_latent"], fixed["belief_latent"])
    actions_equal = source["actions"] == fixed["actions"]
    report = FixedDeckParityReport(
        rows=len(rows),
        actions_equal=actions_equal,
        maximum_policy_logit_error=policy_error,
        maximum_root_value_error=value_error,
        maximum_belief_latent_error=belief_error,
        absolute_tolerance=absolute_tolerance,
    )
    if (
        not actions_equal
        or policy_error > absolute_tolerance
        or value_error > absolute_tolerance
        or belief_error > absolute_tolerance
    ):
        raise ValueError("fixed-deck deployment parity tolerance exceeded")
    return report


def load_fixed_deck_checkpoint(
    path: Path,
) -> tuple[SimpleStatelessPolicyValueNet, Mapping[str, Any]]:
    """Construct a strict fixed-route model from a deployment checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("fixed-deck checkpoint payload must be a mapping")
    if payload.get("format") != "simple_stateless_fixed_deck_policy_v1":
        raise ValueError("unsupported fixed-deck checkpoint format")
    config = SimpleStatelessModelConfig.model_validate(payload["model_config"])
    if config.export_mode != "fixed":
        raise ValueError("deployment checkpoint does not contain a fixed route")
    if model_config_fingerprint(config) != str(payload["model_config_fingerprint"]):
        raise ValueError("deployment model config fingerprint mismatch")
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("deployment checkpoint model state is missing")
    state = {
        str(name): tensor
        for name, tensor in raw_state.items()
        if isinstance(tensor, Tensor)
    }
    if len(state) != len(raw_state):
        raise ValueError("deployment state contains a non-tensor value")
    if canonical_model_state_fingerprint(state) != str(payload["model_fingerprint"]):
        raise ValueError("deployment model-state fingerprint mismatch")
    model = SimpleStatelessPolicyValueNet(
        config,
        load_static_features=False,
        initialize=False,
    )
    model.half()
    model.load_state_dict(state, strict=True)
    return model.eval(), payload


def _deployment_outputs(
    model: SimpleStatelessPolicyValueNet,
    rows: Sequence[SimpleStatelessActorRow],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device).eval()
    batch = collate_simple_stateless_actor_rows(rows, device=device)
    routes = resolve_simple_exact_routes(
        batch.deck_signatures,
        model.config,
        device=device,
    )
    use_autocast = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_autocast,
    ):
        state = model.encode_observation_state(
            state=batch.states,
            unique_deck_card_ids=batch.unique_deck_card_ids,
            deck_counts=batch.deck_counts,
            deck_valid_mask=batch.deck_valid_mask,
            belief_summary=batch.belief_summary,
            route_plan=routes,
        )
        options = model.encode_legal_options(
            state,
            batch.options,
            route_plan=routes,
        )
        selected = torch.zeros_like(batch.options.valid_options)
        counts = torch.zeros_like(batch.options.min_counts)
        history = state.policy.new_zeros(state.policy.shape)
        first = model.heads.step(
            state.policy,
            state.opponent_belief,
            options,
            batch.options,
            selected_mask=selected,
            selected_counts=counts,
            ordered_history=history,
            route_plan=routes,
        )
        actions = model.heads.sample_decode(
            state.policy,
            state.opponent_belief,
            options,
            batch.options,
            route_plan=routes,
            temperature=1.0,
            generator=torch.Generator(device=device).manual_seed(1729),
        )
        roots = model.heads.root_value(
            state.value,
            state.opponent_belief,
            route_plan=routes,
        )
    return {
        "logits": first.logits.float().cpu(),
        "actions": actions,
        "root_values": roots.float().cpu(),
        "belief_latent": state.opponent_belief.float().cpu(),
    }


def _maximum_error(left: Any, right: Any) -> float:
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        raise TypeError("parity outputs must be tensors")
    if left.shape != right.shape:
        raise ValueError("parity output shapes differ")
    if not left.numel():
        return 0.0
    if bool(torch.isnan(left).any()) or bool(torch.isnan(right).any()):
        raise ValueError("parity output contains NaN")
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    if not torch.equal(left_finite, right_finite):
        raise ValueError("parity output finite masks differ")
    non_finite = ~left_finite
    if bool(non_finite.any()) and not torch.equal(
        torch.signbit(left[non_finite]),
        torch.signbit(right[non_finite]),
    ):
        raise ValueError("parity output infinities differ")
    if not bool(left_finite.any()):
        return 0.0
    return float((left[left_finite] - right[right_finite]).abs().max())


def _private_route_key(name: str) -> str | None:
    reference = _private_route_reference(name)
    return reference[1] if reference is not None else None


def _private_route_reference(name: str) -> tuple[str, str] | None:
    """Return the immutable bank and route key for one private state tensor."""
    for bank in (
        "heads.policy_residuals",
        "heads.option_residuals",
        "heads.value_residuals",
        "backbone.v2_adapters.prompts",
        *(
            "backbone.v2_adapters.stages."
            f"after_layer_{layer:02d}.exact_capsules"
            for layer in tuple(
                sorted(
                    set(SIMPLE_STATELESS_V2_CAPSULE_STAGES)
                    | set(GENERALIST_SEQUENCE_V2_CAPSULE_STAGES)
                )
            )
        ),
    ):
        prefix = f"{bank}."
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            route_key = suffix.split(".", maxsplit=1)[0]
            if not route_key:
                raise ValueError("exact-route tensor has an empty module key")
            return bank, route_key
    return None


def _family_route_key(name: str) -> str | None:
    """Return the physical family module key for one private-tail tensor."""
    prefix = "backbone.family_private.tails."
    if not name.startswith(prefix):
        return None
    suffix = name.removeprefix(prefix)
    family_key = suffix.split(".", maxsplit=1)[0]
    if not family_key:
        raise ValueError("family-private tensor has an empty module key")
    return family_key


def _expected_private_route_banks(
    config: SimpleStatelessModelConfig,
) -> tuple[str, ...]:
    """Enumerate all route-keyed banks required by the declared topology."""
    banks = [
        "heads.policy_residuals",
        "heads.value_residuals",
    ]
    if uses_exact_v2_topology(config):
        banks.append("heads.option_residuals")
        banks.append("backbone.v2_adapters.prompts")
        banks.extend(
            "backbone.v2_adapters.stages."
            f"after_layer_{layer:02d}.exact_capsules"
            for layer in (
                GENERALIST_SEQUENCE_V2_CAPSULE_STAGES
                if uses_temporal_prefusion(config)
                else SIMPLE_STATELESS_V2_CAPSULE_STAGES
            )
        )
    return tuple(banks)


def _component_inventory(
    state: Mapping[str, Tensor],
) -> tuple[ExportComponentRecord, ...]:
    counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for name, tensor in state.items():
        component = _component_name(name)
        counts[component][0] += 1
        counts[component][1] += tensor.numel()
        counts[component][2] += tensor.numel() * tensor.element_size()
    return tuple(
        ExportComponentRecord(
            component=name,
            tensors=values[0],
            tensor_elements=values[1],
            storage_bytes=values[2],
        )
        for name, values in sorted(counts.items())
    )


def _component_name(name: str) -> str:
    if ".card_encoder." in f".{name}.":
        return "card_encoder"
    if name.startswith("backbone.input_encoder.deck_encoder."):
        return "deck_encoder"
    if name.startswith("backbone.family_private.generic_upper."):
        return "family_private_generic_upper"
    if ".cloned_layers." in name and name.startswith(
        "backbone.family_private.tails."
    ):
        return "family_private_cloned_upper"
    if ".appended_layers." in name and name.startswith(
        "backbone.family_private.tails."
    ):
        return "family_private_appended_upper"
    if name.startswith("backbone.input_encoder.belief_"):
        return "belief_input"
    if name.startswith("backbone.v2_adapters.prompts."):
        return "v2_exact_prompts"
    if ".exact_capsules." in name:
        return "v2_exact_capsules"
    if name.startswith("backbone.v2_adapters.stages."):
        return "v2_shared_bases"
    if name.startswith("backbone.input_encoder.raw_state_encoder."):
        return "public_state_encoder"
    if name.startswith("backbone.trunk."):
        return "shared_transformer"
    if name.startswith(
        ("heads.entity_temporal_fusion.", "heads.option_temporal_fusion.")
    ):
        return "temporal_prefusion"
    if name.startswith("heads.shared_root_wdl."):
        return "distributional_wdl_critic"
    if name.startswith("belief_head."):
        return "belief_head"
    if _private_route_key(name) is not None:
        return "fixed_exact_residual"
    if name.startswith("heads."):
        return "shared_policy_value_heads"
    return "special_tokens"


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} fingerprint must be SHA-256")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ExportComponentRecord",
    "FixedDeckExportReport",
    "FixedDeckParityReport",
    "export_fixed_deck_checkpoint",
    "fixed_deck_model_config",
    "fp16_storage_state",
    "load_fixed_deck_checkpoint",
    "prune_fixed_deck_state",
    "validate_fixed_deck_parity",
    "write_fixed_deck_export_report",
]
