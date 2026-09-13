"""Runtime checkpoint export helpers for Kaggle submission archives."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn

from ptcg_rl.cards.card_encoder import CardEncoder, build_card_encoder
from ptcg_rl.checkpoint_storage import (
    BYTE_SHUFFLE_FORMAT,
    byte_shuffle_state_dict,
    direct_recurrent_policy_metadata,
    strip_direct_recurrent_policy_state,
    unpack_checkpoint_state_dict,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks import canonicalize_deck
from ptcg_rl.decks.registry import (
    DeckExpertRoute,
    PrivateDeckProfile,
    deck_expert_registry_fingerprint,
    private_registry_fingerprint,
)
from ptcg_rl.evaluation.search_identity import file_sha256, fingerprint_payload
from ptcg_rl.model import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
    AgentNetworkConfig,
    strip_training_only_action_value,
)
from ptcg_rl.model.deck_conditioning import DeckEncoder
from ptcg_rl.model.deck_lora import (
    RoutedLoRAMergeSpec,
    routed_lora_merge_specs,
)
from ptcg_rl.model.state_encoder import TOKEN_KIND_TO_INDEX
from ptcg_rl.submission.compositional_export import (
    compositional_materialization_inventory,
    fix_selected_compositional_strategy,
    fixed_compositional_schema_sha256,
)

RuntimeCheckpointPrecision = Literal["fp16", "fp32"]
RuntimeCheckpointPrivateProfileMode = Literal["full", "pruned", "merged", "fixed"]
RuntimeCheckpointTensorStorage = Literal["native", "byte_shuffle_v1"]

_CHECKPOINT_METADATA_KEYS = (
    "model_config",
    "policy_input_schema",
    "agent_network_config",
    "network_config",
    "config",
    "training_config",
    "metadata",
    "epoch",
    "global_step",
    "step",
    "metrics",
)
_PRIVATE_STATE_PREFIXES = (
    "state_encoder.private_adapters.",
    "state_encoder.private_lora.",
    "state_encoder.private_strategy_stacks.",
    "policy_head.private_lora.",
    "policy_head.private_strategies.",
    "private_policy_adapters.",
    "private_root_value_heads.",
    "private_prefix_value_heads.",
    "dense_private_root_value_heads.",
    "dense_private_prefix_value_heads.",
    "exact_capsules.",
)
_DENSE_PRIVATE_GENERIC_POLICY_PREFIXES = (
    "policy_head.scalar_projection.0.",
    "policy_head.scalar_projection.3.",
    "policy_head.dynamic_effect_projection.0.",
    "policy_head.dynamic_effect_projection.3.",
    "policy_head.option_projection.",
    "policy_head.selected_projection.",
    "policy_head.ordered_history_projection.",
    "policy_head.decoder_cardinality_projection.",
    "policy_head.query_projection.0.",
    "policy_head.query_projection.3.",
    "policy_head.count_state_projection.",
    "policy_head.count_feature_projection.",
    "policy_head.count_output_projection.",
)
_DENSE_PRIVATE_GENERIC_VALUE_PREFIXES = (
    "value_head.",
    "prefix_value_delta_head.",
)


class RuntimeCheckpointExportConfig(BaseModel):
    """Config for exporting a compact runtime-loadable checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_checkpoint: Path
    output_checkpoint: Path
    precision: RuntimeCheckpointPrecision = "fp16"
    deck_path: Path | None = None
    private_profile_mode: RuntimeCheckpointPrivateProfileMode = "full"
    tensor_storage: RuntimeCheckpointTensorStorage = "native"
    direct_policy_only: bool = False

    @field_validator("source_checkpoint", "output_checkpoint")
    @classmethod
    def valid_checkpoint_path(cls, value: Path) -> Path:
        """Reject non-checkpoint-looking paths."""
        if value.suffix not in {".pt", ".pth", ".ckpt"}:
            raise ValueError(
                f"checkpoint path must end with .pt, .pth, or .ckpt: {value}"
            )
        return value


class ReleaseRuntimeCheckpointExportConfig(BaseModel):
    """Hash-preserving export of an immutable release checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    release_manifest_path: Path
    output_checkpoint: Path

    @field_validator("release_manifest_path")
    @classmethod
    def immutable_release_path(cls, value: Path) -> Path:
        """Reject moving latest aliases in release asset inputs."""
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("release export paths cannot contain 'latest'")
        return value

    @field_validator("output_checkpoint")
    @classmethod
    def valid_output_checkpoint(cls, value: Path) -> Path:
        """Require a runtime checkpoint filename."""
        if value.suffix not in {".pt", ".pth", ".ckpt"}:
            raise ValueError("approved checkpoint output must be a checkpoint file")
        return value


def export_runtime_checkpoint(
    config: RuntimeCheckpointExportConfig,
) -> Mapping[str, Any]:
    """Export a compact checkpoint containing only runtime model state and metadata."""
    if not config.source_checkpoint.is_file():
        raise FileNotFoundError(
            f"source checkpoint does not exist: {config.source_checkpoint}"
        )

    checkpoint = torch.load(config.source_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping or contain a mapping state_dict")
    checkpoint_mapping = cast(Mapping[str, Any], checkpoint)

    raw_state_dict = _checkpoint_state_dict(checkpoint_mapping)
    model_config = _checkpoint_model_config(checkpoint_mapping)
    prepared_state_dict, model_config, deck_conditioning = (
        _prepare_deck_conditioned_export(
            state_dict=raw_state_dict,
            model_config=model_config,
            config=config,
        )
    )
    stripped_action_value_keys: tuple[str, ...] = ()
    stripped_direct_policy_keys: tuple[str, ...] = ()
    action_value_state_present = any(
        str(key).startswith("action_value_head.") for key in prepared_state_dict
    )
    action_value_declared = bool(
        model_config is not None and model_config.action_value.enabled
    )
    if model_config is not None and (
        action_value_declared or action_value_state_present
    ):
        direct_actor = strip_training_only_action_value(
            prepared_state_dict,
            model_config,
        )
        prepared_state_dict = direct_actor.state_dict
        model_config = direct_actor.model_config
        stripped_action_value_keys = direct_actor.stripped_keys
    if config.direct_policy_only:
        if model_config is None or model_config.recurrent is None:
            raise ValueError(
                "direct policy-only export requires a recurrent checkpoint"
            )
        prepared_state_dict, stripped_direct_policy_keys = (
            strip_direct_recurrent_policy_state(prepared_state_dict)
        )
    if deck_conditioning is not None and model_config is not None:
        deck_conditioning = _refresh_fixed_compositional_schema_identity(
            deck_conditioning,
            model_config=model_config,
            state_dict=prepared_state_dict,
        )
    state_dict = _convert_state_dict(
        prepared_state_dict,
        precision=config.precision,
    )
    exported = _runtime_checkpoint_payload(
        checkpoint_mapping,
        state_dict=state_dict,
        config=config,
        model_config=model_config,
        deck_conditioning=deck_conditioning,
        replace_model_config=bool(
            deck_conditioning is not None
            or action_value_declared
            or action_value_state_present
        ),
    )
    exported["export"]["stripped_action_value_tensors"] = len(
        stripped_action_value_keys
    )
    exported["export"]["reanalysis_runtime_included"] = False
    if stripped_direct_policy_keys:
        exported["export"]["direct_policy_only"] = direct_recurrent_policy_metadata(
            stripped_direct_policy_keys
        )
    if config.tensor_storage == BYTE_SHUFFLE_FORMAT:
        packed_state, storage_metadata = byte_shuffle_state_dict(state_dict)
        exported["model_state_dict"] = packed_state
        exported["export"]["tensor_storage"] = storage_metadata
    output_diagnostics: Mapping[str, Any] | None = None
    pruning_equivalence: Mapping[str, Any] | None = None
    source_snapshot: Mapping[str, Any] | None = None
    if deck_conditioning is not None:
        source_snapshot = _policy_output_snapshot(
            config.source_checkpoint,
            deck_path=cast(Path, config.deck_path),
            direct_policy_only=config.direct_policy_only,
        )

    config.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_checkpoint.with_suffix(
        config.output_checkpoint.suffix + ".tmp"
    )
    try:
        torch.save(exported, temporary)
        if deck_conditioning is not None:
            validated_source_snapshot = cast(
                Mapping[str, Any],
                source_snapshot,
            )
            _strict_validate_conditioned_artifact(
                temporary,
                deck_path=cast(Path, config.deck_path),
            )
            final_snapshot = _policy_output_snapshot(
                temporary,
                deck_path=cast(Path, config.deck_path),
                action=cast(
                    tuple[int, ...],
                    validated_source_snapshot["greedy_action"],
                ),
                direct_policy_only=config.direct_policy_only,
            )
            output_diagnostics = _output_delta_diagnostics(
                validated_source_snapshot,
                final_snapshot,
            )
            if bool(deck_conditioning["pruning_applied"]):
                pruning_equivalence = _validate_pruned_fp32_equivalence(
                    checkpoint_mapping,
                    prepared_state_dict=prepared_state_dict,
                    model_config=cast(AgentNetworkConfig, model_config),
                    deck_conditioning=deck_conditioning,
                    config=config,
                    source_snapshot=validated_source_snapshot,
                    stripped_direct_policy_keys=stripped_direct_policy_keys,
                )
        temporary.replace(config.output_checkpoint)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    conversion_diagnostics = _conversion_diagnostics(
        prepared_state_dict,
        state_dict,
    )
    return {
        "source_checkpoint": str(config.source_checkpoint),
        "output_checkpoint": str(config.output_checkpoint),
        "precision": config.precision,
        "parameter_tensors": len(state_dict),
        "floating_tensors": sum(
            1
            for value in state_dict.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
        "numel": sum(
            int(value.numel())
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        ),
        "tensor_bytes": sum(
            int(value.numel()) * int(value.element_size())
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        ),
        "bytes": config.output_checkpoint.stat().st_size,
        "metadata_keys": [key for key in _CHECKPOINT_METADATA_KEYS if key in exported],
        "conversion_diagnostics": conversion_diagnostics,
        "output_conversion_diagnostics": output_diagnostics,
        "pruning_equivalence": pruning_equivalence,
        "deck_conditioning": deck_conditioning,
        "direct_policy_only": config.direct_policy_only,
        "stripped_direct_policy_tensors": len(stripped_direct_policy_keys),
        "tensor_storage": config.tensor_storage,
        "validation": {
            "strict_load": deck_conditioning is not None,
            "prewarm": deck_conditioning is not None,
        },
    }


def export_release_runtime_checkpoint(
    config: ReleaseRuntimeCheckpointExportConfig,
) -> Mapping[str, Any]:
    """Copy the immutable FP16 release asset without changing its bytes."""
    from ptcg_rl.submission.release_assets import load_release_bundle

    bundle = load_release_bundle(config.release_manifest_path)
    source = records.repo_path(bundle.checkpoint_path).resolve()
    output = config.output_checkpoint.resolve()
    if source == output:
        raise ValueError("approved checkpoint export source and output must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != bundle.checkpoint_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("copied release checkpoint SHA256 changed unexpectedly")
    temporary.replace(output)
    return {
        "release_manifest_path": str(config.release_manifest_path),
        "bundle_id": bundle.bundle_id,
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "runtime_sha256": bundle.runtime_sha256,
        "output_checkpoint": str(config.output_checkpoint),
        "bytes": output.stat().st_size,
        "storage_precision": bundle.storage_precision,
        "compute_precision": bundle.compute_precision,
    }


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict = _strip_lightning_model_prefix(unpack_checkpoint_state_dict(checkpoint))
    _verify_tensor_state_dict(state_dict)
    return state_dict


def _strip_lightning_model_prefix(
    state_dict: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _convert_state_dict(
    state_dict: Mapping[str, Any],
    *,
    precision: RuntimeCheckpointPrecision,
) -> dict[str, Any]:
    return {
        str(key): _convert_state_value(value, precision=precision)
        for key, value in state_dict.items()
    }


def _convert_state_value(value: Any, *, precision: RuntimeCheckpointPrecision) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    tensor = value.detach().cpu()
    if not tensor.is_floating_point():
        return tensor
    if precision == "fp16":
        return tensor.to(dtype=torch.float16)
    return tensor.to(dtype=torch.float32)


def _verify_tensor_state_dict(state_dict: Mapping[str, Any]) -> None:
    if not any(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("checkpoint mapping does not contain tensor state_dict values")


def _runtime_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    state_dict: Mapping[str, Any],
    config: RuntimeCheckpointExportConfig,
    model_config: AgentNetworkConfig | None,
    deck_conditioning: Mapping[str, Any] | None,
    replace_model_config: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_state_dict": dict(state_dict)}
    for key in _CHECKPOINT_METADATA_KEYS:
        if key in checkpoint:
            payload[key] = checkpoint[key]
    if model_config is not None and replace_model_config:
        payload["model_config"] = _portable_model_config_payload(model_config)
    if model_config is not None and deck_conditioning is not None:
        conditioning = model_config.deck_conditioning
        if (
            conditioning is not None
            and conditioning.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ):
            for key in ("config", "training_config", "metadata"):
                if key in payload:
                    payload[key] = _remove_v3_lora_schema(payload[key])
    for key in ("config", "training_config"):
        if key in payload:
            payload[key] = _remove_training_only_runtime_schema(payload[key])
    payload["export"] = {
        "precision": config.precision,
        "storage_precision": config.precision,
        "compute_precision": "fp32",
        "source_checkpoint": str(config.source_checkpoint),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if deck_conditioning is not None:
        payload["export"]["deck_conditioning"] = dict(deck_conditioning)
    return payload


def _portable_model_config_payload(
    model_config: AgentNetworkConfig,
) -> dict[str, Any]:
    """Serialize runtime model config without a historical LoRA v3 field."""
    payload = model_config.model_dump(mode="json")
    conditioning = model_config.deck_conditioning
    if (
        conditioning is not None
        and conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
    ):
        raw_conditioning = payload.get("deck_conditioning")
        if not isinstance(raw_conditioning, dict):
            raise ValueError("dense-private model config has no conditioning mapping")
        raw_conditioning.pop("lora", None)
    return payload


def _remove_training_only_runtime_schema(value: Any) -> Any:
    """Remove learner-only service configuration from copied checkpoint metadata."""
    if not isinstance(value, Mapping):
        return value
    payload = {str(key): item for key, item in value.items()}
    payload.pop("amortized_policy_iteration", None)
    raw_model = payload.get("model")
    if isinstance(raw_model, Mapping):
        model_payload = {str(key): item for key, item in raw_model.items()}
        raw_action_value = model_payload.get("action_value")
        if isinstance(raw_action_value, Mapping):
            model_payload["action_value"] = {
                **dict(raw_action_value),
                "enabled": False,
            }
        payload["model"] = model_payload
    return payload


def _remove_v3_lora_schema(value: Any) -> Any:
    """Remove historical LoRA fields only from architecture-v3 mappings."""
    if isinstance(value, BaseModel):
        return _remove_v3_lora_schema(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        architecture_version = value.get("architecture_version")
        return {
            str(key): _remove_v3_lora_schema(item)
            for key, item in value.items()
            if not (
                architecture_version
                == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
                and str(key) == "lora"
            )
        }
    if isinstance(value, tuple):
        return tuple(_remove_v3_lora_schema(item) for item in value)
    if isinstance(value, list):
        return [_remove_v3_lora_schema(item) for item in value]
    return value


def inspect_runtime_checkpoint_deck_conditioning(
    checkpoint_path: Path,
    *,
    deck_path: Path,
    strict_runtime_load: bool = False,
) -> Mapping[str, Any]:
    """Verify and return the exact deck binding carried by a runtime artifact."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("runtime checkpoint must be a mapping")
    checkpoint_mapping = cast(Mapping[str, Any], checkpoint)
    model_config = _checkpoint_model_config(checkpoint_mapping)
    if model_config is None:
        raise ValueError("deck-conditioned runtime checkpoint has no model_config")
    conditioning = model_config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        raise ValueError("runtime checkpoint does not enable deck conditioning")
    deck = canonicalize_deck(records.read_deck(deck_path))
    profile = conditioning.profile_by_signature.get(deck.signature)
    if profile is None:
        raise ValueError("packaged deck has no exact private checkpoint profile")
    export_metadata = checkpoint_mapping.get("export")
    if not isinstance(export_metadata, Mapping):
        raise ValueError("deck-conditioned runtime checkpoint has no export metadata")
    raw_identity = export_metadata.get("deck_conditioning")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("runtime checkpoint has no deck-conditioning identity")
    identity = dict(raw_identity)
    expected_common = {
        "architecture_version": conditioning.architecture_version,
        "canonical_deck_signature": deck.signature,
        "deck_digest": deck.deck_digest,
        "selected_private_profile_module_key": profile.module_key,
        "packaged_private_profile_count": len(conditioning.active_routes),
    }
    for key, expected in expected_common.items():
        if identity.get(key) != expected:
            raise ValueError(f"runtime checkpoint deck binding mismatch: {key}")
    source_registry = identity.get("source_registry_sha256")
    source_checkpoint = identity.get("source_checkpoint_sha256")
    if not _is_sha256(source_registry) or not _is_sha256(source_checkpoint):
        raise ValueError("runtime checkpoint source identities must be SHA256")

    state_dict = _checkpoint_state_dict(checkpoint_mapping)
    configured_keys = {item.module_key for item in conditioning.active_routes}
    state_module_keys = _private_state_module_keys(state_dict)
    if state_module_keys != configured_keys:
        raise ValueError("runtime checkpoint private state/config registry mismatch")
    pruning_applied = identity.get("pruning_applied")
    if not isinstance(pruning_applied, bool):
        raise ValueError("runtime checkpoint pruning identity must be boolean")
    if pruning_applied:
        if configured_keys != {profile.module_key}:
            raise ValueError("pruned checkpoint retains non-selected private profiles")
        if conditioning.architecture_version in {
            DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
        }:
            expected_registry = deck_expert_registry_fingerprint(
                (cast(DeckExpertRoute, profile),)
            )
        else:
            expected_registry = private_registry_fingerprint(
                (cast(PrivateDeckProfile, profile),)
            )
        if conditioning.resolved_registry_sha256 != expected_registry:
            raise ValueError("pruned checkpoint registry fingerprint is invalid")
    elif conditioning.resolved_registry_sha256 != source_registry:
        raise ValueError("full checkpoint registry differs from its source registry")
    raw_lora_merged = identity.get("lora_merged", False)
    if not isinstance(raw_lora_merged, bool):
        raise ValueError("runtime checkpoint LoRA merge flag must be boolean")
    lora_merged = raw_lora_merged
    configured_merged = (
        conditioning.architecture_version == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        and conditioning.lora is not None
        and conditioning.lora.export_mode == "merged"
    )
    if configured_merged != lora_merged:
        raise ValueError("runtime checkpoint LoRA merge config/identity mismatch")
    raw_fixed_strategy = identity.get("fixed_strategy", False)
    if not isinstance(raw_fixed_strategy, bool):
        raise ValueError("runtime checkpoint fixed-strategy flag must be boolean")
    fixed_strategy = raw_fixed_strategy
    configured_fixed_dense = (
        conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        and conditioning.dense_private is not None
        and conditioning.dense_private.export_mode == "fixed"
    )
    configured_fixed_compositional = (
        conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        and conditioning.compositional is not None
        and conditioning.compositional.export_mode == "fixed"
    )
    configured_fixed = configured_fixed_dense or configured_fixed_compositional
    if configured_fixed != fixed_strategy:
        raise ValueError("runtime checkpoint fixed strategy config/identity mismatch")
    if (conditioning.deck_context_mode == "folded") != (lora_merged or fixed_strategy):
        raise ValueError("runtime checkpoint deck-context fold identity mismatch")
    if lora_merged:
        if (
            conditioning.architecture_version
            != DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
            or conditioning.lora is None
            or conditioning.lora.export_mode != "merged"
        ):
            raise ValueError("merged checkpoint config does not disable routed LoRA")
        route = cast(DeckExpertRoute, profile)
        specs, schema_sha256 = _lora_merge_specs_and_schema(model_config)
        expected_merge_identity = {
            "selected_expert_id": route.expert_id,
            "merged_target_count": len(specs),
            "lora_schema_sha256": schema_sha256,
            "deck_context_folded": True,
            "card_encoder_alias_pruned": True,
            "residual_route_retained": True,
        }
        for key, expected in expected_merge_identity.items():
            if identity.get(key) != expected:
                raise ValueError(f"runtime checkpoint LoRA merge mismatch: {key}")
        if any(
            str(key).startswith(
                ("state_encoder.private_lora.", "policy_head.private_lora.")
            )
            for key in state_dict
        ):
            raise ValueError("merged checkpoint still contains routed LoRA factors")
        if conditioning.deck_context_mode != "folded":
            raise ValueError("merged checkpoint did not fold its fixed deck context")
    elif any(
        key in identity
        for key in (
            "selected_expert_id",
            "merged_target_count",
            "lora_schema_sha256",
            "residual_route_retained",
        )
    ):
        raise ValueError("non-merged checkpoint declares LoRA merge identity")
    if fixed_strategy and configured_fixed_dense:
        route = cast(DeckExpertRoute, profile)
        schema_sha256 = _dense_private_strategy_schema_sha256(
            model_config,
            state_dict,
            module_key=route.module_key,
        )
        expected_fixed_identity = {
            "selected_strategy_id": route.expert_id,
            "strategy_schema_sha256": schema_sha256,
            "generic_upper_pruned": True,
            "generic_policy_decision_pruned": True,
            "generic_value_heads_pruned": True,
            "lora_runtime_absent": True,
            "deck_context_folded": True,
            "card_encoder_alias_pruned": True,
        }
        for key, expected in expected_fixed_identity.items():
            if identity.get(key) != expected:
                raise ValueError(f"runtime checkpoint fixed strategy mismatch: {key}")
        dense_config = conditioning.dense_private
        if dense_config is None or dense_config.export_mode != "fixed":
            raise ValueError("fixed checkpoint has no fixed dense-private config")
        if conditioning.lora is not None or _has_lora_state(state_dict):
            raise ValueError("fixed checkpoint retains historical LoRA state")
        raw_model_config = checkpoint_mapping.get("model_config")
        if not isinstance(raw_model_config, Mapping):
            raise ValueError("fixed checkpoint has no portable model config")
        raw_conditioning = raw_model_config.get("deck_conditioning")
        if not isinstance(raw_conditioning, Mapping) or "lora" in raw_conditioning:
            raise ValueError("fixed checkpoint retains a LoRA runtime schema field")
        shared_layers = dense_config.shared_transformer_layers
        removed_prefixes = (
            *(
                f"state_encoder.transformer.layers.{layer_index}."
                for layer_index in range(
                    shared_layers,
                    model_config.state_encoder.num_layers,
                )
            ),
            "state_encoder.layer_norm.",
            *_DENSE_PRIVATE_GENERIC_POLICY_PREFIXES,
            *_DENSE_PRIVATE_GENERIC_VALUE_PREFIXES,
        )
        if any(str(key).startswith(removed_prefixes) for key in state_dict):
            raise ValueError("fixed checkpoint retains generic private duplicates")
        if "policy_head.stop_embedding" in state_dict:
            raise ValueError("fixed checkpoint retains generic STOP embedding")
    elif fixed_strategy and configured_fixed_compositional:
        route = cast(DeckExpertRoute, profile)
        schema_sha256 = fixed_compositional_schema_sha256(
            model_config,
            cast(Mapping[str, torch.Tensor], state_dict),
            module_key=route.module_key,
        )
        merged_projection_count, fixed_film_site_count = (
            compositional_materialization_inventory(model_config)
        )
        expected_fixed_identity = {
            "selected_strategy_id": route.expert_id,
            "strategy_schema_sha256": schema_sha256,
            "fixed_compositional": True,
            "merged_projection_count": merged_projection_count,
            "fixed_film_site_count": fixed_film_site_count,
            "dynamic_router_pruned": True,
            "shared_basis_pruned": True,
            "non_selected_capsules_pruned": True,
            "lora_runtime_absent": True,
            "deck_context_folded": True,
            "card_encoder_alias_pruned": True,
        }
        for key, expected in expected_fixed_identity.items():
            if identity.get(key) != expected:
                raise ValueError(
                    f"runtime checkpoint fixed compositional mismatch: {key}"
                )
        dynamic_prefixes = (
            "state_encoder.compositional_projections.",
            "policy_head.compositional_linears.",
        )
        if any(str(key).startswith(dynamic_prefixes) for key in state_dict):
            raise ValueError("fixed v4 checkpoint retains dynamic basis/router state")
        selected_capsule_prefix = f"exact_capsules.{route.module_key}."
        allowed_capsule_prefixes = tuple(
            selected_capsule_prefix + suffix
            for suffix in (
                "option_adapter.",
                "policy_global_residual.",
                "value_calibrators.",
            )
        )
        if any(
            str(key).startswith("exact_capsules.")
            and not str(key).startswith(allowed_capsule_prefixes)
            for key in state_dict
        ):
            raise ValueError("fixed v4 checkpoint retains mergeable capsule state")
    elif any(
        key in identity
        for key in (
            "selected_strategy_id",
            "strategy_schema_sha256",
            "generic_upper_pruned",
            "generic_policy_decision_pruned",
            "generic_value_heads_pruned",
            "lora_runtime_absent",
        )
    ):
        raise ValueError("non-fixed checkpoint declares fixed-strategy identity")
    if (lora_merged or fixed_strategy) and any(
        str(key).startswith(
            (
                "deck_encoder.",
                "deck_input_projection.",
                "state_encoder.card_encoder.",
            )
        )
        for key in state_dict
    ):
        raise ValueError("folded checkpoint retains removable deck context state")
    if not (lora_merged or fixed_strategy) and any(
        key in identity for key in ("deck_context_folded", "card_encoder_alias_pruned")
    ):
        raise ValueError("non-fixed checkpoint declares deck-context folding")
    if strict_runtime_load:
        _strict_validate_conditioned_artifact(
            checkpoint_path,
            deck_path=deck_path,
        )
    return identity


def _prepare_deck_conditioned_export(
    *,
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig | None,
    config: RuntimeCheckpointExportConfig,
) -> tuple[Mapping[str, Any], AgentNetworkConfig | None, Mapping[str, Any] | None]:
    if model_config is None:
        if config.private_profile_mode != "full":
            raise ValueError("private profile specialization requires a model_config")
        return state_dict, None, None
    conditioning = model_config.deck_conditioning
    if conditioning is None or not conditioning.enabled:
        if config.private_profile_mode != "full":
            raise ValueError("cannot specialize a checkpoint without deck conditioning")
        return state_dict, model_config, None
    if (
        conditioning.architecture_version == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        and conditioning.lora is not None
        and conditioning.lora.export_mode == "merged"
    ):
        raise ValueError("an already merged runtime checkpoint cannot be re-exported")
    if (
        conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        and conditioning.dense_private is not None
        and conditioning.dense_private.export_mode == "fixed"
    ):
        raise ValueError("an already fixed runtime checkpoint cannot be re-exported")
    if (
        conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        and conditioning.compositional is not None
        and conditioning.compositional.export_mode == "fixed"
    ):
        raise ValueError(
            "an already fixed compositional checkpoint cannot be re-exported"
        )
    if config.deck_path is None:
        raise ValueError("enabled checkpoint export requires deck_path")
    deck_path = records.repo_path(config.deck_path).resolve()
    if not deck_path.is_file():
        raise FileNotFoundError(f"runtime export deck does not exist: {deck_path}")
    deck = canonicalize_deck(records.read_deck(deck_path))
    profile = conditioning.profile_by_signature.get(deck.signature)
    if profile is None:
        raise ValueError(
            "selected release deck has no exact private checkpoint profile"
        )
    source_registry = conditioning.resolved_registry_sha256
    if source_registry is None:
        raise ValueError("enabled checkpoint has no resolved private registry")

    prepared_state_dict = state_dict
    prepared_config = model_config
    pruning_applied = config.private_profile_mode in {
        "pruned",
        "merged",
        "fixed",
    }
    merge_identity: Mapping[str, Any] | None = None
    if pruning_applied:
        if conditioning.architecture_version in {
            DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
            DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
        }:
            selected_expert_routes = (cast(DeckExpertRoute, profile),)
            registry_update = {
                "expert_routes": selected_expert_routes,
                "resolved_registry_sha256": deck_expert_registry_fingerprint(
                    selected_expert_routes
                ),
            }
        else:
            selected_private_profiles = (cast(PrivateDeckProfile, profile),)
            registry_update = {
                "private_profiles": selected_private_profiles,
                "resolved_registry_sha256": private_registry_fingerprint(
                    selected_private_profiles
                ),
            }
        pruned_conditioning = conditioning.model_copy(
            update=registry_update,
        )
        prepared_config = model_config.model_copy(
            update={"deck_conditioning": pruned_conditioning}
        )
        prepared_state_dict = {
            str(key): value
            for key, value in state_dict.items()
            if not str(key).startswith(_PRIVATE_STATE_PREFIXES)
            or f".{profile.module_key}." in str(key)
        }
    if config.private_profile_mode == "merged":
        if (
            conditioning.architecture_version
            != DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        ):
            raise ValueError("merged export is reserved for architecture-v2 LoRA")
        prepared_state_dict, prepared_config, merge_identity = (
            _merge_selected_deck_lora(
                state_dict=prepared_state_dict,
                model_config=prepared_config,
                module_key=profile.module_key,
            )
        )
        prepared_state_dict, prepared_config, fold_identity = (
            _fold_selected_deck_context(
                state_dict=prepared_state_dict,
                model_config=prepared_config,
                route=cast(DeckExpertRoute, profile),
            )
        )
        merge_identity = {**merge_identity, **fold_identity}
    elif config.private_profile_mode == "fixed":
        if (
            conditioning.architecture_version
            == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        ):
            prepared_state_dict, prepared_config, merge_identity = (
                _fix_selected_dense_private_strategy(
                    state_dict=prepared_state_dict,
                    model_config=prepared_config,
                    route=cast(DeckExpertRoute, profile),
                )
            )
        elif (
            conditioning.architecture_version
            == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        ):
            prepared_state_dict, prepared_config, merge_identity = (
                fix_selected_compositional_strategy(
                    state_dict=prepared_state_dict,
                    model_config=prepared_config,
                    route=cast(DeckExpertRoute, profile),
                )
            )
        else:
            raise ValueError("fixed export requires architecture-v3 or v4")
        prepared_state_dict, prepared_config, fold_identity = (
            _fold_selected_deck_context(
                state_dict=prepared_state_dict,
                model_config=prepared_config,
                route=cast(DeckExpertRoute, profile),
            )
        )
        merge_identity = {**merge_identity, **fold_identity}

    packaged_conditioning = cast(
        Any,
        prepared_config.deck_conditioning,
    )
    identity: dict[str, Any] = {
        "architecture_version": conditioning.architecture_version,
        "canonical_deck_signature": deck.signature,
        "deck_digest": deck.deck_digest,
        "selected_private_profile_module_key": profile.module_key,
        "source_registry_sha256": source_registry,
        "source_checkpoint_sha256": file_sha256(config.source_checkpoint),
        "packaged_private_profile_count": len(packaged_conditioning.active_routes),
        "pruning_applied": pruning_applied,
    }
    if merge_identity is not None:
        identity.update(merge_identity)
    return prepared_state_dict, prepared_config, identity


def _refresh_fixed_compositional_schema_identity(
    identity: Mapping[str, Any],
    *,
    model_config: AgentNetworkConfig,
    state_dict: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind a fixed-v4 schema identity to the final direct-actor topology."""
    if identity.get("fixed_compositional") is not True:
        return identity
    conditioning = model_config.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or conditioning.compositional is None
        or conditioning.compositional.export_mode != "fixed"
        or len(conditioning.expert_routes) != 1
    ):
        raise ValueError("fixed compositional identity has no unique fixed route")
    route = conditioning.expert_routes[0]
    tensor_state = cast(Mapping[str, torch.Tensor], state_dict)
    return {
        **identity,
        "strategy_schema_sha256": fixed_compositional_schema_sha256(
            model_config,
            tensor_state,
            module_key=route.module_key,
        ),
    }


def _merge_selected_deck_lora(
    *,
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
    module_key: str,
) -> tuple[Mapping[str, Any], AgentNetworkConfig, Mapping[str, Any]]:
    """Merge one selected architecture-v2 LoRA expert into FP32 base weights."""
    conditioning = model_config.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        or conditioning.lora is None
    ):
        raise ValueError("merged export requires architecture-v2 LoRA")
    if conditioning.lora.export_mode != "routed":
        raise ValueError("source LoRA checkpoint must use routed export mode")
    specs, schema_sha256 = _lora_merge_specs_and_schema(model_config)
    expected_factor_keys = {
        spec.factor_key(module_key, factor)
        for spec in specs
        for factor in ("lora_a", "lora_b")
    }
    actual_factor_keys = {
        str(key)
        for key in state_dict
        if str(key).startswith(
            ("state_encoder.private_lora.", "policy_head.private_lora.")
        )
    }
    if actual_factor_keys != expected_factor_keys:
        missing = sorted(expected_factor_keys - actual_factor_keys)
        extra = sorted(actual_factor_keys - expected_factor_keys)
        raise ValueError(
            f"LoRA merge factor inventory mismatch: missing={missing}, extra={extra}"
        )

    merged_state = dict(state_dict)
    for spec in specs:
        _merge_lora_spec(
            merged_state,
            spec=spec,
            module_key=module_key,
            rank=conditioning.lora.rank,
            scaling=conditioning.lora.alpha / conditioning.lora.rank,
        )
    for key in actual_factor_keys:
        merged_state.pop(key)

    merged_lora = conditioning.lora.model_copy(update={"export_mode": "merged"})
    merged_conditioning = conditioning.model_copy(update={"lora": merged_lora})
    merged_config = model_config.model_copy(
        update={"deck_conditioning": merged_conditioning}
    )
    route = next(
        (item for item in conditioning.active_routes if item.module_key == module_key),
        None,
    )
    if route is None or not hasattr(route, "expert_id"):
        raise ValueError("selected LoRA module has no expert identity")
    return (
        merged_state,
        merged_config,
        {
            "selected_expert_id": cast(Any, route).expert_id,
            "lora_merged": True,
            "merged_target_count": len(specs),
            "lora_schema_sha256": schema_sha256,
        },
    )


def _fix_selected_dense_private_strategy(
    *,
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
    route: DeckExpertRoute,
) -> tuple[Mapping[str, Any], AgentNetworkConfig, Mapping[str, Any]]:
    """Prune generic duplicates from one fixed architecture-v3 strategy."""
    conditioning = model_config.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        or conditioning.dense_private is None
    ):
        raise ValueError("fixed strategy export requires architecture-v3 conditioning")
    if conditioning.dense_private.export_mode != "routed":
        raise ValueError("source dense-private checkpoint must use routed export mode")
    if len(conditioning.expert_routes) != 1:
        raise ValueError("fixed strategy export requires one pruned exact-deck route")
    if conditioning.lora is not None or _has_lora_state(state_dict):
        raise ValueError("architecture-v3 fixed export cannot contain LoRA state")
    if conditioning.expert_routes[0].expert_id != route.expert_id:
        raise ValueError("fixed strategy route does not match the pruned registry")

    shared_layers = conditioning.dense_private.shared_transformer_layers
    upper_layer_prefixes = tuple(
        f"state_encoder.transformer.layers.{layer_index}."
        for layer_index in range(
            shared_layers,
            model_config.state_encoder.num_layers,
        )
    )
    removable_prefixes = (
        *upper_layer_prefixes,
        "state_encoder.layer_norm.",
        *_DENSE_PRIVATE_GENERIC_POLICY_PREFIXES,
        *_DENSE_PRIVATE_GENERIC_VALUE_PREFIXES,
    )
    removable_exact = {"policy_head.stop_embedding"}
    _require_state_prefix_inventory(
        state_dict,
        prefixes=removable_prefixes,
        exact=removable_exact,
        label="generic dense-private duplicate",
    )
    selected_prefixes = tuple(
        f"{prefix}{route.module_key}."
        for prefix in (
            "state_encoder.private_strategy_stacks.",
            "policy_head.private_strategies.",
            "dense_private_root_value_heads.",
            "dense_private_prefix_value_heads.",
        )
    )
    _require_state_prefix_inventory(
        state_dict,
        prefixes=selected_prefixes,
        exact=set(),
        label="selected dense-private strategy",
    )

    fixed_state = {
        str(key): value
        for key, value in state_dict.items()
        if str(key) not in removable_exact
        and not str(key).startswith(removable_prefixes)
    }
    fixed_dense_config = conditioning.dense_private.model_copy(
        update={"export_mode": "fixed"}
    )
    fixed_conditioning = conditioning.model_copy(
        update={"dense_private": fixed_dense_config}
    )
    fixed_config = model_config.model_copy(
        update={"deck_conditioning": fixed_conditioning}
    )
    schema_sha256 = _dense_private_strategy_schema_sha256(
        fixed_config,
        fixed_state,
        module_key=route.module_key,
    )
    return (
        fixed_state,
        fixed_config,
        {
            "selected_strategy_id": route.expert_id,
            "fixed_strategy": True,
            "strategy_schema_sha256": schema_sha256,
            "generic_upper_pruned": True,
            "generic_policy_decision_pruned": True,
            "generic_value_heads_pruned": True,
            "lora_runtime_absent": True,
        },
    )


def _require_state_prefix_inventory(
    state_dict: Mapping[str, Any],
    *,
    prefixes: tuple[str, ...],
    exact: set[str],
    label: str,
) -> None:
    """Require every declared removable/retained state component to exist."""
    state_keys = {str(key) for key in state_dict}
    missing_prefixes = tuple(
        prefix
        for prefix in prefixes
        if not any(key.startswith(prefix) for key in state_keys)
    )
    missing_exact = tuple(sorted(exact - state_keys))
    if missing_prefixes or missing_exact:
        raise ValueError(
            f"{label} inventory mismatch: "
            f"missing_prefixes={missing_prefixes}, missing_exact={missing_exact}"
        )


def _dense_private_strategy_schema_sha256(
    model_config: AgentNetworkConfig,
    state_dict: Mapping[str, Any],
    *,
    module_key: str,
) -> str:
    """Fingerprint the route-independent fixed private parameter topology."""
    conditioning = model_config.deck_conditioning
    if conditioning is None or conditioning.dense_private is None:
        raise ValueError("dense-private schema requires dense-private configuration")
    selected_prefixes = tuple(
        f"{prefix}{module_key}."
        for prefix in (
            "state_encoder.private_strategy_stacks.",
            "policy_head.private_strategies.",
            "dense_private_root_value_heads.",
            "dense_private_prefix_value_heads.",
        )
    )
    normalized_tensors: list[dict[str, Any]] = []
    for raw_key, raw_value in state_dict.items():
        key = str(raw_key)
        matching_prefix = next(
            (prefix for prefix in selected_prefixes if key.startswith(prefix)),
            None,
        )
        if matching_prefix is None:
            continue
        if not isinstance(raw_value, torch.Tensor):
            raise ValueError(f"dense-private state contains a non-tensor: {key}")
        normalized_tensors.append(
            {
                "key": key.replace(module_key, "{strategy}", 1),
                "shape": list(raw_value.shape),
            }
        )
    if not normalized_tensors:
        raise ValueError("fixed dense-private strategy has no private tensors")
    normalized_tensors.sort(key=lambda item: cast(str, item["key"]))
    return fingerprint_payload(
        {
            "schema": "dense-private-fixed-v1",
            "state_encoder": model_config.state_encoder.model_dump(mode="json"),
            "policy": model_config.policy.model_dump(mode="json"),
            "value_hidden_dim": model_config.value_hidden_dim,
            "adapter_layer_indices": list(conditioning.adapter_layer_indices),
            "adapter_bottleneck_dim": conditioning.adapter_bottleneck_dim,
            "policy_bottleneck_dim": conditioning.policy_bottleneck_dim,
            "value_bottleneck_dim": conditioning.value_bottleneck_dim,
            "dense_private": conditioning.dense_private.model_dump(
                mode="json",
                exclude={"export_mode"},
            ),
            "private_tensors": normalized_tensors,
        }
    )


def _has_lora_state(state_dict: Mapping[str, Any]) -> bool:
    """Return whether any historical routed LoRA tensor remains."""
    return any(
        str(key).startswith(
            ("state_encoder.private_lora.", "policy_head.private_lora.")
        )
        for key in state_dict
    )


def _lora_merge_specs_and_schema(
    model_config: AgentNetworkConfig,
) -> tuple[tuple[RoutedLoRAMergeSpec, ...], str]:
    """Rebuild the declared merge targets and their portable schema identity."""
    conditioning = model_config.deck_conditioning
    if conditioning is None or conditioning.lora is None:
        raise ValueError("LoRA merge schema requires architecture-v2 conditioning")
    specs = routed_lora_merge_specs(
        transformer_layer_indices=(
            conditioning.lora.resolved_transformer_layers(
                num_layers=model_config.state_encoder.num_layers
            )
        ),
        transformer_targets=conditioning.lora.transformer_targets,
        policy_targets=conditioning.lora.policy_targets,
    )
    schema_payload = {
        "schema": "deck-lora-merge-v1",
        "rank": conditioning.lora.rank,
        "alpha": conditioning.lora.alpha,
        "targets": [
            {
                "target_id": spec.target_id,
                "base_weight_key": spec.base_weight_key,
                "adapter_bank_prefix": spec.adapter_bank_prefix,
            }
            for spec in specs
        ],
    }
    return specs, fingerprint_payload(schema_payload)


def _fold_selected_deck_context(
    *,
    state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
    route: DeckExpertRoute,
) -> tuple[Mapping[str, Any], AgentNetworkConfig, Mapping[str, Any]]:
    """Fold one fixed deck's shared encoder residual into the global token kind."""
    conditioning = model_config.deck_conditioning
    merged_lora = (
        conditioning is not None
        and conditioning.architecture_version
        == DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        and conditioning.lora is not None
        and conditioning.lora.export_mode == "merged"
    )
    fixed_dense_private = (
        conditioning is not None
        and conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        and conditioning.dense_private is not None
        and conditioning.dense_private.export_mode == "fixed"
    )
    fixed_compositional = (
        conditioning is not None
        and conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        and conditioning.compositional is not None
        and conditioning.compositional.export_mode == "fixed"
    )
    if (
        conditioning is None
        or not (merged_lora or fixed_dense_private or fixed_compositional)
        or len(conditioning.expert_routes) != 1
    ):
        raise ValueError("deck-context folding requires one fixed private route")
    d_model = model_config.state_encoder.d_model
    card_encoder = (
        build_card_encoder(model_config.card_encoder)
        if model_config.card_encoder is not None
        else CardEncoder(d_model=d_model)
    )
    deck_encoder = DeckEncoder(d_model, conditioning.encoder_hidden_dim)
    deck_projection = nn.Linear(d_model, d_model)
    _load_prefixed_module_state(
        card_encoder,
        state_dict,
        prefix="card_encoder.",
    )
    _load_prefixed_module_state(
        deck_encoder,
        state_dict,
        prefix="deck_encoder.",
    )
    _load_prefixed_module_state(
        deck_projection,
        state_dict,
        prefix="deck_input_projection.",
    )
    card_encoder.eval().float()
    deck_encoder.eval().float()
    deck_projection.eval().float()
    cards = torch.tensor((route.canonical_card_ids,), dtype=torch.long)
    with torch.no_grad():
        deck_embedding = deck_encoder(cards, card_encoder=card_encoder)
        residual = deck_projection(deck_embedding).squeeze(0)
    if not torch.isfinite(residual).all():
        raise ValueError("fixed deck context produced a non-finite residual")

    kind_key = "state_encoder.kind_embedding.weight"
    raw_kind_weight = state_dict.get(kind_key)
    if not isinstance(raw_kind_weight, torch.Tensor):
        raise ValueError("checkpoint has no state token-kind embedding")
    kind_weight = raw_kind_weight.detach().cpu().float().clone()
    global_kind_index = TOKEN_KIND_TO_INDEX["global"]
    if kind_weight.ndim != 2 or tuple(kind_weight.shape[1:]) != (d_model,):
        raise ValueError("state token-kind embedding has an invalid shape")
    kind_weight[global_kind_index] += residual
    if not torch.isfinite(kind_weight).all():
        raise ValueError("folded global token embedding is non-finite")

    folded_state = {
        str(key): value
        for key, value in state_dict.items()
        if not str(key).startswith(
            (
                "deck_encoder.",
                "deck_input_projection.",
                "state_encoder.card_encoder.",
            )
        )
    }
    folded_state[kind_key] = kind_weight
    folded_conditioning = conditioning.model_copy(
        update={"deck_context_mode": "folded"}
    )
    folded_config = model_config.model_copy(
        update={"deck_conditioning": folded_conditioning}
    )
    fold_identity = {
        "deck_context_folded": True,
        "card_encoder_alias_pruned": True,
    }
    if merged_lora:
        fold_identity["residual_route_retained"] = True
    return (
        folded_state,
        folded_config,
        fold_identity,
    )


def _load_prefixed_module_state(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
) -> None:
    """Strictly load one module subtree as FP32 for offline folding."""
    expected_keys = set(module.state_dict())
    selected = {
        str(key).removeprefix(prefix): value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }
    if set(selected) != expected_keys:
        raise ValueError(
            f"checkpoint subtree inventory mismatch for {prefix}: "
            f"missing={sorted(expected_keys - set(selected))}, "
            f"extra={sorted(set(selected) - expected_keys)}"
        )
    module.load_state_dict(selected, strict=True)


def _merge_lora_spec(
    state_dict: dict[str, Any],
    *,
    spec: RoutedLoRAMergeSpec,
    module_key: str,
    rank: int,
    scaling: float,
) -> None:
    """Validate and merge one declared A/B pair into its base weight."""
    base = state_dict.get(spec.base_weight_key)
    lora_a = state_dict.get(spec.factor_key(module_key, "lora_a"))
    lora_b = state_dict.get(spec.factor_key(module_key, "lora_b"))
    if not all(isinstance(value, torch.Tensor) for value in (base, lora_a, lora_b)):
        raise ValueError(f"LoRA merge target contains a non-tensor: {spec.target_id}")
    source_tensors = tuple(
        cast(torch.Tensor, value) for value in (base, lora_a, lora_b)
    )
    if any(not value.is_floating_point() for value in source_tensors):
        raise ValueError(
            f"LoRA merge tensors must be real floating point: {spec.target_id}"
        )
    if not all(torch.isfinite(value).all() for value in source_tensors):
        raise ValueError(f"LoRA merge tensors must be finite: {spec.target_id}")
    base_tensor, a_tensor, b_tensor = (
        value.detach().cpu().float() for value in source_tensors
    )
    if base_tensor.ndim != 2:
        raise ValueError(f"LoRA base weight must be two-dimensional: {spec.target_id}")
    expected_a_shape = (rank, int(base_tensor.shape[1]))
    expected_b_shape = (int(base_tensor.shape[0]), rank)
    if tuple(a_tensor.shape) != expected_a_shape:
        raise ValueError(f"LoRA A shape mismatch: {spec.target_id}")
    if tuple(b_tensor.shape) != expected_b_shape:
        raise ValueError(f"LoRA B shape mismatch: {spec.target_id}")
    delta = scaling * torch.matmul(b_tensor, a_tensor)
    merged = base_tensor + delta
    if not torch.isfinite(delta).all() or not torch.isfinite(merged).all():
        raise ValueError(f"LoRA merge overflowed FP32: {spec.target_id}")
    state_dict[spec.base_weight_key] = merged


def _checkpoint_model_config(
    checkpoint: Mapping[str, Any],
) -> AgentNetworkConfig | None:
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        value = full_config.get("model")
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    return None


def _private_state_module_keys(state_dict: Mapping[str, Any]) -> set[str]:
    configured: set[str] = set()
    for raw_key in state_dict:
        key = str(raw_key)
        if not key.startswith(_PRIVATE_STATE_PREFIXES):
            continue
        for chunk in key.split("."):
            if chunk.startswith("deck_") and len(chunk) == 69:
                configured.add(chunk)
    return configured


def _strict_validate_conditioned_artifact(
    checkpoint_path: Path,
    *,
    deck_path: Path,
) -> None:
    from ptcg_rl.actions.selection import is_legal_action
    from ptcg_rl.agent.runtime import CheckpointPolicy

    deck = records.read_deck(deck_path)
    policy = CheckpointPolicy(checkpoint_path, device="cpu", own_deck=deck)
    if policy.selected_private_profile_module_key is None:
        raise ValueError("runtime artifact did not select its private deck module")
    policy.prewarm()
    observation = _integrity_observation()
    _bind_integrity_runtime_context(policy, observation)
    try:
        action = policy.select_action(observation)
        logits = policy.first_step_logits(observation)
        value = (
            None
            if policy.direct_policy_only
            else policy.value(observation, root_player_index=0)
        )
    finally:
        policy.abort_runtime_decision()
    if not is_legal_action(observation["select"], action):
        raise ValueError("runtime artifact decoded an illegal integrity action")
    if len(logits) != 3 or not all(math.isfinite(value) for value in logits[:2]):
        raise ValueError("runtime artifact emitted invalid legal-option logits")
    if math.isfinite(logits[2]):
        raise ValueError("runtime artifact failed the minimum-count STOP mask")
    if value is not None and not math.isfinite(value):
        raise ValueError("runtime artifact emitted a non-finite root value")


def _integrity_observation() -> dict[str, Any]:
    return {
        "current": {
            "yourIndex": 0,
            "result": -1,
            "players": [{"prize": [None] * 6}, {"prize": [None] * 6}],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}, {"type": 2}],
        },
    }


def _bind_integrity_runtime_context(policy: Any, observation: Any) -> None:
    """Bind an empty production-shaped event delta for recurrent asset probes."""
    if not bool(getattr(policy, "recurrent_enabled", False)):
        return
    from ptcg_rl.context import GameContext

    context = GameContext(player_index=0)
    context_features = context.update(observation)
    policy.bind_runtime_context(
        context_snapshot=context.snapshot(),
        context_features=context_features,
        deadline_monotonic=float("inf"),
    )


def _conversion_diagnostics(
    source: Mapping[str, Any],
    converted: Mapping[str, Any],
) -> Mapping[str, float | int]:
    total_absolute_error = 0.0
    total_numel = 0
    maximum_absolute_error = 0.0
    for key, source_value in source.items():
        converted_value = converted.get(key)
        if (
            not isinstance(source_value, torch.Tensor)
            or not source_value.is_floating_point()
            or not isinstance(converted_value, torch.Tensor)
        ):
            continue
        difference = (
            source_value.detach().cpu().float() - converted_value.detach().cpu().float()
        ).abs()
        total_absolute_error += float(difference.sum().item())
        total_numel += int(difference.numel())
        if difference.numel():
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(difference.max().item()),
            )
    return {
        "floating_numel": total_numel,
        "max_abs": maximum_absolute_error,
        "mean_abs": (total_absolute_error / total_numel if total_numel else 0.0),
    }


def _policy_output_snapshot(
    checkpoint_path: Path,
    *,
    deck_path: Path,
    action: tuple[int, ...] | None = None,
    direct_policy_only: bool = False,
) -> Mapping[str, Any]:
    from ptcg_rl.agent.runtime import CheckpointPolicy

    policy = CheckpointPolicy(
        checkpoint_path,
        device="cpu",
        own_deck=records.read_deck(deck_path),
    )
    observation = _integrity_observation()
    _bind_integrity_runtime_context(policy, observation)
    try:
        greedy_action = policy.select_action(observation)
        evaluated_action = greedy_action if action is None else action
        snapshot: dict[str, Any] = {
            "greedy_action": greedy_action,
            "first_step_logits": policy.first_step_logits(observation),
        }
        if not direct_policy_only:
            snapshot["conditioned"] = policy.conditioned_diagnostics(
                observation,
                evaluated_action,
            )
        return snapshot
    finally:
        policy.abort_runtime_decision()


def _validate_pruned_fp32_equivalence(
    checkpoint: Mapping[str, Any],
    *,
    prepared_state_dict: Mapping[str, Any],
    model_config: AgentNetworkConfig,
    deck_conditioning: Mapping[str, Any],
    config: RuntimeCheckpointExportConfig,
    source_snapshot: Mapping[str, Any],
    stripped_direct_policy_keys: tuple[str, ...],
) -> Mapping[str, Any]:
    validation_path = config.output_checkpoint.with_suffix(
        config.output_checkpoint.suffix + ".pruned-fp32.tmp"
    )
    fp32_config = config.model_copy(update={"precision": "fp32"})
    fp32_payload = _runtime_checkpoint_payload(
        checkpoint,
        state_dict=_convert_state_dict(prepared_state_dict, precision="fp32"),
        config=fp32_config,
        model_config=model_config,
        deck_conditioning=deck_conditioning,
        replace_model_config=True,
    )
    if stripped_direct_policy_keys:
        fp32_payload["export"]["direct_policy_only"] = direct_recurrent_policy_metadata(
            stripped_direct_policy_keys
        )
    try:
        torch.save(fp32_payload, validation_path)
        pruned_snapshot = _policy_output_snapshot(
            validation_path,
            deck_path=cast(Path, config.deck_path),
            action=cast(tuple[int, ...], source_snapshot["greedy_action"]),
            direct_policy_only=config.direct_policy_only,
        )
    finally:
        validation_path.unlink(missing_ok=True)
    diagnostics = _output_delta_diagnostics(source_snapshot, pruned_snapshot)
    if diagnostics["nonfinite_mismatches"] != 0:
        raise ValueError("FP32 pruned checkpoint changed output masks")
    lora_merged = bool(deck_conditioning.get("lora_merged", False))
    deck_context_folded = bool(deck_conditioning.get("deck_context_folded", False))
    allows_fp32_roundoff = lora_merged or deck_context_folded
    maximum_tolerance = 1.0e-5 if allows_fp32_roundoff else 0.0
    if float(diagnostics["max_abs"]) > maximum_tolerance:
        raise ValueError(
            "FP32 specialized checkpoint changed selected-deck outputs: "
            f"{dict(diagnostics)}"
        )
    # Merging LoRA projections and folding the deck residual into the global
    # token-kind embedding both change FP32 addition order without changing the
    # selected-deck function. Direct-policy exports expose only the two finite
    # integrity-probe logits, so keep the mean allowance five times below the
    # per-value maximum while covering their expected roundoff.
    mean_tolerance = 2.0e-6 if allows_fp32_roundoff else 0.0
    if float(diagnostics["mean_abs"]) > mean_tolerance:
        raise ValueError(
            "FP32 specialized checkpoint accumulated excessive error: "
            f"{dict(diagnostics)}"
        )
    if diagnostics["greedy_disagreement"]:
        raise ValueError("FP32 pruned checkpoint changed the greedy action")
    return diagnostics


def _output_delta_diagnostics(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> Mapping[str, float | int | bool]:
    source_values = tuple(_nested_float_values(source))
    target_values = tuple(_nested_float_values(target))
    if len(source_values) != len(target_values):
        raise ValueError("checkpoint output diagnostics have different shapes")
    total_absolute_error = 0.0
    finite_values = 0
    maximum_absolute_error = 0.0
    nonfinite_mismatches = 0
    for expected, actual in zip(source_values, target_values, strict=True):
        if not math.isfinite(expected) or not math.isfinite(actual):
            if expected != actual:
                nonfinite_mismatches += 1
            continue
        difference = abs(expected - actual)
        total_absolute_error += difference
        finite_values += 1
        maximum_absolute_error = max(maximum_absolute_error, difference)
    return {
        "finite_values": finite_values,
        "nonfinite_mismatches": nonfinite_mismatches,
        "max_abs": maximum_absolute_error,
        "mean_abs": (total_absolute_error / finite_values if finite_values else 0.0),
        "greedy_disagreement": (
            source.get("greedy_action") != target.get("greedy_action")
        ),
    }


def _nested_float_values(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        flattened: list[float] = []
        for key in sorted(value):
            if key == "greedy_action":
                continue
            flattened.extend(_nested_float_values(value[key]))
        return flattened
    if isinstance(value, (tuple, list)):
        flattened = []
        for item in value:
            flattened.extend(_nested_float_values(item))
        return flattened
    if isinstance(value, (float, int)):
        return [float(value)]
    raise TypeError(f"unsupported checkpoint diagnostic value: {type(value)!r}")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ReleaseRuntimeCheckpointExportConfig",
    "RuntimeCheckpointExportConfig",
    "RuntimeCheckpointPrecision",
    "RuntimeCheckpointPrivateProfileMode",
    "RuntimeCheckpointTensorStorage",
    "export_release_runtime_checkpoint",
    "export_runtime_checkpoint",
    "inspect_runtime_checkpoint_deck_conditioning",
]
