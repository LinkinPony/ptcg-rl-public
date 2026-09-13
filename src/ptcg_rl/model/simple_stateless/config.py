"""Validated topology and identity for the clean stateless policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.cards.card_encoder import CardEncoderConfig
from ptcg_rl.decks.registry import (
    DeckExpertRoute,
    DeckFamilyRoute,
    ResolvedDeckExpertRegistry,
    ResolvedDeckFamilyRegistry,
    validate_active_exact_strategy_routes,
)
from ptcg_rl.model.network import AgentNetworkConfig
from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_ARCHITECTURE,
    GENERALIST_SEQUENCE_V2_ARCHITECTURE,
    GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    GeneralistSequenceConfig,
)

SIMPLE_STATELESS_ARCHITECTURE = "simple_stateless_v1"
SIMPLE_STATELESS_V2_ARCHITECTURE = "simple_stateless_v2"
LEGACY_POINTER_ARCHITECTURE = "legacy_pointer"
SIMPLE_BELIEF_TARGET_SEMANTICS = "unidentified_cards_sparse_v1"


class SimpleStatelessModelConfig(BaseModel):
    """Immutable model topology for the fresh stateless policy lineage.

    Architecture dimensions are literals on purpose. Experiments that need a
    different topology must introduce a new discriminator rather than silently
    changing the meaning of an existing checkpoint identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: Literal[
        "simple_stateless_v1",
        "simple_stateless_v2",
        "generalist_sequence_v1",
        "generalist_sequence_v2",
        "generalist_sequence_v3",
    ] = "simple_stateless_v1"
    d_model: Literal[512] = 512
    num_layers: Literal[20, 23, 34] = 34
    attention_heads: Literal[8] = 8
    feedforward_dim: Literal[1024] = 1024
    scratch_tokens: Literal[8] = 8
    dropout: float = Field(default=0.0, ge=0.0, le=0.0)
    norm_first: Literal[True] = True
    activation: Literal["gelu"] = "gelu"
    option_comparator_layers: Literal[1] = 1
    option_entity_slots: Literal[2] = 2
    deck_size: Literal[60] = 60
    deck_hidden_dim: Literal[512] = 512
    exact_residual_bottleneck_dim: Literal[64] = 64
    policy_output_gain: float = Field(default=0.01, gt=0.0)
    belief_target_semantics: Literal["unidentified_cards_sparse_v1"] = (
        "unidentified_cards_sparse_v1"
    )
    public_deck_catalog_fingerprint: str | None = None
    card_encoder: CardEncoderConfig = Field(
        default_factory=lambda: CardEncoderConfig(
            d_model=512,
            hidden_dim=512,
            dropout=0.0,
        )
    )
    exact_routes: tuple[DeckExpertRoute, ...] = ()
    resolved_registry_sha256: str | None = None
    family_routes: tuple[DeckFamilyRoute, ...] = ()
    resolved_family_registry_sha256: str | None = None
    export_mode: Literal["routed", "fixed"] = "routed"
    sequence: GeneralistSequenceConfig | None = None

    @field_validator("public_deck_catalog_fingerprint")
    @classmethod
    def valid_optional_catalog_fingerprint(cls, value: str | None) -> str | None:
        """Validate the resolved catalog identity persisted in checkpoints."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("public catalog fingerprint must contain 64 hex digits")
        return normalized

    @model_validator(mode="after")
    def consistent_architecture(self) -> SimpleStatelessModelConfig:
        """Bind shared widths and the path-free exact-route registry."""
        expected_snapshot_layers = {
            GENERALIST_SEQUENCE_V2_ARCHITECTURE: 20,
            GENERALIST_SEQUENCE_V3_ARCHITECTURE: 23,
        }.get(self.architecture, 34)
        if self.num_layers != expected_snapshot_layers:
            raise ValueError(
                f"{self.architecture} requires exactly "
                f"{expected_snapshot_layers} snapshot layers"
            )
        if self.card_encoder.d_model != self.d_model:
            raise ValueError("CardEncoder width must equal the shared model width")
        if self.card_encoder.dropout != 0.0:
            raise ValueError("simple stateless CardEncoder dropout must be zero")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("shared width must be divisible by attention heads")
        if self.export_mode == "fixed" and len(self.exact_routes) != 1:
            raise ValueError("fixed export requires exactly one exact-deck route")
        sequence_architectures = {
            GENERALIST_SEQUENCE_ARCHITECTURE,
            GENERALIST_SEQUENCE_V2_ARCHITECTURE,
            GENERALIST_SEQUENCE_V3_ARCHITECTURE,
        }
        if self.sequence is None:
            if self.architecture in sequence_architectures:
                raise ValueError(
                    "generalist sequence architecture requires sequence config"
                )
        elif self.architecture not in sequence_architectures:
            raise ValueError(
                "sequence config is exclusive to generalist sequence architecture"
            )
        elif self.sequence.d_model != self.d_model:
            raise ValueError("temporal and snapshot widths must match")
        if uses_exact_v2_topology(self) and self.exact_routes:
            validate_active_exact_strategy_routes(self.exact_routes)
        if self.resolved_registry_sha256 is not None:
            ResolvedDeckExpertRegistry(
                routes=self.exact_routes,
                resolved_registry_sha256=self.resolved_registry_sha256,
            )
        elif self.exact_routes:
            raise ValueError("exact routes require a resolved registry fingerprint")
        if uses_family_private_topology(self):
            unresolved = (
                not self.exact_routes
                and self.resolved_registry_sha256 is None
                and not self.family_routes
                and self.resolved_family_registry_sha256 is None
            )
            if unresolved:
                return self
            if not self.family_routes:
                raise ValueError("family-private topology requires family routes")
            exact_digests = tuple(route.deck_digest for route in self.exact_routes)
            family_digests = tuple(route.deck_digest for route in self.family_routes)
            if exact_digests != family_digests:
                raise ValueError("exact and family routes must cover the same decks")
            if self.resolved_family_registry_sha256 is None:
                raise ValueError(
                    "family routes require a resolved family registry fingerprint"
                )
            ResolvedDeckFamilyRegistry(
                routes=self.family_routes,
                resolved_registry_sha256=self.resolved_family_registry_sha256,
            )
            if self.export_mode == "fixed" and len(self.family_routes) != 1:
                raise ValueError("fixed export requires exactly one family route")
        elif self.family_routes or self.resolved_family_registry_sha256 is not None:
            raise ValueError("family routes are exclusive to family-private topology")
        return self


PolicyModelConfig: TypeAlias = AgentNetworkConfig | SimpleStatelessModelConfig


def parse_policy_model_config(
    value: PolicyModelConfig | Mapping[str, Any],
) -> PolicyModelConfig:
    """Parse a new discriminated config or a discriminator-free legacy config."""
    if isinstance(value, (AgentNetworkConfig, SimpleStatelessModelConfig)):
        return value
    architecture = value.get("architecture")
    if architecture is None:
        return AgentNetworkConfig.model_validate(value)
    if architecture in {
        SIMPLE_STATELESS_ARCHITECTURE,
        SIMPLE_STATELESS_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_ARCHITECTURE,
        GENERALIST_SEQUENCE_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    }:
        return SimpleStatelessModelConfig.model_validate(value)
    raise ValueError(f"unsupported model architecture discriminator: {architecture!r}")


def model_architecture_name(config: PolicyModelConfig) -> str:
    """Return the stable architecture identity used in artifact manifests."""
    if isinstance(config, SimpleStatelessModelConfig):
        return config.architecture
    return LEGACY_POINTER_ARCHITECTURE


def uses_exact_v2_topology(config: SimpleStatelessModelConfig) -> bool:
    """Return whether the model carries dense-private V2 exact-route paths."""
    return config.architecture in {
        SIMPLE_STATELESS_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_ARCHITECTURE,
        GENERALIST_SEQUENCE_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    }


def uses_generalist_sequence(config: SimpleStatelessModelConfig) -> bool:
    """Return whether the post-backbone causal temporal branch is active."""
    return config.architecture in {
        GENERALIST_SEQUENCE_ARCHITECTURE,
        GENERALIST_SEQUENCE_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    }


def uses_temporal_prefusion(config: SimpleStatelessModelConfig) -> bool:
    """Return whether temporal context conditions entities and options early."""
    return config.architecture in {
        GENERALIST_SEQUENCE_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    }


def uses_wdl_critic(config: SimpleStatelessModelConfig) -> bool:
    """Return whether root value uses the distributional WDL critic."""
    return config.architecture in {
        GENERALIST_SEQUENCE_V2_ARCHITECTURE,
        GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    }


def uses_family_private_topology(config: SimpleStatelessModelConfig) -> bool:
    """Return whether exact decks share routed private upper Transformer tails."""
    return config.architecture == GENERALIST_SEQUENCE_V3_ARCHITECTURE
