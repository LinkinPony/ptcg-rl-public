"""Privacy-safe, policy-independent prospective engine facts."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context.game import GameContextFeatures
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_NAMES,
    DYNAMIC_EFFECT_FEATURE_SIZE,
)

if TYPE_CHECKING:
    from ptcg_rl.agent.probe import RuntimeProbeResult
    from ptcg_rl.engine.native_probe import (
        NativeProbeBackend,
        NativeProbeStats,
    )

ENGINE_FACT_SCHEMA_VERSION = 2
_ENGINE_FACT_DOMAIN = b"ptcg-rl/exact-prospective-engine-facts/v2\x00"
_ENGINE_FACT_CONTRACT_DOMAIN = (
    b"ptcg-rl/exact-prospective-engine-fact-contract/v1\x00"
)
_NATIVE_FACT_CONTRACT_VERSION = 1
_REQUEST_RNG_DOMAIN = b"ptcg-rl/prospective-engine-fact-request-rng/v1\x00"


class ProspectiveEngineFactConfig(BaseModel):
    """Validated fixed-budget producer configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    worlds: int = Field(default=2, ge=2, le=16)
    seed: int = Field(default=0, ge=0)
    manual_coin: bool = False
    require_byte_identical_worlds: bool = True
    sampler: BeliefSamplerConfig = Field(default_factory=BeliefSamplerConfig)


@dataclass(frozen=True)
class ProspectiveEngineFactResult:
    """Exact fact rows and auditable producer diagnostics."""

    features: tuple[tuple[float, ...], ...]
    masks: tuple[bool, ...]
    producer_fingerprint: str
    stats: NativeProbeStats

    def __post_init__(self) -> None:
        """Distinguish unavailable rows from resolved zero-effect rows."""
        if len(self.features) != len(self.masks):
            raise ValueError("engine fact features and masks are misaligned")
        if any(len(row) != DYNAMIC_EFFECT_FEATURE_SIZE for row in self.features):
            raise ValueError("engine fact feature width differs from schema")
        if len(self.producer_fingerprint) != 64:
            raise ValueError("engine fact producer identity must be SHA-256")

    def as_probe_result(self) -> RuntimeProbeResult:
        """Return the existing option-tensor injection boundary."""
        from ptcg_rl.agent.probe import RuntimeProbeResult

        return RuntimeProbeResult(
            features=self.features,
            masks=self.masks,
            world_vectors={},
            worlds_requested=self.stats.worlds,
            unresolved_options=self.stats.unresolved_options,
            unresolved_worlds=self.stats.unresolved_worlds,
        )


class ProspectiveEngineFactProducer:
    """Produce exact facts through the required native multi-root ABI."""

    def __init__(
        self,
        *,
        sampler: BeliefSampler,
        config: ProspectiveEngineFactConfig,
    ) -> None:
        """Bind one train/serve producer identity and deterministic requests."""
        from ptcg_rl.engine.native_probe import (
            NativeProbeBackend,
        )

        self.sampler = sampler
        self.config = config
        self.backend: NativeProbeBackend = NativeProbeBackend(
            manual_coin=config.manual_coin
        )
        contract_descriptor = {
            "schema_version": ENGINE_FACT_SCHEMA_VERSION,
            "native_fact_contract_version": _NATIVE_FACT_CONTRACT_VERSION,
            "config": config.model_dump(mode="json"),
            "sampler": sampler.semantic_fingerprint,
            "backend": self.backend.backend_name,
            "feature_names": DYNAMIC_EFFECT_FEATURE_NAMES,
            "aggregation": "byte_identical_float32_worlds_or_unavailable",
            "support": "policy_independent_root_single_option_attack_or_ability",
            "rng": "public_request_hash_v1",
        }
        self.contract_fingerprint = hashlib.sha256(
            _ENGINE_FACT_CONTRACT_DOMAIN
            + json.dumps(
                contract_descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        descriptor = {
            **contract_descriptor,
            "engine_abi_fingerprint": self.backend.engine_abi_fingerprint,
        }
        self.fingerprint = hashlib.sha256(
            _ENGINE_FACT_DOMAIN
            + json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    def run(
        self,
        observation: dict[str, Any],
        context_features: GameContextFeatures,
        *,
        your_deck: tuple[int, ...],
    ) -> ProspectiveEngineFactResult | None:
        """Return exact facts, leaving hidden/stochastic outcomes unavailable."""
        if not self.config.enabled:
            return None
        result = self.backend.run_exact_facts(
            observation,
            context_features,
            your_deck=your_deck,
            sampler=self.sampler,
            worlds=self.config.worlds,
            rng=_request_rng(
                observation,
                context_features,
                your_deck=your_deck,
                seed=self.config.seed,
            ),
            require_byte_identical_worlds=(
                self.config.require_byte_identical_worlds
            ),
        )
        if result is None:
            return None
        return ProspectiveEngineFactResult(
            features=result.features,
            masks=result.masks,
            producer_fingerprint=self.fingerprint,
            stats=result.stats,
        )


def _request_rng(
    observation: dict[str, Any],
    context_features: GameContextFeatures,
    *,
    your_deck: tuple[int, ...],
    seed: int,
) -> random.Random:
    """Return a scheduling-independent RNG for one public decision request."""
    raw_search_input = observation.get("search_begin_input")
    if isinstance(raw_search_input, bytes):
        request_bytes = raw_search_input
    elif isinstance(raw_search_input, str):
        request_bytes = raw_search_input.encode("ascii")
    else:
        request_bytes = json.dumps(
            observation.get("select"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    descriptor = {
        "seed": seed,
        "search_input_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "public_context": context_features.as_observation_dict(),
        "your_deck": list(your_deck),
    }
    payload = json.dumps(
        descriptor,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(_REQUEST_RNG_DOMAIN + payload).digest()
    return random.Random(digest)


__all__ = [
    "ENGINE_FACT_SCHEMA_VERSION",
    "ProspectiveEngineFactConfig",
    "ProspectiveEngineFactProducer",
    "ProspectiveEngineFactResult",
]
