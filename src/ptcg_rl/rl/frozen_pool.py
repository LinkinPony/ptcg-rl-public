"""Frozen checkpoint policy pool for curriculum rollout."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    build_agent_policy_value_net,
)
from ptcg_rl.rl.collection import ModelRolloutPolicy
from ptcg_rl.rl.curriculum import FrozenPoolMember
from ptcg_rl.rl.inference_snapshot_router import InferencePolicySnapshotRouter
from ptcg_rl.rl.rollout import RolloutPolicy

FrozenPolicyPrecision = Literal["bf16", "fp32"]
_SAMPLING_SEED_DOMAIN = b"ptcg-rl/frozen-policy-sampling/v1\x00"


class FrozenPolicyPoolConfig(BaseModel):
    """Config for frozen-checkpoint policy loading."""

    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    precision: FrozenPolicyPrecision = "bf16"
    strict: bool = False
    load_timeout_seconds: float = 25.0
    max_recurrent_sequence_leases: int = 8192
    recurrent_replay_cache_capacity: int = 4096

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("device must be non-empty")
        return cleaned

    @field_validator(
        "max_recurrent_sequence_leases",
        "recurrent_replay_cache_capacity",
    )
    @classmethod
    def valid_positive_capacity(cls, value: int) -> int:
        """Require positive bounded recurrent runtime capacities."""
        if value <= 0:
            raise ValueError("recurrent frozen-policy capacities must be positive")
        return value

    @field_validator("load_timeout_seconds")
    @classmethod
    def valid_load_timeout_seconds(cls, value: float) -> float:
        """Require one finite positive deadline for staged policy loading."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("frozen-policy load timeout must be finite and positive")
        return value


@dataclass(frozen=True)
class FrozenPolicyPoolUpdate:
    """Summary from syncing loaded policies with frozen members."""

    loaded: tuple[str, ...]
    unloaded: tuple[str, ...]
    kept: tuple[str, ...]


@dataclass(frozen=True)
class PreparedFrozenPolicyPoolUpdate:
    """Fully loaded next pool generation awaiting one atomic commit."""

    entries: tuple[tuple[str, _LoadedPolicy], ...]
    update: FrozenPolicyPoolUpdate


@dataclass(frozen=True)
class _LoadedPolicy:
    member: FrozenPoolMember
    policy: RolloutPolicy


FrozenPolicyLoader = Callable[
    [FrozenPoolMember, FrozenPolicyPoolConfig, AgentNetworkConfig],
    RolloutPolicy,
]


class FrozenPolicyPool:
    """Keep frozen rollout policies loaded for the active curriculum pool."""

    def __init__(
        self,
        *,
        model_config: AgentNetworkConfig,
        config: FrozenPolicyPoolConfig | None = None,
        loader: FrozenPolicyLoader | None = None,
    ) -> None:
        """Initialize an empty loaded-policy pool."""
        self.model_config = model_config
        self.config = config or FrozenPolicyPoolConfig()
        self._loader = loader or load_frozen_rollout_policy
        self._loaded: dict[str, _LoadedPolicy] = {}
        self._known_identities: dict[str, tuple[Path, bool]] = {}
        self._lock = threading.RLock()

    @property
    def policies(self) -> Mapping[str, RolloutPolicy]:
        """Return loaded policies keyed by frozen opponent id."""
        with self._lock:
            return {
                opponent_id: loaded.policy
                for opponent_id, loaded in self._loaded.items()
            }

    def sync(self, members: Sequence[FrozenPoolMember]) -> FrozenPolicyPoolUpdate:
        """Load new members and unload removed members."""
        prepared = self.prepare_sync(members)
        return self.commit_prepared(prepared)

    def prepare_sync(
        self,
        members: Sequence[FrozenPoolMember],
    ) -> PreparedFrozenPolicyPoolUpdate:
        """Load a complete next generation without mutating live routes."""
        desired = {member.opponent_id: member for member in members}
        with self._lock:
            current = dict(self._loaded)
            known_identities = dict(self._known_identities)
        for opponent_id, member in desired.items():
            known_identity = known_identities.get(opponent_id)
            if known_identity is not None and known_identity != _member_identity(
                member
            ):
                raise ValueError(
                    "frozen opponent identity is immutable; use a new opponent_id "
                    f"for checkpoint or topology changes: {opponent_id}"
                )
        unloaded: list[str] = []
        next_loaded: dict[str, _LoadedPolicy] = {}
        for opponent_id, loaded in current.items():
            desired_member = desired.get(opponent_id)
            if desired_member is not None:
                next_loaded[opponent_id] = loaded
                continue
            unloaded.append(opponent_id)

        loaded_ids: list[str] = []
        kept: list[str] = []
        for opponent_id, member in desired.items():
            if opponent_id in next_loaded:
                kept.append(opponent_id)
                continue
            policy = self._loader(member, self.config, self.model_config)
            next_loaded[opponent_id] = _LoadedPolicy(member=member, policy=policy)
            loaded_ids.append(opponent_id)

        return PreparedFrozenPolicyPoolUpdate(
            entries=tuple(next_loaded.items()),
            update=FrozenPolicyPoolUpdate(
                loaded=tuple(loaded_ids),
                unloaded=tuple(unloaded),
                kept=tuple(kept),
            ),
        )

    def commit_prepared(
        self,
        prepared: PreparedFrozenPolicyPoolUpdate,
    ) -> FrozenPolicyPoolUpdate:
        """Atomically publish one fully prepared pool generation."""
        with self._lock:
            for opponent_id, loaded in prepared.entries:
                identity = _member_identity(loaded.member)
                known_identity = self._known_identities.get(opponent_id)
                if known_identity is not None and known_identity != identity:
                    raise ValueError(
                        "frozen opponent identity changed before staged commit; "
                        f"use a new opponent_id: {opponent_id}"
                    )
                self._known_identities[opponent_id] = identity
            self._loaded = dict(prepared.entries)
        return prepared.update

    def policy_for(self, opponent_id: str) -> RolloutPolicy:
        """Return one loaded frozen policy by opponent id."""
        with self._lock:
            try:
                return self._loaded[opponent_id].policy
            except KeyError as exc:
                raise KeyError(f"frozen policy is not loaded: {opponent_id}") from exc


def _checkpoint_path_identity(path: Path) -> Path:
    """Resolve one configured checkpoint path for immutable route identity."""
    return deck_records.repo_path(path).resolve()


def _member_identity(member: FrozenPoolMember) -> tuple[Path, bool]:
    """Return the immutable wire/runtime identity of a frozen route."""
    return (_checkpoint_path_identity(member.checkpoint_path), member.recurrent)


def load_frozen_rollout_policy(
    member: FrozenPoolMember,
    config: FrozenPolicyPoolConfig,
    fallback_model_config: AgentNetworkConfig,
    *,
    sampling_seed: int | None = None,
) -> RolloutPolicy:
    """Load one frozen checkpoint as an eval-mode rollout policy."""
    device = _resolve_device(config.device)
    checkpoint_path = deck_records.repo_path(member.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = _checkpoint_model_config(checkpoint) or fallback_model_config
    model = build_agent_policy_value_net(model_config).to(device)
    incompatible = model.load_state_dict(
        _checkpoint_state_dict(checkpoint),
        strict=False,
    )
    _raise_unexpected_incompatibility(
        incompatible,
        allow_legacy_aux_head=config.strict is False,
    )
    model.eval()
    generator = (
        None
        if sampling_seed is None
        else _frozen_policy_generator(
            device,
            base_seed=sampling_seed,
            opponent_id=member.opponent_id,
        )
    )
    policy = ModelRolloutPolicy(
        model,
        policy_version=0,
        autocast="bf16" if config.precision == "bf16" else "off",
        generator=generator,
    )
    if not member.recurrent:
        return policy
    if not policy.recurrent_enabled:
        raise RuntimeError(
            "frozen member declares recurrence but checkpoint is stateless"
        )
    return InferencePolicySnapshotRouter(
        policy,
        max_resident_snapshots=1,
        max_in_flight_leases=config.max_recurrent_sequence_leases,
        max_recurrent_sequence_leases=config.max_recurrent_sequence_leases,
        recurrent_replay_cache_capacity=config.recurrent_replay_cache_capacity,
    )


def _frozen_policy_generator(
    device: torch.device,
    *,
    base_seed: int,
    opponent_id: str,
) -> torch.Generator:
    digest = hashlib.sha256()
    digest.update(_SAMPLING_SEED_DOMAIN)
    digest.update(base_seed.to_bytes(16, "big", signed=True))
    digest.update(opponent_id.encode("utf-8"))
    generator = torch.Generator(device=device)
    generator.manual_seed(int.from_bytes(digest.digest()[:8], "big"))
    return generator


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    elif normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "agent_network_config", "network_config", "model"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _raise_unexpected_incompatibility(
    incompatible: Any,
    *,
    allow_legacy_aux_head: bool,
) -> None:
    allowed_missing = set(LEGACY_STATE_ENCODER_MISSING_KEYS)
    if allow_legacy_aux_head:
        allowed_missing.update(
            {
                "opponent_hand_head.weight",
                "opponent_hand_head.bias",
            }
        )
    missing = {str(key) for key in incompatible.missing_keys}
    unexpected = {str(key) for key in incompatible.unexpected_keys}
    if missing - allowed_missing or unexpected:
        raise RuntimeError("checkpoint state dict is incompatible with rollout model")
