"""Shared opponent-pool registry types and lookup helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.engine.protocols import ObservationInput

OpponentSource = Literal["builtin", "third_party", "public_opponent", "policy"]


class BattleAgent(Protocol):
    """Unified local-battle agent interface."""

    name: str

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Return selected option indices for the current prompt."""

    def reset(self) -> None:
        """Reset any per-game state before starting a fresh battle."""


class OpponentSpec(BaseModel):
    """One registered opponent entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tier: int
    source: OpponentSource
    deck_path: Path | None = None
    vector_safe: bool = False
    requires_search: bool = False
    checkpoint_path: Path | None = None
    device: str = "cpu"
    belief_summary_path: Path | None = None

    @field_validator("tier")
    @classmethod
    def valid_tier(cls, value: int) -> int:
        """Reject tier values outside the planned gauntlet range."""
        if value < 0 or value > 4:
            raise ValueError("tier must be in [0, 4]")
        return value

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("opponent device must be non-empty")
        return cleaned


class PolicyOpponentConfig(BaseModel):
    """Hydra-configurable frozen checkpoint opponent."""

    model_config = ConfigDict(extra="forbid")

    name: str
    checkpoint_path: Path
    deck_path: Path | None = None
    tier: int = 2
    vector_safe: bool = True
    requires_search: bool = False
    device: str = "cpu"
    belief_summary_path: Path | None = None

    @field_validator("tier")
    @classmethod
    def valid_tier(cls, value: int) -> int:
        """Reject tier values outside the planned gauntlet range."""
        if value < 0 or value > 4:
            raise ValueError("tier must be in [0, 4]")
        return value

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("policy opponent device must be non-empty")
        return cleaned


class OpponentPoolConfig(BaseModel):
    """Hydra-facing opponent-pool filter."""

    model_config = ConfigDict(extra="forbid")

    include_tiers: tuple[int, ...] = (0, 1, 2, 3, 4)
    include_names: tuple[str, ...] | None = None
    exclude_names: tuple[str, ...] = ()
    policy_opponents: tuple[PolicyOpponentConfig, ...] = Field(default_factory=tuple)

    @field_validator("include_tiers")
    @classmethod
    def valid_include_tiers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject tier filters outside the planned gauntlet range."""
        invalid = [tier for tier in value if tier < 0 or tier > 4]
        if invalid:
            raise ValueError(f"include_tiers contains invalid tiers: {invalid}")
        return value


def build_opponent(spec: OpponentSpec, *, seed: int = 0) -> BattleAgent:
    """Build a fresh agent for one opponent spec."""
    if spec.source in {"builtin", "policy"}:
        from ptcg_rl.opponents.builtin import build_builtin_opponent

        return build_builtin_opponent(spec, seed=seed)
    if spec.source == "third_party":
        from ptcg_rl.opponents import third_party

        if spec.name == "heuristic":
            return third_party.build_heuristic_agent(spec.name)
        if spec.name.startswith("archaludon_lean_"):
            return third_party.build_archaludon_lean_agent(spec.name)
        if spec.name == "mcts":
            return third_party.build_mcts_agent(spec.name)
        raise KeyError(f"unknown third-party opponent: {spec.name}")
    if spec.source == "public_opponent":
        from ptcg_rl.opponents import third_party

        return third_party.build_public_agent(spec.name, seed=seed)
    raise ValueError(f"unsupported opponent source: {spec.source}")


def opponent_registry(
    *,
    policy_opponents: Sequence[PolicyOpponentConfig] = (),
) -> dict[str, OpponentSpec]:
    """Return registered opponents keyed by globally unique name."""
    from ptcg_rl.opponents import third_party

    specs: list[OpponentSpec] = [
        OpponentSpec(
            name="random",
            tier=0,
            source="builtin",
            vector_safe=True,
        ),
        OpponentSpec(
            name="end",
            tier=0,
            source="builtin",
            vector_safe=True,
        ),
        OpponentSpec(
            name="mixed25",
            tier=1,
            source="builtin",
            vector_safe=True,
        ),
        OpponentSpec(
            name="mixed50",
            tier=1,
            source="builtin",
            vector_safe=True,
        ),
        OpponentSpec(
            name="mixed75",
            tier=1,
            source="builtin",
            vector_safe=True,
        ),
        OpponentSpec(
            name="heuristic",
            tier=2,
            source="third_party",
            vector_safe=True,
        ),
        OpponentSpec(
            name="archaludon_lean_db1a",
            tier=2,
            source="third_party",
            deck_path=Path(
                "docs/experiments/rl_dynamic_deck_pool_20260708/decks/"
                "35_archaludon_relicanth_xerosic_db1a949a2bca.csv"
            ),
            vector_safe=True,
        ),
        OpponentSpec(
            name="archaludon_lean_cf28",
            tier=2,
            source="third_party",
            deck_path=Path(
                "docs/experiments/rl_71eb_archaludon_dual_mainline_20260713/"
                "decks/cf28f33153b7.csv"
            ),
            vector_safe=True,
        ),
        OpponentSpec(
            name="archaludon_lean_ea6a",
            tier=2,
            source="third_party",
            deck_path=Path(
                "docs/experiments/rl_71eb_recent_sparring_20260712/"
                "decks/ea6a498ffa65.csv"
            ),
            vector_safe=True,
        ),
        OpponentSpec(
            name="archaludon_lean_9f25",
            tier=2,
            source="third_party",
            deck_path=Path(
                "docs/experiments/rl_71eb_recent_sparring_20260712/"
                "decks/9f25834ed732.csv"
            ),
            vector_safe=True,
        ),
    ]
    specs.extend(
        OpponentSpec(
            name=name,
            tier=3,
            source="public_opponent",
            deck_path=third_party.public_deck_path(name),
            vector_safe=False,
            requires_search=third_party.public_requires_search(name),
        )
        for name in sorted(third_party.PUBLIC_OPPONENT_FIXTURES)
    )
    specs.append(
        OpponentSpec(
            name="mcts",
            tier=4,
            source="third_party",
            vector_safe=False,
            requires_search=True,
        )
    )
    specs.extend(_policy_spec(config) for config in policy_opponents)
    return _spec_dict(specs)


def opponents_by_tier(
    max_tier: int | None = None,
    *,
    registry: dict[str, OpponentSpec] | None = None,
) -> list[OpponentSpec]:
    """Return registered opponents ordered by tier and name."""
    specs = list((registry or opponent_registry()).values())
    if max_tier is not None:
        specs = [spec for spec in specs if spec.tier <= max_tier]
    return sorted(specs, key=lambda spec: (spec.tier, spec.name))


def select_opponents(config: OpponentPoolConfig) -> list[OpponentSpec]:
    """Apply Hydra pool filters and return ordered opponent specs."""
    registry = opponent_registry(policy_opponents=config.policy_opponents)
    excluded = set(config.exclude_names)
    if config.include_names is not None:
        names = [name for name in config.include_names if name not in excluded]
        return [registry[name] for name in names]

    included_tiers = set(config.include_tiers)
    return [
        spec
        for spec in opponents_by_tier(registry=registry)
        if spec.tier in included_tiers and spec.name not in excluded
    ]


def _policy_spec(config: PolicyOpponentConfig) -> OpponentSpec:
    return OpponentSpec(
        name=config.name,
        tier=config.tier,
        source="policy",
        deck_path=config.deck_path,
        vector_safe=config.vector_safe,
        requires_search=config.requires_search,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        belief_summary_path=config.belief_summary_path,
    )


def _spec_dict(specs: Sequence[OpponentSpec]) -> dict[str, OpponentSpec]:
    output: dict[str, OpponentSpec] = {}
    for spec in specs:
        if spec.name in output:
            raise ValueError(f"duplicate opponent name: {spec.name}")
        output[spec.name] = spec
    return output
