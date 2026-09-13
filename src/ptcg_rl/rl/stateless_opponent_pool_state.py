"""Versioned exact-resume state for stateless historical opponent pools."""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.rl.opponent_pool import LeagueState, QuotaStratum
from ptcg_rl.rl.opponent_pool._identity import (
    cached_fingerprint,
    canonical_fingerprint,
    normalize_sha256,
)
from ptcg_rl.rl.opponent_pool.adaptive import (
    AdaptiveMatchupEvidence,
    AdaptiveTargetWeight,
    PortfolioName,
)
from ptcg_rl.rl.opponent_pool.role_budget import (
    ROLE_ORDER,
    ROLE_TARGET_SHARES,
    RoleBudgetName,
)


class StratumDecisionTotal(BaseModel):
    """Accepted trainable decisions in one effective quota epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stratum: QuotaStratum
    decisions: int = Field(default=0, ge=0)


class StatelessOpponentPoolLineageState(BaseModel):
    """Exact-resume envelope for full-lineage scheduling state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    league_state: LeagueState
    founder_policy_sha256: str
    effective_strata: tuple[QuotaStratum, ...] = ()
    stratum_decisions: tuple[StratumDecisionTotal, ...] = ()

    @field_validator("founder_policy_sha256")
    @classmethod
    def valid_founder(cls, value: str) -> str:
        """Require an immutable source policy identity."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("founder policy identity must be lowercase SHA-256")
        return normalized

    @model_validator(mode="after")
    def canonical_coverage(self) -> Self:
        """Keep effective quota rows aligned and canonical."""
        names = tuple(item.stratum for item in self.stratum_decisions)
        if names != self.effective_strata or len(set(names)) != len(names):
            raise ValueError("lineage stratum coverage differs from effective strata")
        return self

    @property
    def fingerprint(self) -> str:
        """Bind active graph, founder, quota epoch, and accepted decisions."""
        domain = "stateless-opponent-pool-lineage-state-v2"
        return cached_fingerprint(
            domain,
            self,
            lambda: canonical_fingerprint(domain, self.model_dump(mode="json")),
        )


class StatelessOpponentPoolAdaptiveState(BaseModel):
    """Exact-resume envelope for joint evidence-driven allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3, 4, 5] = 3
    league_state: LeagueState
    founder_policy_sha256: str
    evidence: tuple[AdaptiveMatchupEvidence, ...] = ()
    allocation_decision_clock: int = Field(default=0, ge=0)
    last_target_weights: tuple[AdaptiveTargetWeight, ...] = ()
    candidate_target_shares: dict[str, float] = Field(default_factory=dict)
    portfolio_target_mass: dict[PortfolioName, float] = Field(default_factory=dict)
    portfolio_decisions: dict[PortfolioName, int] = Field(default_factory=dict)
    role_target_mass: dict[RoleBudgetName, float] = Field(default_factory=dict)
    role_decisions: dict[RoleBudgetName, int] = Field(default_factory=dict)
    last_target_fingerprint: str | None = None

    @field_validator("founder_policy_sha256")
    @classmethod
    def valid_founder(cls, value: str) -> str:
        """Require an immutable source policy identity."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("founder policy identity must be lowercase SHA-256")
        return normalized

    @field_validator("last_target_fingerprint")
    @classmethod
    def valid_optional_target(cls, value: str | None) -> str | None:
        """Validate the last committed target identity when present."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("adaptive target identity must be lowercase SHA-256")
        return normalized

    @field_validator("candidate_target_shares")
    @classmethod
    def valid_candidate_targets(cls, value: dict[str, float]) -> dict[str, float]:
        """Require one optional positive candidate simplex."""
        if not value:
            return value
        normalized = {normalize_sha256(key): item for key, item in value.items()}
        if (
            len(normalized) != len(value)
            or any(
                not math.isfinite(item) or item <= 0.0 for item in normalized.values()
            )
            or abs(sum(normalized.values()) - 1.0) > 1e-9
        ):
            raise ValueError("adaptive candidate targets must form a simplex")
        return normalized

    @field_validator("portfolio_target_mass")
    @classmethod
    def valid_portfolio_targets(
        cls,
        value: dict[PortfolioName, float],
    ) -> dict[PortfolioName, float]:
        """Require an optional normalized target diagnostic."""
        if not value:
            return value
        expected = {"counter", "frontier", "probe", "rehearsal", "staleness"}
        if (
            set(value) != expected
            or any(not math.isfinite(item) or item < 0.0 for item in value.values())
            or abs(sum(value.values()) - 1.0) > 1e-9
        ):
            raise ValueError("adaptive portfolio targets must form a simplex")
        return value

    @field_validator("portfolio_decisions")
    @classmethod
    def non_negative_portfolio_decisions(
        cls,
        value: dict[PortfolioName, int],
    ) -> dict[PortfolioName, int]:
        """Reject corrupt decision totals."""
        expected = {"counter", "frontier", "probe", "rehearsal", "staleness"}
        if value and (
            set(value) != expected or any(item < 0 for item in value.values())
        ):
            raise ValueError("adaptive portfolio decision diagnostics are invalid")
        return value

    @field_validator("role_target_mass")
    @classmethod
    def valid_role_targets(
        cls,
        value: dict[RoleBudgetName, float],
    ) -> dict[RoleBudgetName, float]:
        """Require the fixed V4 role budget when it has been committed."""
        if value and (
            set(value) != set(ROLE_ORDER)
            or any(
                not math.isclose(
                    value[name],
                    ROLE_TARGET_SHARES[name],
                    abs_tol=1e-12,
                )
                for name in ROLE_ORDER
            )
        ):
            raise ValueError("role-budget target diagnostics are invalid")
        return value

    @field_validator("role_decisions")
    @classmethod
    def non_negative_role_decisions(
        cls,
        value: dict[RoleBudgetName, int],
    ) -> dict[RoleBudgetName, int]:
        """Reject incomplete or corrupt V4 decision ledgers."""
        if value and (
            set(value) != set(ROLE_ORDER) or any(item < 0 for item in value.values())
        ):
            raise ValueError("role-budget decision diagnostics are invalid")
        return value

    @model_validator(mode="after")
    def canonical_rows(self) -> Self:
        """Keep long-lived evidence and target rows unique and deterministic."""
        evidence_keys = tuple(item.identity.evidence_key for item in self.evidence)
        if evidence_keys != tuple(sorted(evidence_keys)) or len(evidence_keys) != len(
            set(evidence_keys)
        ):
            raise ValueError("adaptive evidence must be canonical")
        target_keys = tuple(item.matchup_key for item in self.last_target_weights)
        if target_keys != tuple(sorted(target_keys)) or len(target_keys) != len(
            set(target_keys)
        ):
            raise ValueError("adaptive target weights must be canonical")
        if (self.last_target_fingerprint is None) != (not self.last_target_weights):
            raise ValueError("adaptive target identity differs from target rows")
        if self.schema_version == 3:
            if self.role_target_mass or self.role_decisions:
                raise ValueError("V3 adaptive state cannot contain V4 role ledgers")
            if self.last_target_fingerprint is not None and (
                not self.candidate_target_shares or not self.portfolio_target_mass
            ):
                raise ValueError("adaptive committed target diagnostics are incomplete")
            if (
                self.last_target_fingerprint is not None
                and not self.portfolio_decisions
            ):
                raise ValueError(
                    "adaptive portfolio decision diagnostics are incomplete"
                )
            return self
        if self.portfolio_target_mass or self.portfolio_decisions:
            raise ValueError("role-budget state cannot contain V3 portfolios")
        if set(self.role_decisions) != set(ROLE_ORDER):
            raise ValueError("role-budget decision ledger is incomplete")
        if self.last_target_fingerprint is not None and (
            not self.candidate_target_shares or not self.role_target_mass
        ):
            raise ValueError("role-budget committed target diagnostics are incomplete")
        return self

    @property
    def fingerprint(self) -> str:
        """Bind the active graph and every allocation sufficient statistic."""
        domain = f"stateless-opponent-pool-adaptive-state-v{self.schema_version}"

        def compute() -> str:
            payload = self.model_dump(mode="json")
            if self.schema_version == 3:
                # Preserve exact fingerprints for checkpoints written before V4
                # added its separate role ledger to this shared envelope class.
                payload.pop("role_target_mass")
                payload.pop("role_decisions")
            return canonical_fingerprint(domain, payload)

        return cached_fingerprint(domain, self, compute)


OpponentPoolCheckpointState = (
    LeagueState | StatelessOpponentPoolLineageState | StatelessOpponentPoolAdaptiveState
)


def opponent_pool_checkpoint_state(value: Any) -> OpponentPoolCheckpointState:
    """Dispatch versioned state without importing the runtime bridge."""
    if not isinstance(value, dict):
        raise ValueError("opponent-pool state must be a mapping")
    schema_version = int(value.get("schema_version", 1))
    if schema_version in {3, 4, 5}:
        return StatelessOpponentPoolAdaptiveState.model_validate(value)
    if schema_version == 2:
        return StatelessOpponentPoolLineageState.model_validate(value)
    return LeagueState.model_validate(value)


__all__ = [
    "OpponentPoolCheckpointState",
    "StatelessOpponentPoolAdaptiveState",
    "StatelessOpponentPoolLineageState",
    "StratumDecisionTotal",
    "opponent_pool_checkpoint_state",
]
