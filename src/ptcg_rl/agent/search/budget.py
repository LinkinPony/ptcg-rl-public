"""Whole-act deadlines, cumulative search accounting, and arena ActTime."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.agent.search.config import SearchBudgetConfig

Clock = Callable[[], float]
QuotaClass = Literal["ordinary_main", "first_high_value_main", "later_high_value_main"]


@dataclass(frozen=True)
class SearchBudgetPlan:
    """One absolute-deadline search allowance computed at an act boundary."""

    act_started_at: float
    whole_act_deadline: float
    search_soft_deadline: float
    remaining_overage_time: float
    nominal_quota_seconds: float
    quota_seconds: float
    taper_scale: float
    bank_before_seconds: float
    bank_left_seconds: float
    stop_reason: str | None = None

    def can_start(self, now: float, *, call_guard_seconds: float = 0.0) -> bool:
        """Return whether another non-interruptible unit fits the soft deadline."""
        return (
            self.stop_reason is None
            and self.quota_seconds > 0.0
            and now + max(0.0, call_guard_seconds) < self.search_soft_deadline
        )


class SearchBudgetManager:
    """Track the per-game search bank and compute deadline-safe quotas."""

    def __init__(
        self,
        config: SearchBudgetConfig,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.config = config
        self._clock = clock
        self._spent_seconds = 0.0

    @property
    def spent_seconds(self) -> float:
        """Return actual search wall time charged in this game."""
        return self._spent_seconds

    @property
    def bank_left_seconds(self) -> float:
        """Return the non-negative unspent search bank."""
        return max(0.0, self.config.bank_limit_seconds - self._spent_seconds)

    def reset(self) -> None:
        """Reset cumulative accounting for a new game."""
        self._spent_seconds = 0.0

    def plan(
        self,
        *,
        remaining_overage_time: float,
        act_started_at: float,
        quota_class: QuotaClass = "ordinary_main",
        now: float | None = None,
    ) -> SearchBudgetPlan:
        """Compute the documented reserve/taper quota and absolute deadlines."""
        current = self._clock() if now is None else float(now)
        remaining = max(0.0, float(remaining_overage_time))
        nominal = self._nominal_quota(quota_class)
        scale = min(
            1.0,
            max(
                0.0,
                (remaining - self.config.hard_reserve_seconds)
                / self.config.taper_window_seconds,
            ),
        )
        reserve_capacity = max(
            0.0,
            remaining
            - self.config.hard_reserve_seconds
            - self.config.uninterruptible_guard_seconds,
        )
        whole_act_capacity = max(
            0.0,
            remaining
            - self.config.hard_reserve_seconds
            - self.config.return_guard_seconds,
        )
        whole_act_deadline = act_started_at + min(
            self.config.whole_act_limit_seconds,
            whole_act_capacity,
        )
        deadline_capacity = max(
            0.0,
            whole_act_deadline - self.config.return_guard_seconds - current,
        )
        bank_before = self._spent_seconds
        bank_left = self.bank_left_seconds
        quota = min(
            nominal * scale,
            bank_left,
            reserve_capacity,
            deadline_capacity,
        )
        stop_reason: str | None = None
        if bank_left <= 0.0:
            stop_reason = "bank_exhausted"
        elif scale <= 0.0 or reserve_capacity <= 0.0:
            stop_reason = "hard_reserve"
        elif deadline_capacity <= 0.0:
            stop_reason = "whole_act_deadline"
        elif quota <= 0.0:
            stop_reason = "zero_quota"
        return SearchBudgetPlan(
            act_started_at=float(act_started_at),
            whole_act_deadline=whole_act_deadline,
            search_soft_deadline=current + quota,
            remaining_overage_time=remaining,
            nominal_quota_seconds=nominal,
            quota_seconds=quota,
            taper_scale=scale,
            bank_before_seconds=bank_before,
            bank_left_seconds=bank_left,
            stop_reason=stop_reason,
        )

    def track_search(self) -> SearchCharge:
        """Return an idempotent context manager that charges actual wall time once."""
        return SearchCharge(self, clock=self._clock)

    def charge(self, elapsed_seconds: float) -> None:
        """Charge one completed search interval."""
        elapsed = max(0.0, float(elapsed_seconds))
        if not math.isfinite(elapsed):
            raise ValueError("elapsed search time must be finite")
        self._spent_seconds += elapsed

    def _nominal_quota(self, quota_class: QuotaClass) -> float:
        if quota_class == "first_high_value_main":
            return self.config.first_high_value_main_quota_seconds
        if quota_class == "later_high_value_main":
            return self.config.later_high_value_main_quota_seconds
        return self.config.ordinary_main_quota_seconds


class SearchCharge:
    """Idempotent actual-time charge for one search callback."""

    def __init__(self, manager: SearchBudgetManager, *, clock: Clock) -> None:
        self._manager = manager
        self._clock = clock
        self._started_at: float | None = None
        self._elapsed_seconds = 0.0
        self._finished = False

    @property
    def elapsed_seconds(self) -> float:
        """Return the charged interval, or zero before completion."""
        return self._elapsed_seconds

    def __enter__(self) -> SearchCharge:
        self._started_at = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self.finish()

    def finish(self) -> float:
        """Charge elapsed time once even if cleanup paths call this repeatedly."""
        if self._finished:
            return self._elapsed_seconds
        if self._started_at is None:
            raise RuntimeError("search charge was not entered")
        self._elapsed_seconds = max(0.0, self._clock() - self._started_at)
        self._manager.charge(self._elapsed_seconds)
        self._finished = True
        return self._elapsed_seconds


class ActTimeLedgerConfig(BaseModel):
    """Kaggle-like per-seat cumulative overage configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    total_seconds: float = 600.0
    startup_charge_seconds: float = 20.0

    @field_validator("total_seconds", "startup_charge_seconds")
    @classmethod
    def finite_non_negative(cls, value: float) -> float:
        """Reject invalid ledger limits."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("ActTime ledger limits must be finite and non-negative")
        return value


class ActTimeTimeoutError(RuntimeError):
    """Raised when a seat exhausts its cumulative ActTime pool."""

    def __init__(self, seat: int, used_seconds: float) -> None:
        super().__init__(f"seat {seat} exhausted ActTime after {used_seconds:.6f}s")
        self.seat = int(seat)
        self.used_seconds = float(used_seconds)


class ActTimeLedger:
    """Maintain isolated per-seat overage and inject it before every callback."""

    def __init__(self, config: ActTimeLedgerConfig) -> None:
        self.config = config
        startup = config.startup_charge_seconds if config.enabled else 0.0
        self._used_seconds = [startup, startup]

    def remaining(self, seat: int) -> float:
        """Return the current non-negative remaining overage for one seat."""
        self._validate_seat(seat)
        return max(0.0, self.config.total_seconds - self._used_seconds[seat])

    def used(self, seat: int) -> float:
        """Return cumulative charged seconds, including startup."""
        self._validate_seat(seat)
        return self._used_seconds[seat]

    def observation_for(self, observation: Mapping[str, Any], seat: int) -> dict[str, Any]:
        """Copy an observation and inject the seat's live overage value."""
        if self.config.enabled and self.remaining(seat) <= 0.0:
            raise ActTimeTimeoutError(seat, self.used(seat))
        copied = dict(observation)
        copied["remainingOverageTime"] = self.remaining(seat)
        return copied

    def charge(self, seat: int, elapsed_seconds: float) -> None:
        """Charge a callback and fail immediately when it exhausts the pool."""
        self._validate_seat(seat)
        elapsed = max(0.0, float(elapsed_seconds))
        if not math.isfinite(elapsed):
            raise ValueError("ActTime elapsed seconds must be finite")
        if not self.config.enabled:
            return
        self._used_seconds[seat] += elapsed
        if self._used_seconds[seat] > self.config.total_seconds:
            raise ActTimeTimeoutError(seat, self._used_seconds[seat])

    @staticmethod
    def _validate_seat(seat: int) -> None:
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat}")
