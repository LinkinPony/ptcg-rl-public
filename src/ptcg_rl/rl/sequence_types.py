"""Small immutable identities shared by sequence collection and runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SequenceDecisionIdentity(BaseModel):
    """Absolute decision coordinates for one game and policy seat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    game_id: str
    seat: Literal[0, 1]
    decision_index: int = Field(ge=0)
    request_id: str

    @field_validator("game_id", "request_id")
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Reject ambiguous cache and duplicate-request identities."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("sequence decision identity cannot be empty")
        return normalized


__all__ = ["SequenceDecisionIdentity"]
