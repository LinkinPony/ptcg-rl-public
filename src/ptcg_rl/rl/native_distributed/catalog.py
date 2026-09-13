"""Metadata-only past-self route ownership for the H200 coordinator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ptcg_rl.rl.stateless_curriculum import PfspMember


@dataclass(frozen=True, slots=True)
class _PreparedMetadataRoutes:
    catalog: MetadataPastSelfRouteCatalog
    members: tuple[PfspMember, ...]

    def commit(self) -> None:
        """Publish member metadata without constructing an inference model."""
        self.catalog._commit(self.members)

    def abort(self) -> None:
        """Discard uncommitted metadata."""
        return None


class MetadataPastSelfRouteCatalog:
    """Bind controller-verified policies without learner-side model residency."""

    def __init__(self) -> None:
        self._members: dict[str, PfspMember] = {}

    def prepare(
        self,
        members: Sequence[PfspMember],
    ) -> _PreparedMetadataRoutes:
        """Stage metadata whose bytes the curriculum controller already verified."""
        staged = tuple(member for member in members if member.source == "past_self")
        for member in staged:
            if member.pair is None:
                raise ValueError("past-self metadata member has no policy artifact")
        return _PreparedMetadataRoutes(self, staged)

    def unload(self, member_ids: Sequence[str]) -> None:
        """Forget retired member metadata after controller lease drain."""
        for member_id in member_ids:
            self._members.pop(member_id, None)

    @property
    def loaded_member_ids(self) -> frozenset[str]:
        """Return metadata bindings visible to the controller."""
        return frozenset(self._members)

    def _commit(self, members: Sequence[PfspMember]) -> None:
        staged = dict(self._members)
        for member in members:
            existing = staged.get(member.member_id)
            if existing is not None and existing != member:
                raise ValueError("past-self metadata member changed in place")
            staged[member.member_id] = member
        self._members = staged


__all__ = ["MetadataPastSelfRouteCatalog"]
