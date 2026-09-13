"""Deduplicated actual-trajectory rows for root-information value learning."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ptcg_rl.agent.search.root_information import RootInformationLeaf

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ExecutedEndpointValueRow:
    """One actual endpoint paired with its final root-perspective W/D/L."""

    game_fingerprint: str
    root_player: int
    semantic_endpoint_fingerprint: str
    leaf: RootInformationLeaf
    final_root_outcome: int

    def __post_init__(self) -> None:
        """Validate actual, public endpoint supervision."""
        _require_fingerprint(self.game_fingerprint)
        _require_fingerprint(self.semantic_endpoint_fingerprint)
        if self.root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        if self.semantic_endpoint_fingerprint != (
            self.leaf.information_history_fingerprint
        ):
            raise ValueError("semantic endpoint fingerprint differs from leaf")
        if self.final_root_outcome not in (-1, 0, 1):
            raise ValueError("final_root_outcome must be W/D/L from the root seat")


@dataclass(frozen=True, slots=True)
class ExecutedEndpointValueTable:
    """Frozen compact endpoint table referenced by decision-row indices."""

    rows: tuple[ExecutedEndpointValueRow, ...]


@dataclass(frozen=True, slots=True)
class _PendingEndpoint:
    game_fingerprint: str
    root_player: int
    leaf: RootInformationLeaf


class ExecutedEndpointValueTableBuilder:
    """Capture actual endpoints once, then attach final game outcomes."""

    def __init__(self) -> None:
        self._pending: list[_PendingEndpoint] = []
        self._index_by_key: dict[tuple[str, int, str], int] = {}
        self._outcomes: dict[tuple[str, int], int] = {}

    def register(
        self,
        *,
        game_fingerprint: str,
        root_player: int,
        leaf: RootInformationLeaf,
    ) -> int:
        """Return one stable index for an actually reached semantic endpoint."""
        _require_fingerprint(game_fingerprint)
        if root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        endpoint_fingerprint = leaf.information_history_fingerprint
        key = (game_fingerprint, root_player, endpoint_fingerprint)
        existing = self._index_by_key.get(key)
        if existing is not None:
            if self._pending[existing].leaf != leaf:
                raise ValueError(
                    "one semantic endpoint has different value-row inputs"
                )
            return existing
        index = len(self._pending)
        self._pending.append(
            _PendingEndpoint(
                game_fingerprint=game_fingerprint,
                root_player=root_player,
                leaf=leaf,
            )
        )
        self._index_by_key[key] = index
        return index

    def finalize_game(
        self,
        *,
        game_fingerprint: str,
        root_player_zero_outcome: int,
    ) -> None:
        """Attach one final result and its negation to both root perspectives."""
        _require_fingerprint(game_fingerprint)
        if root_player_zero_outcome not in (-1, 0, 1):
            raise ValueError("root_player_zero_outcome must be W/D/L")
        first = (game_fingerprint, 0)
        second = (game_fingerprint, 1)
        expected = {
            first: root_player_zero_outcome,
            second: -root_player_zero_outcome,
        }
        for key, outcome in expected.items():
            previous = self._outcomes.get(key)
            if previous is not None and previous != outcome:
                raise ValueError("game was finalized with a different outcome")
            self._outcomes[key] = outcome

    def freeze(self) -> ExecutedEndpointValueTable:
        """Return finalized rows or reject missing trajectory outcomes."""
        rows: list[ExecutedEndpointValueRow] = []
        for pending in self._pending:
            outcome = self._outcomes.get(
                (pending.game_fingerprint, pending.root_player)
            )
            if outcome is None:
                raise ValueError("executed endpoint is missing its final outcome")
            rows.append(
                ExecutedEndpointValueRow(
                    game_fingerprint=pending.game_fingerprint,
                    root_player=pending.root_player,
                    semantic_endpoint_fingerprint=(
                        pending.leaf.information_history_fingerprint
                    ),
                    leaf=pending.leaf,
                    final_root_outcome=outcome,
                )
            )
        return ExecutedEndpointValueTable(rows=tuple(rows))


def _require_fingerprint(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("expected a lowercase SHA-256 fingerprint")


__all__ = [
    "ExecutedEndpointValueRow",
    "ExecutedEndpointValueTable",
    "ExecutedEndpointValueTableBuilder",
]
