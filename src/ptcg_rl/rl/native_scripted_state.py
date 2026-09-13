"""Minimal native-column state access for the frozen scripted opponent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptcg_rl.engine.constants import AreaType, OptionType
from ptcg_rl.engine.native_training import NativeTrainingBatchView
from ptcg_rl.rl.native_scripted_catalog import MIXED75_71EB_CARD_IDS

_VIRTUAL_AREA = 0
_MISSING_ROW = np.iinfo(np.uint32).max
_READY_STATUS = 1
_NO_ERROR = 0


@dataclass(frozen=True)
class NativeScriptedOption:
    """One legal engine option backed by the native fixed-width parameters."""

    option_type: int
    p0: int
    p1: int
    p2: int
    p3: int
    p4: int

    @property
    def number(self) -> int | None:
        return self.p0 if self.option_type == int(OptionType.NUMBER) else None

    @property
    def attack_id(self) -> int | None:
        return self.p0 if self.option_type == int(OptionType.ATTACK) else None

    @property
    def card_id(self) -> int | None:
        return self.p0 if self.option_type == int(OptionType.SKILL) else None

    @property
    def area(self) -> int | None:
        if self.option_type in {
            int(OptionType.CARD),
            int(OptionType.TOOL_CARD),
            int(OptionType.ENERGY_CARD),
            int(OptionType.ENERGY),
            int(OptionType.ATTACH),
            int(OptionType.EVOLVE),
            int(OptionType.ABILITY),
            int(OptionType.DISCARD),
        }:
            return self.p0
        return None

    @property
    def index(self) -> int | None:
        if self.option_type == int(OptionType.PLAY):
            return self.p0
        if self.area is not None:
            return self.p1
        return None

    @property
    def player_index(self) -> int | None:
        if self.option_type in {
            int(OptionType.CARD),
            int(OptionType.TOOL_CARD),
            int(OptionType.ENERGY_CARD),
            int(OptionType.ENERGY),
        }:
            return self.p2
        return None

    @property
    def in_play_area(self) -> int | None:
        if self.option_type in {int(OptionType.ATTACH), int(OptionType.EVOLVE)}:
            return self.p2
        return None

    @property
    def in_play_index(self) -> int | None:
        if self.in_play_area is not None:
            return self.p3
        return None

    @property
    def count(self) -> int | None:
        return self.p4 if self.option_type == int(OptionType.ENERGY) else None


class NativeScriptedRow:
    """Validated lightweight view over one native arena output row."""

    def __init__(self, view: NativeTrainingBatchView, row: int) -> None:
        if row < 0 or row >= view.batch_size:
            raise IndexError(f"native scripted row is out of range: {row}")
        if (
            int(view.status[row]) != _READY_STATUS
            or int(view.error[row]) != _NO_ERROR
        ):
            raise ValueError("native scripted policy requires a ready engine row")
        perspective = int(view.select_player[row])
        if perspective not in (0, 1):
            raise ValueError("native scripted row has no acting perspective")
        self.view = view
        self.row = row
        self.perspective = perspective
        self.opponent = 1 - perspective
        self.select_type = int(view.select_type[row]) - 1
        self.context = int(view.select_context[row]) - 1
        self.minimum = int(view.select_min[row])
        self.maximum = int(view.select_max[row])
        self._option_start = int(view.option_offsets[row])
        self._option_stop = int(view.option_offsets[row + 1])
        self._visible_start = int(view.visible_card_offsets[row])
        self._visible_stop = int(view.visible_card_offsets[row + 1])
        self._attachment_start = int(view.attachment_offsets[row])
        self._attachment_stop = int(view.attachment_offsets[row + 1])
        self._card_rows: dict[tuple[int, int, int], int] | None = None
        self._area_rows: dict[tuple[int, int], tuple[int, ...]] | None = None
        self._attachment_rows: dict[int, tuple[int, ...]] | None = None
        self._validate_bounds()
        self._validate_own_identity()

    @property
    def option_count(self) -> int:
        return self._option_stop - self._option_start

    def option(self, index: int) -> NativeScriptedOption:
        """Return one legal option without allocating an observation object."""
        if index < 0 or index >= self.option_count:
            raise IndexError(f"native option is out of range: {index}")
        absolute = self._option_start + index
        params = self.view.option_params
        return NativeScriptedOption(
            option_type=int(self.view.option_type[absolute]),
            p0=int(params[0][absolute]),
            p1=int(params[1][absolute]),
            p2=int(params[2][absolute]),
            p3=int(params[3][absolute]),
            p4=int(params[4][absolute]),
        )

    def options(self) -> tuple[NativeScriptedOption, ...]:
        """Return all legal options in stable engine order."""
        return tuple(self.option(index) for index in range(self.option_count))

    def player_deck_count(self, player: int) -> int:
        return int(self.view.player_deck_counts[player][self.row])

    def player_hand_count(self, player: int) -> int:
        return int(self.view.player_hand_counts[player][self.row])

    def player_prize_count(self, player: int) -> int:
        return int(self.view.player_prize_counts[player][self.row])

    def player_bench_max(self, player: int) -> int:
        return int(self.view.player_bench_max[player][self.row])

    def card_row(
        self,
        area: int,
        index: int,
        player: int | None = None,
    ) -> int | None:
        """Resolve a public card pointer exactly as the legacy chooser does."""
        if area == int(AreaType.STADIUM):
            return self._row_for_area_index(area, index)
        if area == int(AreaType.LOOKING):
            return self._row_for_area_index(area, index)
        if area == int(AreaType.DECK):
            return self._row_for_area_index(area, index)
        resolved_player = self.perspective if player is None else int(player)
        self._ensure_card_index()
        assert self._card_rows is not None
        return self._card_rows.get((resolved_player, area, index))

    def card_id(
        self,
        area: int,
        index: int,
        player: int | None = None,
    ) -> int | None:
        row = self.card_row(area, index, player)
        if row is None:
            return None
        card_id = int(self.view.visible_card_id[row])
        return card_id if card_id > 0 else None

    def card_id_from_option(self, option: NativeScriptedOption) -> int | None:
        """Match ``HeuristicChooser._card_id_from_option`` exactly."""
        if option.card_id:
            return option.card_id
        if option.option_type == int(OptionType.PLAY) and option.index is not None:
            return self.card_id(
                int(AreaType.HAND),
                option.index,
                self.perspective,
            )
        if option.area is None or option.index is None:
            return None
        return self.card_id(option.area, option.index, option.player_index)

    def pokemon_from_option(
        self,
        option: NativeScriptedOption,
    ) -> int | None:
        """Return the absolute visible row of an option's Pokémon pointer."""
        if (
            option.in_play_area is not None
            and option.in_play_index is not None
        ):
            return self.pokemon_row(
                option.in_play_area,
                option.in_play_index,
                option.player_index,
            )
        if option.area is None or option.index is None:
            return None
        return self.pokemon_row(
            option.area,
            option.index,
            option.player_index,
        )

    def pokemon_row(
        self,
        area: int,
        index: int,
        player: int | None = None,
    ) -> int | None:
        if area not in {int(AreaType.ACTIVE), int(AreaType.BENCH)}:
            return None
        row = self.card_row(area, index, player)
        if row is None or int(self.view.visible_card_id[row]) <= 0:
            return None
        return row

    def pokemon_rows(self, player: int) -> tuple[int, ...]:
        """Return non-hidden Active then Bench Pokémon in engine list order."""
        self._ensure_area_index()
        assert self._area_rows is not None
        return (
            *self._area_rows.get((player, int(AreaType.ACTIVE)), ()),
            *self._area_rows.get((player, int(AreaType.BENCH)), ()),
        )

    def active_row(self, player: int) -> int | None:
        rows = self._rows_for_player_area(player, int(AreaType.ACTIVE))
        return rows[0] if rows else None

    def bench_rows(self, player: int) -> tuple[int, ...]:
        return self._rows_for_player_area(player, int(AreaType.BENCH))

    def hand_card_ids(self, player: int) -> tuple[int, ...]:
        return self._card_ids_for_player_area(player, int(AreaType.HAND))

    def discard_card_ids(self, player: int) -> tuple[int, ...]:
        return self._card_ids_for_player_area(player, int(AreaType.DISCARD))

    def pokemon_id(self, pokemon_row: int) -> int:
        self._require_pokemon_row(pokemon_row)
        return int(self.view.visible_card_id[pokemon_row])

    def pokemon_serial(self, pokemon_row: int) -> int:
        self._require_pokemon_row(pokemon_row)
        return int(self.view.visible_card_serial[pokemon_row])

    def pokemon_hp(self, pokemon_row: int) -> int:
        self._require_pokemon_row(pokemon_row)
        return int(self.view.visible_card_hp[pokemon_row])

    def pokemon_max_hp(self, pokemon_row: int) -> int:
        self._require_pokemon_row(pokemon_row)
        return int(self.view.visible_card_max_hp[pokemon_row])

    def pokemon_energies(self, pokemon_row: int) -> tuple[int, ...]:
        """Return effective energy types expanded by engine-projected units."""
        energies: list[int] = []
        for attachment in self._attachments_for(pokemon_row):
            if int(self.view.attachment_kind[attachment]) != 1:
                continue
            energy_type = int(self.view.attachment_energy_type[attachment])
            units = int(self.view.attachment_energy_units[attachment])
            if energy_type < 0 or units < 0:
                raise ValueError("native energy attachment is malformed")
            energies.extend([energy_type] * units)
        return tuple(energies)

    def pokemon_attachment_ids(
        self,
        pokemon_row: int,
        *,
        kind: int,
    ) -> tuple[int, ...]:
        return tuple(
            int(self.view.attachment_card_id[attachment])
            for attachment in self._attachments_for(pokemon_row)
            if int(self.view.attachment_kind[attachment]) == kind
        )

    def is_your_active(self, pokemon_row: int) -> bool:
        active = self.active_row(self.perspective)
        return (
            active is not None
            and self.pokemon_serial(active) == self.pokemon_serial(pokemon_row)
        )

    def context_card_id(self) -> int | None:
        absolute = int(self.view.context_card_row[self.row])
        if absolute == _MISSING_ROW:
            return None
        if absolute < self._visible_start or absolute >= self._visible_stop:
            raise ValueError("native context-card pointer crosses its row")
        if (
            int(self.view.visible_card_area[absolute]) != _VIRTUAL_AREA
            or int(self.view.visible_card_area_index[absolute]) != 0
        ):
            raise ValueError("native context-card pointer is malformed")
        card_id = int(self.view.visible_card_id[absolute])
        return card_id if card_id > 0 else None

    def _rows_for_player_area(
        self,
        player: int,
        area: int,
    ) -> tuple[int, ...]:
        self._ensure_area_index()
        assert self._area_rows is not None
        return self._area_rows.get((player, area), ())

    def _card_ids_for_player_area(
        self,
        player: int,
        area: int,
    ) -> tuple[int, ...]:
        return tuple(
            int(self.view.visible_card_id[row])
            for row in self._rows_for_player_area(player, area)
            if int(self.view.visible_card_id[row]) > 0
        )

    def _row_for_area_index(self, area: int, index: int) -> int | None:
        matches = [
            row
            for row in range(self._visible_start, self._visible_stop)
            if int(self.view.visible_card_area[row]) == area
            and int(self.view.visible_card_area_index[row]) == index
        ]
        if len(matches) > 1:
            raise ValueError("native public card pointer is ambiguous")
        return matches[0] if matches else None

    def _attachments_for(self, pokemon_row: int) -> tuple[int, ...]:
        self._require_pokemon_row(pokemon_row)
        if self._attachment_rows is None:
            by_parent: dict[int, list[int]] = {}
            for attachment in range(
                self._attachment_start,
                self._attachment_stop,
            ):
                parent = int(self.view.attachment_parent[attachment])
                if parent < self._visible_start or parent >= self._visible_stop:
                    raise ValueError(
                        "native attachment pointer crosses its state row"
                    )
                by_parent.setdefault(parent, []).append(attachment)
            self._attachment_rows = {
                parent: tuple(rows) for parent, rows in by_parent.items()
            }
        return self._attachment_rows.get(pokemon_row, ())

    def _ensure_card_index(self) -> None:
        if self._card_rows is not None:
            return
        rows: dict[tuple[int, int, int], int] = {}
        for absolute in range(self._visible_start, self._visible_stop):
            owner = int(self.view.visible_card_owner[absolute])
            area = int(self.view.visible_card_area[absolute])
            index = int(self.view.visible_card_area_index[absolute])
            key = (owner, area, index)
            if key in rows:
                raise ValueError("native public card pointer is duplicated")
            rows[key] = absolute
        self._card_rows = rows

    def _ensure_area_index(self) -> None:
        if self._area_rows is not None:
            return
        grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for absolute in range(self._visible_start, self._visible_stop):
            owner = int(self.view.visible_card_owner[absolute])
            area = int(self.view.visible_card_area[absolute])
            if area not in {int(AreaType.ACTIVE), int(AreaType.BENCH)}:
                continue
            index = int(self.view.visible_card_area_index[absolute])
            if int(self.view.visible_card_id[absolute]) <= 0:
                continue
            grouped.setdefault((owner, area), []).append((index, absolute))
        self._area_rows = {
            key: tuple(row for _index, row in sorted(values))
            for key, values in grouped.items()
        }

    def _require_pokemon_row(self, row: int) -> None:
        if row < self._visible_start or row >= self._visible_stop:
            raise ValueError("native Pokémon pointer crosses its state row")
        if int(self.view.visible_card_area[row]) not in {
            int(AreaType.ACTIVE),
            int(AreaType.BENCH),
        }:
            raise ValueError("native pointer does not reference a Pokémon")
        if int(self.view.visible_card_id[row]) <= 0:
            raise ValueError("native pointer references a hidden Pokémon")

    def _validate_bounds(self) -> None:
        if not (0 <= self.minimum <= self.maximum <= self.option_count):
            raise ValueError("native select cardinality is malformed")
        if not (
            0 <= self._visible_start <= self._visible_stop
            <= self.view.visible_card_count
        ):
            raise ValueError("native visible-card offsets are malformed")
        if not (
            0 <= self._attachment_start <= self._attachment_stop
            <= self.view.attachment_count
        ):
            raise ValueError("native attachment offsets are malformed")

    def _validate_own_identity(self) -> None:
        for absolute in range(self._visible_start, self._visible_stop):
            if int(self.view.visible_card_owner[absolute]) != self.perspective:
                continue
            card_id = int(self.view.visible_card_id[absolute])
            if card_id > 0 and card_id not in MIXED75_71EB_CARD_IDS:
                raise ValueError(
                    "mixed75_71eb public state contains a non-artifact own "
                    f"card: {card_id}"
                )
        for attachment in range(
            self._attachment_start,
            self._attachment_stop,
        ):
            parent = int(self.view.attachment_parent[attachment])
            if (
                self._visible_start <= parent < self._visible_stop
                and int(self.view.visible_card_owner[parent]) == self.perspective
            ):
                card_id = int(self.view.attachment_card_id[attachment])
                if card_id not in MIXED75_71EB_CARD_IDS:
                    raise ValueError(
                        "mixed75_71eb attachment contains a non-artifact own "
                        f"card: {card_id}"
                    )


__all__ = ["NativeScriptedOption", "NativeScriptedRow"]
