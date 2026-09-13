"""Pointer-style option encoding for Kaggle select prompts.

This module turns engine-provided legal ``Option`` objects into structural
features for a future policy head. It does not infer legality or card effects;
the bundled engine remains the source of truth for those semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ptcg_rl.context import GameContextFeatures, context_features_from_observation
from ptcg_rl.engine.constants import AreaType, OptionType
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.protocols import ObservationInput, SelectDataLike

MISSING_TOKEN_INDEX = -1
VIRTUAL_AREA = 0
OWN_UNSEEN_CONTEXT_INDEX_BASE = 10_000
OPP_REVEALED_CONTEXT_INDEX_BASE = 20_000
OPP_BELIEF_CONTEXT_INDEX_BASE = 30_000
LEGACY_SCALAR_FEATURE_SIZE = 5
ATTACHMENT_IDENTITY_FEATURE_SIZE = 4
SCALAR_FEATURE_SIZE = LEGACY_SCALAR_FEATURE_SIZE + ATTACHMENT_IDENTITY_FEATURE_SIZE
ENTITY_SLOT_FEATURE_SIZE = 2
ZERO_SCALARS = (0.0,) * SCALAR_FEATURE_SIZE
ZERO_DYNAMIC_EFFECT_FEATURES = (0.0,) * DYNAMIC_EFFECT_FEATURE_SIZE


@dataclass(frozen=True)
class StateTokenKey:
    """Stable lookup key for one entity token in the encoded state."""

    area: int
    player_index: int
    index: int
    sub_index: int = 0


@dataclass(frozen=True)
class StateToken:
    """A visible or virtual entity token before tensor projection."""

    kind: str
    key: StateTokenKey
    card_id: int = 0
    serial: int = 0
    count: float = 0.0
    last_attack_id: int = 0


@dataclass(frozen=True)
class AttachmentIdentity:
    """Public identity of one attached card within its parent Pokemon."""

    card_id: int
    serial: int


@dataclass(frozen=True)
class StateTokenLayout:
    """Entity-token lookup shared by state encoding and option pointers."""

    tokens: tuple[StateToken, ...]
    token_by_key: Mapping[StateTokenKey, int]
    token_by_serial: Mapping[tuple[int, int], int]
    attachment_by_key: Mapping[tuple[int, int, int], AttachmentIdentity]
    your_index: int
    global_token: int
    special_condition_token: int
    context_features: GameContextFeatures

    @classmethod
    def from_observation(
        cls,
        observation: ObservationInput,
        *,
        context_features: GameContextFeatures | None = None,
    ) -> StateTokenLayout:
        """Build a layout from an engine observation or observation mapping."""
        return build_state_token_layout(observation, context_features=context_features)

    def token_index(
        self,
        area: int | None,
        player_index: int | None,
        index: int | None,
        *,
        sub_index: int = 0,
    ) -> int | None:
        """Return a token index for an area pointer, if it can be resolved."""
        if area is None or index is None:
            return None
        normalized_player = self._normalize_player(player_index)
        keys = (
            StateTokenKey(int(area), normalized_player, int(index), sub_index),
            StateTokenKey(int(area), -1, int(index), sub_index),
        )
        for key in keys:
            token_index = self.token_by_key.get(key)
            if token_index is not None:
                return token_index
        return None

    def token_for_serial(
        self,
        card_id: int,
        serial: int | None,
        *,
        player_index: int | None = None,
    ) -> int | None:
        """Return a visible-card token for a serial, falling back to any owner."""
        if serial is None or serial <= 0:
            return None
        normalized_player = self._normalize_player(player_index)
        token_index = self.token_by_serial.get(
            (normalized_player, int(serial)),
            self.token_by_serial.get((-1, int(serial))),
        )
        if token_index is None:
            return None
        token_card_id = self.tokens[token_index].card_id
        if card_id > 0 and token_card_id not in (0, card_id):
            return None
        return token_index

    def attachment_identity(
        self,
        parent_token: int | None,
        attachment_kind: int,
        attachment_index: int | None,
    ) -> AttachmentIdentity | None:
        """Resolve one public attached-card identity from an engine option."""
        if parent_token is None or attachment_index is None:
            return None
        return self.attachment_by_key.get(
            (int(parent_token), int(attachment_kind), int(attachment_index))
        )

    def _normalize_player(self, player_index: int | None) -> int:
        if player_index is None or player_index < 0:
            return self.your_index
        return int(player_index)


@dataclass(frozen=True)
class EncodedOption:
    """One legal option encoded for pointer policy scoring."""

    option_type: int
    context: int
    entity_slots: tuple[int, ...]
    attack_id: int = 0
    card_id: int = 0
    scalars: tuple[float, ...] = ZERO_SCALARS
    dynamic_effect_features: tuple[float, ...] = ZERO_DYNAMIC_EFFECT_FEATURES
    dynamic_effect_mask: bool = False


@dataclass(frozen=True)
class EncodedOptionArrayFeatures:
    """Numpy-backed encoded legal options for one select prompt."""

    option_types: np.ndarray
    contexts: np.ndarray
    entity_slots: np.ndarray
    entity_slot_mask: np.ndarray
    attack_ids: np.ndarray
    card_ids: np.ndarray
    scalars: np.ndarray
    dynamic_effect_features: np.ndarray
    dynamic_effect_masks: np.ndarray

    def __len__(self) -> int:
        """Return encoded option count."""
        return int(self.option_types.shape[0])

    def to_options(self) -> tuple[EncodedOption, ...]:
        """Return tuple-backed compatibility options."""
        options: list[EncodedOption] = []
        for row_index in range(len(self)):
            slots = tuple(
                int(slot)
                for slot, present in zip(
                    self.entity_slots[row_index],
                    self.entity_slot_mask[row_index],
                    strict=True,
                )
                if bool(present)
            )
            options.append(
                EncodedOption(
                    option_type=int(self.option_types[row_index]),
                    context=int(self.contexts[row_index]),
                    entity_slots=slots,
                    attack_id=int(self.attack_ids[row_index]),
                    card_id=int(self.card_ids[row_index]),
                    scalars=tuple(float(value) for value in self.scalars[row_index]),
                    dynamic_effect_features=tuple(
                        float(value)
                        for value in self.dynamic_effect_features[row_index]
                    ),
                    dynamic_effect_mask=bool(self.dynamic_effect_masks[row_index]),
                )
            )
        return tuple(options)


EncodedOptionInput = Sequence[EncodedOption] | EncodedOptionArrayFeatures


SelectInput = SelectDataLike | Mapping[str, Any]


def build_state_token_layout(
    observation: ObservationInput,
    *,
    context_features: GameContextFeatures | None = None,
) -> StateTokenLayout:
    """Build deterministic entity-token indices for visible observation state."""
    current = _field(observation, "current")
    select = _field(observation, "select")
    active_context = context_features or context_features_from_observation(observation)
    your_index = _int_field(current, "yourIndex", 0)
    builder = _LayoutBuilder(your_index=your_index)
    builder.context_last_attack_by_serial = active_context.last_attack_by_serial()
    global_token = builder.add_virtual("global", index=0)
    special_token = builder.add_virtual("special_condition", index=1)

    _add_current_state_tokens(builder, current)
    _add_select_tokens(builder, select)
    _add_context_tokens(builder, active_context)
    return StateTokenLayout(
        tokens=tuple(builder.tokens),
        token_by_key=dict(builder.token_by_key),
        token_by_serial=dict(builder.token_by_serial),
        attachment_by_key=dict(builder.attachment_by_key),
        your_index=your_index,
        global_token=global_token,
        special_condition_token=special_token,
        context_features=active_context,
    )


def encode_options(
    select: SelectInput | None,
    layout: StateTokenLayout,
) -> tuple[EncodedOption, ...]:
    """Encode every legal engine option for a select prompt."""
    if select is None:
        return ()
    context = _int_field(select, "context", 0)
    return tuple(
        _encode_option(option, context=context, layout=layout)
        for option in _sequence(_field(select, "option", ()))
    )


def encode_option_arrays(
    select: SelectInput | None,
    layout: StateTokenLayout,
) -> EncodedOptionArrayFeatures:
    """Encode every legal engine option into numpy arrays."""
    if select is None:
        return _empty_option_arrays()
    context = _int_field(select, "context", 0)
    raw_options = tuple(_sequence(_field(select, "option", ())))
    option_count = len(raw_options)
    option_types = np.zeros(option_count, dtype=np.int64)
    contexts = np.zeros(option_count, dtype=np.int64)
    entity_slots = np.zeros(
        (option_count, ENTITY_SLOT_FEATURE_SIZE),
        dtype=np.int64,
    )
    entity_slot_mask = np.zeros(
        (option_count, ENTITY_SLOT_FEATURE_SIZE),
        dtype=np.bool_,
    )
    attack_ids = np.zeros(option_count, dtype=np.int64)
    card_ids = np.zeros(option_count, dtype=np.int64)
    scalars = np.zeros((option_count, SCALAR_FEATURE_SIZE), dtype=np.float32)
    dynamic_effect_features = np.zeros(
        (option_count, DYNAMIC_EFFECT_FEATURE_SIZE),
        dtype=np.float32,
    )
    dynamic_effect_masks = np.zeros(option_count, dtype=np.bool_)
    for option_index, option in enumerate(raw_options):
        encoded = _encode_option(option, context=context, layout=layout)
        option_types[option_index] = int(encoded.option_type)
        contexts[option_index] = int(encoded.context)
        slots = tuple(
            int(slot) for slot in encoded.entity_slots[:ENTITY_SLOT_FEATURE_SIZE]
        )
        if slots:
            entity_slots[option_index, : len(slots)] = slots
            entity_slot_mask[option_index, : len(slots)] = True
        attack_ids[option_index] = max(0, int(encoded.attack_id))
        card_ids[option_index] = max(0, int(encoded.card_id))
        scalars[option_index, :] = encoded.scalars
        dynamic_effect_features[option_index, :] = encoded.dynamic_effect_features
        dynamic_effect_masks[option_index] = bool(encoded.dynamic_effect_mask)
    return EncodedOptionArrayFeatures(
        option_types=option_types,
        contexts=contexts,
        entity_slots=entity_slots,
        entity_slot_mask=entity_slot_mask,
        attack_ids=attack_ids,
        card_ids=card_ids,
        scalars=scalars,
        dynamic_effect_features=dynamic_effect_features,
        dynamic_effect_masks=dynamic_effect_masks,
    )


def _empty_option_arrays() -> EncodedOptionArrayFeatures:
    return EncodedOptionArrayFeatures(
        option_types=np.zeros(0, dtype=np.int64),
        contexts=np.zeros(0, dtype=np.int64),
        entity_slots=np.zeros((0, ENTITY_SLOT_FEATURE_SIZE), dtype=np.int64),
        entity_slot_mask=np.zeros((0, ENTITY_SLOT_FEATURE_SIZE), dtype=np.bool_),
        attack_ids=np.zeros(0, dtype=np.int64),
        card_ids=np.zeros(0, dtype=np.int64),
        scalars=np.zeros((0, SCALAR_FEATURE_SIZE), dtype=np.float32),
        dynamic_effect_features=np.zeros(
            (0, DYNAMIC_EFFECT_FEATURE_SIZE),
            dtype=np.float32,
        ),
        dynamic_effect_masks=np.zeros(0, dtype=np.bool_),
    )


def _encode_option(
    option: Any,
    *,
    context: int,
    layout: StateTokenLayout,
) -> EncodedOption:
    option_type = _int_field(option, "type", -1)
    card_id = _int_field(option, "cardId", 0)
    attack_id = _int_field(option, "attackId", 0)
    attachment_serial = 0
    attachment_serial_present = False
    slots: tuple[int, ...]

    if option_type == int(OptionType.PLAY):
        slots = _slots(
            layout.token_index(
                AreaType.HAND,
                layout.your_index,
                _int_or_none(option, "index"),
            )
        )
    elif option_type in {
        int(OptionType.CARD),
        int(OptionType.TOOL_CARD),
        int(OptionType.ENERGY_CARD),
        int(OptionType.ENERGY),
    }:
        parent_token = layout.token_index(
            _int_or_none(option, "area"),
            _int_or_none(option, "playerIndex"),
            _int_or_none(option, "index"),
        )
        slots = _slots(parent_token)
        attachment_kind, attachment_index = _option_attachment_pointer(
            option,
            option_type=option_type,
        )
        attachment = layout.attachment_identity(
            parent_token,
            attachment_kind,
            attachment_index,
        )
        if attachment is not None:
            card_id = attachment.card_id
            attachment_serial = attachment.serial
            attachment_serial_present = True
    elif option_type in {int(OptionType.ATTACH), int(OptionType.EVOLVE)}:
        slots = _slots(
            layout.token_index(
                _int_or_none(option, "area"),
                layout.your_index,
                _int_or_none(option, "index"),
            ),
            layout.token_index(
                _int_or_none(option, "inPlayArea"),
                layout.your_index,
                _int_or_none(option, "inPlayIndex"),
            ),
        )
    elif option_type in {int(OptionType.ABILITY), int(OptionType.DISCARD)}:
        slots = _slots(
            layout.token_index(
                _int_or_none(option, "area"),
                layout.your_index,
                _int_or_none(option, "index"),
            )
        )
    elif option_type == int(OptionType.ATTACK):
        slots = _slots(layout.token_index(AreaType.ACTIVE, layout.your_index, 0))
    elif option_type == int(OptionType.SKILL):
        slots = (
            (layout.special_condition_token,)
            if card_id == 0
            else _slots(
                layout.token_for_serial(
                    card_id,
                    _int_or_none(option, "serial"),
                    player_index=layout.your_index,
                )
            )
        )
    else:
        slots = ()

    scalars = _scalar_fields(
        option,
        attachment_serial=attachment_serial,
        attachment_serial_present=attachment_serial_present,
    )

    return EncodedOption(
        option_type=option_type,
        context=context,
        entity_slots=slots,
        attack_id=attack_id,
        card_id=card_id,
        scalars=scalars,
    )


def _add_current_state_tokens(builder: _LayoutBuilder, current: Any) -> None:
    players = list(_sequence(_field(current, "players", ())))
    last_attack_by_serial = builder.context_last_attack_by_serial
    for player_index in _player_iteration_order(players, builder.your_index):
        player = players[player_index]
        _add_card_sequence(
            builder,
            kind="active",
            area=int(AreaType.ACTIVE),
            cards=_sequence(_field(player, "active", ())),
            default_player=player_index,
            last_attack_by_serial=last_attack_by_serial,
        )
        _add_card_sequence(
            builder,
            kind="bench",
            area=int(AreaType.BENCH),
            cards=_sequence(_field(player, "bench", ())),
            default_player=player_index,
            last_attack_by_serial=last_attack_by_serial,
        )
        _add_card_sequence(
            builder,
            kind="hand",
            area=int(AreaType.HAND),
            cards=_sequence(_field(player, "hand", ())),
            default_player=player_index,
        )
        _add_card_sequence(
            builder,
            kind="discard",
            area=int(AreaType.DISCARD),
            cards=_sequence(_field(player, "discard", ())),
            default_player=player_index,
        )
        _add_card_sequence(
            builder,
            kind="prize",
            area=int(AreaType.PRIZE),
            cards=_sequence(_field(player, "prize", ())),
            default_player=player_index,
        )

    _add_card_sequence(
        builder,
        kind="stadium",
        area=int(AreaType.STADIUM),
        cards=_sequence(_field(current, "stadium", ())),
        default_player=-1,
        owner_aliases=(-1, builder.your_index, 1 - builder.your_index),
    )
    _add_card_sequence(
        builder,
        kind="looking",
        area=int(AreaType.LOOKING),
        cards=_sequence(_field(current, "looking", ())),
        default_player=builder.your_index,
    )


def _player_iteration_order(players: Sequence[Any], your_index: int) -> tuple[int, ...]:
    if 0 <= your_index < len(players):
        return (your_index,) + tuple(
            player_index
            for player_index in range(len(players))
            if player_index != your_index
        )
    return tuple(range(len(players)))


def _add_select_tokens(builder: _LayoutBuilder, select: Any) -> None:
    _add_card_sequence(
        builder,
        kind="deck",
        area=int(AreaType.DECK),
        cards=_sequence(_field(select, "deck", ())),
        default_player=builder.your_index,
    )
    for index, field_name in enumerate(("contextCard", "effect")):
        card = _object_or_none(_field(select, field_name))
        if card is None:
            continue
        builder.add_card(
            kind=field_name,
            key=StateTokenKey(
                area=VIRTUAL_AREA,
                player_index=_int_field(card, "playerIndex", builder.your_index),
                index=index,
            ),
            card=card,
        )


def _add_context_tokens(
    builder: _LayoutBuilder,
    context_features: GameContextFeatures,
) -> None:
    for index, unseen_item in enumerate(context_features.own_unseen):
        builder.add_card(
            kind="own_unseen",
            key=StateTokenKey(
                area=VIRTUAL_AREA,
                player_index=builder.your_index,
                index=OWN_UNSEEN_CONTEXT_INDEX_BASE + index,
            ),
            card=None,
            card_id=unseen_item.card_id,
            count=unseen_item.count,
        )
    opponent_index = 1 - builder.your_index if builder.your_index in (0, 1) else -1
    for index, revealed_item in enumerate(context_features.opponent_revealed):
        builder.add_card(
            kind="opponent_revealed",
            key=StateTokenKey(
                area=VIRTUAL_AREA,
                player_index=opponent_index,
                index=OPP_REVEALED_CONTEXT_INDEX_BASE + index,
            ),
            card=None,
            card_id=revealed_item.card_id,
            count=revealed_item.count,
        )
    for index, belief_item in enumerate(context_features.opponent_belief):
        builder.add_card(
            kind="opponent_belief",
            key=StateTokenKey(
                area=VIRTUAL_AREA,
                player_index=opponent_index,
                index=OPP_BELIEF_CONTEXT_INDEX_BASE + index,
            ),
            card=None,
            card_id=belief_item.card_id,
            count=belief_item.expected_count,
        )


def _add_card_sequence(
    builder: _LayoutBuilder,
    *,
    kind: str,
    area: int,
    cards: Sequence[Any],
    default_player: int,
    owner_aliases: Sequence[int] = (),
    last_attack_by_serial: Mapping[int, int] | None = None,
) -> None:
    for index, raw_card in enumerate(cards):
        card = _object_or_none(raw_card)
        player_index = (
            _int_field(card, "playerIndex", default_player)
            if card is not None
            else default_player
        )
        token_index = builder.add_card(
            kind=kind,
            key=StateTokenKey(area=area, player_index=player_index, index=index),
            card=card,
            last_attack_id=(
                0
                if card is None or last_attack_by_serial is None
                else last_attack_by_serial.get(_int_field(card, "serial", 0), 0)
            ),
        )
        if card is not None and kind in {"active", "bench"}:
            builder.add_attachments(token_index, card)
        aliases = set(owner_aliases)
        aliases.add(default_player)
        aliases.add(player_index)
        for alias_player in aliases:
            builder.add_alias(
                StateTokenKey(area=area, player_index=alias_player, index=index),
                token_index,
            )


class _LayoutBuilder:
    """Mutable helper used only while building an immutable layout."""

    def __init__(self, *, your_index: int) -> None:
        self.your_index = your_index
        self.tokens: list[StateToken] = []
        self.token_by_key: dict[StateTokenKey, int] = {}
        self.token_by_serial: dict[tuple[int, int], int] = {}
        self.attachment_by_key: dict[tuple[int, int, int], AttachmentIdentity] = {}
        self.context_last_attack_by_serial: Mapping[int, int] = {}

    def add_virtual(self, kind: str, *, index: int) -> int:
        """Add a virtual non-card token."""
        return self.add_card(
            kind=kind,
            key=StateTokenKey(area=VIRTUAL_AREA, player_index=-1, index=index),
            card=None,
        )

    def add_card(
        self,
        *,
        kind: str,
        key: StateTokenKey,
        card: Any | None,
        card_id: int | None = None,
        count: float = 0.0,
        last_attack_id: int = 0,
    ) -> int:
        """Add a card-like token, returning the existing index if present."""
        existing = self.token_by_key.get(key)
        if existing is not None:
            return existing
        token_index = len(self.tokens)
        token = StateToken(
            kind=kind,
            key=key,
            card_id=_int_field(card, "id", 0) if card_id is None else int(card_id),
            serial=_int_field(card, "serial", 0),
            count=max(0.0, float(count)),
            last_attack_id=max(0, int(last_attack_id)),
        )
        self.tokens.append(token)
        self.token_by_key[key] = token_index
        if token.serial > 0:
            self.token_by_serial.setdefault(
                (key.player_index, token.serial),
                token_index,
            )
            self.token_by_serial.setdefault((-1, token.serial), token_index)
        return token_index

    def add_alias(self, key: StateTokenKey, token_index: int) -> None:
        """Register another lookup key for an existing token."""
        self.token_by_key.setdefault(key, token_index)

    def add_attachments(self, parent_token: int, pokemon: Any) -> None:
        """Index every visible attachment by parent, kind, and engine index."""
        attachment_fields = (
            ("energyCards", int(OptionType.ENERGY_CARD)),
            ("tools", int(OptionType.TOOL_CARD)),
        )
        for field_name, attachment_kind in attachment_fields:
            for attachment_index, card in enumerate(
                _sequence(_field(pokemon, field_name, ()))
            ):
                card_id = _int_field(card, "id", 0)
                serial = _int_field(card, "serial", 0)
                if card_id <= 0:
                    continue
                self.attachment_by_key[
                    (parent_token, attachment_kind, attachment_index)
                ] = AttachmentIdentity(card_id=card_id, serial=serial)


def _slots(*values: int | None) -> tuple[int, ...]:
    return tuple(value for value in values if value is not None)


def _option_attachment_pointer(
    option: Any,
    *,
    option_type: int,
) -> tuple[int, int | None]:
    """Return the canonical attachment kind and engine-local index."""
    if option_type == int(OptionType.TOOL_CARD):
        return int(OptionType.TOOL_CARD), _int_or_none(option, "toolIndex")
    if option_type in {int(OptionType.ENERGY_CARD), int(OptionType.ENERGY)}:
        return int(OptionType.ENERGY_CARD), _int_or_none(option, "energyIndex")
    return -1, None


def _scalar_fields(
    option: Any,
    *,
    attachment_serial: int,
    attachment_serial_present: bool,
) -> tuple[float, ...]:
    energy_index = _field(option, "energyIndex")
    tool_index = _field(option, "toolIndex")
    return (
        _normalized(_field(option, "number"), 10.0),
        _normalized(_field(option, "count"), 10.0),
        _normalized(energy_index, 16.0),
        _normalized(tool_index, 8.0),
        _normalized(_field(option, "specialConditionType"), 4.0),
        1.0 if energy_index is not None else 0.0,
        1.0 if tool_index is not None else 0.0,
        float(attachment_serial) / 128.0,
        1.0 if attachment_serial_present else 0.0,
    )


def _normalized(value: Any, scale: float) -> float:
    if value is None:
        return 0.0
    return float(value) / scale


def _object_or_none(value: Any) -> Any | None:
    return None if value is None else value


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default


def _int_or_none(value: Any, name: str) -> int | None:
    field_value = _field(value, name)
    return int(field_value) if field_value is not None else None
